"""
model.py — Alpamayo VLA Model  (DGX / V100 build)

Dtype policy (CANONICAL — see CLAUDE.MD):
  - Frozen Cosmos backbone (LM + visual encoder) stays float16.
  - Trainable adapters (LoRA A/B, ego_encoder, traj_embed, output_head) are
    float32 MASTER WEIGHTS. Mandatory because torch.amp.GradScaler refuses to
    unscale fp16 grads ("Attempting to unscale FP16 gradients"). The fp16<->fp32
    boundary is handled by explicit casts in LoRALinear.forward / AlpamayoVLA.forward.
  - No bf16 (V100/Volta has no bf16 hardware). No FlashAttention (no Volta support).

Augmentation:
  - The token-space perturbation below (token dropout / gaussian / shuffle) is the
    LEGACY approach and is OFF by default (self.augment=False). The canonical
    augmentation is now IMAGE-SPACE photometric jitter applied in the dataloader
    BEFORE the frozen encoder (see vision_live.py / dataset.py).
"""

import sys
import math
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')

COSMOS_PATH  = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'
TEXT_DIM     = 3584
EGO_DIM      = 4
TRAJ_VOCAB   = 129
TRAJ_LEN     = 24
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
EGO_NOISE    = 0.05   # Gaussian noise std on ego features during training

# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, linear, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)
        self.lora_A  = nn.Linear(linear.in_features,  rank,                bias=False)
        self.lora_B  = nn.Linear(rank,                linear.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x):
        # base path runs in the frozen backbone's dtype (fp16); LoRA path runs in
        # the adapter's fp32 master dtype, then casts back to the base dtype.
        base = self.linear(x)
        lora = self.lora_B(self.dropout(self.lora_A(x.to(self.lora_A.weight.dtype))))
        return base + (lora * self.scaling).to(base.dtype)


def apply_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
    """Apply LoRA to q_proj, v_proj, o_proj in all LM attention layers."""
    count = 0
    for layer in model.model.language_model.layers:
        attn = layer.self_attn
        for name in ('q_proj', 'v_proj', 'o_proj'):
            if hasattr(attn, name):
                original = getattr(attn, name)
                lora     = LoRALinear(original, rank, alpha, dropout)
                dev      = next(original.parameters()).device
                # adapters are fp32 master weights (NOT the backbone dtype)
                lora.lora_A.to(device=dev, dtype=torch.float32)
                lora.lora_B.to(device=dev, dtype=torch.float32)
                setattr(attn, name, lora)
                count += 1
    return count

# ── Ego Encoder ───────────────────────────────────────────────────────────────

class EgoEncoderMLP(nn.Module):
    """[B, 4, 4] -> [B, 4, 3584]"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EGO_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, TEXT_DIM),
        )

    def forward(self, x):
        return self.net(x)

# ── Main Model ────────────────────────────────────────────────────────────────

class AlpamayoVLA(nn.Module):
    """
    Forward:
        visual_tokens [B, 1536, 3584]  float16 (live-encoded, frozen Cosmos visual)
        ego_state     [B, 4, 4]
        traj_tokens   [B, 24]          long
    Returns:
        logits [B, 24, 129]
    """

    def __init__(self, cosmos):
        super().__init__()
        self.cosmos      = cosmos
        self.ego_encoder = EgoEncoderMLP()
        self.traj_embed  = nn.Embedding(TRAJ_VOCAB, TEXT_DIM)
        self.output_head = nn.Linear(TEXT_DIM, TRAJ_VOCAB, bias=False)
        # Legacy token-space augmentation. OFF by default — canonical aug is
        # image-space photometric (dataloader). Kept for ablation only.
        self.augment = False

    def _build_context(self, visual_tokens, ego_state):
        vis = visual_tokens.to(dtype=torch.float16)

        if self.training and self.augment:
            # 1. Token dropout — zero out 15% of visual tokens
            mask = (torch.rand(vis.shape[0], vis.shape[1], 1,
                    device=vis.device) > 0.15).to(dtype=vis.dtype)
            vis = vis * mask

            # 2. Gaussian noise sigma=0.02
            vis = vis + torch.randn_like(vis) * 0.02

            # 3. Shuffle 5% of token positions
            B, N, D = vis.shape
            n_shuffle = max(1, int(N * 0.05))
            for b in range(B):
                idx = torch.randperm(N, device=vis.device)[:n_shuffle]
                shuffled = idx[torch.randperm(n_shuffle, device=vis.device)]
                vis[b, idx] = vis[b, shuffled]

            # 4. Ego noise
            ego_state = ego_state + torch.randn_like(ego_state) * EGO_NOISE

        # ego_encoder is fp32; run it in fp32 then cast back to backbone dtype
        ego = self.ego_encoder(ego_state.float()).to(vis.dtype)
        return torch.cat([vis, ego], dim=1)

    def forward(self, visual_tokens, ego_state, traj_tokens):
        context = self._build_context(visual_tokens, ego_state)
        ctx_len = context.shape[1]   # 1540

        # traj_embed is fp32; cast embeddings to the LM (fp16) input dtype
        gt_embeds = self.traj_embed(traj_tokens).to(context.dtype)
        lm_input  = torch.cat([context, gt_embeds[:, :-1, :]], dim=1)  # [B, 1563, 3584]

        lm_out = self.cosmos.model.language_model(
            inputs_embeds=lm_input,
            use_cache=False,
        )
        hidden      = lm_out.last_hidden_state
        traj_hidden = hidden[:, ctx_len - 1 : ctx_len + TRAJ_LEN - 1, :]
        # output_head is fp32; compute logits in fp32 for a numerically clean CE
        logits      = self.output_head(traj_hidden.float())   # [B, 24, 129]
        return logits

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

# ── Factory ───────────────────────────────────────────────────────────────────

def load_model(cosmos_path=COSMOS_PATH, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
               lora_dropout=LORA_DROPOUT, device='cuda:0'):
    print(f"[model] Loading Cosmos-Reason from {cosmos_path} ...")

    # V100/Volta: no FlashAttention, no bf16. Load fp16 with default attention.
    cosmos = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cosmos_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    print("[model] Loaded fp16 (default attention)")

    print("[model] Freezing all weights...")
    for p in cosmos.parameters():
        p.requires_grad_(False)

    # Gradient checkpointing — trades 30% compute for ~60% activation memory
    cosmos.model.language_model.gradient_checkpointing_enable()
    print("[model] Gradient checkpointing enabled on LM")

    n_lora = apply_lora(cosmos, lora_rank, lora_alpha, lora_dropout)
    print(f"[model] LoRA on {n_lora} projections (q,v,o) "
          f"rank={lora_rank} alpha={lora_alpha} dropout={lora_dropout}")

    model = AlpamayoVLA(cosmos)
    # trainable adapters are fp32 master weights
    model.ego_encoder.to(device=device, dtype=torch.float32)
    model.traj_embed.to(device=device,  dtype=torch.float32)
    model.output_head.to(device=device, dtype=torch.float32)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.to(device=device, dtype=torch.float32)
            m.lora_B.to(device=device, dtype=torch.float32)

    stats = model.count_parameters()
    print(f"[model] Total: {stats['total']:,} | Trainable: {stats['trainable']:,} "
          f"({100*stats['trainable']/stats['total']:.2f}%)")
    return model


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda:0'
    model  = load_model(device=device)
    model.train()

    B             = 1
    visual_tokens = torch.randn(B, 1536, TEXT_DIM, dtype=torch.float16, device=device)
    ego_state     = torch.randn(B, 4, EGO_DIM,     dtype=torch.float32, device=device)
    traj_tokens   = torch.randint(0, TRAJ_VOCAB, (B, TRAJ_LEN),          device=device)

    print(f"[smoke test] Forward pass B={B}...")
    logits = model(visual_tokens, ego_state, traj_tokens)
    print(f"[smoke test] logits: {logits.shape}")
    assert logits.shape == (B, TRAJ_LEN, TRAJ_VOCAB)

    loss = nn.CrossEntropyLoss()(logits.reshape(-1, TRAJ_VOCAB).float(), traj_tokens.reshape(-1))
    print(f"[smoke test] loss: {loss.item():.4f}")
    loss.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"[smoke test] params with grad: {n_grads}")
    print("[smoke test] PASSED")
