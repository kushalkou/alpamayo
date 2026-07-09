"""
model.py — Alpamayo VLA Model (DGX / V100 edition)

Changes vs the RTX-workstation version:
  - V100 (sm70): fp16 everywhere for the frozen backbone, NEVER bf16.
  - No FlashAttention (needs Ampere+). Default ("eager"/sdpa) attention only.
  - Mixed-precision dtype policy (STANDING ARCHITECTURE):
      * Frozen Cosmos backbone (LM + visual encoder) stays fp16.
      * Trainable adapters (LoRA A/B, ego MLP, traj_embed, output_head) are
        fp32 master weights. GradScaler refuses to unscale fp16 grads, and
        fp32 masters give stable AdamW on V100.
      * The fp16<->fp32 boundary is handled by explicit casts inside
        LoRALinear.forward and AlpamayoVLA.forward.
  - `augment` flag gates the LEGACY token-space aug path (default OFF — the
    real augmentation now lives in the dataloader / vision_live.py, image-space).
  - Paths repointed drive1 -> DGX (/home/dgx1user/Alpamayo-Kushal/Alpamayo).
  - Gradient checkpointing on the LM for training (disable for inference).
  - DDP compatible (no next(parameters()) calls in forward).
"""

import sys
import math
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration

COSMOS_PATH  = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'
TEXT_DIM     = 3584
EGO_DIM      = 4
TRAJ_VOCAB   = 129
TRAJ_LEN     = 24
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
EGO_NOISE    = 0.05   # Gaussian noise std on ego features during training

# Dtype policy
BACKBONE_DTYPE = torch.float16   # frozen Cosmos (LM + visual encoder)
ADAPTER_DTYPE  = torch.float32   # trainable master weights

# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Frozen fp16 base linear + fp32 LoRA update. Cast at the boundary."""
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
        # Base path runs in the frozen backbone dtype (fp16).
        base = self.linear(x)
        # LoRA update runs in the fp32 adapter dtype, then casts back.
        lora_dtype = self.lora_A.weight.dtype
        update = self.lora_B(self.dropout(self.lora_A(x.to(lora_dtype)))) * self.scaling
        return base + update.to(base.dtype)


def apply_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
    """Apply LoRA to q_proj, v_proj, o_proj in all LM attention layers.
    LoRA A/B are placed as fp32 master weights on the layer's device."""
    count = 0
    for layer in model.model.language_model.layers:
        attn = layer.self_attn
        for name in ('q_proj', 'v_proj', 'o_proj'):
            if hasattr(attn, name):
                original = getattr(attn, name)
                lora     = LoRALinear(original, rank, alpha, dropout)
                dev      = next(original.parameters()).device
                lora.lora_A.to(device=dev, dtype=ADAPTER_DTYPE)
                lora.lora_B.to(device=dev, dtype=ADAPTER_DTYPE)
                setattr(attn, name, lora)
                count += 1
    return count

# ── Ego Encoder ───────────────────────────────────────────────────────────────

class EgoEncoderMLP(nn.Module):
    """[B, 4, 4] -> [B, 4, 3584].  Trainable fp32 master weights."""
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
        visual_tokens [B, 1536, 3584]  fp16 (from the live frozen encoder)
        ego_state     [B, 4, 4]        fp32
        traj_tokens   [B, 24]          long
    Returns:
        logits [B, 24, 129]            fp32 (ready for CE)

    `augment` gates the LEGACY token-space aug (default OFF; superseded by the
    image-space photometric aug in the dataloader / vision_live.py). Ego noise
    is applied in training regardless (it is not part of the legacy path).
    """

    def __init__(self, cosmos, augment=False):
        super().__init__()
        self.cosmos      = cosmos
        self.augment     = augment
        self.ego_encoder = EgoEncoderMLP()
        self.traj_embed  = nn.Embedding(TRAJ_VOCAB, TEXT_DIM)
        self.output_head = nn.Linear(TEXT_DIM, TRAJ_VOCAB, bias=False)

    def _build_context(self, visual_tokens, ego_state):
        vis = visual_tokens.to(dtype=BACKBONE_DTYPE)

        if self.training and self.augment:
            # ── LEGACY token-space augmentation (superseded, default OFF) ──
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

        if self.training:
            # Ego noise (always on in training).
            ego_state = ego_state + torch.randn_like(ego_state) * EGO_NOISE

        # Ego MLP runs in fp32 (master weights), then casts to the context dtype.
        ego = self.ego_encoder(ego_state.to(dtype=ADAPTER_DTYPE))
        ego = ego.to(dtype=vis.dtype)
        return torch.cat([vis, ego], dim=1)

    def forward(self, visual_tokens, ego_state, traj_tokens):
        context = self._build_context(visual_tokens, ego_state)
        ctx_len = context.shape[1]   # 1540

        # traj_embed is fp32; cast embeds to the context (fp16) dtype for the LM.
        gt_embeds = self.traj_embed(traj_tokens).to(dtype=context.dtype)
        lm_input  = torch.cat([context, gt_embeds[:, :-1, :]], dim=1)  # [B, 1563, 3584]

        lm_out = self.cosmos.model.language_model(
            inputs_embeds=lm_input,
            use_cache=False,
        )
        hidden      = lm_out.last_hidden_state
        traj_hidden = hidden[:, ctx_len - 1 : ctx_len + TRAJ_LEN - 1, :]
        # output_head is fp32; cast LM hidden to fp32 so logits are fp32 for CE.
        logits      = self.output_head(traj_hidden.to(dtype=ADAPTER_DTYPE))   # [B, 24, 129]
        return logits

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

# ── Factory ───────────────────────────────────────────────────────────────────

def load_model(cosmos_path=COSMOS_PATH, lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
               lora_dropout=LORA_DROPOUT, device='cuda:0', augment=False,
               grad_checkpointing=True):
    print(f"[model] Loading Cosmos-Reason from {cosmos_path} ...")

    # V100: fp16 + default attention. NEVER bf16, NEVER flash-attn.
    cosmos = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cosmos_path,
        torch_dtype=BACKBONE_DTYPE,
        device_map=device,
    )
    print("[model] Loaded fp16 with default attention (no flash-attn on V100)")

    print("[model] Freezing all backbone weights...")
    for p in cosmos.parameters():
        p.requires_grad_(False)

    if grad_checkpointing:
        cosmos.model.language_model.gradient_checkpointing_enable()
        print("[model] Gradient checkpointing enabled on LM")

    n_lora = apply_lora(cosmos, lora_rank, lora_alpha, lora_dropout)
    print(f"[model] LoRA on {n_lora} projections (q,v,o) "
          f"rank={lora_rank} alpha={lora_alpha} dropout={lora_dropout}")

    model = AlpamayoVLA(cosmos, augment=augment)
    # Trainable adapters live as fp32 master weights.
    model.ego_encoder.to(device=device, dtype=ADAPTER_DTYPE)
    model.traj_embed.to(device=device,  dtype=ADAPTER_DTYPE)
    model.output_head.to(device=device, dtype=ADAPTER_DTYPE)

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
    visual_tokens = torch.randn(B, 1536, TEXT_DIM, dtype=BACKBONE_DTYPE, device=device)
    ego_state     = torch.randn(B, 4, EGO_DIM,     dtype=ADAPTER_DTYPE,  device=device)
    traj_tokens   = torch.randint(0, TRAJ_VOCAB, (B, TRAJ_LEN),          device=device)

    print(f"[smoke test] Forward pass B={B}...")
    logits = model(visual_tokens, ego_state, traj_tokens)
    print(f"[smoke test] logits: {logits.shape}  dtype={logits.dtype}")
    assert logits.shape == (B, TRAJ_LEN, TRAJ_VOCAB)
    assert logits.dtype == torch.float32

    loss = nn.CrossEntropyLoss()(logits.reshape(-1, TRAJ_VOCAB), traj_tokens.reshape(-1))
    print(f"[smoke test] loss: {loss.item():.4f}")
    loss.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"[smoke test] params with grad: {n_grads}")
    print("[smoke test] PASSED")
