# CLAUDE.md — Alpamayo VLA (DGX edition)

Persistent context for this repo. Read fully before acting. The current task lives in `INITIAL_INSTRUCTIONS.md` (migrating this project from the RTX workstation to this DGX). This file is the *why* and the *rules*; that file is the *what to do now*.

---

## 1. What this project is

Fine-tune **Cosmos-Reason1-7B** (Qwen2.5-VL based, 8.3B params) on **nuScenes** to predict 6-second future driving trajectories. Trajectories are emitted as discrete tokens (acceleration + curvature), decoded through a unicycle dynamics model into (x, y, yaw). Long-term goal: a reasoning-augmented VLA planner; eventual deployment target is the college motorsports buggies. This is a REU project (advisor Dr. Uday Kulkarni) + GenAI course project at KLE Technological University.

Pipeline:
```
6 cameras (448x448) ─► Cosmos native vision encoder (FROZEN) ─► 1,536 visual tokens [3584-dim]
ego state [4 steps x 4 feats] ─► MLP (4→256→256→3584) ─► 4 ego tokens
        [1536 visual | 4 ego | 23 GT traj tokens] = 1,563-token context
                          Cosmos LM (28 layers, LoRA on q,v,o)
                          output head (3584 → 129) ─► 24 traj tokens
                          unicycle model (dt=0.5s, 12 steps) ─► (x, y, yaw) over 6s
```

---

## 2. Operating model (READ THIS)

Kushal + a planner-Claude (separate chat) make the decisions. **You (Claude Code) are the executor.** You run work autonomously between **gates**. A gate is a hard stop with explicit pass/fail criteria; at a gate you STOP and report only the specified numbers (not full logs) so the planner can decide the next move. Do not blow past a gate. Do not freelance around a blocker — if something fights you (a version conflict, a missing file, an ambiguous result), stop and report it rather than improvising a workaround that changes the experiment.

---

## 3. Current state & objective

The validated pipeline already exists (best model: val loss 1.9098 @ epoch 1, token accuracy 41.49% vs 0.78% random — real learning). The **one open problem is overfitting** (best val at epoch 1, rises every epoch after). We are migrating to this DGX specifically to fix it.

**The fix — live vision, NOT unfreezing.** On the old box, visual tokens were precomputed once and frozen (a 246GB cache), so they were identical every epoch and the LM memorized `visual_fingerprint → trajectory`. The fix is to run the frozen encoder **live** every step on **augmented** images, so visual tokens vary epoch-to-epoch. We do **NOT** make the encoder weights trainable — two external reviews confirmed that unfreezing ~600M+ vision params on 16K samples makes overfitting *worse*. The encoder stays frozen and runs under `no_grad`; only LoRA + the ego MLP train. The cache is being abandoned (do not copy it to the DGX).

---

## 4. Hard constraints (non-negotiable)

- **This DGX is V100 (Volta, sm70). No bf16 in hardware.** Always load models `torch_dtype=torch.float16`. Never `bfloat16`/`auto`.
- **No FlashAttention** (needs Ampere+). Let attention fall back to default. Do not install `flash-attn`.
- **fp16 numerics are fragile here.** Cross-entropy must be computed in **float32** (cast logits before CE — fp16 CE caused nan before). Use a GradScaler for mixed precision.
- **Shared machine.** Other users' conda envs (`video_quality`, `vqa`, `i3d_pooja`) and k3s pods live here. Use only the `alpamayo` env. Check `nvidia-smi` for other users' jobs before claiming GPUs. Before any long training run, confirm bare-metal use is permitted vs. going through k8s.
- **Disk:** stage on `/home` (~1.5T free), never `/` (~80G). NAS at `/home/NAS/{kletech,dgx-v100}` is the fallback/relay.
- **Always run long jobs in tmux** (a dropped SSH session killed a run once).

---

## 5. Repo & data layout

Code + docs (this git repo): `/home/dgx1user/Alpamayo-Kushal/`
Heavy data/models (NOT in git): `/home/dgx1user/Alpamayo-Kushal/Alpamayo/` — everything lives INSIDE the repo tree. The code points **directly** at `/home/dgx1user/Alpamayo-Kushal/Alpamayo/...` (no symlink, no external `/home/dgx1user/Alpamayo`). **Never** create a folder named `Alpamayo` outside `Alpamayo-Kushal/`. `.gitignore` excludes `Alpamayo/{nuscenes,models,data,_ref_tokens,results}` and `*.pt/*.pkl/*.safetensors`. (History: the original migration used an external `/home/dgx1user/Alpamayo` symlink; on 2026-06-18 a stray mv/rm of that symlink destroyed the local tree — it was re-pulled from WS directly into this path with no symlink.)

```
Alpamayo-Kushal/            <- git repo
  CLAUDE.md  INITIAL_INSTRUCTIONS.md
  code/  dataset.py model.py finetune.py inference.py tokenizer.py precompute/
Alpamayo/                   <- data, gitignored
  nuscenes/                 <- dataroot (samples/CAM_* x6, sweeps/ [empty], maps/, v1.0-trainval/)
  models/cosmos_reason/     <- Cosmos weights (~16G)
  models/checkpoints/       <- alpamayo_best.pt, alpamayo_latest.pt
  data/trajectories_full.pkl
```

Config must point `dataroot`, cosmos weights, checkpoint dir, and trajectories at the paths above. **torchrun is invoked as** `python -m torch.distributed.run` (torchrun may not be on PATH).

---

## 6. Architecture details (don't break these)

**Vision (frozen, runs live).** Cosmos-Reason **native** encoder (NOT CLIP). API: `cosmos.model.visual(hidden_states, grid_thw)`. Input `[total_patches, 1176]`, 1176 = 3×2×14×14. Unfold each image into 14×14 patches, duplicate temporally (Conv3d needs T_in=2), `grid_thw=[[1,32,32]]` per image. Use `pooler_output` (post-merger, [N×256, 3584]), NOT `last_hidden_state`. 6 cams × 256 = 1,536 tokens × 3584-dim. The live path must reproduce this preprocessing exactly — equivalence to the old cache is Gate 2 of the migration.

**Ego (trainable).** Per-step MLP `4→256→GELU→Dropout(.1)→256→GELU→Dropout(.1)→3584` → 4 ego tokens. Features `[speed, yaw, yaw_rate, accel]` from pose dicts. yaw_rate uses **wrapped** diff `arctan2(sin(dyaw),cos(dyaw))/dt` (prevents ±π spikes). Pad short history with oldest pose. Ego noise σ=0.05 in training.

**LM + LoRA (trainable).** 28 layers, hidden 3584. LoRA on `self_attn.{q_proj,v_proj,o_proj}` (84 modules), rank=16, alpha=32, dropout=0.1 → 10.2M trainable (0.12%). lora_A init normal std 0.02, lora_B init zeros. Gradient checkpointing ON for training, **OFF for inference** (left on once → hung).

**Forward (training).** Single-pass teacher forcing: feed `[context(1540)|gt[:23]]` = 1563 tokens in ONE pass, extract hidden states at positions `ctx_len-1 .. ctx_len+22` → all 24 traj tokens at once. Loss in float32. **Inference** uses true autoregressive decode (24 sequential passes); clamp predicted tokens to 0–63 before detokenize.

**Dtype policy (STANDING ARCHITECTURE — must be identical in finetune.py and inference.py).** Frozen Cosmos backbone (LM + visual encoder) stays **fp16**. The trainable adapters — **LoRA A/B, ego MLP, traj_embed, output_head — are fp32 master weights.** This is the textbook stable mixed-precision recipe and is *mandatory*: `torch.amp.GradScaler` refuses to unscale fp16 gradients (`"Attempting to unscale FP16 gradients"`), and fp32 master weights give stable AdamW updates on V100. The fp16↔fp32 boundary is handled by explicit casts inside `LoRALinear.forward` (cast x into the LoRA dtype, cast the update back to base dtype) and `AlpamayoVLA.forward` (ego/traj embeds cast to context dtype before the LM; LM hidden cast to fp32 before output_head). `load_model()` sets this up. Proven clean over Gate 3's 200 steps (0 nan/inf). DDP only all-reduces the fp32 `requires_grad` params, ignoring the frozen fp16 backbone.

**Augmentation (live vision).** Image-space, applied **before** the frozen encoder (NOT the old token-space dropout/noise/shuffle, which was a cached-regime hack). **Photometric only** to start — brightness/contrast/saturation/hue jitter + light gaussian noise — which is geometry-preserving and already enough to break `fingerprint→trajectory` memorization. **Defer geometric aug** (flips/crops/scale): these driving images carry the trajectory in the ego frame, so a flip silently inverts steering (would require flipping traj labels + swapping L/R cameras) and crops break image↔geometry correspondence. `AlpamayoVLA.augment` flag gates the *legacy* token-space path (kept default-on for back-compat but superseded); the real aug now lives in the dataloader (`vision_live.py` photometric jitter, train split only).

**Tokenizer (validated — do not change).** `tokenize` → 12 (accel, curv) tuples, both 0–63. 64 accel bins over [-11,+10] m/s²; 64 curv bins over [-0.8,+0.65] rad/m; +1 STOP(=128) → **vocab 129**. Stored flattened `[a0..a11, k0..k11]` = 24 tokens. Roundtrip reconstruction ADE = **0.885m** (immovable floor). Unicycle: dt=0.5s, 12 steps.

---

## 7. Data

nuScenes v1.0-trainval, 850 scenes / 34,149 samples → **23,949 trajectories** (2s/4-frame history, 6s/12-frame future @2Hz). **Scene-based split (CRITICAL — never sample-split):** 595/127/128 scenes = **16,763 / 3,572 / 3,614** samples, seed=42. `scene_token` is recovered at runtime from the nuScenes API (needs the `v1.0-trainval` metadata tables). Camera-only pipeline — LIDAR/RADAR/sweeps are not used.

---

## 8. Training config (old best run — retune for 8 GPUs)

DDP via `torch.distributed.run`. On the old box: `--nproc_per_node=3`, batch=1/GPU × grad_accum=8 = effective 24. **On this DGX use `--nproc_per_node=8`** and re-tune grad_accum/effective batch (a throughput gate, later — don't guess now). AdamW, lr=5e-5 peak → 100-step warmup → cosine to 5e-6, weight_decay=0.05, grad_clip=1.0. Weighted CE: sqrt inverse-frequency (computed rank 0, broadcast). Ego noise σ=0.05. Early stopping patience 7 when testing augmentation.

---

## 9. Results to beat / reference points

- Best valid run: val **1.9098 @ epoch 1**; token acc **41.49%** (random 0.78%), seq acc 11.34%.
- Test (3,614 traj): ADE@1s mean 1.097 / median **0.489m**; ADE@6s mean 7.356 / median 4.887; FDE@6s mean 15.97 / median 11.5.
- Entropy baseline 2.33 nats. Roundtrip floor 0.885m. Alpamayo-R1 reference: ADE 0.94m, FDE 1.87m.
- A val loss *below* 2.33 with unweighted loss = modal collapse, not success. Use ADE/FDE as the real metric, never BERTScore (meaningless for bin-index tokens).

---

## 10. Mistakes already made — do NOT repeat

1. **Sample-level split → data leakage.** Always split by SCENE.
2. **Unweighted loss → modal collapse** (token 35 ≈ constant speed = 37.75% of tokens). Use sqrt inv-freq weights.
3. **CLIP encoder was wrong** (768-dim vs LM's 3584). Use the Cosmos native encoder, no projection.
4. **DataParallel OOMs** on an 8B model (GPU0 holds everything). Use DDP (symmetric all-reduce).
5. **fp16 cross-entropy → nan.** Cast logits to float32 before CE.
6. **BERTScore is meaningless** for trajectory tokens. Use ADE/FDE.
7. **Grad checkpointing left on during inference** → hang. Disable for inference.
8. **Lost a run to a dropped SSH session.** Always tmux.

---

## 11. People

- **Kushal Koujalagi** (01fe23bcs106) — lead, this implementation.
- **Darshan, Pooja, Avanthika** — co-authors / labmates on related pieces.
- **Amogh** — classmate running a parallel Qwen2.5-VL pipeline; the Cosmos-vs-Qwen comparison (same scene split, same metrics) is the headline research contribution, not yet done.
- **Prateek** — labmate; on the *old* workstation he used GPU 0 (check `nvidia-smi` before launching). This DGX is shared more broadly.

---

## 12. Status

Done: tokenizer/unicycle validated; dataset/model/finetune/inference scripts working; one valid trained run + full test inference; defense PDF; paper drafted (Overleaf).
In progress: migration to this DGX (see `INITIAL_INSTRUCTIONS.md`); the visual-token-augmentation idea, now upgraded to true live-vision augmentation.
Not started: Amogh comparison (core contribution); ablation table; Chain-of-Causation dataset; collision/off-road metrics; ego-only baseline.

---

## 13. Rules for you (Claude Code)

- Stop at every gate; report concise numbers, not log dumps.
- Never use bf16; never install flash-attn; keep CE in float32.
- Never touch other users' conda envs or k3s pods; check `nvidia-smi` before claiming GPUs.
- Long jobs in tmux, always.
- Ask before anything irreversible (deleting data, overwriting checkpoints, modifying shared/system config). Migration copies are additive — never delete source data on the workstation.
- If a result looks too good (val below the 2.33 entropy baseline, near-zero loss fast), suspect leakage or collapse and report it rather than celebrating.
- When unsure whether a change alters the experiment's meaning, stop and ask the planner.