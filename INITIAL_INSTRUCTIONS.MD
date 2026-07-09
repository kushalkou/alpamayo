# INITIAL_INSTRUCTIONS.md — Recovery + Gate 4 (post third code-loss)

**Situation:** The DGX code tree was lost a third time. No git repo, no `code/` dir. **Data is intact** (6 cameras × 34,149; cosmos_reason 16G). **WS is intact and reachable** (`user@10.2.0.27`, `/home/drive1/Alpamayo`, 962G) and is the ground-truth source. Do NOT reconstruct code from anyone's memory — pull WS originals and re-apply the documented session changes on top.

**Operating model:** You (Claude Code) execute; a planner-Claude decides. Stop at gates, report only the asked numbers. Read `CLAUDE.md` for full project context and hard constraints (V100: fp16 only, no bf16, no flash-attn, CE in float32).

---

## STEP 0 — Git safety net FIRST (before any recovery)

The root cause of three losses is unversioned work in one location. Fix that before touching code.

```bash
cd ~/Alpamayo-Kushal
ls -la                          # confirm which docs survived (CLAUDE.md, this file)
git init
cat > .gitignore <<'EOF'
Alpamayo/nuscenes/
Alpamayo/models/
Alpamayo/data/
Alpamayo/_ref_tokens/
__pycache__/
*.pt
*.pkl
.env
EOF
git add .gitignore CLAUDE.md INITIAL_INSTRUCTIONS.md
git commit -m "Recovery baseline: docs + gitignore"
```

Create a **private** remote (GitHub `kushalkou`, a NEW private repo — not travel-world-backend) and push. Auth (PAT or SSH key) is Kushal's to provide; do not enter credentials for him.
```bash
git remote add origin <PRIVATE_REPO_URL>
git branch -M main
git push -u origin main
```
**After this, commit + push after every file lands below. Non-negotiable.**

---

## STEP 1 — Pull WS originals verbatim (the base)

```bash
mkdir -p code
WS=user@10.2.0.27
rsync -aP "$WS:/home/drive1/Alpamayo/tokenization/tokenizer.py" \
          "$WS:/home/drive1/Alpamayo/training/inference.py" \
          "$WS:/home/drive1/Alpamayo/training/model.py" \
          "$WS:/home/drive1/Alpamayo/data/dataset.py" \
          "$WS:/home/drive1/Alpamayo/training/finetune.py" \
          "$WS:/home/drive1/Alpamayo/precompute/precompute_visual_tokens.py" \
          code/
git add code/ && git commit -m "WS-verbatim baseline (validated originals)" && git push
```
`tokenizer.py` and `inference.py` are DONE here — they were never modified. Verify `tokenizer.py` checksum matches WS (its bin arrays are load-bearing for the 0.885m floor).

---

## STEP 2 — Re-apply this session's documented changes (commit each)

Apply on top of the WS base. Because you're editing the real originals, the diff *is* the fidelity audit — every change below should be the only differences vs WS.

**`model.py`** — dtype policy: frozen Cosmos backbone fp16; trainable adapters (LoRA A/B, ego, traj-embed, head) as **fp32 master weights** with casts at the fp16/fp32 boundary inside `LoRALinear.forward` and `AlpamayoVLA.forward`. fp16 load + default attention (no flash on V100). Add an `augment` flag (default OFF). Path repoint drive1→DGX.

**`dataset.py`** — live 6-image loading (not cache/zeros); resolve the 6 camera paths once during the scene split; photometric-aug flag threaded through; cache path removed. Keep SCENE split (595/127/128, seed 42).

**`finetune.py`** — GradScaler wired into train + validate; live encode in both; smoke args `--batch_size --grad_accum_steps --max_steps --augment --find_unused`; per-step VRAM + sec/step reporting. lr 5e-5→cosine 5e-6, wd 0.05, grad_clip 1.0, weighted CE (sqrt inv-freq), ego noise σ0.05.

**`vision_live.py`** (new — no WS original, so anchor its pieces):
- patchify: **copy verbatim from `code/precompute_visual_tokens.py`** (do not rewrite from memory).
- normalization: read from the Qwen2VL processor config in `Alpamayo/models/cosmos_reason` (deterministic).
- resize: PIL **BILINEAR → 448**.
- output: frozen encoder → `[1536, 3584]` fp16, `grid_thw=[[1,32,32]]` per cam, `pooler_output`.
- photometric aug (image-space, before encoder): `ColorJitter(brightness/contrast/saturation=0.3, hue=0.05)` + Gaussian noise σ0.02. **No geometric aug** (no flips/crops — they'd desync trajectory geometry).
- Pin PIL/torchvision/torch/transformers/numpy versions; warn-on-drift.

Commit + push after each file.

---

## STEP 3 — Re-verify Gate 3, then Gate 4

**Gate 3 re-verify (single GPU):** overfit-one-batch, aug OFF, ~200 steps. Confirms the rebuilt stack runs clean. Report: final loss, monotonic drop y/n, **nan/inf count (must be 0)**, peak VRAM.

**Gate 4 (8-GPU DDP smoke):**
```
--nproc_per_node=8 --batch_size 2 --grad_accum_steps 1 --max_steps 60 --augment
```
Report ONLY: peak VRAM/GPU (must clear 32G with headroom), sec/step, NCCL no-hang confirm, loss-moves confirm. If it *hangs* (not errors), suspect DDP unused-params → retry `--find_unused`. Stop and report.

---

## Standing rule going forward
Never do unversioned work. Commit + push at every meaningful step. WS (10.2.0.27) remains the ground-truth mirror — if in doubt, diff against it.