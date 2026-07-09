"""
precompute_visual_tokens.py

Runs all nuScenes images through Cosmos-Reason's native vision encoder
and saves the output visual tokens to disk.

Each sample produces: [6*256, 3584] = [1536, 3584] float16 tensor
Saved as: /home/drive1/Alpamayo/data/visual_tokens/<sample_token>.pt

This is NOT the same mistake as before (frozen CLIP):
  - We use Cosmos-Reason's OWN encoder — same embedding space as the LM
  - Vision encoder is frozen during training anyway
  - Precomputing saves ~8s per training step

Usage:
    cd /home/drive1/Alpamayo
    /opt/miniconda3/bin/python precompute/precompute_visual_tokens.py
"""

import os
import sys
import pickle
import time
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, '/home/drive1/Alpamayo/data')
from dataset import NuScenesVLADataset, split_trajectories, preprocess_image, CAMERAS, IMAGE_H, IMAGE_W

# ── Config ────────────────────────────────────────────────────────────────────

COSMOS_PATH    = '/home/drive1/Alpamayo/models/cosmos_reason'
TRAJ_PATH      = '/home/drive1/Alpamayo/data/trajectories_full.pkl'
NUSCENES_ROOT  = '/home/drive1/Alpamayo/nuscenes_full'
OUTPUT_DIR     = Path('/home/drive1/Alpamayo/data/visual_tokens')
DEVICE         = 'cuda:0'
BATCH_SIZE     = 6       # process all 6 cameras of one sample at once
PATCH_SIZE     = 14
TEMPORAL_SIZE  = 2
H_OUT          = IMAGE_H // PATCH_SIZE   # 32
W_OUT          = IMAGE_W // PATCH_SIZE   # 32
T_OUT          = 1
PATCHES_PER_IMG = T_OUT * H_OUT * W_OUT  # 1024
FLAT_DIM       = 3 * TEMPORAL_SIZE * PATCH_SIZE * PATCH_SIZE  # 1176

# ── Load model (vision encoder only, on one GPU) ──────────────────────────────

def load_vision_encoder():
    print(f"[precompute] Loading Cosmos-Reason vision encoder from {COSMOS_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        COSMOS_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
    )
    visual = model.model.visual
    visual.eval()
    for p in visual.parameters():
        p.requires_grad_(False)
    print("[precompute] Vision encoder loaded and frozen.")
    # Free LM memory — we only need the visual encoder
    del model.model.language_model
    del model.lm_head
    torch.cuda.empty_cache()
    print("[precompute] LM freed. GPU memory available for batch processing.")
    return visual

# ── Encode one sample's 6 images ─────────────────────────────────────────────

def encode_images(visual, cam_paths: dict) -> torch.Tensor:
    """
    Load 6 camera images, run through vision encoder.
    Returns: [1536, 3584] float16 tensor (6 cams * 256 merged tokens)
    """
    images = []
    for cam in CAMERAS:
        path = cam_paths.get(cam)
        if path and os.path.exists(path):
            img = preprocess_image(path)
        else:
            img = torch.zeros(3, IMAGE_H, IMAGE_W, dtype=torch.float32)
        images.append(img)

    # [6, 3, H, W] -> [6, C, 2, H, W] for temporal duplication
    imgs = torch.stack(images, dim=0).to(DEVICE, dtype=torch.bfloat16)  # [6, 3, 448, 448]

    P = PATCH_SIZE

    # Extract patches: [6, H_out, W_out, C, P, P]
    imgs_unf = imgs.unfold(2, P, P).unfold(3, P, P)
    imgs_unf = imgs_unf.permute(0, 2, 3, 1, 4, 5).contiguous()
    # [6, 1024, C*P*P]
    imgs_unf = imgs_unf.reshape(6, PATCHES_PER_IMG, 3 * P * P)
    # Duplicate temporal: [6, 1024, 2*C*P*P] = [6, 1024, 1176]
    imgs_unf = imgs_unf.unsqueeze(2).expand(-1, -1, TEMPORAL_SIZE, -1).contiguous()
    imgs_unf = imgs_unf.reshape(6, PATCHES_PER_IMG, FLAT_DIM)
    # Flatten: [6*1024, 1176]
    hidden_states = imgs_unf.reshape(6 * PATCHES_PER_IMG, FLAT_DIM)

    grid_thw = torch.tensor(
        [[T_OUT, H_OUT, W_OUT]] * 6,
        dtype=torch.long, device=DEVICE,
    )

    with torch.no_grad():
        visual_out = visual(hidden_states=hidden_states, grid_thw=grid_thw)

    # pooler_output: [6*256, 3584]
    merged = visual_out.pooler_output   # [1536, 3584]
    return merged.to(dtype=torch.float16).cpu()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all trajectories
    print(f"[precompute] Loading trajectories from {TRAJ_PATH}...")
    with open(TRAJ_PATH, 'rb') as f:
        all_trajs = pickle.load(f)
    print(f"[precompute] {len(all_trajs)} trajectories loaded.")

    # Build sample_token -> cam_paths index (reuse dataset logic)
    print("[precompute] Building camera path index...")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_ROOT, verbose=False)

    cam_paths_index = {}
    for sample in nusc.sample:
        token = sample['token']
        paths = {}
        for cam in CAMERAS:
            if cam not in sample['data']:
                continue
            sd = nusc.get('sample_data', sample['data'][cam])
            paths[cam] = str(Path(NUSCENES_ROOT) / sd['filename'])
        cam_paths_index[token] = paths
    print(f"[precompute] Index built: {len(cam_paths_index)} samples.")

    # Load vision encoder
    visual = load_vision_encoder()

    # Process each trajectory
    n_total   = len(all_trajs)
    n_done    = 0
    n_skipped = 0
    t_start   = time.time()

    print(f"[precompute] Processing {n_total} samples -> {OUTPUT_DIR}")
    print("[precompute] Already-processed samples will be skipped.\n")

    for traj in all_trajs:
        token = traj['sample_token']
        out_path = OUTPUT_DIR / f"{token}.pt"

        if out_path.exists():
            n_skipped += 1
            n_done += 1
            continue

        cam_paths = cam_paths_index.get(token, {})
        try:
            tokens = encode_images(visual, cam_paths)   # [1536, 3584]
            torch.save(tokens, out_path)
        except Exception as e:
            print(f"[precompute] ERROR on {token}: {e}")
            n_done += 1
            continue

        n_done += 1

        if n_done % 100 == 0:
            elapsed  = time.time() - t_start
            rate     = (n_done - n_skipped) / max(elapsed, 1)
            remaining = (n_total - n_done) / max(rate, 1e-6)
            print(f"  [{n_done}/{n_total}] "
                  f"{rate:.1f} samples/s | "
                  f"~{remaining/60:.0f} min remaining | "
                  f"skipped {n_skipped}")

    elapsed = time.time() - t_start
    print(f"\n[precompute] Done. {n_done} samples in {elapsed/60:.1f} min.")
    print(f"[precompute] Output: {OUTPUT_DIR}  ({n_done - n_skipped} new files)")


if __name__ == '__main__':
    main()