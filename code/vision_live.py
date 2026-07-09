"""
vision_live.py — Live frozen-encoder vision path (DGX / V100 edition)

Replaces the precomputed 246GB visual-token cache. Every training step runs the
6 camera images through the FROZEN Cosmos-Reason native encoder under no_grad, so
visual tokens vary epoch-to-epoch (with photometric aug) and the LM can no longer
memorize `visual_fingerprint -> trajectory`. The encoder weights stay frozen —
we do NOT unfreeze the ~600M vision params.

Anchors (do not rewrite from memory):
  - patchify: copied VERBATIM from precompute_visual_tokens.py (encode_images).
  - normalization: read from the Qwen2VL processor config in the Cosmos weights.
  - resize: PIL BILINEAR -> 448.
  - output: frozen encoder -> [1536, 3584] fp16, grid_thw=[[1,32,32]] per cam,
    using pooler_output (post-merger), NOT last_hidden_state.
  - photometric aug (image-space, BEFORE the encoder): ColorJitter
    (brightness/contrast/saturation=0.3, hue=0.05) + Gaussian noise sigma=0.02.
    NO geometric aug (flips/crops would desync the ego-frame trajectory).
"""

import json
import warnings
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

COSMOS_PATH = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'

# 6 cameras, fixed order (visual tokens are concatenated in this order).
CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

IMAGE_H = 448
IMAGE_W = 448
PATCH_SIZE      = 14
TEMPORAL_SIZE   = 2
H_OUT           = IMAGE_H // PATCH_SIZE   # 32
W_OUT           = IMAGE_W // PATCH_SIZE   # 32
T_OUT           = 1
PATCHES_PER_IMG = T_OUT * H_OUT * W_OUT               # 1024
FLAT_DIM        = 3 * TEMPORAL_SIZE * PATCH_SIZE * PATCH_SIZE  # 1176

# ── Version pinning (validated stack — warn on drift) ─────────────────────────

PINNED_VERSIONS = {
    'torch':        '2.7.1',
    'torchvision':  '0.22.1',
    'transformers': '5.5.4',
    'numpy':        '1.26.4',
    'PIL':          '12.2.0',
}

def check_versions():
    import importlib
    for name, pinned in PINNED_VERSIONS.items():
        try:
            mod = importlib.import_module(name)
            cur = getattr(mod, '__version__', '?').split('+')[0]
        except Exception as e:
            warnings.warn(f"[vision_live] could not import {name}: {e}")
            continue
        if cur != pinned:
            warnings.warn(f"[vision_live] {name} version drift: pinned {pinned}, "
                          f"got {cur}. Encoder preprocessing may differ.")

# ── Normalization (deterministic, from the processor config) ──────────────────

def _load_norm(cosmos_path=COSMOS_PATH):
    cfg_path = Path(cosmos_path) / 'preprocessor_config.json'
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg['image_mean'], cfg['image_std']

IMAGE_MEAN, IMAGE_STD = _load_norm()

# ── Image-space transforms ────────────────────────────────────────────────────

# Photometric-only aug (geometry-preserving). Applied on [0,1] tensors, train only.
_color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
NOISE_SIGMA = 0.02

_to_tensor = T.ToTensor()
_normalize = T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)


def preprocess_image(path, augment=False):
    """Load one camera image -> normalized [3, 448, 448] float32 tensor.
    PIL BILINEAR resize to 448; photometric aug (train only) BEFORE normalize."""
    img = Image.open(path).convert('RGB').resize((IMAGE_W, IMAGE_H), Image.BILINEAR)
    x = _to_tensor(img)                       # [3, 448, 448] in [0, 1]
    if augment:
        x = _color_jitter(x)
        x = (x + torch.randn_like(x) * NOISE_SIGMA).clamp_(0.0, 1.0)
    x = _normalize(x)
    return x.to(dtype=torch.float32)


def load_camera_images(cam_paths: dict, augment=False):
    """cam_paths: {CAM_NAME: filepath}. Returns [6, 3, 448, 448] float32.
    Missing cameras fall back to a normalized zero (black) image."""
    imgs = []
    for cam in CAMERAS:
        p = cam_paths.get(cam)
        if p and Path(p).exists():
            imgs.append(preprocess_image(p, augment=augment))
        else:
            warnings.warn(f"[vision_live] missing image for {cam}: {p}")
            black = _normalize(torch.zeros(3, IMAGE_H, IMAGE_W))
            imgs.append(black.to(dtype=torch.float32))
    return torch.stack(imgs, dim=0)           # [6, 3, 448, 448]

# ── Patchify (VERBATIM from precompute_visual_tokens.py::encode_images) ───────

def patchify(imgs):
    """imgs: [N, 3, 448, 448] on device -> (hidden_states [N*1024, 1176],
    grid_thw [N, 3]). Same unfold/temporal-duplication as the cache builder."""
    N = imgs.shape[0]
    P = PATCH_SIZE

    # Extract patches: [N, H_out, W_out, C, P, P]
    imgs_unf = imgs.unfold(2, P, P).unfold(3, P, P)
    imgs_unf = imgs_unf.permute(0, 2, 3, 1, 4, 5).contiguous()
    # [N, 1024, C*P*P]
    imgs_unf = imgs_unf.reshape(N, PATCHES_PER_IMG, 3 * P * P)
    # Duplicate temporal: [N, 1024, 2*C*P*P] = [N, 1024, 1176]
    imgs_unf = imgs_unf.unsqueeze(2).expand(-1, -1, TEMPORAL_SIZE, -1).contiguous()
    imgs_unf = imgs_unf.reshape(N, PATCHES_PER_IMG, FLAT_DIM)
    # Flatten: [N*1024, 1176]
    hidden_states = imgs_unf.reshape(N * PATCHES_PER_IMG, FLAT_DIM)

    grid_thw = torch.tensor(
        [[T_OUT, H_OUT, W_OUT]] * N,
        dtype=torch.long, device=imgs.device,
    )
    return hidden_states, grid_thw

# ── Live encode ───────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(visual, images):
    """Run the FROZEN encoder live.
    visual : the Cosmos native encoder (model.cosmos.model.visual)
    images : [B, 6, 3, 448, 448]  (or [6, 3, 448, 448] for a single sample)
    Returns: [B, 1536, 3584] fp16 visual tokens (pooler_output, post-merger)."""
    squeeze = (images.dim() == 4)
    if squeeze:
        images = images.unsqueeze(0)
    B = images.shape[0]

    device = next(visual.parameters()).device
    dtype  = next(visual.parameters()).dtype          # fp16
    flat   = images.reshape(B * 6, 3, IMAGE_H, IMAGE_W).to(device=device, dtype=dtype)

    hidden_states, grid_thw = patchify(flat)
    out = visual(hidden_states=hidden_states, grid_thw=grid_thw)
    merged = out.pooler_output                         # [B*6*256, 3584]
    tokens = merged.reshape(B, 6 * 256, merged.shape[-1]).to(dtype=torch.float16)
    return tokens[0] if squeeze else tokens


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    check_versions()
    print(f"[vision_live] mean={IMAGE_MEAN} std={IMAGE_STD}")
    # Fake images to exercise patchify shapes without loading the encoder.
    imgs = torch.randn(2, 6, 3, IMAGE_H, IMAGE_W)
    flat = imgs.reshape(12, 3, IMAGE_H, IMAGE_W)
    hs, grid = patchify(flat)
    print(f"[vision_live] hidden_states {tuple(hs.shape)} grid_thw {tuple(grid.shape)}")
    assert hs.shape == (12 * PATCHES_PER_IMG, FLAT_DIM)
    assert grid.shape == (12, 3)
    print("[vision_live] patchify smoke PASSED")
