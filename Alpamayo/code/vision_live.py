"""vision_live.py — CANONICAL live vision encoder for Alpamayo VLA.

Live path (cache abandoned): PIL BILINEAR resize -> 448 + Qwen2VL normalization
(rescale 1/255, CLIP mean/std) + verbatim patchify -> frozen Cosmos visual encoder
-> pooler_output [1536, 3584] fp16. Validated at cos_mean 0.9955 vs the old cache.

Augmentation is IMAGE-SPACE photometric ONLY (brightness/contrast/saturation/hue
jitter + light Gaussian noise), applied to the [0,1] image BEFORE normalization and
BEFORE the frozen encoder. Geometric aug (crops/flips/scale) is DEFERRED — flips
invert steering geometry, crops break image->geometry correspondence. Photometric is
geometry-preserving and breaks fingerprint->trajectory memorization.

Division of labour for training: CPU dataloader workers do image IO + photometric
aug + normalize (return [6,3,448,448] tensors); the GPU runs the frozen encoder live
each step (patchify is a cheap reshape) via encode_normalized_images().
"""
import os, warnings
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

_PINNED = {'PIL': '12.2.0', 'torchvision': '0.22.1+cu118',
           'torch': '2.7.1+cu118', 'transformers': '5.5.4', 'numpy': '1.26.4'}

def _check_versions():
    import PIL, torchvision, transformers
    have = {'PIL': PIL.__version__, 'torchvision': torchvision.__version__,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'numpy': np.__version__}
    drift = {k: (have[k], v) for k, v in _PINNED.items() if have[k] != v}
    if drift:
        warnings.warn(f"[vision_live] version drift from pinned: {drift}", stacklevel=2)
_check_versions()

CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
IMG = 448; P = 14; H_OUT = W_OUT = IMG // P
PATCHES_PER_IMG = H_OUT * W_OUT; TEMPORAL = 2; FLAT_DIM = 3 * TEMPORAL * P * P
RESAMPLE = Image.BILINEAR
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

# ── Image-space photometric augmentation (geometry-preserving) ──────────────────
JITTER_BRIGHTNESS = 0.3
JITTER_CONTRAST   = 0.3
JITTER_SATURATION = 0.3
JITTER_HUE        = 0.05
NOISE_STD         = 0.02

def _make_jitter():
    return T.ColorJitter(brightness=JITTER_BRIGHTNESS, contrast=JITTER_CONTRAST,
                         saturation=JITTER_SATURATION, hue=JITTER_HUE)


def preprocess_image(path, augment=False, jitter=None):
    """Load -> resize 448 (PIL bilinear) -> [optional photometric aug] -> normalize.
    Returns [3, 448, 448] float32, normalized. Photometric aug acts on the [0,1]
    image BEFORE normalization so it stays in valid image space."""
    img = Image.open(path).convert('RGB').resize((IMG, IMG), resample=RESAMPLE)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # [3,H,W] in [0,1]
    if augment:
        if jitter is None:
            jitter = _make_jitter()
        t = jitter(t)                                   # brightness/contrast/sat/hue
        t = t + torch.randn_like(t) * NOISE_STD         # light sensor-style noise
        t = t.clamp_(0.0, 1.0)
    return (t - MEAN) / STD


def patchify(imgs):
    """imgs [n,3,448,448] -> (hidden [n*1024, 1176], grid_thw [n,3]). Verbatim recipe."""
    n = imgs.shape[0]
    u = imgs.unfold(2, P, P).unfold(3, P, P)
    u = u.permute(0, 2, 3, 1, 4, 5).contiguous()
    u = u.reshape(n, PATCHES_PER_IMG, 3 * P * P)
    u = u.unsqueeze(2).expand(-1, -1, TEMPORAL, -1).contiguous()
    hidden = u.reshape(n * PATCHES_PER_IMG, FLAT_DIM)
    grid = torch.tensor([[1, H_OUT, W_OUT]] * n, dtype=torch.long, device=imgs.device)
    return hidden, grid


def get_cam_paths(nusc, sample_token):
    s = nusc.get('sample', sample_token)
    return {c: nusc.get_sample_data_path(s['data'][c]) for c in CAMERAS}


@torch.no_grad()
def encode_normalized_images(visual, images, dtype=torch.float16):
    """GPU helper for the train loop. images: [n,3,448,448] ALREADY normalized, on
    the visual encoder's device. Runs the frozen encoder live -> pooler [n*256, 3584].
    `n` is typically 6 (one sample's cameras); pass [B*6,...] for a batch."""
    images = images.to(dtype=dtype)
    hidden, grid = patchify(images)
    out = visual(hidden_states=hidden, grid_thw=grid)
    return out.pooler_output.to(dtype)


class LiveVisionEncoder:
    """Wraps the frozen Cosmos visual tower. Build from an already-loaded model
    (model.cosmos.model.visual) to avoid a second 16GB load."""

    def __init__(self, visual, device='cuda:0', dtype=torch.float16):
        self.visual = visual.eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode_sample(self, cam_paths, augment=False):
        missing = [c for c in CAMERAS if not os.path.exists(cam_paths[c])]
        if missing:
            raise FileNotFoundError(f"missing camera images: {missing}")
        jitter = _make_jitter() if augment else None
        imgs = torch.stack([preprocess_image(cam_paths[c], augment=augment, jitter=jitter)
                            for c in CAMERAS], 0)
        imgs = imgs.to(self.device, dtype=self.dtype)
        hidden, grid = patchify(imgs)
        out = self.visual(hidden_states=hidden, grid_thw=grid)
        return out.pooler_output.to(self.dtype)   # [1536, 3584]

    @torch.no_grad()
    def encode_batch(self, cam_paths_list, augment=False):
        return torch.stack([self.encode_sample(p, augment=augment)
                            for p in cam_paths_list], 0)
