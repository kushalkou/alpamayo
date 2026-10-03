"""ar1/vision_ar1.py -- R1.1: AR1-style image input (448x280 -> 160 tokens per image).

Each image: PIL bilinear resize 1600x900 -> 448x280 (W x H; full field of view, aspect
1.78 -> 1.60, same anisotropic-resize policy as the week-1 448x448 path), rescale 1/255,
CLIP mean / std, then the Qwen2-VL NATIVE patch layout, verified element-wise against
transformers' Qwen2VLImageProcessor in test_vision.py:
  per image grid_thw = [1, 20, 32] (280/14, 448/14); the frame is repeated to fill the
  temporal patch (T = 2, as the processor does for a single image); patches are ordered
  in 2x2 merge blocks and each row is flattened as (C, T, 14, 14).
After the 2x2 merger: (20/2) x (32/2) = 160 tokens per image, 3584-dim.
Every frame of the history is encoded as its own image (AR1: 160 tokens per image), so
one sample = 3 cameras x 4 frames x 160 = 1,920 visual tokens. Order: camera-major,
oldest -> newest within a camera (CAMS order of hist_index.py).
The frozen week-1 path (vision_live.py) is NOT modified.
"""
import numpy as np, torch
from PIL import Image

W_IMG, H_IMG, P, M, TP = 448, 280, 14, 2, 2
GH, GW = H_IMG // P, W_IMG // P                     # 20, 32
TOK_PER_IMG = (GH // M) * (GW // M)                 # 160
FLAT = 3 * TP * P * P                               # 1176
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def preprocess(path):
    """-> [3, 280, 448] float32 normalized."""
    img = Image.open(path).convert('RGB').resize((W_IMG, H_IMG), resample=Image.BILINEAR)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return (t - MEAN) / STD


def patchify_native(imgs):
    """imgs [n,3,H,W] -> (hidden [n*GH*GW, 1176], grid_thw [n,3]) in Qwen2-VL order."""
    n, C, H, W = imgs.shape
    gh, gw = H // P, W // P
    x = imgs.unsqueeze(1).expand(n, TP, C, H, W)                       # repeat frame
    x = x.reshape(n, 1, TP, C, gh // M, M, P, gw // M, M, P)
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)       # n,t,gh/M,gw/M,M,M,C,TP,P,P
    hidden = x.reshape(n * gh * gw, C * TP * P * P)
    grid = torch.tensor([[1, gh, gw]] * n, dtype=torch.long, device=imgs.device)
    return hidden, grid


@torch.no_grad()
def encode(visual, imgs, dtype=torch.float16):
    """imgs [n,3,280,448] normalized -> [n*160, 3584]."""
    hidden, grid = patchify_native(imgs.to(dtype))
    return visual(hidden_states=hidden, grid_thw=grid).pooler_output.to(dtype)
