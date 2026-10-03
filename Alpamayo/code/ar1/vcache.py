"""ar1/vcache.py -- R2.2: load cached native-order vision features.
gather(hist_entry, frames) -> float16 tensor [3 * len(frames) * 160, 3584], camera-major,
oldest -> newest (the order of hist_index / bench_inputs). frames = (3,) for t0 only,
(0, 1, 2, 3) for the 4-keyframe history.
Unit test: python ar1/vcache.py --test N   (cached == live encode, one GPU)."""
import sys, numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
from build_vcache import cache_path, unique_images, MP
CAMS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


def gather(h, frames=(0, 1, 2, 3)):
    return torch.from_numpy(np.concatenate(
        [np.load(cache_path(h['cams'][c][k][0])) for c in CAMS for k in frames], 0))


if __name__ == '__main__' and '--test' in sys.argv:
    import os, time, vision_ar1 as V
    from transformers import Qwen2_5_VLForConditionalGeneration
    n = int(sys.argv[sys.argv.index('--test') + 1])
    vis = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MP, torch_dtype=torch.float16, device_map='cuda:0').model.visual.eval()
    allp = unique_images()
    miss = sum(not os.path.exists(cache_path(p)) for p in allp)
    rs = np.random.RandomState(0)
    worst_cos, worst_abs, exact = 1.0, 0.0, 0
    for i in rs.choice(len(allp), n, replace=False):
        p = allp[i]
        live = V.encode(vis, V.preprocess(p).unsqueeze(0).cuda()).float().cpu()
        cached = torch.from_numpy(np.load(cache_path(p))).float()
        cs = torch.nn.functional.cosine_similarity(live, cached, -1).min().item()
        worst_cos = min(worst_cos, cs); worst_abs = max(worst_abs, (live - cached).abs().max().item())
        exact += int(torch.equal(live, cached))
    print(f'files missing {miss} of {len(allp)}; {n} random images: min token cosine '
          f'{worst_cos:.6f}, max |diff| {worst_abs:.4g}, bit-exact {exact}/{n}')
    print('PASS' if miss == 0 and worst_cos >= 0.9999 else 'FAIL')
