"""ar1/vcache.py -- R2.2: load cached native-order vision features.
gather(hist_entry, frames) -> float16 tensor [3 * len(frames) * 160, 3584], camera-major,
oldest -> newest (the order of hist_index / bench_inputs). frames = (3,) for t0 only,
(0, 1, 2, 3) for the 4-keyframe history.
Unit test: python ar1/vcache.py --test N (one GPU): bit-exact vs recomputed build batches;
single-image live vs cache reported (fp16 batch-shape noise)."""
import sys, numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
from build_vcache import cache_path, unique_images, MP
CAMS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


def gather(h, frames=(0, 1, 2, 3)):
    return torch.from_numpy(np.concatenate(
        [np.load(cache_path(h['cams'][c][k][0])) for c in CAMS for k in frames], 0))


if __name__ == '__main__' and '--test' in sys.argv:
    # (a) EXACTNESS: recompute whole build batches (rank r's shard, batch 16, the order
    #     build_vcache used) and require bit-exact equality with the cache.
    # (b) single-image live encode vs cache: fp16 results depend on the batch shape, so
    #     this is reported (per-image mean / min token cosine), not required to be exact.
    #     For scale, fp16 vs fp32 tower on the same image is also reported.
    import os, vision_ar1 as V
    from transformers import Qwen2_5_VLForConditionalGeneration
    n = int(sys.argv[sys.argv.index('--test') + 1])
    vis = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MP, torch_dtype=torch.float16, device_map='cuda:0').model.visual.eval()
    allp = unique_images()
    miss = sum(not os.path.exists(cache_path(p)) for p in allp)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), -1)
    exact = True
    for r, j in ((0, 0), (3, 37), (7, 400)):        # (rank, batch index) of the build
        P = allp[r::8][16 * j:16 * j + 16]
        x = torch.stack([V.preprocess(p) for p in P]).cuda()
        f = V.encode(vis, x).reshape(len(P), 160, -1).cpu()
        c = torch.stack([torch.from_numpy(np.load(cache_path(p))) for p in P])
        exact &= torch.equal(f, c)
    print(f'(a) files missing {miss} of {len(allp)}; 3 build batches (48 images) recomputed: '
          f'bit-exact {exact}')
    rs = np.random.RandomState(0); m_, mn_ = [], []
    for i in rs.choice(len(allp), n, replace=False):
        p = allp[i]
        live = V.encode(vis, V.preprocess(p).unsqueeze(0).cuda()).cpu()
        cs = cos(live, torch.from_numpy(np.load(cache_path(p))))
        m_.append(cs.mean().item()); mn_.append(cs.min().item())
    print(f'(b) {n} random images, single-image live vs cache: per-image mean cos min '
          f'{min(m_):.5f} median {np.median(m_):.5f}; per-token min cos min {min(mn_):.4f} '
          f'median {np.median(mn_):.4f}')
    f32 = vis.float(); q, w = [], []
    for i in rs.choice(len(allp), 10, replace=False):
        x = V.preprocess(allp[i]).unsqueeze(0).cuda()
        cs = cos(V.encode(f32, x, dtype=torch.float32).cpu(), torch.from_numpy(np.load(cache_path(allp[i]))))
        q.append(cs.mean().item()); w.append(cs.min().item())
    print(f'    scale: fp32 tower vs cache (fp16), 10 images: per-image mean cos min {min(q):.4f} '
          f'median {np.median(q):.4f}; per-token min {min(w):.3f}')
    print('PASS' if miss == 0 and exact else 'FAIL')
