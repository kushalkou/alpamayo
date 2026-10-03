"""ar1/build_vcache.py -- R2.2: vision cache, correct (native) patch order, 448x280.

One file per UNIQUE keyframe image of CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT that
appears in ar1_hist.pkl (any slot, any split): the frozen Cosmos vision tower output
[160, 3584] fp16 (vision_ar1.encode = HF-equivalent, see ar1/r20_patch_check.py), saved
as Alpamayo/data/ar1_vcache/<CAM>/<image basename>.npy. No augmentation (the cache
replaces live encoding for un-augmented training; photometric aug is not available on
cached runs). Samples gather their 12 (or 3) files at load time (vcache.py).
  python -m torch.distributed.run --nproc_per_node=8 ar1/build_vcache.py
Each rank encodes a disjoint shard (sorted list, stride = world size), batch 16 images,
skips existing files (resumable), writes via tmp + rename.
"""
import os, sys, time, pickle
import numpy as np, torch
import torch.distributed as dist
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import vision_ar1 as V

MP = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'
HIST = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_hist.pkl'
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_vcache'


def cache_path(img_path):
    cam = img_path.split('/')[-2]
    return f'{OUT}/{cam}/{os.path.basename(img_path)[:-4]}.npy'


def unique_images():
    H = pickle.load(open(HIST, 'rb'))
    return sorted({p for h in H.values() for seq in h['cams'].values() for p, _ in seq})


class DS(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.p = paths

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return V.preprocess(self.p[i]), i


def main():
    dist.init_process_group('nccl')
    r = int(os.environ['LOCAL_RANK']); ws = dist.get_world_size()
    torch.cuda.set_device(r)
    from transformers import Qwen2_5_VLForConditionalGeneration
    vis = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MP, torch_dtype=torch.float16, device_map=f'cuda:{r}').model.visual.eval()
    allp = unique_images()
    for c in ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'):
        os.makedirs(f'{OUT}/{c}', exist_ok=True)
    mine = [p for p in allp[r::ws] if not os.path.exists(cache_path(p))]
    if r == 0:
        print(f'[vcache] unique images {len(allp)}; rank 0 todo {len(mine)}', flush=True)
    dl = torch.utils.data.DataLoader(DS(mine), batch_size=16, num_workers=6)
    t = time.time(); n = 0
    for x, idx in dl:
        f = V.encode(vis, x.cuda()).reshape(len(idx), V.TOK_PER_IMG, -1).cpu().numpy()
        for k, i in enumerate(idx.tolist()):
            q = cache_path(mine[i]); np.save(q + '.tmp.npy', f[k]); os.replace(q + '.tmp.npy', q)
        n += len(idx)
        if r == 0 and n % 1600 == 0:
            print(f'  rank0 {n}/{len(mine)} {time.time() - t:.0f}s', flush=True)
    dist.barrier()
    if r == 0:
        print(f'[vcache] done in {time.time() - t:.0f}s (rank 0 encode loop)', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
