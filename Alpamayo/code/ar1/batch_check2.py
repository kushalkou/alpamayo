"""ar1/batch_check2.py -- reproduce one build_vcache batch (rank 0, first 16 images)."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import vision_ar1 as V
from build_vcache import unique_images, cache_path, MP
from transformers import Qwen2_5_VLForConditionalGeneration
vis = Qwen2_5_VLForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float16,
                                                         device_map='cuda:0').model.visual.eval()
P = unique_images()[0::8][:16]
x = torch.stack([V.preprocess(p) for p in P]).cuda()
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), -1)
b16 = V.encode(vis, x).reshape(16, 160, -1).cpu()
c = torch.stack([torch.from_numpy(np.load(cache_path(p))) for p in P])
print('batch-16 recompute vs cache: bit-exact', torch.equal(b16, c), 'min cos', cos(b16, c).min().item())
one = torch.stack([V.encode(vis, x[i:i + 1]).cpu() for i in range(16)])
cs = cos(one, c)
print('single vs cache: per-image min cos', [round(v, 4) for v in cs.min(1).values.tolist()])
print('single vs cache: per-image mean cos', [round(v, 5) for v in cs.mean(1).tolist()])
f32 = vis.float()
o32 = torch.stack([V.encode(f32, x[i:i + 1].float(), dtype=torch.float32).cpu() for i in range(4)])
print('fp32 single vs fp16 single, min cos', cos(o32, one[:4]).min(1).values.tolist())
print('fp32 single vs fp16 batch16 (cache), min cos', cos(o32, c[:4]).min(1).values.tolist())
