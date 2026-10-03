"""ar1/batch_check.py -- does encoding images in a batch change each image's features?"""
import sys, pickle, torch, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import vision_ar1 as V
from transformers import Qwen2_5_VLForConditionalGeneration
MP = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'
m = Qwen2_5_VLForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float16, device_map='cuda:0')
vis = m.model.visual.eval()
print('attn impl:', m.config._attn_implementation, getattr(vis.config, '_attn_implementation', None))
H = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_hist.pkl', 'rb'))
h = list(H.values())[5000]
imgs = torch.stack([V.preprocess(p) for c in h['cams'] for p, _ in h['cams'][c]]).cuda()
cos = lambda a, b: torch.nn.functional.cosine_similarity(a.float(), b.float(), -1)
one = [V.encode(vis, imgs[i:i + 1]) for i in range(4)]
for n in (1, 2, 4, 12):
    f = V.encode(vis, imgs[:n]).reshape(n, 160, -1)
    print(f'batch {n:2d}: image0 cos vs alone min {cos(f[0], one[0]).min():.5f}; '
          f'image{min(n,4)-1} min {cos(f[min(n,4)-1], one[min(n,4)-1]).min():.5f}')
f32 = vis.float()
a = V.encode(f32, imgs[:1], dtype=torch.float32); b = V.encode(f32, imgs[:4], dtype=torch.float32).reshape(4, 160, -1)
print(f'fp32 tower: batch 4 vs alone, image0 min cos {cos(b[0], a).min():.6f}')
