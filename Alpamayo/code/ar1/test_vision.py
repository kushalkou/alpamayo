"""ar1/test_vision.py -- R1.1: verify the native patch layout and the token count.
(1) CPU: patchify_native(preprocess(img)) == Qwen2VLImageProcessor(pixel_values) for a
    448x280 image, element-wise; grid_thw identical.
(2) Same comparison for the FROZEN week-1 recipe (vision_live.patchify, 448x448), as a
    side check only (nothing in the week-1 path is changed).
(3) --gpu: run the frozen Cosmos visual tower on 12 images (3 cams x 4 frames) of one
    sample; report the token count, and cos(features) native vs week-1 layout."""
import sys, pickle, numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
from PIL import Image
from transformers import Qwen2VLImageProcessor
import vision_ar1 as V
import vision_live as VL

MP = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'
H = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_hist.pkl', 'rb'))
st = next(k for k, v in H.items() if v['split'] == 'val' and v['n_pad'] == 0)
path = H[st]['cams']['CAM_FRONT'][-1][0]
proc = Qwen2VLImageProcessor.from_pretrained(MP)

img = Image.open(path).convert('RGB').resize((V.W_IMG, V.H_IMG), resample=Image.BILINEAR)
ref = proc(images=[img], do_resize=False, return_tensors='pt')
mine, grid = V.patchify_native(V.preprocess(path).unsqueeze(0))
print('(1) 448x280 native: grid', grid.tolist(), 'vs processor', ref['image_grid_thw'].tolist(),
      '| shape', tuple(mine.shape), tuple(ref['pixel_values'].shape),
      '| max |diff|', float((mine - ref['pixel_values']).abs().max()))
assert grid.tolist() == ref['image_grid_thw'].tolist()
assert float((mine - ref['pixel_values']).abs().max()) < 1e-4
print(f'    tokens per image after 2x2 merge: {int(grid[0, 1] * grid[0, 2] // 4)}')

img2 = Image.open(path).convert('RGB').resize((448, 448), resample=Image.BILINEAR)
ref2 = proc(images=[img2], do_resize=False, return_tensors='pt')['pixel_values']
leg, _ = VL.patchify(VL.preprocess_image(path).unsqueeze(0))
nat2, _ = V.patchify_native(VL.preprocess_image(path).unsqueeze(0))
print('(2) week-1 448x448 recipe vs processor: max |diff|', float((leg - ref2).abs().max()),
      '| native layout vs processor:', float((nat2 - ref2).abs().max()))
same_set = torch.allclose(leg.sort(0).values.sum(1), ref2.sort(0).values.sum(1), atol=1e-3)
print('    same multiset of patch values (order-only difference):', same_set)

if '--gpu' in sys.argv:
    from transformers import Qwen2_5_VLForConditionalGeneration
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(MP, torch_dtype=torch.float16,
                                                          device_map='cuda:0')
    vis = m.model.visual.eval()
    imgs = torch.stack([V.preprocess(p) for c in H[st]['cams'] for p, _ in H[st]['cams'][c]])
    torch.cuda.reset_peak_memory_stats()
    f = V.encode(vis, imgs.cuda())
    print(f'(3) 12 images -> {tuple(f.shape)} = {f.shape[0] // 12} tokens per image; finite '
          f'{bool(torch.isfinite(f).all())}; encoder peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')
    x = VL.preprocess_image(path).unsqueeze(0).cuda().half()
    hl, gl = VL.patchify(x); hn, gn = V.patchify_native(x)
    a = vis(hidden_states=hl, grid_thw=gl).pooler_output.float()
    b = vis(hidden_states=hn, grid_thw=gn).pooler_output.float()
    cs = torch.nn.functional.cosine_similarity(a, b, -1)
    print(f'    448x448 features, week-1 layout vs native: token cos mean {cs.mean():.3f} '
          f'min {cs.min():.3f}')
