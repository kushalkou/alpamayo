"""ar1/r20_patch_check.py -- R2.0a: R1 image path vs HF Qwen2.5-VL processor + vision tower.
5 val keyframe images (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, from 5 scenes).
  NEW  vision_ar1 (448x280, native layout) vs HF processor (same 448x280 PIL image,
       do_resize=False) -> frozen tower, fp16; per-token cosine; pass if min >= 0.999.
  OLD  vision_live.patchify (448x448, week-1 recipe) vs HF processor on the same 448x448
       image; reported only.
python ar1/r20_patch_check.py   (one GPU)"""
import sys, pickle, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
from PIL import Image
from transformers import Qwen2VLImageProcessor, Qwen2_5_VLForConditionalGeneration
import vision_ar1 as V, vision_live as VL

MP = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'
H = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_hist.pkl', 'rb'))
proc = Qwen2VLImageProcessor.from_pretrained(MP)
vis = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MP, torch_dtype=torch.float16, device_map='cuda:0').model.visual.eval()
cams = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT')
val = [h for h in H.values() if h['split'] == 'val' and h['n_pad'] == 0]
picks = [val[i * (len(val) // 5)]['cams'][cams[i]][-1][0] for i in range(5)]   # 5 scenes

@torch.no_grad()
def hf(img):
    r = proc(images=[img], do_resize=False, return_tensors='pt')
    return vis(hidden_states=r['pixel_values'].cuda().half(),
               grid_thw=r['image_grid_thw'].cuda()).pooler_output.float()


cos = lambda a, b: torch.nn.functional.cosine_similarity(a, b, -1)
ok = True
print(f'{"image":58} {"NEW mean":>9} {"NEW min":>8} {"OLD mean":>9} {"OLD min":>8}')
for p in picks:
    im = Image.open(p).convert('RGB')
    a = hf(im.resize((448, 280), Image.BILINEAR))
    b = V.encode(vis, V.preprocess(p).unsqueeze(0).cuda()).float()
    c = hf(im.resize((448, 448), Image.BILINEAR))
    x = VL.preprocess_image(p).unsqueeze(0).cuda().half()
    hl, gl = VL.patchify(x)
    d = vis(hidden_states=hl, grid_thw=gl).pooler_output.float()
    n, o = cos(a, b), cos(c, d)
    ok &= bool(n.min() >= 0.999)
    print(f'{p.split("/")[-1][:58]:58} {n.mean():9.5f} {n.min():8.5f} {o.mean():9.3f} {o.min():8.3f}')
print('NEW path vs HF: ' + ('PASS (every token cosine >= 0.999)' if ok else 'FAIL'))
