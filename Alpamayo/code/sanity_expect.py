"""sanity_expect.py — GATE 1 pre-check: is expectation decoding biased?

(a) mean of the 64 accel bin centers  — the [-11,+10] range is asymmetric, so a
    UNIFORM slot distribution decodes to a spurious braking value.
(b) the accel value implied by the model's ACTUAL averaged predicted distribution
    over 100 val samples (all 12 accel slots), i.e. E_p[a] under mean p.

Also verifies decode_trajectory_ex(decode_mode='argmax') is token-identical to the
existing ar_eval.ar_decode path (the argmax path must not have changed).
"""
import sys, pickle, numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset
from tokenizer import TrajectoryTokenizer
from ar_eval import encode_live_one, ar_decode, fixed_val_indices
import inference as INF
from inference import decode_trajectory_ex

CKPT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints/_y1_full_turnw/alpamayo_best.pt'
DEV = 'cuda:0'; N = 100

tok = TrajectoryTokenizer()
ac, cc = tok.accel_centers, tok.curv_centers
print(f"\n(a) mean of 64 ACCEL bin centers = {ac.mean():+.4f} m/s²   "
      f"(range [{ac[0]:.3f},{ac[-1]:.3f}], width {ac[1]-ac[0]:.4f})")
print(f"    mean of 64 CURV  bin centers = {cc.mean():+.4f} rad/m")

with open(INF.TRAJECTORIES_PATH, 'rb') as f: allt = pickle.load(f)
_, val, _ = build_scene_split(allt, INF.NUSCENES_ROOT)
ds = NuScenesVLADataset(val, split='val', augment=False)
idx = [int(i) for i in fixed_val_indices(len(val), k=400)][:N]

model = load_model(device=DEV); model.cosmos.model.language_model.gradient_checkpointing_disable()
model.eval(); model.zero_ego = False
ck = torch.load(CKPT, map_location='cpu'); model.load_state_dict(ck['model_state'], strict=False)
visual = model.cosmos.model.visual
torch.set_grad_enabled(False)

P = np.zeros((24, 64)); mism = 0
import time; t0 = time.time()
for c, i in enumerate(idx):
    item = ds[i]
    vt = encode_live_one(visual, item['images'], DEV)
    toks_e, _, _, st = decode_trajectory_ex(model, vt, item['ego_state'], tok, device=DEV,
                                            decode_mode='argmax', collect_stats=True)
    for s in range(24): P[s] += st[s]['p']
    if c < 8:
        toks_r, _, _ = ar_decode(model, vt, item['ego_state'], tok, DEV)
        mism += int(list(toks_r) != list(toks_e))
    if c % 25 == 0: print(f"  {c}/{N}  {time.time()-t0:.0f}s", flush=True)
P /= len(idx)

pa = P[:12].mean(0); pk = P[12:].mean(0)
print(f"\n(b) E[accel] under the model's averaged predicted distribution "
      f"(n={len(idx)} val samples, 12 accel slots) = {float((pa*ac).sum()):+.4f} m/s²")
print(f"    E[curv]  under the averaged predicted distribution              "
      f"= {float((pk*cc).sum()):+.5f} rad/m")
print(f"    per-slot E[accel]: " + " ".join(f"{float((P[s]*ac).sum()):+.3f}" for s in range(12)))
print(f"    per-slot E[curv] : " + " ".join(f"{float((P[s]*cc).sum()):+.4f}" for s in range(12, 24)))
print(f"\n    reference: uniform-distribution decode -> accel {ac.mean():+.4f}, curv {cc.mean():+.4f}")
print(f"\n[argmax equivalence] decode_trajectory_ex(argmax) vs ar_decode: "
      f"{8-mism}/8 identical token sequences")
