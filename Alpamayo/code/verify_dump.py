"""verify_dump.py — GATE 3 VERIFICATION (required before any offline sweep is trusted).

The offline sweeps assume the argmax decode and the STOP-aware expectation decode
share one forward pass, because both feed back the ARGMAX token. This re-decodes
100 test samples the SLOW way -- one independent GPU pass per variant, through the
same decode_trajectory_ex path used in Gates 1/2 -- and compares the resulting
trajectories against the ones reconstructed offline from dump_test.pkl.

Reports max |ADE_slow - ADE_offline| over the 100 samples, per variant.
If this is not < 1e-6, every offline sweep result is invalid.

Single GPU. Usage: python verify_dump.py [n]
"""
import sys, pickle, math
import numpy as np, torch

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset
from tokenizer import TrajectoryTokenizer
from ar_eval import encode_live_one
import inference as INF
from inference import decode_trajectory_ex

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
MODELS = {
    'y1_full': (f'{CK}/_y1_full_turnw/alpamayo_best.pt',    False, False),
    'y1_ego':  (f'{CK}/_y1_egoonly_turnw/alpamayo_best.pt', True,  False),
    'zeroboth_jul12': (f'{CK}/_zeroboth_run_jul12/alpamayo_best_e1_val2.0392.pt', True, True),
}
DEV = 'cuda:0'; DT, N = 0.5, 12


def rollout(acc, cur, v0, yaw0):
    x = y = 0.0; yaw = yaw0; v = v0; out = np.empty((N, 2))
    for j in range(N):
        v = max(0.0, v + acc[j] * DT)
        yaw = yaw + v * cur[j] * DT
        x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
        out[j] = (x, y)
    return out


def ade(p, g):
    return float(np.linalg.norm(p - g, axis=1).mean())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    D = pickle.load(open(f'{RES}/dump_test.pkl', 'rb'))
    meta = D['meta']
    with open(INF.TRAJECTORIES_PATH, 'rb') as f: allt = pickle.load(f)
    _, _, test = build_scene_split(allt, INF.NUSCENES_ROOT)
    ds = NuScenesVLADataset(test, split='test', augment=False)
    tok = TrajectoryTokenizer()
    model = load_model(device=DEV)
    model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual = model.cosmos.model.visual
    torch.set_grad_enabled(False)
    idx = sorted(meta)[:n]

    print(f"[verify] re-decoding {len(idx)} test samples the SLOW way "
          f"(one GPU pass per variant)\n")
    worst = {}
    for mname, (path, zv, ze) in MODELS.items():
        ck = torch.load(path, map_location='cpu')
        model.load_state_dict(ck['model_state'], strict=False); model.zero_ego = ze
        dmax = {'argmax': 0.0, 'V1s': 0.0}
        vmax = {'argmax': 0.0, 'V1s': 0.0}
        for c, i in enumerate(idx):
            item = ds[i]
            vt = encode_live_one(visual, item['images'], DEV)
            if zv: vt = torch.zeros_like(vt)
            ego = item['ego_state']
            g = np.array(meta[i]['gt']); v0 = meta[i]['v0']; yaw0 = meta[i]['yaw0']
            rec = D['data'][mname][i]
            for lab, kw in (('argmax', dict(decode_mode='argmax')),
                            ('V1s', dict(decode_mode='expect', stop_aware=True))):
                # SLOW path: an independent forward pass for this variant
                _, acc, cur, _ = decode_trajectory_ex(model, vt, ego, tok, device=DEV, **kw)
                slow = rollout(acc, cur, v0, yaw0)
                off_v = np.array(rec['argmax' if lab == 'argmax' else 'expect'])
                offline = rollout(off_v[:12], off_v[12:], v0, yaw0)
                dmax[lab] = max(dmax[lab], abs(ade(slow, g) - ade(offline, g)))
                vmax[lab] = max(vmax[lab], float(np.abs(
                    np.concatenate([acc, cur]) - off_v).max()))
            if c % 25 == 0:
                print(f"  [{mname}] {c}/{len(idx)}", flush=True)
        worst[mname] = (dmax, vmax)
        print(f"  [{mname}] max|ADE_slow - ADE_offline|: "
              f"argmax {dmax['argmax']:.3e}  V1s {dmax['V1s']:.3e}   "
              f"max|value diff|: argmax {vmax['argmax']:.3e}  V1s {vmax['V1s']:.3e}")

    allmax = max(max(d.values()) for d, _ in worst.values())
    print(f"\n[verify] WORST max|ADE_slow - ADE_offline| over all models/variants = "
          f"{allmax:.3e}")
    print(f"[verify] {'PASS' if allmax < 1e-6 else 'FAIL'} (threshold 1e-6)")


if __name__ == '__main__':
    main()
