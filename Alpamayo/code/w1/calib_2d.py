"""w1/calib_2d.py -- STEP 2d: loss calibration of the OLD causal_ego checkpoint (custom split,
causal val subset, FREE-RUNNING: argmax fed back). Per accel slot: KL(pred marginal || GT
marginal) over the 65-way support (64 bins + STOP), where pred marginal = mean of the
per-sample softmax and GT marginal = token histogram (add 1e-6); mean predicted accel
(STOP-aware expectation) vs mean GT accel (continuous target; STOP steps count as 0, the
tokenizer's value). Report only."""
import sys, pickle, contextlib, io, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from tokenizer import TrajectoryTokenizer, STOP_TOKEN
with contextlib.redirect_stdout(io.StringIO()): tok = TrajectoryTokenizer()
R = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
D = pickle.load(open(f'{R}/dump_val_causal_ego_probs.pkl', 'rb'))
S = pickle.load(open(f'{R}/leak_split.pkl', 'rb'))['val']
C = D['ce']['causal_ego']; idx = sorted(C)
Q = np.stack([C[i]['q65'] for i in idx])                     # [n,24,65]
G = np.array([[64 if g == STOP_TOKEN else g for g in C[i]['gt']] for i in idx])
EV = np.array([D['data']['causal_ego'][i]['expect'] for i in idx])
GA = np.array([[0.0 if C[i]['gt'][j] == STOP_TOKEN else S[i]['future_accelerations'][j]
                for j in range(12)] for i in idx])
print(f'n={len(idx)} (custom-split val, >=2 past poses), free-running')
print(' slot  KL(pred||GT)  H(pred marg)  H(GT marg)  mean pred accel  mean GT accel  pred-GT  p(STOP) pred/GT')
for j in range(12):
    p = Q[:, j].mean(0); g = np.bincount(G[:, j], minlength=65) / len(idx) + 1e-6; g /= g.sum()
    kl = float((p * np.log(np.maximum(p, 1e-12) / g)).sum())
    H = lambda x: float(-(x * np.log(np.maximum(x, 1e-12))).sum())
    print(f'  a{j:<3} {kl:12.4f} {H(p):13.3f} {H(g):11.3f} {EV[:, j].mean():16.3f} {GA[:, j].mean():14.3f} '
          f'{EV[:, j].mean()-GA[:, j].mean():+8.3f}   {p[64]:.3f}/{g[64]:.3f}')
