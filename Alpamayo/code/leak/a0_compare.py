"""leak/a0_compare.py -- WEEK1 STAGE A0: visual tokens REMOVED vs ZEROED, causal ego-only,
custom split, old recipe (Y1 ego, seed 42). Paired on the causal test subset (n=3358).
Compares standalone V1s ADE@6s, the blend (alpha fit on val) vs CV, and blend[A0] -
blend[causal_ego]. The AR-val selection ADEs come from the two training logs."""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/leak')
import causal_eval as CE
from gate41 import ade_arr, fit_alpha

R = CE.RES


def ld(split):
    D, idx = CE.load(split, 'causal_ego')
    A = pickle.load(open(f'{R}/dump_{split}_A0_causal_ego_novis.pkl', 'rb'))
    D['data'].update(A['data'])
    assert sorted(A['data']['A0_causal_ego_novis']) == idx
    return D, idx


Dv, iv = ld('val'); Dt, it = ld('test')
_, Ct, Gt = CE.arrs(Dt, it, 'causal_ego', 'argmax')
cv = ade_arr(Ct, Gt)
print(f'causal test subset n={len(it)}  CV {cv.mean():.3f}')
BL, ST = {}, {}
for m in ('causal_ego', 'A0_causal_ego_novis'):
    Mv, Cv, Gv = CE.arrs(Dv, iv, m, 'expect'); Mt, _, _ = CE.arrs(Dt, it, m, 'expect')
    a, _ = fit_alpha(Mv, Cv, Gv)
    BL[m] = ade_arr(a * Mt + (1 - a) * Ct, Gt); ST[m] = ade_arr(Mt, Gt)
    print(f'  {m:22} V1s {ST[m].mean():.3f} (med {np.median(ST[m]):.3f})  alpha* {a:.2f}  '
          f'blend {BL[m].mean():.3f}  blend-CV {CE.ci(BL[m], cv)}')
print(f'  V1s   removed - zeroed: {CE.ci(ST["A0_causal_ego_novis"], ST["causal_ego"])}')
print(f'  blend removed - zeroed: {CE.ci(BL["A0_causal_ego_novis"], BL["causal_ego"])}')
