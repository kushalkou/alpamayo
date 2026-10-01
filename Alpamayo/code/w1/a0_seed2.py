"""w1/a0_seed2.py -- QUEUE v2 G9: A0 with a second seed. As leak/a0_compare.py (causal test
subset, alpha fit on the custom val), adding A0 seed 123 (visual tokens REMOVED). Rows:
zeroed s42 (causal_ego), removed s42 (A0), removed s123; pairwise paired CIs (CE.ci); the
seed-to-seed difference of the removed variant is the yardstick for the A0 flag."""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/leak')
import causal_eval as CE
from gate41 import ade_arr, fit_alpha

R = CE.RES
MS = ('causal_ego', 'A0_causal_ego_novis', 'A0_causal_ego_novis_s123')


def ld(split):
    D, idx = CE.load(split, 'causal_ego')
    for m in MS[1:]:
        A = pickle.load(open(f'{R}/dump_{split}_{m}.pkl', 'rb'))
        D['data'].update(A['data']); assert sorted(A['data'][m]) == idx
    return D, idx


Dv, iv = ld('val'); Dt, it = ld('test')
_, Ct, Gt = CE.arrs(Dt, it, 'causal_ego', 'argmax')
cv = ade_arr(Ct, Gt)
print(f'causal test subset n={len(it)}  CV {cv.mean():.3f}')
BL, ST = {}, {}
for m in MS:
    Mv, Cv, Gv = CE.arrs(Dv, iv, m, 'expect'); Mt, _, _ = CE.arrs(Dt, it, m, 'expect')
    a, _ = fit_alpha(Mv, Cv, Gv)
    BL[m] = ade_arr(a * Mt + (1 - a) * Ct, Gt); ST[m] = ade_arr(Mt, Gt)
    print(f'  {m:26} V1s mean {ST[m].mean():.3f} med {np.median(ST[m]):.3f} p95 {np.percentile(ST[m], 95):.3f}'
          f'  alpha* {a:.2f}  blend {BL[m].mean():.3f}  blend-CV {CE.ci(BL[m], cv)}')
for a_, b_ in ((MS[1], MS[0]), (MS[2], MS[0]), (MS[2], MS[1])):
    print(f'  V1s   {a_} - {b_}: {CE.ci(ST[a_], ST[b_])}')
    print(f'  blend {a_} - {b_}: {CE.ci(BL[a_], BL[b_])}')
