"""ar1/r6_verify_eval.py -- R6 Part 1 (V1, V2, gates GV1-GV3) from the batched dumps.
Reference = the batched 'gen' decode (same decoder as every variant). tau 0.3 (M1-v2a holdout).
  python ar1/r6_verify_eval.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r33_eval import derive, controls

TAU = 0.3
DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); ff = C['ff']
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    G = np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva]); g = np.linalg.norm(G, axis=2).max(1)
    stat = C['st'] == 'stationary'; gs = stat & (g < 0.5)
    res = {}
    print('V1 / V2  M1-v2a s42, batched decoder, all 5,119 val (force_s1: 676 GT-stopped stationary)')
    print(f'  {"variant":34} {"n":>5} {"L2@3s":>7} {"d vs own":>9} {"false-go":>9} {"consistency":>12}')
    for var, lab in (('gen', 'own words (reference)'), ('cams', 'V1 cameras shuffled'), ('ego', 'V1 ego shuffled'),
                     ('cmd', 'V1 nav command shuffled'), ('force_gt', 'V2 (i) GT 2 Hz words [ORACLE]'),
                     ('force_maj', 'V2 (ii) all maintain / straight'), ('force_s1', 'V2 (iii) slot-1 = gentle_acc')):
        d = load(f'M1v2a_{var}', 'val')
        R = [r for r in rva if r['sample_token'] in d['data']]
        k = np.array([r['sample_token'] in d['data'] for r in rva])
        P = preds(d, R, 'hybrid', TAU)
        l2 = np.linalg.norm(P[:, 5] - G[k][:, 5], axis=1)
        dm = np.linalg.norm(P[:, :6], axis=2).max(1)
        fg = (dm[gs[k]] > 1.0).mean()
        words = np.array([d['data'][r['sample_token']]['meta_gen'] for r in R])
        der = np.array([derive(a, kk, T[r['sample_token']]['v0']) for (a, kk), r in zip(controls(d, R, TAU), R)])
        cons = (der == words).all(1).mean()
        res[var] = dict(l2=l2.mean(), fg=fg, cons=cons, k=k, l2v=l2)
        dl = f'{l2.mean() - res["gen"]["l2"]:+.3f}' if var not in ('gen', 'force_s1') else '-'
        print(f'  {lab:34} {k.sum():5d} {l2.mean():7.3f} {dl:>9} {fg:9.3f} {cons:12.3f}')
        if var in ('cams', 'ego', 'cmd', 'force_gt', 'force_maj'):
            c = E.scene_boot(rva, l2, res['gen']['l2v'])
            print(f'  {"":34} paired scene bootstrap vs own: {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]')
        if var == 'gen':
            for nm, m in (('excl. first frames', ~ff), ('excl. WSS', ~w), ('WSS only', w)):
                print(f'  {"   " + nm:34} {m.sum():5d} {l2[m].mean():7.3f}')
    own_s1 = res['gen']['l2v']
    gv1 = res['gen']['l2'] - res['force_gt']['l2']
    gv2 = res['ego']['l2'] - res['gen']['l2']
    fg_own = res['gen']['fg']
    print(f'\nGV1 own - forced-GT L2@3s = {gv1:+.3f} m (>= 0.10 required) -> {"PASS" if gv1 >= 0.10 else "FAIL (leak suspect, STOP)"}')
    print(f'GV2 ego-shuffle - own L2@3s = {gv2:+.3f} m (>= 0.30 required) -> {"PASS" if gv2 >= 0.30 else "FAIL (STOP)"}')
    print(f'GV3 forced slot-1 gentle_acc false-go = {res["force_s1"]["fg"]:.3f} (own on the same samples {fg_own:.3f}) -> '
          f'{"words causally drive the start decision" if res["force_s1"]["fg"] >= 0.50 else "words not causal at standstill"}')


if __name__ == '__main__':
    main()
