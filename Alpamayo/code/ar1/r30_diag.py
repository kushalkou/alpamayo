"""ar1/r30_diag.py -- R3.0 camera-harm diagnosis (CPU; existing dumps; no training).

a. Stationary val samples (stratum: v0_can < 0.5 m/s; 993 all, 853 excl. first frames).
   Pre-registered thresholds (fixed before looking), over the first 3 s (steps 1..6):
     GT stopped  = max displacement from the origin < 0.5 m; GT moves = > 1.0 m
     pred goes   = max displacement > 1.0 m;               pred stays = < 0.5 m
   false-go   = share of GT-stopped samples where the prediction goes
   false-stop = share of GT-moving samples where the prediction stays
   Predictions: hybrid decode with tau fit on holdout (q2_traj convention).
   Models: no-camera VLA seeds 42 / 123 / 2024 (A3, G6C_s123, G6C_s2024), B1, B2.
b. Overfitting: AR ADE@6s (mean / median) on a FIXED 1,000-sample train subset (n_fut =
   12, seed 0; dump_w1 --rand_subset 1000) vs official val (n_fut = 12), for no camera
   seed 42 (A3) and B1; gap = val - train. Same decode and tau as on val.
  python ar1/r30_diag.py [a|b]
"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import evalw1 as E, records
import q2_traj as Q
from gate_a import load, preds, ade12, TAUS

MODELS = (('no cam s42 (A3)', 'A3'), ('no cam s123', 'G6C_s123'), ('no cam s2024', 'G6C_s2024'),
          ('B1 3 cams t0', 'B1'), ('B2 3 cams x 4 kf', 'B2'))


def tau_of(tag):
    return Q.run(tag)['tau']


def part_a():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data()
    G = np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva])
    gmax = np.linalg.norm(G, axis=2).max(1)
    stat = C['st'] == 'stationary'
    print('R3.0a STATIONARY: false-go = P(pred > 1 m within 3 s | GT < 0.5 m); '
          'false-stop = P(pred < 0.5 m | GT > 1 m)')
    for blk, m in Q.blocks():
        s = stat & m
        gs, gm = s & (gmax < 0.5), s & (gmax > 1.0)
        print(f'\n  {blk}: stationary n={int(s.sum())}; GT stopped {int(gs.sum())}, GT moves {int(gm.sum())}, '
              f'in between {int((s & ~(gmax < 0.5) & ~(gmax > 1.0)).sum())}')
        print(f'    {"model":20} {"false-go":>9} {"false-stop":>11} {"pred median travel 3s, GT stopped (m)":>40}')
        fg = {}
        for nm, tag in MODELS:
            P = preds(load(tag, 'val'), rva, 'hybrid', tau_of(tag))[:, :6]
            pmax = np.linalg.norm(P, axis=2).max(1)
            fg[nm] = ((pmax[gs] > 1.0).mean(), (pmax[gm] < 0.5).mean())
            print(f'    {nm:20} {fg[nm][0]:9.3f} {fg[nm][1]:11.3f} {np.median(pmax[gs]):40.3f}')
        nc = np.mean([fg[k] for k in fg if k.startswith('no cam')], 0)
        print(f'    {"no cam 3-seed mean":20} {nc[0]:9.3f} {nc[1]:11.3f}')


def part_b():
    C = Q.ctx(); W, V0, T = E.data()
    print('R3.0b OVERFITTING: AR ADE@6s, fixed 1,000-sample train subset vs val (n_fut = 12); '
          'hybrid decode, tau from holdout')
    rva12 = [r for r in C['rva'] if r['n_fut'] == 12]
    print(f'    {"model":20} {"train mean / med":>18} {"val mean / med":>16} {"gap val-train (mean / med)":>28}')
    for nm, tag, sub in (('no cam s42 (A3)', 'A3', 'A3_trsub'), ('B1 3 cams t0', 'B1', 'B1_trsub')):
        tau = tau_of(tag)
        dt = load(sub, 'train')
        rtr = [r for r in E.split_records('train') if r['sample_token'] in dt['data']]
        assert len(rtr) == 1000, len(rtr)
        ptr = preds(dt, rtr, 'hybrid', tau)
        etr = np.array([np.linalg.norm(ptr[j] - T[r['sample_token']]['P'][:12], axis=1).mean() for j, r in enumerate(rtr)])
        pva = preds(load(tag, 'val'), rva12, 'hybrid', tau)
        eva = np.array([np.linalg.norm(pva[j] - T[r['sample_token']]['P'][:12], axis=1).mean() for j, r in enumerate(rva12)])
        print(f'    {nm:20} {etr.mean():8.3f} / {np.median(etr):6.3f} {eva.mean():7.3f} / {np.median(eva):6.3f}'
              f' {eva.mean() - etr.mean():+12.3f} / {np.median(eva) - np.median(etr):+6.3f}')


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'ab'
    if 'a' in arg: part_a()
    if 'b' in arg: part_b()
