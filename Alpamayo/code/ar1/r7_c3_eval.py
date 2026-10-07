"""ar1/r7_c3_eval.py -- R7 C3 blank-image control: B1 s42 and M1-v2a s42, BATCHED decoder,
variants real / cameras shuffled / blank (per-dim train mean). Gate GBL. Rules in the
R7A_REPORT.md header (C2 strata adopted after R4).
  python ar1/r7_c3_eval.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); tokv = [r['sample_token'] for r in rva]
    G = np.stack([np.asarray(T[t]['P'])[:6] for t in tokv]); g = np.linalg.norm(G, axis=2).max(1)
    v0 = np.array([V0[t]['v0_can'] for t in tokv])
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    ST = {'all': np.ones(len(rva), bool), 'STOPPED': (v0 <= 0.2) & (g <= 1.0), 'START': (v0 <= 0.2) & (g > 1.0),
          'MOVING': v0 > 0.2, 'WSS (sec.)': w}

    # decoder check: B1 batched vs per-sample dump, first 200 val records
    d = load('B1_real_check200', 'val'); ps = load('B1', 'val'); tau = Q.run('B1')['tau']
    R = [r for r in rva if r['sample_token'] in d['data']]
    same = np.mean([d['data'][r['sample_token']]['tok'] == ps['data'][r['sample_token']]['tok'] for r in R])
    a, b = preds(d, R, 'hybrid', tau), preds(ps, R, 'hybrid', tau)
    k = np.array([r['sample_token'] in d['data'] for r in rva])
    print(f'B1 decoder check, {len(R)} val: identical 24-token sequences {same:.3f}; L2@3s batched '
          f'{np.linalg.norm(a[:, 5] - G[k][:, 5], axis=1).mean():.4f} vs per-sample {np.linalg.norm(b[:, 5] - G[k][:, 5], axis=1).mean():.4f}')

    print('\nC3 (BATCHED decoder; C2 strata adopted after R4). L2@3s NoAvg | FG (STOPPED: pred > 1.0 m) | '
          'MG (START: pred <= 1.0 m)')
    print('  n: ' + ', '.join(f'{s} {m.sum()}' for s, m in ST.items()))
    res = {}
    for model, tau, V in (('B1 s42', Q.run('B1')['tau'], (('real', 'B1_real'), ('cams shuffled', 'B1_cams'), ('blank', 'B1_blank'))),
                          ('M1-v2a s42', 0.3, (('real', 'M1v2a_gen'), ('cams shuffled', 'M1v2a_cams'), ('blank', 'M1v2a_blank')))):
        print(f'  {model} (tau {tau})')
        for nm, tag in V:
            P = preds(load(tag, 'val'), rva, 'hybrid', tau)
            l2 = np.linalg.norm(P[:, 5] - G[:, 5], axis=1); dm = np.linalg.norm(P[:, :6], axis=2).max(1)
            res[(model, nm)] = l2
            print(f'    {nm:14} ' + ' '.join(f'{s} {l2[m].mean():.3f}' for s, m in ST.items()) +
                  f' | FG {(dm[ST["STOPPED"]] > 1.0).mean():.3f} | MG {(dm[ST["START"]] <= 1.0).mean():.3f}')
        for nm in ('cams shuffled', 'blank'):
            for s, m in ST.items():
                sub = [r for r, x in zip(rva, m) if x]
                c = E.scene_boot(sub, res[(model, nm)][m], res[(model, 'real')][m])
                print(f'      {nm} - real {s:10} {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]')
    c = E.scene_boot(rva, res[('M1-v2a s42', 'blank')], res[('M1-v2a s42', 'real')])
    print(f'\nGBL M1-v2a blank - real, all val: {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}] -> ' +
          ('"cameras unused by M1-v2a"' if c[1] <= 0 <= c[2] else '"cameras used by M1-v2a"'))


if __name__ == '__main__':
    main()
