"""ar1/r7a_eval.py -- R7 Part A: no-route arms (B1-NR, M1-v2a-NR, Ego-MLP-NR), gates GN1, GN2
and per-run diagnostics. Rules in the R7A_REPORT.md header. Per-sample dumps only.
  python ar1/r7a_eval.py"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r33_eval import derive, controls
from r5_a5c1 import coll_rates
from meta_actions import LON, LAT

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
SEEDS = (42, 123, 2024)


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); ff = C['ff']; occ = C['occ']
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    S = {'all': np.ones(len(rva), bool), 'excl. first frames': ~ff, 'excl. WSS': ~w, 'WSS only': w}
    g = np.linalg.norm(np.stack([np.asarray(T[r['sample_token']]['P'])[:6] for r in rva]), axis=2).max(1)
    stat = C['st'] == 'stationary'; gs = stat & (g < 0.5); gm = stat & (g > 1.0)

    # per-run predictions and per-sample metrics
    PR, PS = {}, {}
    for t in ('B1', 'B1_NR', 'M1v2a', 'M1v2a_NR'):
        PS[t] = Q.strip(Q.run(t)); PR[t] = preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau'])
        print(f'tau {t} {Q.run(t)["tau"]}')
    mc = pickle.load(open(f'{RES}/w1_egomlp.pkl', 'rb')); mn = pickle.load(open(f'{RES}/w1_egomlp_nr.pkl', 'rb'))
    assert mc['val_tokens'] == mn['val_tokens'] == [r['sample_token'] for r in rva]
    for s in SEEDS:
        for nm, M in (('EgoMLP', mc), ('EgoMLP_NR', mn)):
            PR[f'{nm} s{s}'] = M['preds'][s]['val']; PS[f'{nm} s{s}'] = E.per_sample(rva, M['preds'][s]['val'], occ)
        print(f'Ego-MLP holdout ADE@6s s{s}: + cmd {mc["preds"][s]["holdout_ade"]:.3f}, NR {mn["preds"][s]["holdout_ade"]:.3f}')
    PS['EgoMLP'] = Q.avg([PS[f'EgoMLP s{s}'] for s in SEEDS]); PS['EgoMLP_NR'] = Q.avg([PS[f'EgoMLP_NR s{s}'] for s in SEEDS])

    print('\nL2@3s NoAvg / ADE@6s (n_fut = 12) per stratum')
    print(f'  {"run":12} ' + ' '.join(f'{k:>20}' for k in S))
    for t in ('B1', 'B1_NR', 'M1v2a', 'M1v2a_NR', 'EgoMLP', 'EgoMLP_NR'):
        print(f'  {t:12} ' + ' '.join(f'{PS[t]["l2"][m, 5].mean():9.3f} {np.nanmean(PS[t]["ade"][m]):10.3f}' for m in S.values()))
    for nm in ('EgoMLP', 'EgoMLP_NR'):
        print(f'  {nm} per seed L2@3s ' + ' '.join(f'{PS[f"{nm} s{s}"]["l2"][:, 5].mean():.3f}' for s in SEEDS))

    G = {}
    print('\nPaired scene bootstrap (10,000), L2@3s NoAvg; ADE@6s after')
    for pair, a, b in (('GN1 B1-NR - B1 s42', 'B1_NR', 'B1'), ('GN1 M1-v2a-NR - M1-v2a s42', 'M1v2a_NR', 'M1v2a'),
                       ('GN2 M1-v2a-NR - B1-NR', 'M1v2a_NR', 'B1_NR'), ('ref M1-v2a - B1 s42 (route)', 'M1v2a', 'B1'),
                       ('Ego-MLP NR - + cmd (3 seeds)', 'EgoMLP_NR', 'EgoMLP')):
        for sn, m in S.items():
            sub = [r for r, k in zip(rva, m) if k]
            c = E.scene_boot(sub, PS[a]['l2'][m, 5], PS[b]['l2'][m, 5])
            c2 = E.scene_boot(sub, PS[a]['ade'][m], PS[b]['ade'][m]); G[(pair, sn)] = c
            print(f'  {pair:30} {sn:19} n {m.sum():4d}  {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]   '
                  f'ADE6 {c2[0]:+.3f} [{c2[1]:+.3f},{c2[2]:+.3f}]')

    print('\nPer run: false-go all / WSS / non-WSS (n GT-stopped stationary %d), missed-go (n %d); collision @3 s '
          'NoAvg / TemAvg vehicles, VAD port (aa) | yaw-aware' % (gs.sum(), gm.sum()))
    for t in ('B1', 'B1_NR', 'M1v2a', 'M1v2a_NR') + tuple(f'{n} s{s}' for n in ('EgoMLP', 'EgoMLP_NR') for s in SEEDS):
        d = np.linalg.norm(PR[t][:, :6], axis=2).max(1)
        fg = lambda m: (d[m & gs] > 1.0).mean()
        ca = coll_rates(rva, PR[t], occ, 'aa'); cy = coll_rates(rva, PR[t], occ, 'yaw')
        print(f'  {t:14} FG {fg(np.ones(len(rva), bool)):.3f} / {fg(w):.3f} / {fg(~w):.3f}  MG {(d[gm] < 0.5).mean():.3f}  '
              f'Col aa {ca[0]:.2f}/{ca[1]:.2f} | yaw {cy[0]:.2f}/{cy[1]:.2f}')

    print('\nWords (val n_fut = 12, 2 Hz labels): collapse share, consistency, per-class F1 pooled over slots')
    lab = pickle.load(open(f'{DATA}/ar1_meta2hz.pkl', 'rb')); r12 = [r for r in rva if r['n_fut'] == 12]
    y = np.array([lab[r['sample_token']] for r in r12]); maj = np.array([4] * 6 + [6] * 6)
    for t in ('M1v2a', 'M1v2a_NR'):
        dv = load(t, 'val'); gen = np.array([dv['data'][r['sample_token']]['meta_gen'] for r in r12])
        f = []
        for o, nm in ((0, LON), (6, LAT)):
            yy, pp = y[:, o:o + 6].ravel(), gen[:, o:o + 6].ravel()
            for c in range(7):
                n = (yy == c).sum()
                if n < 30:
                    continue
                tp = ((pp == c) & (yy == c)).sum(); P_ = tp / max((pp == c).sum(), 1); R_ = tp / n
                f.append(f'{nm[c][:9]} {2 * P_ * R_ / max(P_ + R_, 1e-9):.2f}')
        ga = np.array([dv['data'][r['sample_token']]['meta_gen'] for r in rva])
        der = np.array([derive(a, k, T[r['sample_token']]['v0']) for (a, k), r in zip(controls(dv, rva, Q.run(t)['tau']), rva)])
        print(f'  {t:10} collapse {(gen == maj).all(1).mean():.3f} (labels {(y == maj).all(1).mean():.3f}); consistency '
              f'{(der == ga).all(1).mean():.3f}; F1: ' + ', '.join(f))

    c = G[('GN2 M1-v2a-NR - B1-NR', 'all')]
    v = ('"meta helps without route"' if c[2] < 0 else '"meta hurts without route (AR1 direction)"' if c[1] > 0
         else '"flat without route"')
    a1 = G[('GN1 B1-NR - B1 s42', 'all')]; a2 = G[('GN1 M1-v2a-NR - M1-v2a s42', 'all')]
    print('\nGATES (one seed, indicative)')
    print(f'  GN1 B1-NR - B1 s42 {a1[0]:+.3f} [{a1[1]:+.3f},{a1[2]:+.3f}]; '
          f'M1-v2a-NR - M1-v2a s42 {a2[0]:+.3f} [{a2[1]:+.3f},{a2[2]:+.3f}]')
    print(f'  GN2 M1-v2a-NR - B1-NR {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}] -> {v}')


if __name__ == '__main__':
    main()
