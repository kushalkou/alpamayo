"""ar1/r23_eval.py -- R2.3 report (our Table 12 analog): token decoder vs flow-matching
expert on the SAME frozen A3 VLM (ego + cmd [P], no cameras), official val.

Rows (all on official val, nothing tuned on val):
  KIN                  kinematic rule (evalw1.kin)
  Ego-MLP + cmd [P]    seed 42 and the 3-seed mean (w1/egomlp)
  A3 token decoder     STOP-aware hybrid decode, tau fit on holdout (q2_traj.run('A3'))
  A3 + expert, 1 sample      noise seed 0 of the 6 (pre-registered single-sample row)
  A3 + expert, mean of 6     average of the 6 trajectories (extra; not selected on)
Metrics: evalw1 table (L2 NoAvg / TemAvg 1-3 s, Col% both conventions, ADE / FDE@6s
mean / med / p95, L2@3s med / p95) + ADE@3s, minADE_6@3s (all, n_fut >= 6) and
minADE_6@6s (n_fut = 12). Deterministic rows: minADE_6 = their ADE (one sample).
Both strata: ALL 5,119 and EXCL. first frames. Paired scene bootstrap (10,000).
  python ar1/r23_eval.py
"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import evalw1 as E
import q2_traj as Q

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data()
    X = pickle.load(open(f'{RES}/ar1_expert_A3.pkl', 'rb'))
    assert X['val']['tokens'] == [r['sample_token'] for r in rva]
    tr = X['val']['traj']                                         # [N,6,12,2]
    G = [np.asarray(T[r['sample_token']]['P']) for r in rva]
    n12 = np.array([r['n_fut'] == 12 for r in rva])
    rows = {'KIN': C['cache']['KIN'], 'Ego-MLP + cmd s42 [P]': C['cache']['egoMLP s42'],
            'Ego-MLP + cmd 3-seed [P]': Q.avg([C['cache'][f'egoMLP s{s}'] for s in Q.SEEDS]),
            'A3 token decoder [P]': Q.strip(Q.run('A3'))}
    if 'exp0' not in X:
        X['exp0'] = E.per_sample(rva, tr[:, 0], C['occ'])
        X['expm'] = E.per_sample(rva, tr.mean(1), C['occ'])
        pickle.dump(X, open(f'{RES}/ar1_expert_A3.pkl', 'wb'))
    rows['A3 + expert, 1 sample [P]'] = X['exp0']
    rows['A3 + expert, mean of 6 [P]'] = X['expm']
    ade3 = {k: v['l2'][:, :6].mean(1) for k, v in rows.items()}
    e3 = np.stack([np.linalg.norm(tr[i, :, :6] - G[i][None, :6], axis=2).mean(1) for i in range(len(rva))])
    e6 = np.full(e3.shape, np.nan)
    for i in np.where(n12)[0]:
        e6[i] = np.linalg.norm(tr[i] - G[i][None, :12], axis=2).mean(1)
    chk = np.abs(e3[:, 0] - ade3['A3 + expert, 1 sample [P]']).max()
    print(f'consistency: expert sample-0 ADE@3s from trajectories vs plan_metrics L2 max |diff| {chk:.2e}')
    minade = {k: (ade3[k], v['ade']) for k, v in rows.items()}
    minade['A3 + expert, 1 sample [P]'] = (e3.min(1), np.nanmin(e6, 1))
    minade['A3 + expert, mean of 6 [P]'] = (e3.min(1), np.nanmin(e6, 1))
    for blk, m in Q.blocks():
        sub = [r for r, k in zip(rva, m) if k]
        cut = lambda p: {k: v[m] for k, v in p.items()}
        print(f'\n===== {blk} =====')
        print(E.HDR + ' | ADE@3s | minADE6 @3s @6s')
        for k, v in rows.items():
            a3, (m3, m6) = ade3[k][m], minade[k]
            m6 = m6[m]
            star = '' if 'expert' in k else '  (= ADE, 1 sample)'
            print(E.row(k, cut(v)) + f' | {a3.mean():.3f} | {m3[m].mean():.3f} {np.nanmean(m6):.3f}{star}')
        print('  paired scene bootstrap (95% CI):')
        tok = cut(rows['A3 token decoder [P]'])
        for k in ('A3 + expert, 1 sample [P]', 'A3 + expert, mean of 6 [P]'):
            print(E.compare(sub, cut(rows[k]), tok, f'{k[:26]} - token'))
        print(E.compare(sub, cut(rows['A3 + expert, 1 sample [P]']), cut(rows['Ego-MLP + cmd s42 [P]']),
                        'expert 1 sample - Ego-MLP s42'))
        print(E.compare(sub, cut(rows['A3 + expert, 1 sample [P]']), cut(rows['KIN']), 'expert 1 sample - KIN'))
        c = E.scene_boot(sub, minade['A3 + expert, 1 sample [P]'][1][m], rows['A3 token decoder [P]']['ade'][m])
        print(f'  {"expert minADE_6@6s - token ADE@6s":34} {E.fmt_ci(c)}')
        st = C['st'][m]
        print('  strata, L2@3s NoAvg mean (n): ' + '; '.join(
            f'{s} ({int((st == s).sum())}): token {tok["l2"][st == s, 5].mean():.3f} / expert '
            f'{cut(rows["A3 + expert, 1 sample [P]"])["l2"][st == s, 5].mean():.3f} / MLP '
            f'{cut(rows["Ego-MLP + cmd s42 [P]"])["l2"][st == s, 5].mean():.3f}'
            for s in ('straight', 'turning', 'stationary')))
    print(f'\nexpert: params {X["params"]:,}; best epoch {X["best_epoch"]} (holdout median ADE@6s '
          f'{X["ho_best_med"]:.4f}); train {X["train_gpu_h"]:.2f} GPU-h; 6-sample inference holdout + val '
          f'{X["infer_gpu_h"] * 60:.1f} GPU-min')
    for h in X['hist']:
        print(f'  epoch {h["epoch"]:3d} fm_loss {h["fm_loss"]:.4f} holdout ADE@6s median {h["ho_ade_med"]:.4f} '
              f'mean {h["ho_ade_mean"]:.4f}')


if __name__ == '__main__':
    main()
