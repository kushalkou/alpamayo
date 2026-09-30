"""w1/ladder.py -- QUEUE C1: mini information ladder on the Ego-MLP (a SMALL-MODEL
demonstration of the ladder logic, not the VLA).

Rungs (Ego-MLP recipe of w1/ego_mlp.py: 512-512 MLP, L1 on 12 lidar-frame waypoints,
100 epochs, holdout selection with TRUE labels; 3 seeds; CPU):
  L0   CV; kinematic rule (KIN)                                   no learning
  L1   Ego-MLP, ego only (16 causal features)
  L2   Ego-MLP, ego + cmd                                         [privileged: cmd]
  L2b  Ego-MLP, cmd only, no ego                                  [privileged: cmd]
  L5   Ego-MLP, ego + oracle meta-action lon(4)+lat(3) one-hot    [privileged: oracle]
  L5n  L5 trained with 10% flips (seed 42, as A2), tested at 0/10/20/40% flips (seed 777)
  OKIN oracle-kinematic rule (no learning), also at the same test flips
Gap closed = (X - L2) / (L5 - L2) on seed-mean ADE@6s and on L2@3s NoAvg.
Eval: evalw1 on official val, ALL 5,119 and EXCL. the 140 first frames (4,979); paired
scene-level bootstrap vs CV, KIN, L2.
"""
import sys, pickle, time
import numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records, evalw1 as E, plan_metrics as PMX
from ego_mlp import train_one

torch.set_num_threads(24)
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/w1_ladder.pkl'
SEEDS = (42, 123, 2024)
FLIPS = (0.0, 0.1, 0.2, 0.4)
EXTRA = ('cmd', 'lat', 'lon')


def feats(R, kind):
    ego = np.stack([t['w1_ego'][:4].numpy().ravel() for t in R])
    cmd = np.eye(3)[[t['shown']['cmd'] for t in R]]
    lon = np.eye(4)[[t['shown']['lon'] for t in R]]
    lat = np.eye(3)[[t['shown']['lat'] for t in R]]
    F = {'L1': [ego], 'L2': [ego, cmd], 'L2b': [cmd], 'L5': [ego, lon, lat]}[kind]
    return np.concatenate(F, 1).astype(np.float32)


def fit(kind, train_flip, seed, tr, ho, Yho, mho):
    Xtr = feats(tr, kind); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: (X - mu) / sd
    Ytr = np.stack([t['future_positions'] for t in tr]).reshape(len(tr), -1).astype(np.float32)
    net, best, bep = train_one(seed, z(Xtr), Ytr, z(feats(ho, kind)), Yho, mho, 'cpu')
    return net, z, best, bep


def main():
    t0 = time.time()
    tr0 = records.build('train', EXTRA, 0.0)
    tr10 = records.build('train', EXTRA, 0.1, flip_seed=42)
    ho = records.build('holdout', EXTRA, 0.0, n_fut=6)
    VA = {f: records.build('val', EXTRA, f, flip_seed=777, n_fut=6) for f in FLIPS}
    Yho = np.stack([np.pad(t['future_positions'], ((0, 12 - len(t['future_positions'])), (0, 0)))
                    for t in ho]).reshape(len(ho), -1)
    mho = np.array([t['n_fut'] == 12 for t in ho])
    P = {}
    for kind, trainset, lab in (('L1', tr0, 'L1'), ('L2', tr0, 'L2'), ('L2b', tr0, 'L2b'),
                                ('L5', tr0, 'L5'), ('L5', tr10, 'L5n')):
        for s in SEEDS:
            net, z, best, bep = fit(kind, None, s, trainset, ho, Yho, mho)
            fl = FLIPS if lab == 'L5n' else (FLIPS if lab == 'L5' else (0.0,))
            for f in fl:
                with torch.no_grad():
                    P[(lab, s, f)] = net(torch.tensor(z(feats(VA[f], kind)))).view(-1, 12, 2).numpy()
            print(f'[ladder] {lab} seed {s}: holdout ADE@6s {best:.3f} (ep {bep})  {time.time()-t0:.0f}s',
                  flush=True)
    recs = E.split_records('val')
    assert [r['sample_token'] for r in recs] == [t['sample_token'] for t in VA[0.0]]
    occ = [PMX.occupancies(r) for r in recs]
    tab = E.oracle_kin_table()
    PER = {'CV': E.per_sample(recs, E.cv(recs), occ), 'KIN': E.per_sample(recs, E.kin(recs), occ)}
    for f in FLIPS:
        PER[('OKIN', f)] = E.per_sample(recs, E.oracle_kin(recs, tab, [t['shown']['lon'] for t in VA[f]],
                                                            [t['shown']['lat'] for t in VA[f]]), occ)
    for k, v in P.items():
        PER[k] = E.per_sample(recs, v, occ)
    ff = np.array([not t['can_ok'] and len(r['past_poses']) == 0 for t, r in zip(VA[0.0], recs)])
    pickle.dump({'P': P, 'PER': PER, 'ff': ff, 'order': [r['sample_token'] for r in recs]}, open(OUT, 'wb'))
    print(f'saved {OUT}; first frames excluded in the second block: n={ff.sum()}')
    report(PER, recs, ff)


def seed_mean(PER, lab, f, m, key):
    vals = []
    for s in SEEDS:
        p = PER[(lab, s, f)]
        vals.append(np.nanmean(p['ade'][m]) if key == 'ade' else p['l2'][m, 5].mean())
    return float(np.mean(vals)), float(np.std(vals))


def report(PER, recs, ff):
    for blk, m in (('ALL 5,119', np.ones(len(recs), bool)), (f'EXCL. FIRST FRAMES ({int((~ff).sum())})', ~ff)):
        sub = [r for r, k in zip(recs, m) if k]
        cut = lambda p: {k: v[m] for k, v in p.items()}
        print(f'\n===== {blk} =====')
        print(E.HDR)
        print(E.row('L0 CV', cut(PER['CV']))); print(E.row('L0 KIN', cut(PER['KIN'])))
        for lab in ('L1', 'L2', 'L2b', 'L5'):
            print(E.row(f'{lab} s42', cut(PER[(lab, 42, 0.0)])))
        for f in FLIPS:
            print(E.row(f'L5n s42 flip{int(f*100)}', cut(PER[('L5n', 42, f)])))
            print(E.row(f'OKIN flip{int(f*100)}', cut(PER[('OKIN', f)])))
        print('\n  seed mean +- sd (3 seeds): ADE@6s | L2@3s NoAvg')
        rows = [('L1', 0.0), ('L2', 0.0), ('L2b', 0.0), ('L5', 0.0)] + [('L5n', f) for f in FLIPS]
        S = {}
        for lab, f in rows:
            a = seed_mean(PER, lab, f, m, 'ade'); l = seed_mean(PER, lab, f, m, 'l2')
            S[(lab, f)] = (a[0], l[0])
            print(f'    {lab:4} flip{int(f*100):<3} ADE {a[0]:.3f} +- {a[1]:.3f} | L2@3s {l[0]:.3f} +- {l[1]:.3f}')
        for nm in ('CV', 'KIN'):
            S[(nm, 0.0)] = (float(np.nanmean(PER[nm]['ade'][m])), float(PER[nm]['l2'][m, 5].mean()))
        for f in FLIPS:
            S[('OKIN', f)] = (float(np.nanmean(PER[('OKIN', f)]['ade'][m])), float(PER[('OKIN', f)]['l2'][m, 5].mean()))
        a2, a5 = S[('L2', 0.0)], S[('L5', 0.0)]
        print('\n  GAP CLOSED (X - L2)/(L5 - L2), seed means [ADE@6s | L2@3s]; L2 = 0, L5 = 1:')
        for k, v in S.items():
            g = [(v[i] - a2[i]) / (a5[i] - a2[i]) for i in (0, 1)]
            print(f'    {k[0]:4} flip{int(k[1]*100):<3} {g[0]:+.3f} | {g[1]:+.3f}')
        print('\n  paired scene-level bootstrap (seed 42):')
        for lab in ('L1', 'L2', 'L2b', 'L5'):
            for b in ('CV', 'KIN'):
                print(E.compare(sub, cut(PER[(lab, 42, 0.0)]), cut(PER[b]), f'{lab} - {b}'))
            if lab != 'L2':
                print(E.compare(sub, cut(PER[(lab, 42, 0.0)]), cut(PER[('L2', 42, 0.0)]), f'{lab} - L2'))
        print(E.compare(sub, cut(PER[('OKIN', 0.0)]), cut(PER[('L2', 42, 0.0)]), 'OKIN - L2'))


if __name__ == '__main__':
    main()
