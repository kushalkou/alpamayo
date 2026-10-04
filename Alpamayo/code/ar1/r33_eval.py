"""ar1/r33_eval.py -- R3.3 report: META-ACTION + TRAJ arm (M1) on the B1 base.

1. Standard tables (q2_traj shared eval; hybrid decode, tau on holdout), both strata:
   KIN, Ego-MLP 3-seed, no-cam VLA 3-seed, B1 (seed 42, or the 3-seed mean when present),
   M1 token decoder, M1 + expert (1 sample, mean of 6), minADE_6 @3s / @6s.
2. Meta-action accuracy / macro-F1 per slot (6 lon, 6 lat at t0+1..6 s) of the GENERATED
   (constrained) words vs the R1.2 labels; val n_fut = 12 subset (labels fully in-scene)
   and all 5,119. Share of words where the unconstrained argmax equals the constrained one.
3. CONSISTENCY (AR1 Table 9 analog): meta-actions DERIVED from the predicted trajectory with
   the R1.2 rules at the same 1 Hz ticks vs the GENERATED ones. Derivation from 2 Hz
   controls: speed after step k (k = 2 t - 1), accel = speed change over that 0.5 s step,
   curvature = cur[k] (0 if speed < 1 m/s); lon_cls / lat_cls from ar1/meta_actions.py.
   Reported: share of samples with all 12 slots consistent, per-slot rate. Reference: the
   same derivation on the GT trajectory vs the GT (10 Hz CAN) labels = the ceiling.
  python ar1/r33_eval.py [TAG]   (default M1)
"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E
import q2_traj as Q
from gate_a import load
from meta_actions import lon_cls, lat_cls, LON, LAT

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
DT = 0.5


def controls(dump, recs, tau):
    out = []
    for r in recs:
        d = dump['data'][r['sample_token']]
        a, e, p = np.array(d['argmax']), np.array(d['expect']), np.array(d['p_stop'])
        v = np.where(p > tau, a, e)
        out.append((v[:12], v[12:]))
    return out


def derive(acc, cur, v0):
    """R1.2 rules at t = 1..6 s from 2 Hz controls -> 6 lon + 6 lat classes."""
    v = v0; sp = []
    for k in range(12):
        v = max(0.0, v + acc[k] * DT); sp.append(v)
    lo, la = [], []
    for t in range(1, 7):
        k = 2 * t - 1
        a = (sp[k] - sp[k - 1]) / DT
        c = lon_cls(sp[k], a)
        kk = cur[k] if sp[k] >= 1.0 else 0.0
        lo.append(c); la.append(lat_cls(c, kk))
    return lo + la


def f1(y, p, n):
    f = []
    for c in range(n):
        if (y == c).sum() == 0 and (p == c).sum() == 0:
            continue
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def main(tag='M1'):
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data()
    LEN, ci, _ = Q.seed_tags()
    b1 = ['B1'] + [t for t in ('B1_s123', 'B1_s2024') if os.path.exists(f'{RES}/w1_dump_{t}_val_f0.0.pkl')]
    rows = {'KIN': C['cache']['KIN'],
            'Ego-MLP + cmd 3-seed [P]': Q.avg([C['cache'][f'egoMLP s{s}'] for s in Q.SEEDS]),
            'VLA no cam 3-seed [P]': Q.avg([Q.run(t) for t in ci.values()]),
            f'B1 ({len(b1)}-seed mean) [P]': Q.avg([Q.run(t) for t in b1]),
            'B1 seed 42 [P]': Q.strip(Q.run('B1')),
            f'{tag} token decoder [P]': Q.strip(Q.run(tag))}
    xp = f'{RES}/ar1_expert_{tag}.pkl'
    X = pickle.load(open(xp, 'rb')) if os.path.exists(xp) else None
    if X is not None:
        assert X['val']['tokens'] == [r['sample_token'] for r in rva]
        if 'exp0' not in X:
            X['exp0'] = E.per_sample(rva, X['val']['traj'][:, 0], C['occ'])
            X['expm'] = E.per_sample(rva, X['val']['traj'].mean(1), C['occ'])
            pickle.dump(X, open(xp, 'wb'))
        rows[f'{tag} + expert, 1 sample'] = X['exp0']
        rows[f'{tag} + expert, mean of 6'] = X['expm']
        G = [np.asarray(T[r['sample_token']]['P']) for r in rva]
        tr = X['val']['traj']
        e3 = np.stack([np.linalg.norm(tr[i, :, :6] - G[i][None, :6], axis=2).mean(1) for i in range(len(rva))])
        e6 = np.full(e3.shape, np.nan)
        for i, r in enumerate(rva):
            if r['n_fut'] == 12:
                e6[i] = np.linalg.norm(tr[i] - G[i][None, :12], axis=2).mean(1)
    print(f'tau (holdout): {tag} {Q.run(tag)["tau"]}; B1 seeds used: {b1}')
    for blk, m in Q.blocks():
        sub = [r for r, k in zip(rva, m) if k]
        cut = lambda p: {k: v[m] for k, v in p.items()}
        R = {k: cut(v) for k, v in rows.items()}
        print(f'\n===== {blk} =====')
        print(E.HDR + ' | ADE@3s | minADE6 @3s @6s')
        for k, v in R.items():
            a3 = v['l2'][:, :6].mean(1)
            if 'expert' in k:
                mm = f'{e3.min(1)[m].mean():.3f} {np.nanmean(np.nanmin(e6, 1)[m]):.3f}'
            else:
                mm = f'{a3.mean():.3f} {np.nanmean(v["ade"]):.3f} (1 sample)'
            print(E.row(k, v) + f' | {a3.mean():.3f} | {mm}')
        print('  paired scene bootstrap (95% CI):')
        tk = f'{tag} token decoder [P]'
        for a_, b_ in ((tk, 'B1 seed 42 [P]'), (tk, f'B1 ({len(b1)}-seed mean) [P]'), (tk, 'VLA no cam 3-seed [P]')) + \
                (((f'{tag} + expert, mean of 6', tk), (f'{tag} + expert, 1 sample', tk)) if X is not None else ()):
            print(E.compare(sub, R[a_], R[b_], f'{a_[:18]} - {b_[:18]}'))
        if X is not None:
            c = E.scene_boot(sub, np.nanmin(e6, 1)[m], R[tk]['ade'])
            print(f'  {"expert minADE_6@6s - token ADE@6s":36} {E.fmt_ci(c)}')
        st = C['st'][m]
        print('  strata L2@3s (straight / turning / stationary): ' + '; '.join(
            f'{k[:22]} ' + ' / '.join(f'{R[k]["l2"][st == s, 5].mean():.3f}' for s in ('straight', 'turning', 'stationary'))
            for k in ('VLA no cam 3-seed [P]', 'B1 seed 42 [P]', tk)))

    # meta-action accuracy, free-argmax agreement, consistency
    dv = load(tag, 'val')['data']
    tau = Q.run(tag)['tau']
    ctl = controls(load(tag, 'val'), rva, tau)
    gen = np.array([dv[r['sample_token']]['meta_gen'] for r in rva])
    gt = np.array([dv[r['sample_token']]['meta_gt'] for r in rva])
    free = np.array([dv[r['sample_token']]['meta_free_ok'] for r in rva])
    n12 = np.array([r['n_fut'] == 12 for r in rva])
    print('\nMETA-ACTION WORDS (generated, constrained) vs R1.2 labels, per slot t0+1..6 s')
    for nm, m in (('val n_fut = 12', n12), ('val all 5,119', np.ones(len(rva), bool))):
        print(f'  {nm} (n={int(m.sum())})')
        for half, off, ncl in (('lon', 0, 7), ('lat', 6, 7)):
            acc = [(gen[m, off + s] == gt[m, off + s]).mean() for s in range(6)]
            fs = [f1(gt[m, off + s], gen[m, off + s], ncl) for s in range(6)]
            print(f'    {half} acc   ' + ' '.join(f'{x:.3f}' for x in acc) + f' | mean {np.mean(acc):.3f}')
            print(f'    {half} mF1   ' + ' '.join(f'{x:.3f}' for x in fs) + f' | mean {np.mean(fs):.3f}')
        print(f'    unconstrained argmax == constrained word: {free[m].mean():.3f} of word slots')
        base = {h: np.bincount(gt[m, o:o + 6].ravel(), minlength=7) / gt[m, o:o + 6].size for h, o in (('lon', 0), ('lat', 6))}
        print('    label shares lon: ' + ' '.join(f'{LON[c]} {base["lon"][c]:.3f}' for c in range(7)))
        print('    label shares lat: ' + ' '.join(f'{LAT[c]} {base["lat"][c]:.3f}' for c in range(7)))
    der = np.array([derive(a, k, T[r['sample_token']]['v0']) for (a, k), r in zip(ctl, rva)])
    gder = np.array([derive(T[r['sample_token']]['acc'][:12], T[r['sample_token']]['cur'][:12], T[r['sample_token']]['v0'])
                     if r['n_fut'] == 12 else [-9] * 12 for r in rva])
    print('\nCONSISTENCY: meta-actions derived from the predicted trajectory == generated words')
    for nm, m in (('val n_fut = 12', n12), ('val all 5,119', np.ones(len(rva), bool))):
        ok = der[m] == gen[m]
        print(f'  {nm}: all 12 slots {ok.all(1).mean():.3f}; lon 6/6 {ok[:, :6].all(1).mean():.3f}; '
              f'lat 6/6 {ok[:, 6:].all(1).mean():.3f}; per-slot lon ' + ' '.join(f'{x:.2f}' for x in ok[:, :6].mean(0))
              + ' | lat ' + ' '.join(f'{x:.2f}' for x in ok[:, 6:].mean(0)))
    okg = gder[n12] == gt[n12]
    print(f'  CEILING (GT trajectory derived vs GT 10 Hz labels, n_fut = 12): all 12 {okg.all(1).mean():.3f}; '
          f'lon 6/6 {okg[:, :6].all(1).mean():.3f}; lat 6/6 {okg[:, 6:].all(1).mean():.3f}; per-slot lon '
          + ' '.join(f'{x:.2f}' for x in okg[:, :6].mean(0)) + ' | lat ' + ' '.join(f'{x:.2f}' for x in okg[:, 6:].mean(0)))
    okt = der[n12] == gt[n12]
    print(f'  predicted-trajectory meta vs GT labels (n_fut = 12): all 12 {okt.all(1).mean():.3f}; '
          f'lon slot acc mean {okt[:, :6].mean():.3f}; lat {okt[:, 6:].mean():.3f}')
    if X is not None:
        cm = X['val']['ctrl'].mean(1)
        dx = np.array([derive(c[:, 0], c[:, 1], T[r['sample_token']]['v0']) for c, r in zip(cm, rva)])
        d0 = np.array([derive(c[:, 0], c[:, 1], T[r['sample_token']]['v0']) for c, r in zip(X['val']['ctrl'][:, 0], rva)])
        for nm, d in (('expert mean-of-6 controls', dx), ('expert sample 0', d0)):
            ok = d[n12] == gen[n12]
            print(f'  {nm} vs generated words (n_fut = 12): all 12 {ok.all(1).mean():.3f}; lon 6/6 '
                  f'{ok[:, :6].all(1).mean():.3f}; lat 6/6 {ok[:, 6:].all(1).mean():.3f}')
        print(f'\nexpert: best epoch {X["best_epoch"]} (holdout median ADE@6s {X["ho_best_med"]:.4f}); train '
              f'{X["train_gpu_h"]:.2f} GPU-h; inference {X["infer_gpu_h"]:.2f} GPU-h')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'M1')
