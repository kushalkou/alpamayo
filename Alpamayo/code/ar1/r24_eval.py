"""ar1/r24_eval.py -- R2.4 report: new camera base (cached, native patch order), A3 recipe.
  B1  3 front cameras at t0 + ego + cmd [P]              (480 visual tokens)
  B2  3 front cameras x 4 keyframes (<= t0) + ego + cmd  (1,920 visual tokens)
  ref no-camera VLA, ego + cmd [P], A3 recipe, 3-seed mean (T9b: A3, G6C_s123, G6C_s2024)
Shared eval (q2_traj): free-running AR dumps, STOP-aware hybrid decode with tau fit on
HOLDOUT, official val, ALL 5,119 and EXCL. first frames; strata; paired scene bootstrap.
Camera-shuffle test: each val sample gets another sample's cached tokens (perm seed 99).
Per-slot argmax accuracy from the dumps (n_fut = 12 samples).
PRE-REGISTERED: base input = B2 if B2 - B1 on L2@3s (NoAvg) has a 95% CI below 0 on ALL
5,119 or on the stationary stratum; else B1.
  python ar1/r24_eval.py
"""
import sys
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import evalw1 as E
import q2_traj as Q
from gate_a import load


def slot_acc(tag, recs):
    d = load(tag, 'val')['data']
    ok = [r['sample_token'] for r in recs if r['n_fut'] == 12]
    tok = np.array([d[s]['tok'] for s in ok]); gt = np.array([d[s]['gt_tok'] for s in ok])
    return (tok == gt).mean(0)


def main():
    C = Q.ctx(); rva = C['rva']
    LEN, ci, _ = Q.seed_tags()
    nc = [Q.run(t) for t in ci.values()]
    rows = {'KIN': C['cache']['KIN'],
            'Ego-MLP + cmd 3-seed [P]': Q.avg([C['cache'][f'egoMLP s{s}'] for s in Q.SEEDS]),
            'VLA no cam, 3-seed [P]': Q.avg(nc),
            'B1 3 cams t0 [P]': Q.strip(Q.run('B1')),
            'B2 3 cams x 4 kf [P]': Q.strip(Q.run('B2')),
            'B1 cams shuffled': Q.strip(Q.run('B1', 'B1_camshuf')),
            'B2 cams shuffled': Q.strip(Q.run('B2', 'B2_camshuf'))}
    print(f'tau (holdout): B1 {Q.run("B1")["tau"]}, B2 {Q.run("B2")["tau"]}; no-cam seeds '
          + ', '.join(f'{t} {Q.run(t)["tau"]}' for t in ci.values()))
    verdict = {}
    for blk, m in Q.blocks():
        sub = [r for r, k in zip(rva, m) if k]
        cut = lambda p: {k: v[m] for k, v in p.items()}
        st = C['st'][m]
        print(f'\n===== {blk} =====')
        print(E.HDR)
        for k, v in rows.items():
            print(E.row(k, cut(v)))
        print('  paired scene bootstrap (95% CI):')
        R = {k: cut(v) for k, v in rows.items()}
        for a, b in (('B1 3 cams t0 [P]', 'VLA no cam, 3-seed [P]'), ('B2 3 cams x 4 kf [P]', 'VLA no cam, 3-seed [P]'),
                     ('B2 3 cams x 4 kf [P]', 'B1 3 cams t0 [P]'), ('B1 cams shuffled', 'B1 3 cams t0 [P]'),
                     ('B2 cams shuffled', 'B2 3 cams x 4 kf [P]'), ('B2 3 cams x 4 kf [P]', 'Ego-MLP + cmd 3-seed [P]')):
            print(E.compare(sub, R[a], R[b], f'{a[:16]} - {b[:16]}'))
        print('  strata, L2@3s NoAvg mean:  ' + '  '.join(f'{s} n={int((st == s).sum())}' for s in
                                                          ('straight', 'turning', 'stationary')))
        for k in ('VLA no cam, 3-seed [P]', 'B1 3 cams t0 [P]', 'B2 3 cams x 4 kf [P]', 'B1 cams shuffled',
                  'B2 cams shuffled'):
            print(f'    {k:26} ' + '  '.join(f'{R[k]["l2"][st == s, 5].mean():.3f}' for s in
                                             ('straight', 'turning', 'stationary')))
        for a, b in (('B2 3 cams x 4 kf [P]', 'B1 3 cams t0 [P]'), ('B1 3 cams t0 [P]', 'VLA no cam, 3-seed [P]'),
                     ('B2 3 cams x 4 kf [P]', 'VLA no cam, 3-seed [P]'), ('B1 cams shuffled', 'B1 3 cams t0 [P]'),
                     ('B2 cams shuffled', 'B2 3 cams x 4 kf [P]')):
            out = []
            for s in ('straight', 'turning', 'stationary'):
                k = st == s
                c = E.scene_boot([r for r, q in zip(sub, k) if q], R[a]['l2'][k, 5], R[b]['l2'][k, 5])
                out.append(f'{s} {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]')
                if a.startswith('B2') and b.startswith('B1') and s == 'stationary':
                    verdict[(blk[:3], 'stat')] = c
            print(f'    {a[:16]} - {b[:16]}, L2@3s: ' + '; '.join(out))
        for nm in ('B1', 'B2'):
            u = R[f'{nm} 3 cams t0 [P]' if nm == 'B1' else 'B2 3 cams x 4 kf [P]']['ade']
            s_ = R[f'{nm} cams shuffled']['ade']
            print(f'  camera shuffle {nm}: ADE@6s x{np.nanmean(s_) / np.nanmean(u):.3f} '
                  f'(rule as G7: within 5% => cameras unused)')
        verdict[(blk[:3], 'all')] = E.scene_boot(sub, R['B2 3 cams x 4 kf [P]']['l2'][:, 5],
                                                 R['B1 3 cams t0 [P]']['l2'][:, 5])
    print('\nper-slot argmax accuracy (val, n_fut = 12): a0 a1 a2 a5 a11 | k0 k1 k2 k5 k11 | mean a / k')
    sl = {'no cam (3-seed mean)': np.mean([slot_acc(t, rva) for t in ci.values()], 0),
          'B1': slot_acc('B1', rva), 'B2': slot_acc('B2', rva),
          'B1 shuffled': slot_acc('B1_camshuf', rva), 'B2 shuffled': slot_acc('B2_camshuf', rva)}
    for k, a in sl.items():
        print(f'  {k:22} ' + ' '.join(f'{a[i]:.3f}' for i in (0, 1, 2, 5, 11)) + ' | '
              + ' '.join(f'{a[12 + i]:.3f}' for i in (0, 1, 2, 5, 11)) + f' | {a[:12].mean():.3f} / {a[12:].mean():.3f}')
    c, s = verdict[('ALL', 'all')], verdict[('ALL', 'stat')]
    b2 = c[2] < 0 or s[2] < 0
    print(f'\nPRE-REGISTERED DECISION (ALL 5,119): B2 - B1 L2@3s {E.fmt_ci(c)}; stationary '
          f'{s[0]:+.3f} [{s[1]:+.3f},{s[2]:+.3f}] -> base input = {"B2" if b2 else "B1"}')


if __name__ == '__main__':
    main()
