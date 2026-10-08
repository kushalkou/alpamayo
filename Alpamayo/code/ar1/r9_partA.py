"""ar1/r9_partA.py -- R9 Part A (CPU): A1 Table 6 analog, A2 Table 7 analog (HARD set), A3
Table 9 analog (consistency). Rules in the R9_REPORT.md header. Per-sample dumps, holdout tau.
  python ar1/r9_partA.py [a1] [a2] [a3]"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r5_a5c1 import coll_rates
from r33_eval import derive, controls

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
SEEDS = (42, 123, 2024)
ARM = {'B1': ('B1', 'B1_s123', 'B1_s2024'), 'B1-NR': ('B1_NR', 'B1_NR_s123', 'B1_NR_s2024'),
       'M1-v2a': ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024'), 'M1-v2a-NR': ('M1v2a_NR', 'M1v2a_NR_s123', 'M1v2a_NR_s2024')}


def main():
    parts = sys.argv[1:] or ['a1', 'a2', 'a3']
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); occ = C['occ']
    tokv = [r['sample_token'] for r in rva]
    g = np.linalg.norm(np.stack([np.asarray(T[t]['P'])[:6] for t in tokv]), axis=2).max(1)
    v0 = np.array([V0[t]['v0_can'] for t in tokv])
    stat = C['st'] == 'stationary'; gs = stat & (g < 0.5); gm = stat & (g > 1.0)
    ST = {'STOPPED': (v0 <= 0.2) & (g <= 1.0), 'START': (v0 <= 0.2) & (g > 1.0), 'MOVING': v0 > 0.2}
    boot = lambda m, A, B, k: E.scene_boot([r for r, x in zip(rva, m) if x], A[k][m] if k == 'ade' else A['l2'][m, 5],
                                            B[k][m] if k == 'ade' else B['l2'][m, 5])
    fmt = lambda c: f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]'
    ALL = np.ones(len(rva), bool)

    if 'a1' in parts:
        print('A1 Table 6 analog: 3-seed mean +- sd | L2@3s NoAvg, ADE@6s, Col aa, Col yaw, R3 FG, R3 MG')
        for nm, tags in ARM.items():
            v = []
            for t in tags:
                P = preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau']); ps = Q.run(t)
                dm = np.linalg.norm(P[:, :6], axis=2).max(1)
                v.append([ps['l2'][:, 5].mean(), np.nanmean(ps['ade']), coll_rates(rva, P, occ, 'aa')[0],
                          coll_rates(rva, P, occ, 'yaw')[0], (dm[gs] > 1.0).mean(), (dm[gm] < 0.5).mean()])
            v = np.array(v); m, s = v.mean(0), v.std(0, ddof=1)
            print(f'  {nm:10} ' + ' '.join(f'{a:.3f}+-{b:.3f}' for a, b in zip(m, s)))
        M = {nm: Q.avg([Q.run(t) for t in tags]) for nm, tags in ARM.items()}
        print('  paired deltas (3-seed means), L2@3s | ADE@6s')
        for lab, a, b in (('+meta - traj, with route', 'M1-v2a', 'B1'), ('+meta - traj, without route', 'M1-v2a-NR', 'B1-NR'),
                          ('without - with route, traj', 'B1-NR', 'B1'), ('without - with route, +meta', 'M1-v2a-NR', 'M1-v2a')):
            print(f'    {lab:30} {fmt(boot(ALL, M[a], M[b], "l2"))} | {fmt(boot(ALL, M[a], M[b], "ade"))}')

    if 'a2' in parts:
        F = pickle.load(open(f'{DATA}/r9_hard_val.pkl', 'rb'))
        tu = np.array([abs(F[t]['dpsi']) > 30 for t in tokv]); ag = np.array([F[t]['agent20'] for t in tokv])
        it = np.array([F[t]['inter0'] or F[t]['sl20'] for t in tokv])
        hard = ST['START'] | tu | (ag & it); rest = ~hard
        print(f'\nA2 HARD set (rule in header): H_START {ST["START"].sum()}, H_TURN {tu.sum()}, H_AGENT {ag.sum()}, '
              f'H_INT {it.sum()}, H_AGENT and H_INT {(ag & it).sum()}; HARD {hard.sum()} ({hard.mean():.3f}), '
              f'REST {rest.sum()}; HARD scenes {len({r["scene_name"] for r, h in zip(rva, hard) if h})}')
        mlp = pickle.load(open(f'{RES}/w1_egomlp.pkl', 'rb')); mlpn = pickle.load(open(f'{RES}/w1_egomlp_nr.pkl', 'rb'))
        def ps(t):
            if t == 'CV': return E.per_sample(rva, E.cv(rva), occ)
            if t == 'KIN': return E.per_sample(rva, E.kin(rva), occ)
            if t.startswith('EgoMLP_NR'): return E.per_sample(rva, mlpn['preds'][int(t.split('s')[-1])]['val'], occ)
            if t.startswith('EgoMLP'): return E.per_sample(rva, mlp['preds'][int(t.split('s')[-1])]['val'], occ)
            return Q.run(t)
        GR = (('CV', ('CV',)), ('KIN', ('KIN',)), ('Ego-MLP + cmd', tuple(f'EgoMLP s{s}' for s in SEEDS)),
              ('Ego-MLP-NR', tuple(f'EgoMLP_NR s{s}' for s in SEEDS)), ('no-cam VLA', ('A3', 'G6C_s123', 'G6C_s2024')),
              ('B1', ARM['B1']), ('B2', ('B2',)), ('M1 (10 Hz)', ('M1', 'M1_s123', 'M1_s2024')), ('M1-v2a', ARM['M1-v2a']),
              ('no-cam M1-v2a', ('M1v2a_nocam',)), ('B1-NR', ARM['B1-NR']), ('M1-v2a-NR', ARM['M1-v2a-NR']))
        print(f'  {"run (seeds)":20} HARD L2@3s ADE@6s | REST L2@3s ADE@6s')
        PS = {}
        for nm, tags in GR:
            P = Q.avg([ps(t) for t in tags]) if len(tags) > 1 else Q.strip(ps(tags[0])); PS[nm] = P
            print(f'  {nm + " (" + str(len(tags)) + ")":20} {P["l2"][hard, 5].mean():6.3f} {np.nanmean(P["ade"][hard]):6.3f} | '
                  f'{P["l2"][rest, 5].mean():6.3f} {np.nanmean(P["ade"][rest]):6.3f}')
        for lab, a, b in (('M1-v2a - B1', 'M1-v2a', 'B1'), ('M1-v2a-NR - B1-NR', 'M1-v2a-NR', 'B1-NR')):
            for sn, m in (('HARD', hard), ('REST', rest)):
                print(f'  {lab:18} {sn}: L2@3s {fmt(boot(m, PS[a], PS[b], "l2"))} | ADE@6s {fmt(boot(m, PS[a], PS[b], "ade"))}')
        for comp, m in (('H_START', ST['START']), ('H_TURN', tu), ('H_AGENT and H_INT', ag & it)):
            print(f'  component {comp:18} n {m.sum():4d}: M1-v2a - B1 L2@3s {fmt(boot(m, PS["M1-v2a"], PS["B1"], "l2"))}; '
                  f'M1-v2a-NR - B1-NR {fmt(boot(m, PS["M1-v2a-NR"], PS["B1-NR"], "l2"))}')

    if 'a3' in parts:
        print('\nA3 Table 9 analog: consistency (trajectory-derived words vs generated words, 2 Hz rules); '
              'all-12-slot match | mean per-slot agreement; overall and per C2 stratum')
        for nm, tags in (('M1 (10 Hz)', ('M1', 'M1_s123', 'M1_s2024')), ('M1-v2a', ARM['M1-v2a']),
                         ('M1-v2a-NR', ARM['M1-v2a-NR']), ('no-cam M1-v2a', ('M1v2a_nocam',))):
            rows = []
            for t in tags:
                dv = load(t, 'val'); ga = np.array([dv['data'][x]['meta_gen'] for x in tokv])
                der = np.array([derive(a, k, T[r['sample_token']]['v0'])
                                for (a, k), r in zip(controls(dv, rva, Q.run(t)['tau']), rva)])
                eq = der == ga
                rows.append([f(eq[m]) for m in [ALL] + list(ST.values()) for f in (lambda e: e.all(1).mean(), lambda e: e.mean())])
            v = np.array(rows).mean(0)
            print(f'  {nm + " (" + str(len(tags)) + ")":18} all {v[0]:.3f} | {v[1]:.3f}; STOPPED {v[2]:.3f} | {v[3]:.3f}; '
                  f'START {v[4]:.3f} | {v[5]:.3f}; MOVING {v[6]:.3f} | {v[7]:.3f}' +
                  ('' if len(tags) == 1 else '  per seed all-12: ' + ' '.join(f'{x[0]:.3f}' for x in rows)))


if __name__ == '__main__':
    main()
