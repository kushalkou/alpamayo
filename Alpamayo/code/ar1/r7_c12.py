"""ar1/r7_c12.py -- R7 Part C1 (mediation, gate GM1) and C2 (rule-defined strata, adopted
after R4). Rules in the R7A_REPORT.md header. Per-sample dumps only; R7 runs included when
their dumps exist.
  python ar1/r7_c12.py"""
import sys, os, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from meta_actions import LON

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
SEEDS = (42, 123, 2024)
GO = ('gentle_acc', 'strong_acc', 'maintain')
GROUPS = (('CV', ('CV',)), ('KIN', ('KIN',)), ('Ego-MLP + cmd', tuple(f'EgoMLP s{s}' for s in SEEDS)),
          ('no-cam VLA', ('A3', 'G6C_s123', 'G6C_s2024')), ('B1', ('B1', 'B1_s123', 'B1_s2024')), ('B2', ('B2',)),
          ('M1 (10 Hz)', ('M1', 'M1_s123', 'M1_s2024')), ('M1-v2a', ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')),
          ('no-cam M1-v2a', ('M1v2a_nocam',)), ('B1-NR', ('B1_NR',)), ('M1-v2a-NR', ('M1v2a_NR',)),
          ('Ego-MLP-NR', tuple(f'EgoMLP_NR s{s}' for s in SEEDS)))


def have(t):
    if t.startswith('EgoMLP_NR'):
        return os.path.exists(f'{RES}/w1_egomlp_nr.pkl')
    return t in ('CV', 'KIN') or t.startswith('EgoMLP') or os.path.exists(f'{RES}/w1_dump_{t}_val_f0.0.pkl')


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); occ = C['occ']
    tokv = [r['sample_token'] for r in rva]
    g = np.linalg.norm(np.stack([np.asarray(T[t]['P'])[:6] for t in tokv]), axis=2).max(1)
    v0 = np.array([V0[t]['v0_can'] for t in tokv])
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])

    # ---------------- C1 mediation ----------------
    gs = (C['st'] == 'stationary') & (g < 0.5)
    print(f'C1 mediation: GT-stopped stationary val (R3), n {gs.sum()} per seed; false-go = pred > 1.0 m')
    res = {}
    for nm, tags in (('M1-v2a (2 Hz)', ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')), ('old M1 (10 Hz)', ('M1', 'M1_s123', 'M1_s2024'))):
        W1, FG = [], []
        for t in tags:
            d = load(t, 'val'); P = preds(d, rva, 'hybrid', Q.run(t)['tau'])
            dm = np.linalg.norm(P[:, :6], axis=2).max(1)
            for i in np.where(gs)[0]:
                W1.append(LON[d['data'][tokv[i]]['meta_gen'][0]]); FG.append(dm[i] > 1.0)
        W1, FG = np.array(W1), np.array(FG)
        print(f'  {nm}: pooled n {len(FG)}, false-go {FG.sum()} ({FG.mean():.3f})')
        print(f'    {"slot-1 word":12} {"n":>5} {"P(FG)":>6} {"share of FG":>11}')
        for wd in LON:
            m = W1 == wd
            if m.sum():
                print(f'    {wd:12} {m.sum():5d} {FG[m].mean():6.3f} {FG[m].sum() / max(FG.sum(), 1):11.3f}')
        go = np.isin(W1, GO)
        sh_go = (FG & go).sum() / max(FG.sum(), 1); sh_st = (FG & (W1 == 'stop')).sum() / max(FG.sum(), 1)
        p_st = FG[W1 == 'stop'].mean() if (W1 == 'stop').any() else float('nan')
        print(f'    false-gos with a go-word {sh_go:.3f}, with stop {sh_st:.3f}, other {1 - sh_go - sh_st:.3f}; '
              f'P(FG | stop) {p_st:.3f}; P(FG | go-word) {FG[go].mean():.3f}')
        res[nm] = (sh_go, p_st)
    sh_go, p_st = res['M1-v2a (2 Hz)']
    print(f'GM1 M1-v2a: go-word share {sh_go:.3f} (>= 0.80), P(FG | stop) {p_st:.3f} (<= 0.05) -> ' +
          ('"start decision is mediated by the generated words"' if sh_go >= 0.80 and p_st <= 0.05 else '"not mediated"'))

    # ---------------- C2 rule-defined strata ----------------
    ST = {'STOPPED': (v0 <= 0.2) & (g <= 1.0), 'START': (v0 <= 0.2) & (g > 1.0), 'MOVING': v0 > 0.2}
    print(f'\nC2 rule-defined strata (ADOPTED AFTER R4): ' + ', '.join(f'{k} n {m.sum()}' for k, m in ST.items()) +
          f'; WSS (secondary) n {w.sum()} (STOPPED {(ST["STOPPED"] & w).sum()}, START {(ST["START"] & w).sum()}, '
          f'MOVING {(ST["MOVING"] & w).sum()})')
    mlp = pickle.load(open(f'{RES}/w1_egomlp.pkl', 'rb'))
    mlpn = pickle.load(open(f'{RES}/w1_egomlp_nr.pkl', 'rb')) if os.path.exists(f'{RES}/w1_egomlp_nr.pkl') else None

    def pr(t):
        if t == 'CV': return E.cv(rva)
        if t == 'KIN': return E.kin(rva)
        if t.startswith('EgoMLP_NR'): return mlpn['preds'][int(t.split('s')[-1])]['val']
        if t.startswith('EgoMLP'): return mlp['preds'][int(t.split('s')[-1])]['val']
        return preds(load(t, 'val'), rva, 'hybrid', Q.run(t)['tau'])
    PS = {}
    print(f'  {"run (seeds)":20} ' + ' | '.join(f'{k}: L2@3s ADE6 {"FG" if k == "STOPPED" else "MG" if k == "START" else ""}'
                                              for k in ST) + ' | WSS L2@3s')
    for nm, tags in GROUPS:
        if not all(have(t) for t in tags):
            continue
        rows = []
        for t in tags:
            P = pr(t); ps = E.per_sample(rva, P, occ); PS[t] = ps
            dm = np.linalg.norm(P[:, :6], axis=2).max(1)
            v = []
            for k, m in ST.items():
                v += [ps['l2'][m, 5].mean(), np.nanmean(ps['ade'][m])]
                v += [(dm[m] > 1.0).mean()] if k == 'STOPPED' else [(dm[m] <= 1.0).mean()] if k == 'START' else []
            rows.append(v + [ps['l2'][w, 5].mean()])
        v = np.mean(rows, 0)
        print(f'  {nm + " (" + str(len(tags)) + ")":20} {v[0]:6.3f} {v[1]:6.3f} {v[2]:5.3f} | {v[3]:6.3f} {v[4]:6.3f} '
              f'{v[5]:5.3f} | {v[6]:6.3f} {v[7]:6.3f} | {v[8]:6.3f}')
    avg = lambda tags: Q.avg([PS[t] for t in tags])
    v2, b1, nc = avg(('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')), avg(('B1', 'B1_s123', 'B1_s2024')), avg(('A3', 'G6C_s123', 'G6C_s2024'))
    print('  paired scene bootstrap (10,000), L2@3s NoAvg; ADE@6s after')
    for pair, A, B in (('M1-v2a 3s - B1 3s', v2, b1), ('M1-v2a 3s - no-cam VLA 3s', v2, nc)):
        for k, m in list(ST.items()) + [('WSS (secondary)', w)]:
            sub = [r for r, x in zip(rva, m) if x]
            c = E.scene_boot(sub, A['l2'][m, 5], B['l2'][m, 5]); c2 = E.scene_boot(sub, A['ade'][m], B['ade'][m])
            print(f'    {pair:26} {k:16} n {m.sum():4d}  {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]   '
                  f'ADE6 {c2[0]:+.3f} [{c2[1]:+.3f},{c2[2]:+.3f}]')


if __name__ == '__main__':
    main()
