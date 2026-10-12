"""ar1/r13_partA.py -- R13 Part A (CPU, CoC seed 42 on v3 traces = tag COC3): A1 causes test (GC6b),
A2 decision confusion and the share of the GC1 gap from wrong decisions. Rules in R13_REPORT.md.
  python ar1/r13_partA.py"""
import sys, re, pickle, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RX = re.compile(r'^Driving decision: ([^;]+); ([^.]+)\. Critical components: (.+)\. Reasoning: (.+)$')
LONS = ('Set speed tracking', 'Lead obstacle following', 'Stop for static constraints', 'Yield (agent right-of-way)',
        'Speed adaptation (road events)', 'Acceleration for passing/overtaking')
LATS = ('Lane keeping & centering', 'None', 'Turn left', 'Turn right', 'Lane change left', 'Lane change right',
        'In-lane nudge left', 'In-lane nudge right')


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); tokv = [r['sample_token'] for r in rva]
    coc = pickle.load(open(f'{DATA}/ar1_coc_2hz_v3.pkl', 'rb'))
    g = np.linalg.norm(np.stack([np.asarray(T[t]['P'])[:6] for t in tokv]), axis=2).max(1)
    v0 = np.array([V0[t]['v0_can'] for t in tokv])
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    S = {'all': np.ones(len(rva), bool), 'excl. first frames': ~C['ff'], 'excl. WSS': ~w, 'WSS only': w,
         'C2 STOPPED': (v0 <= 0.2) & (g <= 1.0), 'C2 START': (v0 <= 0.2) & (g > 1.0), 'C2 MOVING': v0 > 0.2}
    f = lambda c: f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]'
    gi, sh, gt = (Q.strip(Q.run('COC3', src=s)) for s in ('COC3', 'COC3_shuf', 'COC3_gt'))
    v2 = Q.strip(Q.run('M1v2a'))
    print('A1 causes test, seed 42: (iii) shuffled causes [true decision] - (iv) GT CoC [true decision], L2@3s')
    for k, m in S.items():
        c = E.scene_boot([r for r, x in zip(rva, m) if x], sh['l2'][m, 5], gt['l2'][m, 5])
        print(f'  {k:19} n {m.sum():4d}  {f(c)}' + (f'   (ADE@6s {f(E.scene_boot(rva, sh["ade"], gt["ade"]))})' if k == 'all' else ''))
        if k == 'all':
            c6 = c
    print('GC6b ' + f(c6) + ' -> ' + ('"given the true decision, the causes add nothing"' if c6[1] <= 0 <= c6[2] else
                                      '"shuffled causes better (suspect)"' if c6[2] < 0 else '"causes matter"'))

    d = load('COC3', 'val')['data']; mm = [RX.match(d[t]['coc_gen']) for t in tokv]
    lg = np.array([m[1] if m else '(unparsed)' for m in mm]); ag = np.array([m[2] if m else '(unparsed)' for m in mm])
    lt = np.array([coc[t]['lon'] for t in tokv]); at = np.array([coc[t]['lat'] for t in tokv])
    for nm, gtv, gen, cls in (('lon', lt, lg, LONS), ('lat', at, ag, LATS)):
        cls = [c for c in cls if (gtv == c).any() or (gen == c).any()]
        print(f'\nA2 confusion {nm} (rows GT, cols generated; seed 42 mode i):')
        ab = lambda c: ''.join(x[0] for x in re.findall(r'[A-Za-z&]+', c))[:5]
        print('  ' + 'GT / gen'.ljust(24) + ' ' + ' '.join(f'{ab(c):>6}' for c in cls) + '      n  recall')
        for c in cls:
            row = [(gen[gtv == c] == k).sum() for k in cls]
            n = (gtv == c).sum()
            print(f'  {c[:24]:24} ' + ' '.join(f'{x:6d}' for x in row) + f' {n:6d}  {row[cls.index(c)] / max(n, 1):.3f}')
        print('  key: ' + ', '.join(f'{ab(c)} = {c}' for c in cls))
    ok = (lg == lt) & (ag == at)
    li, lv = gi['l2'][:, 5], v2['l2'][:, 5]
    print(f'\nA2 decision correct (lon AND lat): {ok.mean():.3f} ({ok.sum()} of {len(ok)})')
    print(f'  L2@3s CoC (i): correct {li[ok].mean():.3f} | wrong {li[~ok].mean():.3f};  M1-v2a s42 on the same samples: '
          f'correct {lv[ok].mean():.3f} | wrong {lv[~ok].mean():.3f}')
    for nm, m in (('correct', ok), ('wrong', ~ok)):
        c = E.scene_boot([r for r, x in zip(rva, m) if x], li[m], lv[m])
        print(f'  CoC - M1-v2a s42 on {nm:7} decisions (n {m.sum()}): {f(c)}')
    gap = (li - lv).sum(); share = (li - lv)[~ok].sum() / gap
    print(f'  GC1 gap (sum over val) {gap:.1f} m; from wrong decisions {(li - lv)[~ok].sum():.1f} m = share {share:.3f}; '
          f'from correct {(li - lv)[ok].sum():.1f} m')
    for c in LONS[:5]:
        m = ~ok & (lt == c)
        if m.any():
            print(f'    wrong decisions with GT {c[:22]:22}: n {m.sum():4d}, gap contribution {(li - lv)[m].sum() / gap:+.3f}')


if __name__ == '__main__':
    main()
