"""ar1/r13b_eval.py -- R13 Part B: CoC 3 seeds (v3 traces) gates GC1-3, GC4-3, GC5-3, GC6b-3; CoC-AUX
seed 42 gate GX1; per-run collisions, strata, word F1, consistency. Rules in R13_REPORT.md.
  python ar1/r13b_eval.py [--no_aux]"""
import sys, re, pickle, collections
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
RX = re.compile(r'^Driving decision: ([^;]+); ([^.]+)\. Critical components: (.+)\. Reasoning: (.+)$')
COC = ('COC3', 'COC3_s123', 'COC3_s2024'); V2 = ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); occ = C['occ']; tokv = [r['sample_token'] for r in rva]
    coc = pickle.load(open(f'{DATA}/ar1_coc_2hz_v3.pkl', 'rb'))
    g = np.linalg.norm(np.stack([np.asarray(T[t]['P'])[:6] for t in tokv]), axis=2).max(1)
    v0 = np.array([V0[t]['v0_can'] for t in tokv])
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    stat = C['st'] == 'stationary'; gs = stat & (g < 0.5); gm = stat & (g > 1.0)
    S = {'all': np.ones(len(rva), bool), 'excl. first frames': ~C['ff'], 'excl. WSS': ~w, 'WSS only': w,
         'C2 STOPPED': (v0 <= 0.2) & (g <= 1.0), 'C2 START': (v0 <= 0.2) & (g > 1.0), 'C2 MOVING': v0 > 0.2}
    f = lambda c: f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]'
    bt = lambda A, B, m: E.scene_boot([r for r, x in zip(rva, m) if x], A['l2'][m, 5], B['l2'][m, 5])
    lab = pickle.load(open(f'{DATA}/ar1_meta2hz.pkl', 'rb')); r12 = [r for r in rva if r['n_fut'] == 12]
    y = np.array([lab[r['sample_token']] for r in r12]); maj = np.array([4] * 6 + [6] * 6)
    lt = np.array([coc[t]['lon'] for t in tokv])

    def run(tag, src, label, decision=True):
        """per-run diagnostics; returns (per-sample metrics, false-go flags on GT-stopped, generated lon)"""
        R = Q.run(tag, src=src); dv = load(src, 'val'); tau = R['tau']; P = preds(dv, rva, 'hybrid', tau)
        dm = np.linalg.norm(P[:, :6], axis=2).max(1); fgv = dm[gs] > 1.0
        ca = coll_rates(rva, P, occ, 'aa'); cy = coll_rates(rva, P, occ, 'yaw')
        ga = np.array([dv['data'][t]['meta_gen'] for t in tokv])
        der = np.array([derive(a, k, T[r['sample_token']]['v0']) for (a, k), r in zip(controls(dv, rva, tau), rva)])
        weq = (der == ga).all(1)
        gen12 = np.array([dv['data'][r['sample_token']]['meta_gen'] for r in r12]); fs = []
        for o, nms in ((0, LON), (6, LAT)):
            yy, pp = y[:, o:o + 6].ravel(), gen12[:, o:o + 6].ravel()
            for c in range(7):
                n = (yy == c).sum()
                if n >= 30:
                    tp = ((pp == c) & (yy == c)).sum(); P_ = tp / max((pp == c).sum(), 1); R_ = tp / n
                    fs.append(f'{nms[c][:7]} {2 * P_ * R_ / max(P_ + R_, 1e-9):.2f}')
        lg = None; extra = ''
        if decision and 'coc_gen' in dv['data'][tokv[0]] and dv['data'][tokv[0]]['coc_gen']:
            mm = [RX.match(dv['data'][t]['coc_gen']) for t in tokv]
            lg = np.array([m[1] if m else '' for m in mm]); ag = np.array([m[2] if m else '' for m in mm])
            dstop = (der[:, :6] == 5).any(1); dl = np.isin(der[:, 6:], (0, 2)).any(1); dr = np.isin(der[:, 6:], (1, 3)).any(1)
            ok = np.array([m is not None for m in mm])
            agree = ok & np.where(lg == 'Stop for static constraints', dstop, ~dstop) & np.where(ag == 'Turn left', dl, True) & \
                np.where(ag == 'Turn right', dr, True)
            extra = (f' | parse {ok.mean():.3f} dec acc lon {(lg == lt).mean():.3f}; consistency words {weq.mean():.3f} '
                     f'decision {agree.mean():.3f} both {(weq & agree).mean():.3f}')
        else:
            extra = f' | consistency words {weq.mean():.3f}'
        R_ = Q.strip(R)
        print(f'  {label:22} tau {tau} | L2@3s ' + ' '.join(f'{R_["l2"][m, 5].mean():.3f}' for m in S.values()) +
              f' | ADE6 {np.nanmean(R_["ade"]):.3f} | FG {fgv.mean():.3f} / {(dm[gs & w] > 1.0).mean():.3f} / '
              f'{(dm[gs & ~w] > 1.0).mean():.3f} MG {(dm[gm] < 0.5).mean():.3f} | C2 FG {(dm[S["C2 STOPPED"]] > 1.0).mean():.3f} '
              f'MG {(dm[S["C2 START"]] <= 1.0).mean():.3f} | Col {ca[0]:.2f}/{cy[0]:.2f}' + extra)
        print(f'  {"":22} collapse {(gen12 == maj).all(1).mean():.3f}; F1 ' + ', '.join(fs))
        return R_, fgv, lg

    print('Per run (L2@3s strata: ' + ', '.join(S) + '; FG all / WSS / non-WSS)')
    PS = {}; FG = []; LG = []
    for t in COC:
        for mode, src in (('(i) gen', t), ('(iii) shuf', t + '_shuf'), ('(iv) gt', t + '_gt')):
            r_, fg, lg = run(t, src, f'{t} {mode}', decision=(mode == '(i) gen'))
            PS[(t, mode)] = r_
            if mode == '(i) gen':
                FG.append(fg); LG.append(lg)
    for t in V2:
        PS[(t, 'm')] = Q.strip(Q.run(t))
    avg = lambda L: {k: np.mean([x[k] for x in L], 0) for k in L[0]}
    Ci, Cs, Cg = (avg([PS[(t, m)] for t in COC]) for m in ('(i) gen', '(iii) shuf', '(iv) gt'))
    Mv = avg([PS[(t, 'm')] for t in V2])
    print('\n3-seed paired scene bootstrap, L2@3s NoAvg')
    G = {}
    for lab_, A, B in (('GC1-3 CoC (i) - M1-v2a', Ci, Mv), ('GC6b-3 (iii) - (iv)', Cs, Cg), ('(iv) GT CoC - M1-v2a', Cg, Mv)):
        cs = {k: bt(A, B, m) for k, m in S.items()}; G[lab_] = cs['all']
        print(f'  {lab_:24} all {f(cs["all"])} | ADE6 {f(E.scene_boot(rva, A["ade"], B["ade"]))}')
        print(f'  {"":24} ' + '; '.join(f'{k} {f(cs[k])}' for k in list(S)[1:]))
    fg = np.concatenate(FG).mean(); lg = np.concatenate(LG)
    cg = collections.Counter(lg); top = cg.most_common(1)[0]; share = top[1] / len(lg); lshare = (lt == top[0]).mean()
    print(f'  pooled generated lon (3 seeds): ' + ', '.join(f'{k or "(unparsed)"} {v / len(lg):.3f}' for k, v in cg.most_common()))
    print('\nGATES (3 seeds)')
    c = G['GC1-3 CoC (i) - M1-v2a']
    print(f'  GC1-3 {f(c)} -> ' + ('"CoC improves"' if c[2] < 0 else '"CoC hurts"' if c[1] > 0 else '"no measurable CoC effect"'))
    print(f'  GC4-3 pooled false-go {fg:.3f} (<= 0.119) -> ' + ('PASS' if fg <= 0.119 else 'FAIL'))
    print(f'  GC5-3 {top[0]} generated {share:.3f} vs label {lshare:.3f} + 0.10 -> ' + ('PASS' if share <= lshare + 0.10 else 'FAIL (collapse)'))
    c = G['GC6b-3 (iii) - (iv)']
    print(f'  GC6b-3 {f(c)} -> ' + ('"given the true decision, the causes add nothing"' if c[1] <= 0 <= c[2] else
                                    '"shuffled causes better (suspect)"' if c[2] < 0 else '"causes matter"'))
    if '--no_aux' in sys.argv:
        return
    print('\nCoC-AUX seed 42')
    off, _, _ = run('COC3AUX_none', 'COC3AUX_none', 'AUX reasoning OFF', decision=False)
    on, _, _ = run('COC3AUX', 'COC3AUX', 'AUX reasoning ON')
    v42 = PS[('M1v2a', 'm')]
    for lab_, A in (('GX1 AUX (OFF) - M1-v2a s42', off), ('AUX (ON) - M1-v2a s42', on)):
        cs = {k: bt(A, v42, m) for k, m in S.items()}
        print(f'  {lab_:26} all {f(cs["all"])}; ' + '; '.join(f'{k} {f(cs[k])}' for k in list(S)[1:]))
        if lab_.startswith('GX1'):
            c = cs['all']
    print(f'  GX1 {f(c)} -> ' + ('"CoC helps as auxiliary supervision"' if c[2] < 0 else '"hurts"' if c[1] > 0 else '"no effect"'))


if __name__ == '__main__':
    main()
