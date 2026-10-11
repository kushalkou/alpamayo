"""ar1/r11b_eval.py -- R11 Part B: CoC arm seed 42, PROVISIONAL. Gates GC1-GC6 (one seed,
indicative) and diagnostics. Rules in the R11_REPORT.md header. Per-sample dumps, holdout tau
(from the mode-i holdout dump) for every mode.
  python ar1/r11b_eval.py"""
import sys, re, pickle, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from r33_eval import derive, controls
from r5_a5c1 import coll_rates, comfort
from meta_actions import LON, LAT

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RX = re.compile(r'^Driving decision: ([^;]+); ([^.]+)\. Critical components: (.+)\. Reasoning: (.+)$')
TAG = sys.argv[1] if len(sys.argv) > 1 else 'COC'          # R12: COC3 = v3 traces
COCP = f'{DATA}/ar1_coc_2hz_v3.pkl' if TAG == 'COC3' else f'{DATA}/ar1_coc_2hz.pkl'
MODES = (('(i) gen', TAG), ('(ii) none', TAG + '_none'), ('(iii) shuf', TAG + '_shuf'), ('(iv) gt [ORACLE]', TAG + '_gt'))


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); occ = C['occ']; tokv = [r['sample_token'] for r in rva]
    coc = pickle.load(open(COCP, 'rb'))
    g = np.linalg.norm(np.stack([np.asarray(T[t]['P'])[:6] for t in tokv]), axis=2).max(1)
    v0 = np.array([V0[t]['v0_can'] for t in tokv])
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); w = np.array([wss[r['scene_name']] for r in rva])
    stat = C['st'] == 'stationary'; gs = stat & (g < 0.5); gm = stat & (g > 1.0)
    ALL = np.ones(len(rva), bool)
    S = {'all': ALL, 'excl. first frames': ~C['ff'], 'excl. WSS': ~w, 'WSS only': w,
         'C2 STOPPED': (v0 <= 0.2) & (g <= 1.0), 'C2 START': (v0 <= 0.2) & (g > 1.0), 'C2 MOVING': v0 > 0.2}
    tau = Q.run(TAG)['tau']
    print(f'PROVISIONAL (CoC traces not signed off). {TAG} = {COCP}; CoC seed 42; holdout tau {tau} (mode i holdout dump)')
    PS, PR, DV = {}, {}, {}
    for nm, src in MODES:
        PS[nm] = Q.strip(Q.run(TAG, src=src)); DV[nm] = load(src, 'val'); PR[nm] = preds(DV[nm], rva, 'hybrid', tau)
    v2 = Q.strip(Q.run('M1v2a')); v23 = Q.avg([Q.run(t) for t in ('M1v2a', 'M1v2a_s123', 'M1v2a_s2024')]); b1 = Q.avg([Q.run(t) for t in ('B1', 'B1_s123', 'B1_s2024')])
    f = lambda c: f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]'
    bt = lambda A, B, m: E.scene_boot([r for r, x in zip(rva, m) if x], A['l2'][m, 5], B['l2'][m, 5])

    print('\nL2@3s NoAvg / ADE@6s by mode and stratum')
    for nm in PS:
        print(f'  {nm:17} ' + ' | '.join(f'{k} {PS[nm]["l2"][m, 5].mean():.3f}' for k, m in S.items()) +
              f' | ADE6 {np.nanmean(PS[nm]["ade"]):.3f}')
    for nm, P in (('M1-v2a s42', v2), ('M1-v2a 3-seed', v23), ('B1 3-seed', b1)):
        print(f'  {nm:17} ' + ' | '.join(f'{k} {P["l2"][m, 5].mean():.3f}' for k, m in S.items()) + f' | ADE6 {np.nanmean(P["ade"]):.3f}')

    gi = PS['(i) gen']
    print('\nPaired scene bootstrap, L2@3s (ADE@6s for all val after |)')
    pairs = (('GC1 (i) - M1-v2a s42', gi, v2), ('GC1b (i) - M1-v2a 3-seed', gi, v23), ('(i) - B1 3-seed', gi, b1),
             ('GC2 (i) - (ii)', gi, PS['(ii) none']), ('GC6 (iii) - (i)', PS['(iii) shuf'], gi), ('(iv) - (i) [ORACLE]', PS['(iv) gt [ORACLE]'], gi))
    G = {}
    for lab, A, B in pairs:
        cs = {k: bt(A, B, m) for k, m in S.items()}; G[lab] = cs['all']
        c6 = E.scene_boot(rva, A['ade'], B['ade'])
        print(f'  {lab:26} all {f(cs["all"])} | ADE6 {f(c6)}')
        print(f'  {"":26} ' + '; '.join(f'{k} {f(cs[k])}' for k in list(S)[1:]))

    print('\nPer mode: R3 FG all / WSS / non-WSS, MG | C2 FG MG | Col @3 s NoAvg aa / yaw | comfort first-order (n_fut 12)')
    n12 = np.where([r['n_fut'] == 12 for r in rva])[0]
    for nm in PS:
        P = PR[nm]; dm = np.linalg.norm(P[:, :6], axis=2).max(1)
        fg = lambda m: (dm[m & gs] > 1.0).mean()
        ca = coll_rates(rva, P, occ, 'aa'); cy = coll_rates(rva, P, occ, 'yaw')
        cf = np.mean([(lambda c: c['lon_acc'] and c['lat_acc'] and c['yaw_rate'])(comfort(P[i], T[tokv[i]]['v0'])) for i in n12])
        print(f'  {nm:17} FG {fg(ALL):.3f} / {fg(w):.3f} / {fg(~w):.3f}, MG {(dm[gm] < 0.5).mean():.3f} | '
              f'C2 {(dm[S["C2 STOPPED"]] > 1.0).mean():.3f} {(dm[S["C2 START"]] <= 1.0).mean():.3f} | '
              f'{ca[0]:.2f} / {cy[0]:.2f} | {cf:.3f}')
    fg_i = (np.linalg.norm(PR['(i) gen'][:, :6], axis=2).max(1)[gs] > 1.0).mean()

    print('\nGenerated CoC (mode i): parse, decision accuracy, collapse')
    d = DV['(i) gen']['data']; txt = [d[t]['coc_gen'] for t in tokv]; mm = [RX.match(x) for x in txt]
    ok = np.array([m is not None for m in mm]); lon_g = np.array([m[1] if m else '' for m in mm]); lat_g = np.array([m[2] if m else '' for m in mm])
    lon_t = np.array([coc[t]['lon'] for t in tokv]); lat_t = np.array([coc[t]['lat'] for t in tokv])
    L = np.array([d[t]['coc_len'] for t in tokv])
    print(f'  parse {ok.mean():.3f}; length mean {L.mean():.1f} max {L.max()}; decision accuracy lon {(lon_g == lon_t).mean():.3f}, '
          f'lat {(lat_g == lat_t).mean():.3f}, both {((lon_g == lon_t) & (lat_g == lat_t)).mean():.3f}')
    cg, ct = collections.Counter(lon_g), collections.Counter(lon_t)
    print('  generated lon: ' + ', '.join(f'{k or "(unparsed)"} {v / len(tokv):.3f} (label {ct[k] / len(tokv):.3f})' for k, v in cg.most_common()))
    top = cg.most_common(1)[0]
    print('  generated lat: ' + ', '.join(f'{k or "(unparsed)"} {v / len(tokv):.3f}' for k, v in collections.Counter(lat_g).most_common()))
    print(f'  by GT lon class, accuracy: ' + '; '.join(f'{k} n {(lon_t == k).sum()} {(lon_g[lon_t == k] == k).mean():.3f}'
                                                    for k in sorted(set(lon_t))))

    print('\nWords: consistency (12 slots) and decision agreement; per-class F1 (val n_fut = 12)')
    lab = pickle.load(open(f'{DATA}/ar1_meta2hz.pkl', 'rb')); r12 = [r for r in rva if r['n_fut'] == 12]
    y = np.array([lab[r['sample_token']] for r in r12]); maj = np.array([4] * 6 + [6] * 6)
    cons = {}
    for nm, src in MODES:
        dv = DV[nm]; ga = np.array([dv['data'][t]['meta_gen'] for t in tokv])
        der = np.array([derive(a, k, T[r['sample_token']]['v0']) for (a, k), r in zip(controls(dv, rva, tau), rva)])
        weq = (der == ga).all(1)
        if nm == '(i) gen':
            dstop = (der[:, :6] == 5).any(1); dl = np.isin(der[:, 6:], (0, 2)).any(1); dr = np.isin(der[:, 6:], (1, 3)).any(1)
            agree = ok & np.where(lon_g == 'Stop for static constraints', dstop, ~dstop) & \
                np.where(lat_g == 'Turn left', dl, True) & np.where(lat_g == 'Turn right', dr, True)
            cons['gc3'] = (weq & agree).mean()
            ow = ok & np.where(lon_g == 'Stop for static constraints', dstop, True) & \
                np.where(lat_g == 'Turn left', dl, True) & np.where(lat_g == 'Turn right', dr, True)
            print(f'  {nm:17} words {weq.mean():.3f}; decision agrees {agree.mean():.3f} (stop <-> derived stop; turn side); '
                  f'GC3 both {cons["gc3"]:.3f}; [diagnostic, one-way stop -> derived stop: {(weq & ow).mean():.3f}]')
        else:
            print(f'  {nm:17} words {weq.mean():.3f}')
        gen = np.array([dv['data'][r['sample_token']]['meta_gen'] for r in r12]); fs = []
        for o, nms in ((0, LON), (6, LAT)):
            yy, pp = y[:, o:o + 6].ravel(), gen[:, o:o + 6].ravel()
            for c in range(7):
                n = (yy == c).sum()
                if n >= 30:
                    tp = ((pp == c) & (yy == c)).sum(); P_ = tp / max((pp == c).sum(), 1); R_ = tp / n
                    fs.append(f'{nms[c][:9]} {2 * P_ * R_ / max(P_ + R_, 1e-9):.2f}')
        print(f'  {"":17} collapse {(gen == maj).all(1).mean():.3f} (labels {(y == maj).all(1).mean():.3f}); F1 ' + ', '.join(fs))

    print('\nGATES (ONE SEED, INDICATIVE, PROVISIONAL)')
    c = G['GC1 (i) - M1-v2a s42']
    print(f'  GC1 (i) - M1-v2a s42 {f(c)} -> ' + ('"CoC improves on meta-action"' if c[2] < 0 else '"CoC hurts"' if c[1] > 0
                                               else '"no measurable CoC effect"') + f'; vs 3-seed mean {f(G["GC1b (i) - M1-v2a 3-seed"])}')
    c = G['GC2 (i) - (ii)']
    print(f'  GC2 (i) - (ii) {f(c)} -> ' + ('"gain does not need generated reasoning"' if c[1] <= 0 <= c[2] else 'CI excludes 0'))
    print(f'  GC3 consistency {cons["gc3"]:.3f} (>= 0.70) -> ' + ('PASS' if cons['gc3'] >= 0.70 else 'FAIL'))
    print(f'  GC4 false-go (i) {fg_i:.3f} (<= 0.119) -> ' + ('PASS' if fg_i <= 0.119 else 'FAIL'))
    print(f'  GC5 majority generated decision {top[0]} {top[1] / len(tokv):.3f} vs label {ct[top[0]] / len(tokv):.3f} + 0.10 -> ' +
          ('PASS' if top[1] / len(tokv) <= ct[top[0]] / len(tokv) + 0.10 else 'FAIL (collapse)'))
    c = G['GC6 (iii) - (i)']
    print(f'  GC6 (iii) - (i) {f(c)} -> ' + ('"generated causes not used by the action decoder"' if c[1] <= 0 <= c[2] else 'CI excludes 0'))


if __name__ == '__main__':
    main()
