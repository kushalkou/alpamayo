"""analyze_gate2.py — Gate 2.2 / 2.3 / 2.4 tables from the saved per-sample json.

2.2 stratified ADE/FDE (mean AND median) + paired bootstrap CIs
2.3 distribution: win rate vs CV, tail quantiles, oracle-switch ceiling
2.4 deployable turn detectors (past curvature; model's own predicted curvature)

All decodes are CORRECTED argmax (STOP -> 0.0). Y1's published 3.924 and the
Gate 0 reproduction used the legacy STOP -> bin-32 mapping.
"""
import sys, json, numpy as np

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/res_decode_gate2.json'
CURV_T = 0.05
LABS = ['1s', '2s', '3s', '6s']
rng = np.random.RandomState(0)


def paired(a, b, n=20000):
    """mean(a-b) with a paired bootstrap CI. Negative => a better (lower ADE)."""
    d = a - b
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return d.mean(), lo, hi, p


def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),) * 2
    ph = k / n; d = 1 + z*z/n
    c = (ph + z*z/(2*n)) / d
    h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / d
    return c - h, c + h


def main():
    r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else RES))
    idx = sorted(int(i) for i in r['curv'])
    curv = {int(i): v for i, v in r['curv'].items()}
    v0 = {int(i): v for i, v in r['v0'].items()}
    pcv = {int(i): v for i, v in r.get('past_curv', {}).items()}
    nh = {int(i): v for i, v in r.get('n_hist', {}).items()}
    cvad = {int(i): v['ade'] for i, v in r['cv'].items()}
    cvfd = {int(i): v['fde'] for i, v in r['cv'].items()}
    models = list(r['per'].keys())
    variants = list(r['per'][models[0]].keys())

    def A(m, vn, lab='6s'):
        P = r['per'][m][vn]
        return {int(i): P[i][0]['ade'][lab] for i in P}
    def F(m, vn, lab='6s'):
        P = r['per'][m][vn]
        return {int(i): P[i][0]['fde'][lab] for i in P}

    subsets = [('ALL', idx),
               ('STRAIGHT', [i for i in idx if curv[i] <= CURV_T]),
               ('TURNING', [i for i in idx if curv[i] > CURV_T]),
               ('STATIONARY', [i for i in idx if abs(v0[i]) < 0.5])]

    # ── 2.2 ──────────────────────────────────────────────────────────────────
    print("=" * 108)
    print("GATE 2.2 — FULL TEST SET (n=%d). Corrected argmax (STOP->0.0) everywhere." % len(idx))
    print("Y1's published 3.924 and the Gate 0 repro used the legacy STOP->bin-32 mapping.")
    print("=" * 108)
    for sname, sub in subsets:
        print(f"\n-- {sname} (n={len(sub)}) --")
        if not sub:
            print("  (empty subset)"); continue
        print(f"{'model/decode':34} " + " ".join(f"{'ADE'+l:>7}" for l in LABS) +
              f" {'med6s':>7} {'FDE6s':>7} {'medFDE':>7}")
        row = [np.mean([cvad[i][l] for i in sub]) for l in LABS]
        print(f"{'CV (baseline)':34} " + " ".join(f"{v:7.3f}" for v in row) +
              f" {np.median([cvad[i]['6s'] for i in sub]):7.3f}"
              f" {np.mean([cvfd[i]['6s'] for i in sub]):7.3f}"
              f" {np.median([cvfd[i]['6s'] for i in sub]):7.3f}")
        for m in models:
            for vn in variants:
                a = A(m, vn); f = F(m, vn)
                row = [np.mean([A(m, vn, l)[i] for i in sub]) for l in LABS]
                print(f"{m+'/'+vn:34} " + " ".join(f"{v:7.3f}" for v in row) +
                      f" {np.median([a[i] for i in sub]):7.3f}"
                      f" {np.mean([f[i] for i in sub]):7.3f}"
                      f" {np.median([f[i] for i in sub]):7.3f}")

    print("\n" + "=" * 108)
    print("PAIRED BOOTSTRAP, ADE@6s (negative => first term better)")
    print("=" * 108)
    print(f"{'stratum':12} {'comparison':44} {'delta':>8} {'95% CI':>18} {'p':>8}")
    for sname, sub in subsets:
        if not sub: continue
        cmps = []
        for m in models:
            cmps.append((f'{m}: V1s - argmax',
                         np.array([A(m,'V1s_expect')[i] for i in sub]),
                         np.array([A(m,'argmax_corrected')[i] for i in sub])))
            cmps.append((f'{m}: V1s - CV',
                         np.array([A(m,'V1s_expect')[i] for i in sub]),
                         np.array([cvad[i]['6s'] for i in sub])))
        for dec in ['argmax_corrected', 'V1s_expect']:
            if 'zeroboth_jul12' in models and 'y1_full' in models:
                cmps.append((f'zeroboth - y1_full  [{dec}]',
                             np.array([A('zeroboth_jul12',dec)[i] for i in sub]),
                             np.array([A('y1_full',dec)[i] for i in sub])))
            if 'y1_full' in models and 'y1_ego' in models:
                cmps.append((f'y1_full - y1_ego    [{dec}]',
                             np.array([A('y1_full',dec)[i] for i in sub]),
                             np.array([A('y1_ego',dec)[i] for i in sub])))
        for lab, a, b in cmps:
            d, lo, hi, p = paired(a, b)
            print(f"{sname:12} {lab:44} {d:+8.3f}  [{lo:+6.3f},{hi:+6.3f}] {p:8.4f}")

    # ── 2.3 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print("GATE 2.3 — DISTRIBUTION, NOT JUST THE MEAN")
    print("=" * 108)
    print("\n(a) WIN RATE vs CV — fraction of samples with model ADE@6s < CV ADE@6s (Wilson 95%)")
    print(f"{'stratum':12} {'model/decode':30} {'n':>6} {'win rate':>9} {'95% CI':>18} {'mean gap':>9}")
    for sname, sub in subsets:
        if not sub: continue
        for m in models:
            for vn in ['argmax_corrected', 'V1s_expect']:
                a = A(m, vn); w = sum(1 for i in sub if a[i] < cvad[i]['6s'])
                lo, hi = wilson(w, len(sub))
                gap = np.mean([a[i]-cvad[i]['6s'] for i in sub])
                print(f"{sname:12} {m+'/'+vn:30} {len(sub):6d} {w/len(sub):9.4f} "
                      f"[{lo:6.4f},{hi:6.4f}] {gap:+9.3f}")
    print("\n(b) TAIL — per-sample ADE@6s quantiles")
    print(f"{'stratum':12} {'series':30} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'mean':>8}")
    for sname, sub in subsets:
        if not sub: continue
        series = [('CV', np.array([cvad[i]['6s'] for i in sub]))]
        for m in models:
            series.append((m+'/V1s', np.array([A(m,'V1s_expect')[i] for i in sub])))
        for lab, v in series:
            print(f"{sname:12} {lab:30} " + " ".join(f"{np.percentile(v,q):8.3f}"
                  for q in [50,75,90,95,99]) + f" {v.mean():8.3f}")
    print("\n(c) ORACLE SWITCH — per-sample min(model, CV): ceiling on any CV/model selector")
    print(f"{'stratum':12} {'model (V1s)':22} {'CV':>8} {'model':>8} {'oracle':>8} "
          f"{'gain vs CV':>11} {'gain vs model':>13}")
    for sname, sub in subsets:
        if not sub: continue
        c = np.array([cvad[i]['6s'] for i in sub])
        for m in models:
            a = np.array([A(m,'V1s_expect')[i] for i in sub])
            o = np.minimum(a, c)
            print(f"{sname:12} {m:22} {c.mean():8.3f} {a.mean():8.3f} {o.mean():8.3f} "
                  f"{c.mean()-o.mean():11.3f} {a.mean()-o.mean():13.3f}")

    # ── 2.4 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 108)
    print("GATE 2.4 — IS THE TURN ADVANTAGE DEPLOYABLE?")
    print("The TURNING stratum uses max|GT FUTURE curvature| — an ORACLE label.")
    print("=" * 108)
    oracle = set(i for i in idx if curv[i] > CURV_T)
    nohist = [i for i in idx if nh.get(i, 0) == 0]
    print(f"\noracle turning n={len(oracle)} / {len(idx)}   "
          f"samples with NO past_poses: {len(nohist)} ({100*len(nohist)/len(idx):.1f}%)")

    dets = []
    if pcv:
        dets.append(('(i) PAST curvature >0.05', {i: pcv[i] > CURV_T for i in idx}))
    pk = r.get('predcurv', {})
    for m in models:
        if m in pk and 'V1s_expect' in pk[m]:
            d = {int(i): v for i, v in pk[m]['V1s_expect'].items()}
            dets.append((f'(ii) PREDICTED curv >0.05 [{m}/V1s]', {i: d.get(i, 0) > CURV_T for i in idx}))

    for dname, dd in dets:
        pos = [i for i in idx if dd[i]]
        tp = len([i for i in pos if i in oracle]); fp = len(pos) - tp
        fn = len(oracle) - tp; tn = len(idx) - tp - fp - fn
        prec = tp / max(len(pos), 1); rec = tp / max(len(oracle), 1)
        f1 = 2*prec*rec/max(prec+rec, 1e-9)
        print(f"\n-- detector {dname} --")
        print(f"  confusion: TP={tp} FP={fp} FN={fn} TN={tn}   "
              f"precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}   n_positive={len(pos)}")
        if len(pos) < 10:
            print("  (too few positives to evaluate)"); continue
        c = np.array([cvad[i]['6s'] for i in pos])
        print(f"  {'model (on detected subset)':32} {'ADE6s':>8} {'CV':>8} "
              f"{'V1s-CV':>9} {'95% CI':>18} {'p':>8} {'win rate':>9}")
        for m in models:
            a = np.array([A(m,'V1s_expect')[i] for i in pos])
            d_, lo, hi, p = paired(a, c)
            wr = (a < c).mean()
            print(f"  {m+'/V1s':32} {a.mean():8.3f} {c.mean():8.3f} "
                  f"{d_:+9.3f}  [{lo:+6.3f},{hi:+6.3f}] {p:8.4f} {wr:9.4f}")


if __name__ == '__main__':
    main()
