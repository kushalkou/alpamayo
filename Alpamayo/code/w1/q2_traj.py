"""w1/q2_traj.py -- QUEUE v2 evaluation of the VLA TRAJECTORY runs (G5, G6, G7).

Shared eval as gate_a.py: official val, ALL 5,119 and EXCL. first frames; free-running AR
dumps, STOP-aware hybrid decode with tau fit on HOLDOUT; L2 / collision both conventions,
ADE/FDE@6s, mean / median / p95; strata; paired scene bootstrap. Nothing is tuned on val.
  python w1/q2_traj.py G5     G5 (30 epochs) vs A3 (10 epochs), same recipe and seed
  python w1/q2_traj.py T9     3 seeds: (i) ego + cmd (A3 recipe), (ii) ego + oracle meta-action
                              (A2 design + turn weighting) with the 0/10/20/40% flip sweep
  python w1/q2_traj.py G7     vision end-to-end vs its no-camera twin; camera-shuffle test
Per-sample results are cached in results/w1_q2_traj.pkl (for figures / REVIEW_RESULTS).
"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import evalw1 as E, plan_metrics as PMX
from gate_a import load, preds, ade12, TAUS

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
CACHE = f'{RES}/w1_q2_traj.pkl'
SEEDS = (42, 123, 2024)
_C = None


def ctx():
    global _C
    if _C is None:
        rva = E.split_records('val'); rho = E.split_records('holdout')
        occ = [PMX.occupancies(r) for r in rva]
        W, V0, _ = E.data()
        ff = np.array([not V0[r['sample_token']]['can_ok'] and len(r['past_poses']) == 0 for r in rva])
        tab = E.oracle_kin_table()
        base = {'CV': E.cv(rva), 'KIN': E.kin(rva), 'ORACLE-KIN': E.oracle_kin(rva, tab)}
        cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
        mlp = pickle.load(open(f'{RES}/w1_egomlp.pkl', 'rb'))
        for k, v in base.items():
            if k not in cache: cache[k] = E.per_sample(rva, v, occ)
        for s in SEEDS:
            if f'egoMLP s{s}' not in cache and s in mlp['preds']:
                cache[f'egoMLP s{s}'] = E.per_sample(rva, mlp['preds'][s]['val'], occ)
        _C = dict(rva=rva, rho=rho, occ=occ, ff=ff, tab=tab, cache=cache, st=E.strata(rva))
    return _C


def run(tag, src=None, f=0.0):
    """per-sample val metrics of dump `src or tag` at test flip f, tau fit on tag's holdout."""
    C = ctx(); key = (tag, src or tag, f)
    if key not in C['cache']:
        dh = load(tag, 'holdout')
        tau = min(TAUS, key=lambda t: ade12(C['rho'], preds(dh, C['rho'], 'hybrid', t)))
        dv = load(src or tag, 'val', f)
        C['cache'][key] = E.per_sample(C['rva'], preds(dv, C['rva'], 'hybrid', tau), C['occ'])
        C['cache'][key]['tau'] = tau
        s0 = [dv['data'][r['sample_token']]['tok'][0] == dv['data'][r['sample_token']]['gt_tok'][0]
              for r in C['rva'] if r['n_fut'] == 12]
        C['cache'][key]['slot0'] = float(np.mean(s0))
        pickle.dump(C['cache'], open(CACHE, 'wb'))
    return C['cache'][key]


def strip(p):
    return {k: v for k, v in p.items() if k not in ('tau', 'slot0')}


def avg(ps):
    ps = [strip(p) for p in ps]
    return {k: np.mean([p[k] for p in ps], 0) for k in ps[0]}


def blocks():
    C = ctx()
    yield 'ALL 5,119', np.ones(len(C['rva']), bool)
    yield f'EXCL. FIRST FRAMES (n={int((~C["ff"]).sum())})', ~C['ff']


def table(rows, cmps, extra=None):
    C = ctx()
    for blk, m in blocks():
        sub = [r for r, k in zip(C['rva'], m) if k]
        cut = lambda p: {k: v[m] for k, v in strip(p).items()}
        print(f'\n===== official val, {blk} =====')
        print(E.HDR)
        for k, p in rows.items():
            print(E.row(k[:22], cut(p)) + (f'   {k}' if len(k) > 22 else ''))
        print('\n  PAIRED SCENE BOOTSTRAP (negative => first better)')
        for a, b in cmps:
            print(E.compare(sub, cut(rows[a]), cut(rows[b]), f'{a[:16]} - {b[:16]}'))
        if extra: extra(m, sub, cut)


def strata_lines(rows, keys):
    C = ctx()
    print('\nSTRATA (L2@3s NoAvg mean / L2 TemAvg@3s mean), ALL 5,119')
    for s in ('straight', 'turning', 'stationary'):
        m = C['st'] == s
        print(f'  {s:10} n={m.sum():5d} ' + '  '.join(
            f'{k} {strip(rows[k])["l2"][m, 5].mean():.3f}/{strip(rows[k])["l2"][m].mean():.3f}' for k in keys))


def base_rows():
    c = ctx()['cache']
    return {'CV': c['CV'], 'KIN': c['KIN'], 'ORACLE-KIN': c['ORACLE-KIN'], 'egoMLP s42': c['egoMLP s42']}


def shuffle_line(tag, sfx):
    p = run(tag); q = run(tag, f'{tag}_{sfx}')
    a, b = E.summary(strip(p))['ADE6'], E.summary(strip(q))['ADE6']
    return (f'  {tag}: slot-0 {p["slot0"]:.3f} ADE6 {a:.3f} | {sfx.upper()}: slot-0 {q["slot0"]:.3f} '
            f'ADE6 {b:.3f} (x{b/a:.3f}; within 5% -> {"FAILS" if abs(b/a-1) <= .05 else "passes"})')


def g5():
    rows = base_rows(); rows['A3 (10 ep)'] = run('A3'); rows['G5 (30 ep)'] = run('G5')
    print(f'G5: tau A3 {rows["A3 (10 ep)"]["tau"]}, G5 {rows["G5 (30 ep)"]["tau"]}')
    print(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/w1_q2_len.txt').read())
    table(rows, [('G5 (30 ep)', 'A3 (10 ep)'), ('G5 (30 ep)', 'KIN'), ('G5 (30 ep)', 'egoMLP s42')])
    strata_lines(rows, ['KIN', 'egoMLP s42', 'A3 (10 ep)', 'G5 (30 ep)'])
    print('\nSLOT-0 + INPUT-SHUFFLE'); print(shuffle_line('G5', 'shuf'))


def seed_tags():
    LEN = int(open(f'{RES}/Q2_LEN').read())
    ci = {42: 'G5' if LEN == 30 else 'A3', 123: 'G6C_s123', 2024: 'G6C_s2024'}
    mi = {s: f'G6M_s{s}' for s in SEEDS}
    return LEN, ci, mi


def t9():
    LEN, ci, mi = seed_tags()
    print(f'T9 SEEDS (training length {LEN} epochs). (i) ego + cmd: ' + ', '.join(f's{s}={t}' for s, t in ci.items())
          + '; (ii) ego + oracle meta-action, A2 design + turn weighting: ' + ', '.join(f's{s}={t}' for s, t in mi.items()))
    C = ctx(); rows = base_rows()
    rows['egoMLP 3-seed avg'] = avg([C['cache'][f'egoMLP s{s}'] for s in SEEDS])
    for s in SEEDS:
        rows[f'(i) cmd s{s}'] = run(ci[s]); rows[f'(ii) meta s{s}'] = run(mi[s])
    rows['(i) cmd 3-seed avg'] = avg([rows[f'(i) cmd s{s}'] for s in SEEDS])
    rows['(ii) meta 3-seed avg'] = avg([rows[f'(ii) meta s{s}'] for s in SEEDS])
    for f in (0.1, 0.2, 0.4):
        rows[f'(ii) meta flip{int(f*100)} avg'] = avg([run(mi[s], f=f) for s in SEEDS])

    def seedstats(m, sub, cut):
        print('\n  3-seed mean +- sd of the per-seed means (ADE@6s mean | median | p95 || L2@3s NoAvg mean | '
              'L2 TemAvg@3s | slot-0):')
        for lab, tg in (('(i) ego + cmd', ci), ('(ii) ego + meta f0', mi)):
            ps = [run(tg[s]) for s in SEEDS]
            v = np.array([[np.nanmean(p['ade'][m]), np.nanmedian(p['ade'][m]), np.nanpercentile(p['ade'][m], 95),
                           p['l2'][m, 5].mean(), p['l2'][m].mean(), p['slot0']] for p in ps])
            print(f'    {lab:22} ' + '  '.join(f'{a:.3f}+-{b:.3f}' for a, b in zip(v.mean(0), v.std(0))))
        for f in (0.1, 0.2, 0.4):
            v = np.array([[np.nanmean(run(mi[s], f=f)['ade'][m]), run(mi[s], f=f)['l2'][m, 5].mean()] for s in SEEDS])
            print(f'    (ii) meta test flip {int(f*100):2d}%  ADE6 {v[:, 0].mean():.3f}+-{v[:, 0].std():.3f}  '
                  f'L2@3s {v[:, 1].mean():.3f}+-{v[:, 1].std():.3f}')
        tab = C['tab']
        for f in (0.0, 0.1, 0.2, 0.4):
            dv = load(mi[42], 'val', f)
            lab = [dv['data'][r['sample_token']]['shown'] for r in C['rva']]
            ok = E.per_sample(C['rva'], E.oracle_kin(C['rva'], tab, [x['lon'] for x in lab], [x['lat'] for x in lab]),
                              C['occ'])
            print(f'    ORACLE-KIN with the same flipped labels ({int(f*100)}%): ADE6 {np.nanmean(ok["ade"][m]):.3f} '
                  f'L2@3s {ok["l2"][m, 5].mean():.3f}')
    table(rows, [('(i) cmd 3-seed avg', 'KIN'), ('(i) cmd 3-seed avg', 'egoMLP 3-seed avg'),
                 ('(ii) meta 3-seed avg', 'ORACLE-KIN'), ('(ii) meta 3-seed avg', '(i) cmd 3-seed avg'),
                 ('(ii) meta 3-seed avg', 'egoMLP 3-seed avg')], seedstats)
    strata_lines(rows, ['KIN', 'egoMLP 3-seed avg', '(i) cmd 3-seed avg', '(ii) meta 3-seed avg'])
    print('\nSLOT-0 + INPUT-SHUFFLE (new runs)')
    for t in [ci[123], ci[2024]] + list(mi.values()) + (['G5'] if ci[42] == 'G5' else []):
        print(shuffle_line(t, 'shuf'))
    a2 = run('A2')
    print(f'\n  reference: A2 (no turn weighting, s42) ADE6 {E.summary(strip(a2))["ADE6"]:.3f} '
          f'L2@3s {E.summary(strip(a2))["L2_NoAvg_3s"]:.3f}')


def g7():
    LEN, ci, mi = seed_tags()
    twin = ci[42]
    rows = base_rows(); rows[f'no-cam twin {twin}'] = run(twin); rows['G7 cameras'] = run('G7')
    rows['G7 CAMERA-SHUFFLED'] = run('G7', 'G7_camshuf')
    print(f'G7: vision end-to-end (6 cameras + ego + cmd, A3 recipe, {LEN} epochs, seed 42); twin = {twin} '
          f'(identical recipe without visual tokens). tau G7 {rows["G7 cameras"]["tau"]}')
    table(rows, [('G7 cameras', f'no-cam twin {twin}'), ('G7 cameras', 'KIN'), ('G7 cameras', 'egoMLP s42'),
                 ('G7 CAMERA-SHUFFLED', 'G7 cameras')])
    strata_lines(rows, ['KIN', 'egoMLP s42', f'no-cam twin {twin}', 'G7 cameras', 'G7 CAMERA-SHUFFLED'])
    print('\nSLOT-0 + CAMERA-SHUFFLE (images permuted across val samples, seed 99; ego + cmd kept)')
    print(shuffle_line('G7', 'camshuf'))


if __name__ == '__main__':
    {'G5': g5, 'T9': t9, 'G7': g7}[sys.argv[1]]()
