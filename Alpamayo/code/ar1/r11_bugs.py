"""ar1/r11_bugs.py -- R11 Part A: verify the three audit bugs on all 29,049 v2 traces
(rows recomputed by coc_template, speed_src 2hz = data/ar1_coc_2hz.pkl). Rules in R11_REPORT.md.
  python ar1/r11_bugs.py"""
import sys, re, csv, pickle, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import coc_template as CT

ROOT = '/home/dgx1user/Alpamayo-Kushal'
DATA = f'{ROOT}/Alpamayo/data'
ROWS = f'{DATA}/r11_rows_v2.pkl'
CSV = '/home/dgx1user/.claude/uploads/2faf7c73-1073-411e-ab5c-8651e2762b37/a81baafb-coc_audit_results.csv'


def audit_ids():
    m = {}
    for line in open(f'{ROOT}/AR1_COC_AUDIT.txt'):
        x = re.match(r'^(A\d{3})  (scene-\d+)  sample (\w+)$', line.strip('\n'))
        if x:
            m[x[3]] = x[1]
    return m


def main():
    CT._init_globals(); CT._G['speed_src'] = '2hz'
    rows = CT.build_rows(None, procs=40, min_fut=6)
    pickle.dump(rows, open(ROWS, 'wb'))
    coc = pickle.load(open(f'{DATA}/ar1_coc_2hz.pkl', 'rb'))
    assert all((rows[t]['lon'], rows[t]['lat']) == (coc[t]['lon'], coc[t]['lat']) for t in coc), 'rows != v2 pickle'
    aid = audit_ids(); au = {r['sample_id']: r for r in csv.DictReader(open(CSV))}
    assert len(au) == 100 and set(au) <= set(aid)
    sp = {t: coc[t]['split'] for t in coc}
    print(f'v2 rows {len(rows)} (= pickle, decisions identical); audit CSV {len(au)} rows, all mapped to A-ids')

    print('\nA1 command [P] vs v2 lateral turn decision: P(cmd = straight | lat = Turn left / right)')
    for s in ('train', 'holdout', 'val'):
        for side in ('Turn left', 'Turn right'):
            m = [t for t in rows if sp[t] == s and rows[t]['lat'] == side]
            c = collections.Counter(rows[t]['command'] for t in m)
            print(f'  {s:7} {side:10} n {len(m):4d}: straight {c[2] / max(len(m), 1):.3f}, right {c[0] / max(len(m), 1):.3f}, '
                  f'left {c[1] / max(len(m), 1):.3f}')
    bad_cmd = [t for t in au if (rows[t]['lat'].startswith('Turn') and rows[t]['command'] == 2) or
               (rows[t]['lat'] == 'Turn left' and rows[t]['command'] == 0) or (rows[t]['lat'] == 'Turn right' and rows[t]['command'] == 1)]
    print(f'  audit ids with a turn decision but a contradicting command: {sorted(aid[t] for t in bad_cmd)}')
    flag_cmd = sorted(aid[t] for t in au if 'route' in au[t]['note'].lower())
    print(f'  audit ids whose note mentions the route command: {flag_cmd}')

    print('\nA2 lead / yield candidate geometry (v2: rear-axle origin)')
    bad = {}
    for t, x in rows.items():
        for role in ('lead', 'yc'):
            o = x[role]
            if o is None:
                continue
            hw = o['ann']['size'][0] / 2
            why = []
            if o['x'] < 3.5: why.append('x<3.5')
            if abs(o['y']) > hw + 1.5: why.append('|y|>hw+1.5')
            if why:
                bad.setdefault(t, []).append(f'{role} {o["kind"]} x {o["x"]:.1f} y {o["y"]:+.1f} ({",".join(why)})')
    n_l = sum(x['lead'] is not None for x in rows.values()); n_y = sum(x['yc'] is not None for x in rows.values())
    nb = collections.Counter(b.split()[0] + ' ' + b.split('(')[-1].rstrip(')') for v in bad.values() for b in v)
    print(f'  traces with a lead {n_l}, with a yield candidate {n_y}; traces with a flagged lead / yield {len(bad)} '
          f'({len(bad) / len(rows):.3f}); by role and reason: ' + ', '.join(f'{k} {v}' for k, v in nb.most_common()))
    yl = [t for t in bad if any(b.startswith('yc') for b in bad[t])]
    print(f'  yield decisions whose cause is a flagged yield candidate: {sum(rows[t]["lon"].startswith("Yield") for t in yl)} '
          f'of {sum(x["lon"].startswith("Yield") for x in rows.values())} yield decisions')
    for t in sorted((t for t in au if t in bad), key=lambda t: aid[t]):
        print(f'  {aid[t]} {rows[t]["lon"]:28} | ' + '; '.join(bad[t]) + f' | audit: {au[t]["decision"]}/{au[t]["causes"]}')

    print('\nA3 |dpsi over the future| > 45 deg but lateral decision not a turn')
    a3 = [t for t, x in rows.items() if abs(x['F']['dpsi']) > 45 and not x['lat'].startswith('Turn')]
    big = [t for t, x in rows.items() if abs(x['F']['dpsi']) > 45]
    print(f'  {len(a3)} of {len(rows)} traces ({len(a3) / len(rows):.3f}); {len(a3)} of {len(big)} with |dpsi| > 45; '
          f'their lat: ' + ', '.join(f'{k} {v}' for k, v in collections.Counter(rows[t]['lat'] for t in a3).most_common()))
    print(f'  audit ids: ' + ', '.join(f'{aid[t]} (dpsi {rows[t]["F"]["dpsi"]:+.0f}, in-int {rows[t]["F"]["dpsi_int"]:+.0f}, {rows[t]["lat"]})'
                                      for t in sorted((t for t in au if t in a3), key=lambda t: aid[t])))
    print('\nAudit notes, all rows with a non-OK answer or a note:')
    for t in sorted(au, key=lambda t: aid[t]):
        r = au[t]
        if r['decision'] != 'OK' or r['causes'] != 'OK' or r['note']:
            print(f'  {aid[t]} {r["decision"]:6} {r["causes"]:6} {rows[t]["lon"][:22]:22} {rows[t]["lat"][:16]:16} | {r["note"][:70]}')


if __name__ == '__main__':
    main()
