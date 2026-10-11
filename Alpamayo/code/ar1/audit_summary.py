"""ar1/audit_summary.py -- R11 A3: summarise coc_audit_results.csv against the pre-registered
sign-off rule (R11_REPORT.md header). PRINTS ONLY; never writes AUDIT_SIGNOFF.md.
  PASS if wrong decisions <= 2 AND no single cause error type appears >= 3 times
  (cause error = Causes "wrong" or "missing"; type = the trace's cause type in
  audit/coc_audit_meta.json).
  python ar1/audit_summary.py [path/to/coc_audit_results.csv]"""
import sys, csv, json, collections

ROOT = '/home/dgx1user/Alpamayo-Kushal'


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/audit/coc_audit_results.csv'
    meta = {m['id']: m for m in json.load(open(f'{ROOT}/audit/coc_audit_meta.json'))['traces']}
    rows = {r['id']: r for r in csv.DictReader(open(path, newline=''))}
    unknown = sorted(set(rows) - set(meta))
    missing = sorted(i for i in meta if not (rows.get(i, {}).get('decision') and rows.get(i, {}).get('causes')))
    dec = collections.Counter(rows[i]['decision'] or '(blank)' for i in meta if i in rows)
    cau = collections.Counter(rows[i]['causes'] or '(blank)' for i in meta if i in rows)
    print(f'file {path}: {len(rows)} rows; traces {len(meta)}; unanswered {len(missing)}'
          + (f' ({", ".join(missing)})' if missing else '') + (f'; unknown ids {unknown}' if unknown else ''))
    print('Decision: ' + ', '.join(f'{k} {v}' for k, v in sorted(dec.items())))
    print('Causes:   ' + ', '.join(f'{k} {v}' for k, v in sorted(cau.items())))
    wrong_dec = [i for i in meta if rows.get(i, {}).get('decision') == 'wrong']
    errs = [i for i in meta if rows.get(i, {}).get('causes') in ('wrong', 'missing')]
    by_type = collections.Counter(meta[i]['cause_type'] for i in errs)
    print(f'\nwrong decisions: {len(wrong_dec)} ' + str(wrong_dec))
    for i in wrong_dec:
        print(f'  {i} {meta[i]["pool"]:16} {meta[i]["lon"]}; {meta[i]["lat"]} | note: {rows[i]["notes"]}')
    print(f'cause errors (wrong or missing): {len(errs)}; by cause type: ' +
          (', '.join(f'{k} {v}' for k, v in by_type.most_common()) or 'none'))
    for i in errs:
        print(f'  {i} {rows[i]["causes"]:7} {meta[i]["cause_type"]:16} {meta[i]["pool"]:16} | note: {rows[i]["notes"]}')
    by_pool = collections.defaultdict(lambda: [0, 0, 0])
    for i in meta:
        p = by_pool[meta[i]['pool'].split()[0]]; p[0] += 1
        p[1] += rows.get(i, {}).get('decision') == 'wrong'; p[2] += i in errs
    print('per pool (n, wrong decisions, cause errors): ' + '; '.join(f'{k} {v[0]}/{v[1]}/{v[2]}' for k, v in by_pool.items()))
    worst = by_type.most_common(1)[0] if by_type else ('none', 0)
    ok = len(wrong_dec) <= 2 and worst[1] < 3
    print(f'\nRULE: wrong decisions {len(wrong_dec)} (<= 2) AND max cause error type "{worst[0]}" {worst[1]} (< 3) -> '
          + ('PASS' if ok else 'FAIL') + (' [INCOMPLETE: unanswered traces]' if missing else ''))
    print('This script only prints. Kushal writes AUDIT_SIGNOFF.md himself.')


if __name__ == '__main__':
    main()
