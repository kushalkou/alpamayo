"""ar1/audit_summary.py -- summarise a CoC audit CSV against the pre-registered sign-off rule.
PRINTS ONLY; never writes AUDIT_SIGNOFF.md (Kushal writes it himself).
  --v3 (R11 C2): audit/coc_audit_v3.html CSV (sample_id, decision, causes, peeking, note; OK /
       wrong / unsure) + audit/coc_audit_v3_meta.json. PASS if wrong decisions <= 2 AND no
       single cause error type >= 3 among (a)+(b); cause error = Causes "wrong"; type = the
       trace's cause type in the sidecar. "unsure" is counted separately.
  --v3short (R12 A3): audit/coc_audit_v3_short.html CSV (same format as --v3) +
       audit/coc_audit_v3_short_meta.json; same rule over parts (a)-(d).
  default (earlier R11 40-trace page): CSV id, decision, causes, notes (ok / wrong / missing) +
       audit/coc_audit_meta.json; cause error = wrong or missing.
  python ar1/audit_summary.py --v3 path/to/coc_audit_results.csv"""
import sys, csv, json, collections

ROOT = '/home/dgx1user/Alpamayo-Kushal'


def main():
    short = '--v3short' in sys.argv
    v3 = '--v3' in sys.argv or short
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else f'{ROOT}/audit/coc_audit_results.csv'
    meta_p = (f'{ROOT}/audit/coc_audit_v3_short_meta.json' if short else f'{ROOT}/audit/coc_audit_v3_meta.json' if v3
              else f'{ROOT}/audit/coc_audit_meta.json')
    traces = json.load(open(meta_p))['traces']
    key = 'token' if v3 else 'id'
    meta = {m[key]: m for m in traces}
    rows = {(r['sample_id'] if v3 else r['id']): r for r in csv.DictReader(open(path, newline=''))}
    note = (lambda r: r.get('note', '')) if v3 else (lambda r: r.get('notes', ''))
    err_vals = ('wrong',) if v3 else ('wrong', 'missing')
    lab = lambda k: meta[k]['id']
    missing = [lab(k) for k in meta if not (rows.get(k, {}).get('decision') and rows.get(k, {}).get('causes'))]
    unknown = [k for k in rows if k not in meta]
    print(f'file {path}: {len(rows)} rows; traces {len(meta)}; unanswered {len(missing)}'
          + (f' ({", ".join(missing[:20])}{" ..." if len(missing) > 20 else ""})' if missing else '')
          + (f'; rows not on this page {len(unknown)}' if unknown else ''))
    for fld in ('decision', 'causes') + (('peeking',) if v3 else ()):
        c = collections.Counter(rows[k].get(fld) or '(blank)' for k in meta if k in rows)
        print(f'{fld:9}: ' + ', '.join(f'{a} {b}' for a, b in sorted(c.items())))
    wrong_dec = [k for k in meta if rows.get(k, {}).get('decision') == 'wrong']
    errs = [k for k in meta if rows.get(k, {}).get('causes') in err_vals]
    by_type = collections.Counter(meta[k]['cause_type'] for k in errs)
    print(f'\nwrong decisions: {len(wrong_dec)}')
    for k in wrong_dec:
        m = meta[k]
        print(f'  {m["id"]} {m.get("part", m.get("pool", "")):4} {m["lon"]}; {m["lat"]} | note: {note(rows[k])}')
    print(f'cause errors ({" or ".join(err_vals)}): {len(errs)}; by cause type: '
          + (', '.join(f'{a} {b}' for a, b in by_type.most_common()) or 'none'))
    for k in errs:
        m = meta[k]
        print(f'  {m["id"]} {rows[k]["causes"]:7} {m["cause_type"]:16} {m["lon"]}; {m["lat"]} | note: {note(rows[k])}')
    uns = [meta[k]['id'] for k in meta if 'unsure' in (rows.get(k, {}).get('decision'), rows.get(k, {}).get('causes'))]
    if v3:
        print(f'unsure (decision or causes), not counted by the rule: {len(uns)} {uns}')
        for part in (('a', 'b', 'c', 'd') if short else ('a', 'b')):
            ks = [k for k in meta if meta[k]['part'] == part]
            print(f'  part ({part}) n {len(ks)}: wrong decisions {sum(k in wrong_dec for k in ks)}, cause errors '
                  f'{sum(k in errs for k in ks)}')
    worst = by_type.most_common(1)[0] if by_type else ('none', 0)
    ok = len(wrong_dec) <= 2 and worst[1] < 3
    print(f'\nRULE: wrong decisions {len(wrong_dec)} (<= 2) AND max cause error type "{worst[0]}" {worst[1]} (< 3) -> '
          + ('PASS' if ok else 'FAIL') + (' [INCOMPLETE: unanswered traces]' if missing else ''))
    print('This script only prints. Kushal writes AUDIT_SIGNOFF.md himself.')


if __name__ == '__main__':
    main()
