"""ar1/r12_mech.py -- R12 A1: mechanical check of the 68 text-only v3 changes among the 100 audit ids.
Predicted v3 text = v2 row with only (i) route clause deleted, (ii) same element with the distance
reduced by 3.528 m (floored at 0), (iii) a v2 lead / yield candidate dropped because it fails the
v3 rule; composed with the v3 text functions (no stationary wording). PASS iff predicted causes and
trace equal the actual v3 ones. Output: one line per id; JSON list of non-PASS ids.
  python ar1/r12_mech.py"""
import sys, re, json, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import coc_template as CT
import coc_template_v3 as C3
import build_audit_html_v2 as V2

DATA = CT.DATA; ROOT = '/home/dgx1user/Alpamayo-Kushal'
strip = lambda s: re.sub(r'; route: (go straight|turn left|turn right)', '', s)
same = lambda a, b: a is not None and b is not None and a['ann']['translation'] == b['ann']['translation']


def v3_status(st, o2):
    """the v2 candidate re-measured by the v3 rule: (passes, reason)."""
    G = CT._G; r = G['W'][st]; e = CT.ego_pose(r)
    mn = G['loc'][r['scene_name']]
    if mn not in G['maps']:
        G['maps'][mn] = CT.Map(mn)
    objs = C3.objects_v3(r, G['inst'], G['S'], G['A'], G['by_s'], G['maps'][mn], e)
    m = [o for o in objs if o['ann']['translation'] == o2['ann']['translation']]
    if not m:
        return False, 'not an object in v3 (filtered)'
    o = m[0]
    if o['x'] <= 0: return False, f'x_f {o["x"]:.1f} <= 0'
    if not o['in_path']: return False, f'outside corridor (d {o["d_path"]:.1f} > {o["bound"]:.1f})'
    return True, 'still passes v3 rule'


def clauses(a, b):
    A, B = a.split('; '), b.split('; ')
    return [f'-{x}' for x in A if x not in B] + [f'+{x}' for x in B if x not in A]


def main():
    CT._init_globals(); CT._G['speed_src'] = '2hz'
    v2r = pickle.load(open(f'{DATA}/r11_rows_v2.pkl', 'rb')); v3r = pickle.load(open(f'{DATA}/r11_rows_v3.pkl', 'rb'))
    v2 = pickle.load(open(f'{DATA}/ar1_coc_2hz.pkl', 'rb')); v3 = pickle.load(open(f'{DATA}/ar1_coc_2hz_v3.pkl', 'rb'))
    ids = V2.parse_ids(V2.TXT)
    text_only = [(a, t) for a, t in ids if (v2[t]['lon'], v2[t]['lat']) == (v3[t]['lon'], v3[t]['lat'])
                 and (strip(v2[t]['causes']) != v3[t]['causes'] or v2[t]['trace'] != v3[t]['trace'])]
    print(f'text-only changed audit ids: {len(text_only)}')
    C3.FIX.update(route=True, bumper=True, turn=True, nudge=True)
    fails = []
    for a, t in text_only:
        x2, x3 = v2r[t], v3r[t]; p = dict(x2); why = []
        for role in ('lead', 'yc'):
            o2, o3 = x2[role], x3[role]
            if o2 is None and o3 is None:
                continue
            if same(o2, o3):
                p[role] = dict(o2, x=o2['x'] - C3.BUMP); why.append(f'{role}: (ii)')
            elif o2 is not None and o3 is None:
                ok, rs = v3_status(t, o2)
                if not o2['in_path']:
                    p[role] = None; why.append(f'{role}: (iii) crosswalk branch')
                elif not ok:
                    p[role] = None; why.append(f'{role}: (iii) {rs}')
                else:
                    why.append(f'{role}: UNEXPLAINED drop ({rs})')
            elif o2 is None:
                why.append(f'{role}: NEW in v3 ({o3["kind"]} x_f {o3["x"]:.0f})')
            else:
                why.append(f'{role}: DIFFERENT agent in v3 ({o2["kind"]} -> {o3["kind"]})')
        cause = None
        if x2['cause'] is not None:
            cause = p['lead'] if same(x2['cause'], x2['lead']) else p['yc'] if same(x2['cause'], x2['yc']) else x2['cause']
        p['nd'] = None
        pc = C3.components_v3(p)
        pt = C3.compose_v3(x2['lon'], x2['lat'], cause, x2['ego'], x2['F'], x2['sl'], x2['xw'], x2['cons'], False, None)
        if pc == v3[t]['causes'] and pt == v3[t]['trace']:
            print(f'{a} PASS  [{"; ".join(why) or "route / map distances only"}]')
            continue
        diff = []
        if pc != v3[t]['causes']:
            diff.append('causes ' + ' '.join(clauses(pc, v3[t]['causes'])))
            if x2['cons'] != x3['cons']:
                diff.append(f'construction flag {x2["cons"]} -> {x3["cons"]} (cone range now from the bumper)')
        if pt != v3[t]['trace']:
            kind = 'stationary wording' if x3.get('still') and v3[t]['trace'].startswith('Stay stopped') else 'trace text'
            diff.append(f'{kind}: predicted "{pt}" | v3 "{v3[t]["trace"]}"')
        print(f'{a} NOT PASS  [{"; ".join(why)}] ' + ' || '.join(diff))
        fails.append(a)
    json.dump(fails, open(f'{ROOT}/Alpamayo/r12_a1_fail.json', 'w'))
    print(f'\nPASS {len(text_only) - len(fails)} / {len(text_only)}; not PASS: {fails}')


if __name__ == '__main__':
    main()
