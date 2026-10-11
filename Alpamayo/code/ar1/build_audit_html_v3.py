"""ar1/build_audit_html_v3.py -- R11 C1: re-audit page audit/coc_audit_v3.html in the v2 layout
(ar1/build_audit_html_v2.py helpers and CSS reused; v2 file unchanged).
(a) every v2 audit id whose v3 trace differs (decision, or causes / trace beyond the route clause),
    v2 text shown beside v3;
(b) 30 fresh val traces, one per scene, scenes not in the v2 audit, stratified over the v3 lon
    classes (RandomState(11); classes by descending size; even quota, capped by eligible scenes,
    remainder to the largest classes; scenes without replacement, one random sample per scene).
CSV identical to v2: sample_id, decision, causes, peeking, note. Sidecar audit/coc_audit_v3_meta.json.
  python ar1/build_audit_html_v3.py"""
import json, os, pickle, re, html, datetime, collections, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import coc_template as CT
import build_audit_html_v2 as V2

ROOT = '/home/dgx1user/Alpamayo-Kushal'; NUS = V2.NUS; DATA = V2.DATA
OUT = f'{ROOT}/audit/coc_audit_v3.html'
strip = lambda s: re.sub(r'; route: (go straight|turn left|turn right)', '', s)


def cause_type(lon, lat, x):
    if lon.startswith('Yield'): return 'yield agent'
    if lon.startswith('Lead'): return 'lead agent'
    if lat.startswith('Lane change') and x['cause'] is not None: return 'passing lead'
    if lat.startswith('In-lane nudge'): return 'nudge obstacle'
    if lon.startswith('Stop'):
        return 'stop line' if x['sl'] is not None else 'crosswalk' if x['xw'] is not None else 'unexplained stop'
    if lon.startswith('Speed'): return 'turn'
    return 'clear lane'


def tagged(causes):
    out = []
    for c in causes.split('; '):
        tg = 'ego' if c.startswith('ego ') else 'boxes' if c.startswith(('lead ', 'yield candidate', 'nudge ')) else 'map'
        out.append((tg, c))
    return out


def pick_fresh(v3, W, audit_scenes):
    rng = np.random.RandomState(11)
    val = [t for t in sorted(v3) if v3[t]['split'] == 'val' and W[t]['scene_name'] not in audit_scenes]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in val:
        by[v3[t]['lon']][W[t]['scene_name']].append(t)
    classes = sorted(by, key=lambda c: (-sum(len(v) for v in by[c].values()), c))
    cap = {c: len(by[c]) for c in classes}; q = {c: 0 for c in classes}; left = 30
    while left:                                   # even split, capped, remainder to the largest
        open_ = [c for c in classes if q[c] < cap[c]]
        share = max(1, left // len(open_))
        for c in open_:
            if left == 0:
                break
            add = min(share, cap[c] - q[c], left); q[c] += add; left -= add
    used, out = set(), []
    for c in reversed(classes):                    # draw rarest classes first (shared scenes)
        sc = [s for s in sorted(by[c]) if s not in used]
        for i in rng.choice(len(sc), q[c], replace=False):
            s = sc[i]; used.add(s); out.append((by[c][s][rng.randint(len(by[c][s]))], c))
    assert len(out) == 30 and len(used) == 30
    return out, q


def main():
    CT._init_globals()
    v2r = pickle.load(open(f'{DATA}/r11_rows_v2.pkl', 'rb')); v3r = pickle.load(open(f'{DATA}/r11_rows_v3.pkl', 'rb'))
    v2 = pickle.load(open(f'{DATA}/ar1_coc_2hz.pkl', 'rb')); v3 = pickle.load(open(f'{DATA}/ar1_coc_2hz_v3.pkl', 'rb'))
    W = CT._G['W']
    ids = V2.parse_ids(V2.TXT); audit_scenes = {W[t]['scene_name'] for _, t in ids}
    a_list = [(a, t) for a, t in ids if (v2[t]['lon'], v2[t]['lat']) != (v3[t]['lon'], v3[t]['lat'])
              or strip(v2[t]['causes']) != v3[t]['causes'] or v2[t]['trace'] != v3[t]['trace']]
    fresh, quota = pick_fresh(v3, W, audit_scenes)
    print(f'(a) {len(a_list)} changed audit ids; (b) 30 fresh, quota ' + ', '.join(f'{c} {n}' for c, n in quota.items()))
    J = lambda n: json.load(open(f'{NUS}/v1.0-trainval/{n}.json'))
    S = {s['token']: s for s in J('sample')}; EP = {e['token']: e for e in J('ego_pose')}
    CS = {c['token']: c for c in J('calibrated_sensor')}
    cam, lid = {}, {}
    for d in J('sample_data'):
        if d['is_key_frame'] and d['filename'].startswith('samples/CAM_FRONT/'):
            cam[d['sample_token']] = d
        elif d['is_key_frame'] and d['filename'].startswith('samples/LIDAR_TOP/'):
            lid[d['sample_token']] = d
    T = pickle.load(open(f'{DATA}/w1_targets.pkl', 'rb'))['targets']
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    can = NuScenesCanBus(dataroot=NUS)
    out = [V2.HEAD.replace('CoC audit sheet v2 (100 val traces)', f'CoC audit sheet v3 ({len(a_list) + 30} val traces: '
                                                                  f'{len(a_list)} re-checks + 30 new)')]
    out.append('<div class="how"><b>What changed in v3</b>: route command removed from the causes; agent '
               'distances from the front bumper and only agents inside the ego path corridor; turns also when the '
               'heading changes > 45 deg in 6 s; "In-lane nudge" added. Part (a): your v2 audit ids, v2 text in grey '
               'beside the v3 text. Part (b): 30 new traces from scenes you have not seen.</div>')
    meta = []; sec = None
    for aid, st, part in [(a, t, 'a') for a, t in a_list] + [(f'B{k + 1:02d}', t, 'b') for k, (t, _) in enumerate(fresh)]:
        if part != sec:
            out.append('<h2>(a) re-check: your v2 audit ids whose trace changed</h2>' if part == 'a'
                       else '<h2>(b) 30 new traces (one per scene, not in the v2 audit)</h2>'); sec = part
        x = v3r[st]; rec = W[st]; c3 = v3[st]
        t0 = S[st]['timestamp']; frames = []; drawn = []
        for k, lab in V2.STRIP:
            tok = st
            for _ in range(abs(k)):
                tok = S[tok]['prev' if k < 0 else 'next'] if tok else ''
            if not tok:
                frames.append(f'<figure class="f{" now" if k == 0 else ""}"><div class="na">no frame</div>'
                              f'<figcaption>{lab}</figcaption></figure>'); continue
            sd = cam[tok]; im = Image.open(f'{NUS}/{sd["filename"]}').convert('RGB')
            if k == 0:
                drawn = V2.film(None, x, V2.Cam(sd, EP, CS), im)
            w = 480 if k == 0 else 300
            im = im.resize((w, round(900 * w / 1600)), Image.BILINEAR)
            dt = (sd['timestamp'] - t0) / 1e6
            frames.append(f'<figure class="f{" now" if k == 0 else ""}{" fut" if k > 0 else ""}"><img src="{V2.jpg(im)}" alt="{lab}">'
                          f'<figcaption>{lab} <span class="dt">({dt:+.2f} s)</span></figcaption></figure>')
        try:
            m = can.get_messages(x['scene'], 'pose'); ut = np.array([q['utime'] for q in m]); v = np.array([q['vel'][0] for q in m])
        except Exception:
            ut = v = np.array([])
        spd = V2.speed_svg(t0, ut, v) if len(ut) else '<div class="na">no CAN for this scene</div>'
        past, tok = [], st
        for _ in range(5):
            tok = S[tok]['prev']
            if not tok:
                break
            past.append(EP[lid[tok]['ego_pose_token']]['translation'][:2])
        past = past[::-1] + [CT.ego_pose(rec)[:2]]
        xd = dict(x)                                   # top-down plots in the rear-axle frame
        for role in ('lead', 'yc'):
            if x[role] is not None:
                xd[role] = dict(x[role], x=x[role].get('x_rear', x[role]['x']))
        td = V2.topdown_svg(xd, rec, past, V2.getmap(x['scene']))
        when = datetime.datetime.utcfromtimestamp(t0 / 1e6).strftime('%Y-%m-%d %H:%M:%S UTC')
        cz = ''.join(f'<li><span class="tag t{tg}">[{tg}]</span> {html.escape(cc)}</li>' for tg, cc in tagged(c3['causes']))
        evh = ''.join(f'<li>{html.escape(s_)}</li>' for s_ in V2.evidence(x, T[st]))
        old = ''
        if part == 'a':
            c2 = v2[st]
            old = (f'<div class="old" style="color:var(--mut);border-left:3px solid var(--line);padding-left:8px;margin:6px 0">'
                   f'<b>v2 (before)</b>: {html.escape(c2["lon"])}; {html.escape(c2["lat"])}<br>causes: {html.escape(c2["causes"])}'
                   f'<br>trace: {html.escape(c2["trace"])}</div>')
        tog = ''.join(f'<div class="tog"><span>{lab}</span>' + ''.join(
            f'<label><input type="radio" name="{aid}_{k}" value="{val}" data-id="{aid}" data-k="{k}"> {val}</label>'
            for val in ('OK', 'wrong', 'unsure')) + '</div>'
            for k, lab in (('decision', 'Decision'), ('causes', 'Causes'), ('peeking', 'No peeking')))
        out.append(f'''<section class="card" id="{aid}" data-token="{st}">
<div class="hdr"><b>{aid}</b> &middot; {html.escape(x["scene"])} &middot; {when} &middot; <code>{st}</code></div>
<div class="strip">{"".join(frames)}</div>
<p class="futnote">future frames (right of NOW): use only to check the decision. On NOW: {html.escape("; ".join(drawn)) or "no named agents or map items"} (agent distances from the front bumper)</p>
<div class="viz"><div><div class="vt">Ego speed (CAN), t0-1.5 s to t0+6 s</div>{spd}</div>
<div><div class="vt">Top-down, ego frame at t0 (grey past, blue future GT)</div>{td}</div></div>
<div class="vt">Rule evidence</div><ul class="ev">{evh}</ul>
{old}<p><span class="lab">Decision</span> {html.escape(c3["lon"])}; {html.escape(c3["lat"])}</p>
<div class="lab">Causes</div><ul class="cz">{cz}</ul>
<p><span class="lab">Trace</span> {html.escape(c3["trace"])}</p>
<div class="togs">{tog}</div>
<textarea data-id="{aid}" placeholder="note (optional)"></textarea>
</section>''')
        meta.append({'id': aid, 'token': st, 'part': part, 'scene': x['scene'], 'lon': c3['lon'], 'lat': c3['lat'],
                     'cause_type': cause_type(c3['lon'], c3['lat'], x),
                     'v2_audit': None if part == 'b' else True})
    out.append(V2.TAIL.replace("coc_audit_v2", "coc_audit_v3"))
    os.makedirs(f'{ROOT}/audit', exist_ok=True)
    open(OUT, 'w').write('\n'.join(out))
    json.dump({'source': 'data/ar1_coc_2hz_v3.pkl', 'rule': 'R11 C1', 'quota': quota, 'traces': meta},
              open(f'{ROOT}/audit/coc_audit_v3_meta.json', 'w'), indent=1)
    print(f'wrote {OUT}: {os.path.getsize(OUT) / 1e6:.1f} MB, {len(meta)} cards')
    for m in meta[-30:]:
        print(f'  {m["id"]} {m["scene"]} {m["lon"]}; {m["lat"]} ({m["cause_type"]})')


if __name__ == '__main__':
    main()
