"""ar1/build_audit_html_v3short.py -- R12 A2: short re-audit page audit/coc_audit_v3_short.html, v3 layout
and CSV (sample_id, decision, causes, peeking, note). Sections (rules in the R12_REPORT.md header):
(a) the 13 decision-changed audit ids, (b) A1 ids not PASS, both with v2 text beside v3;
(c) 4 turns only from the > 45 deg rule + 3 in-lane nudges, (d) 5 fresh, one per lon class
(the 5 largest with an eligible scene); (c)+(d) val, one per unseen scene, one RandomState(13).
  python ar1/build_audit_html_v3short.py"""
import json, os, pickle, html, datetime, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import coc_template as CT
import build_audit_html_v2 as V2
from build_audit_html_v3 import tagged, cause_type, strip

ROOT = '/home/dgx1user/Alpamayo-Kushal'; NUS = V2.NUS; DATA = V2.DATA
OUT = f'{ROOT}/audit/coc_audit_v3_short.html'
DEC13 = ('A025', 'A034', 'A065', 'A069', 'A082', 'A085', 'A086', 'A087', 'A092', 'A094', 'A096', 'A099', 'A100')
TITLES = {'a': '(a) your v2 audit ids whose DECISION changed in v3', 'b': '(b) your v2 audit ids whose text change was '
          'not purely mechanical (R12 A1)', 'c': '(c) new rules: turns only from the > 45 deg rule, and in-lane nudges',
          'd': '(d) fresh traces, one per decision class'}


def select(v3, v3r, W, seen):
    rng = np.random.RandomState(13); used = set(seen); out = []
    val = sorted(t for t in v3 if v3[t]['split'] == 'val')

    def draw(pool, n, lab):
        for _ in range(n):
            sc = sorted({W[t]['scene_name'] for t in pool if W[t]['scene_name'] not in used})
            assert sc, f'no unseen scene left for {lab}'
            s = sc[rng.randint(len(sc))]; used.add(s)
            ts = [t for t in pool if W[t]['scene_name'] == s]; out.append((ts[rng.randint(len(ts))], lab))
    turn45 = [t for t in val if v3[t]['lat'].startswith('Turn') and abs(v3r[t]['F']['dpsi']) > 45
              and abs(v3r[t]['F']['dpsi_int']) <= 30]
    nud = [t for t in val if v3[t]['lat'].startswith('In-lane nudge')]
    sizes = {}
    draw(turn45, 4, 'c turn > 45 deg')
    # deviation (R12): val has too few unseen nudge scenes; take all of them, then fill from train
    nv = len({W[t]['scene_name'] for t in nud if W[t]['scene_name'] not in used})
    draw(nud, min(3, nv), 'c in-lane nudge (val)')
    if nv < 3:
        ntr = sorted(t for t in v3 if v3[t]['split'] == 'train' and v3[t]['lat'].startswith('In-lane nudge'))
        draw(ntr, 3 - nv, 'c in-lane nudge (TRAIN, val exhausted)')
    import collections
    cls = collections.Counter(v3[t]['lon'] for t in val)
    order = [c for c, _ in cls.most_common()]
    k = 0
    for c in order:
        pool = [t for t in val if v3[t]['lon'] == c and W[t]['scene_name'] not in used]
        sizes[c] = len(pool)
        if pool and k < 5:
            draw(pool, 1, f'd {c}'); k += 1
    return out, {'turn45 pool': len(turn45), 'nudge pool': len(nud), 'class pools': sizes}


def main():
    CT._init_globals(); W = CT._G['W']
    v2r = pickle.load(open(f'{DATA}/r11_rows_v2.pkl', 'rb')); v3r = pickle.load(open(f'{DATA}/r11_rows_v3.pkl', 'rb'))
    v2 = pickle.load(open(f'{DATA}/ar1_coc_2hz.pkl', 'rb')); v3 = pickle.load(open(f'{DATA}/ar1_coc_2hz_v3.pkl', 'rb'))
    ids = dict((a, t) for a, t in V2.parse_ids(V2.TXT))
    assert all((v2[ids[a]]['lon'], v2[ids[a]]['lat']) != (v3[ids[a]]['lon'], v3[ids[a]]['lat']) for a in DEC13)
    fails = json.load(open(f'{ROOT}/Alpamayo/r12_a1_fail.json'))
    seen = {W[t]['scene_name'] for t in ids.values()}
    seen |= {m['scene'] for m in json.load(open(f'{ROOT}/audit/coc_audit_v3_meta.json'))['traces']}
    picks, pools = select(v3, v3r, W, seen)
    print('pools: ' + json.dumps(pools))
    entries = [(a, ids[a], 'a', a) for a in DEC13] + [(a, ids[a], 'b', a) for a in fails] + \
              [(f'S{k + 1:02d}', t, lab[0], lab) for k, (t, lab) in enumerate(picks)]
    n = len(entries)
    J = lambda nm: json.load(open(f'{NUS}/v1.0-trainval/{nm}.json'))
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
    head = V2.HEAD.replace('<title>CoC Audit Sheet</title>', f'<title>CoC re-audit short ({n} cards)</title>')
    out = [head.replace('CoC audit sheet v2 (100 val traces)', f'CoC re-audit, short ({n} cards)')]
    out.append(f'<div class="how"><b>{n} cards.</b> (a) {len(DEC13)} decision changes and (b) {len(fails)} non-mechanical '
               'text changes from your v2 audit, v2 text in grey beside v3; (c) 7 traces for the new turn / nudge rules; '
               '(d) 5 fresh traces. v3: no route command in the causes; agent distances from the front bumper; only agents '
               'inside the ego path corridor; turns also when the heading changes > 45 deg in 6 s; in-lane nudge.</div>')
    meta = []; sec = None
    for aid, st, part, lab in entries:
        if part != sec:
            out.append(f'<h2>{TITLES[part]}</h2>'); sec = part
        x = v3r[st]; rec = W[st]; c3 = v3[st]
        t0 = S[st]['timestamp']; frames = []; drawn = []
        for k, flab in V2.STRIP:
            tok = st
            for _ in range(abs(k)):
                tok = S[tok]['prev' if k < 0 else 'next'] if tok else ''
            if not tok:
                frames.append(f'<figure class="f{" now" if k == 0 else ""}"><div class="na">no frame</div>'
                              f'<figcaption>{flab}</figcaption></figure>'); continue
            sd = cam[tok]; im = Image.open(f'{NUS}/{sd["filename"]}').convert('RGB')
            if k == 0:
                drawn = V2.film(None, x, V2.Cam(sd, EP, CS), im)
            w = 480 if k == 0 else 300
            im = im.resize((w, round(900 * w / 1600)), Image.BILINEAR)
            dt = (sd['timestamp'] - t0) / 1e6
            frames.append(f'<figure class="f{" now" if k == 0 else ""}{" fut" if k > 0 else ""}"><img src="{V2.jpg(im)}" alt="{flab}">'
                          f'<figcaption>{flab} <span class="dt">({dt:+.2f} s)</span></figcaption></figure>')
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
        xd = dict(x)
        for role in ('lead', 'yc'):
            if x[role] is not None:
                xd[role] = dict(x[role], x=x[role].get('x_rear', x[role]['x']))
        td = V2.topdown_svg(xd, rec, past, V2.getmap(x['scene']))
        when = datetime.datetime.utcfromtimestamp(t0 / 1e6).strftime('%Y-%m-%d %H:%M:%S UTC')
        cz = ''.join(f'<li><span class="tag t{tg}">[{tg}]</span> {html.escape(cc)}</li>' for tg, cc in tagged(c3['causes']))
        evh = ''.join(f'<li>{html.escape(s_)}</li>' for s_ in V2.evidence(x, T[st]))
        old = ''
        if part in ('a', 'b'):
            c2 = v2[st]
            old = (f'<div class="old" style="color:var(--mut);border-left:3px solid var(--line);padding-left:8px;margin:6px 0">'
                   f'<b>v2 (before)</b>: {html.escape(c2["lon"])}; {html.escape(c2["lat"])}<br>causes: {html.escape(c2["causes"])}'
                   f'<br>trace: {html.escape(c2["trace"])}</div>')
        tog = ''.join(f'<div class="tog"><span>{tl}</span>' + ''.join(
            f'<label><input type="radio" name="{aid}_{k}" value="{val}" data-id="{aid}" data-k="{k}"> {val}</label>'
            for val in ('OK', 'wrong', 'unsure')) + '</div>'
            for k, tl in (('decision', 'Decision'), ('causes', 'Causes'), ('peeking', 'No peeking')))
        out.append(f'''<section class="card" id="{aid}" data-token="{st}">
<div class="hdr"><b>{aid}</b> &middot; {html.escape(lab)} &middot; {html.escape(x["scene"])} &middot; {when} &middot; <code>{st}</code></div>
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
        meta.append({'id': aid, 'token': st, 'part': part, 'label': lab, 'scene': x['scene'], 'lon': c3['lon'],
                     'lat': c3['lat'], 'cause_type': cause_type(c3['lon'], c3['lat'], x)})
    out.append(V2.TAIL.replace("coc_audit_v2", "coc_audit_v3short"))
    open(OUT, 'w').write('\n'.join(out))
    json.dump({'source': 'data/ar1_coc_2hz_v3.pkl', 'rule': 'R12 A2', 'pools': pools, 'traces': meta},
              open(f'{ROOT}/audit/coc_audit_v3_short_meta.json', 'w'), indent=1)
    print(f'wrote {OUT}: {os.path.getsize(OUT) / 1e6:.1f} MB, {n} cards')
    for m in meta:
        print(f'  {m["id"]} {m["part"]} {m["label"][:28]:28} {m["scene"]} {m["lon"]}; {m["lat"]} ({m["cause_type"]})')


if __name__ == '__main__':
    main()
