"""ar1/build_coc_audit_r11.py -- R11 A1 + A2: 40-trace CoC audit page -> audit/coc_audit.html
(one self-contained file) + audit/coc_audit_meta.json. Rules in the R11_REPORT.md header.
Traces = data/ar1_coc_2hz.pkl (val). Rows recomputed with coc_template (speed_src 2hz) only
for the cause type and the BEV items; the shown text is the pickle's.
  python ar1/build_coc_audit_r11.py"""
import base64, io, json, math, os, pickle, html, sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import coc_template as CT
from build_audit_html_v2 import parse_ids

ROOT = '/home/dgx1user/Alpamayo-Kushal'
NUS = f'{ROOT}/Alpamayo/nuscenes'
DATA = f'{ROOT}/Alpamayo/data'
OUTD = f'{ROOT}/audit'
CHANGED = ('A031', 'A041', 'A042', 'A043', 'A044', 'A048', 'A052', 'A053')
CAMS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT')


def sample_ids(coc, T, V0, J):
    """A1: the 8 changed traces, then 4 per pool with one RandomState(7)."""
    val = sorted(t for t, x in coc.items() if x['split'] == 'val')
    ids = dict(parse_ids(f'{ROOT}/AR1_COC_AUDIT.txt'))
    chosen = [(ids[a], 'CHANGED ' + a) for a in CHANGED]
    wss = pickle.load(open(f'{DATA}/r5_wss.pkl', 'rb')); hf = pickle.load(open(f'{DATA}/r9_hard_val.pkl', 'rb'))
    W = CT._G['W']
    scene_desc = {s['name']: s['description'] for s in J('scene')}
    g = {t: np.linalg.norm(np.asarray(T[t]['P'])[:6], axis=1).max() for t in val}
    v0 = {t: V0[t]['v0_can'] for t in val}
    pools = (('WSS', lambda t: wss[W[t]['scene_name']]),
             ('START', lambda t: v0[t] <= 0.2 and g[t] > 1.0),
             ('TURN', lambda t: abs(hf[t]['dpsi']) > 30),
             ('AGENT', lambda t: hf[t]['agent20']),
             ('INT', lambda t: hf[t]['inter0'] or hf[t]['sl20']),
             ('RAIN/NIGHT', lambda t: any(k in scene_desc[W[t]['scene_name']].lower() for k in ('rain', 'night'))),
             ('MOVING STRAIGHT', lambda t: v0[t] > 0.2 and abs(hf[t]['dpsi']) <= 10),
             ('RANDOM', lambda t: True))
    rng = np.random.RandomState(7); sizes = {}
    for name, f in pools:
        have = {t for t, _ in chosen}
        cand = [t for t in val if f(t) and t not in have]
        sizes[name] = len(cand)
        for i in rng.choice(len(cand), 4, replace=False):
            chosen.append((cand[i], name))
    assert len(chosen) == 40 and len({t for t, _ in chosen}) == 40
    return chosen, sizes


def jpg(im, w, q=72):
    im = im.resize((w, round(im.height * w / im.width)), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=q, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def cause_type(x):
    lon = x['lon']
    if lon.startswith('Yield'): return 'yield agent'
    if lon.startswith('Lead'): return 'lead agent'
    if x['lat'].startswith('Lane change') and x['cause'] is not None: return 'passing lead'
    if lon.startswith('Stop'):
        return 'stop line' if x['sl'] is not None else 'crosswalk' if x['xw'] is not None else 'unexplained stop'
    if lon.startswith('Speed'): return 'turn'
    return 'clear lane'


def bev(rec, past, M, st, W_=340, H_=400):
    """ego frame at t0: x forward -> up, y left -> left; view x in [-15, 55], y in [-28, 28]."""
    e = CT.ego_pose(rec); s = min(W_ / 56.0, H_ / 70.0)
    P = lambda gx, gy: CT.to_ego((gx, gy), e)
    sxy = lambda ex, ey: (W_ / 2 - ey * s, H_ - 15 * s - ex * s)
    pts = lambda seq: ' '.join(f'{u:.1f},{v:.1f}' for u, v in seq)
    parts = []
    for L, cls in (('road_segment', 'int'), ('lane', 'ln'), ('lane_connector', 'ln'), ('stop_line', 'sl'),
                   ('ped_crossing', 'xw')):
        for tok in M.in_radius(e[0], e[1], 70, (L,)):
            poly = M.poly[L][M.tok[L].index(tok)]
            q = [sxy(*P(a, b)) for a, b in poly.exterior.coords]
            if all(not (0 <= u <= W_ and 0 <= v <= H_) for u, v in q):
                continue
            if L == 'stop_line':
                ty = M.rec['stop_line'][tok]['stop_line_type']
                parts.append(f'<polygon points="{pts(q)}" class="{"slp" if ty == "PED_CROSSING" else "sl"}">'
                             f'<title>stop line {ty}</title></polygon>')
                continue
            parts.append(f'<polygon points="{pts(q)}" class="{cls}"/>')
    G = CT._G
    for tok in G['by_s'][st]:
        a = G['A'][tok]; x, y = CT.to_ego(a['translation'], e)
        if math.hypot(x, y) > 50:
            continue
        cat = G['inst'][a['instance_token']]; short = cat.split('.')[-1] if not cat.startswith('human') else 'pedestrian'
        yaw = CT.yaw_of(a['rotation']) - e[2]; l, w = a['size'][1], a['size'][0]
        c, sn = math.cos(yaw), math.sin(yaw)
        corners = [(x + c * dx - sn * dy, y + sn * dx + c * dy) for dx, dy in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2))]
        cls = 'ped' if cat.startswith('human') else 'cyc' if 'cycle' in cat else 'obj' if cat.startswith(('movable', 'static')) else 'veh'
        parts.append(f'<polygon points="{pts([sxy(*p) for p in corners])}" class="{cls}"><title>{html.escape(cat)}</title></polygon>')
        u, v = sxy(x, y)
        if 0 <= u <= W_ and 0 <= v <= H_ and cls != 'obj':
            parts.append(f'<text x="{u + 5:.1f}" y="{v - 3:.1f}" class="lb">{html.escape(short)}</text>')
    if len(past) > 1:
        parts.append(f'<polyline points="{pts([sxy(*P(a, b)) for a, b in past])}" class="past"/>')
    fp = np.asarray(rec['future_positions'])[:12, :2]
    parts.append(f'<polyline points="{pts([sxy(*P(a, b)) for a, b in [(e[0], e[1])] + list(fp)])}" class="fut"/>')
    for a, b in fp:
        u, v = sxy(*P(a, b)); parts.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="2.2" class="futd"/>')
    parts.append(f'<polygon points="{pts([sxy(2.6, 0.95), sxy(2.6, -0.95), sxy(-1.5, -0.95), sxy(-1.5, 0.95)])}" class="ego"/>')
    return (f'<svg viewBox="0 0 {W_} {H_}" class="bev" role="img" aria-label="bird\'s-eye view">'
            f'<rect width="{W_}" height="{H_}" class="bg"/>' + ''.join(parts) +
            f'<text x="6" y="14" class="tk">ego frame at t0, up = forward; 10 m = {10 * s:.0f} px</text></svg>')


def main():
    CT._init_globals(); CT._G['speed_src'] = '2hz'
    J = lambda n: json.load(open(f'{NUS}/v1.0-trainval/{n}.json'))
    coc = pickle.load(open(f'{DATA}/ar1_coc_2hz.pkl', 'rb'))
    T = pickle.load(open(f'{DATA}/w1_targets.pkl', 'rb'))['targets']; V0 = pickle.load(open(f'{DATA}/w1_v0.pkl', 'rb'))
    chosen, sizes = sample_ids(coc, T, V0, J)
    print('pool sizes (after removing already chosen): ' + ', '.join(f'{k} {v}' for k, v in sizes.items()))
    rows = CT.build_rows({t for t, _ in chosen}, procs=20, min_fut=6)
    for t, _ in chosen:          # the recomputed rows must reproduce the pickle exactly
        assert (rows[t]['lon'], rows[t]['lat']) == (coc[t]['lon'], coc[t]['lat']), t
    S = {s['token']: s for s in J('sample')}; EP = {e['token']: e for e in J('ego_pose')}
    sd = {}
    for d in J('sample_data'):
        if d['is_key_frame']:
            ch = d['filename'].split('/')[1]
            if ch in CAMS or ch == 'LIDAR_TOP':
                sd[(d['sample_token'], ch)] = d
    ids2 = {t: a for a, t in parse_ids(f'{ROOT}/AR1_COC_AUDIT.txt')}
    maps = {}; cards = []; meta = []
    for k, (st, pool) in enumerate(chosen):
        rid = f'R{k + 1:02d}'; x = rows[st]; rec = CT._G['W'][st]; c = coc[st]
        mn = CT._G['loc'][x['scene']]
        if mn not in maps:
            maps[mn] = CT.Map(mn)
        img = lambda tk, ch, w: jpg(Image.open(NUS + '/' + sd[(tk, ch)]['filename']).convert('RGB'), w)
        imgs = ''.join(f'<figure><img src="{img(st, ch, 520 if ch == "CAM_FRONT" else 400)}" '
                       f'alt="{ch} at t0"><figcaption>{ch.replace("CAM_", "")} t0</figcaption></figure>' for ch in CAMS)
        fut = st
        for _ in range(6):
            fut = S[fut]['next'] if fut else ''
        if fut:
            dt = (S[fut]['timestamp'] - S[st]['timestamp']) / 1e6
            imgs += (f'<figure class="fut"><img src="{img(fut, "CAM_FRONT", 400)}" '
                     f'alt="CAM_FRONT future"><figcaption>FRONT t0+{dt:.1f} s: future, for checking only</figcaption></figure>')
        else:
            imgs += '<figure class="fut"><div class="na">no frame at t0+3 s (end of scene)</div></figure>'
        past, tok = [], st
        for _ in range(4):
            tok = S[tok]['prev']
            if not tok:
                break
            past.append(EP[sd[(tok, 'LIDAR_TOP')]['ego_pose_token']]['translation'][:2])
        past = past[::-1] + [CT.ego_pose(rec)[:2]]
        ct = cause_type(x)
        meta.append({'id': rid, 'token': st, 'pool': pool, 'scene': x['scene'], 'lon': c['lon'], 'lat': c['lat'],
                     'cause_type': ct})
        a_id = f' &middot; was {ids2[st]}' if st in ids2 else ''
        radios = lambda key, vals: ''.join(f'<label><input type="radio" name="{rid}_{key}" value="{v}" data-id="{rid}" '
                                           f'data-k="{key}"> {v}</label>' for v in vals)
        cards.append(f'''<section class="card" id="{rid}">
<div class="hdr"><b>{rid}</b>{a_id} &middot; pool {html.escape(pool)} &middot; {html.escape(x["scene"])} &middot; <code>{st}</code></div>
<div class="imgs">{imgs}</div>
<div class="row"><div>{bev(rec, past, maps[mn], st)}<div class="leg"><span class="k veh"></span>vehicle <span class="k ped"></span>pedestrian <span class="k cyc"></span>cyclist <span class="k obj"></span>object <span class="k int"></span>intersection <span class="k xw"></span>crosswalk <span class="k sl"></span>stop line <span class="k slp"></span>ped-crossing stop line (hover: type); grey = past 2 s, blue = GT future 6 s</div></div>
<div class="txt"><p><span class="lab">Decision</span> {html.escape(c["lon"])}; {html.escape(c["lat"])}</p>
<p><span class="lab">Causes</span> {html.escape(c["causes"])}</p>
<p><span class="lab">Trace</span> {html.escape(c["trace"])}</p>
<div class="q"><span>Decision</span>{radios("decision", ("ok", "wrong"))}</div>
<div class="q"><span>Causes</span>{radios("causes", ("ok", "wrong", "missing"))}</div>
<textarea data-id="{rid}" placeholder="Notes (optional)"></textarea></div></div>
</section>''')
    os.makedirs(OUTD, exist_ok=True)
    page = HEAD + ''.join(cards) + TAIL.replace('__N__', str(len(cards)))
    open(f'{OUTD}/coc_audit.html', 'w').write(page)
    json.dump({'source': 'data/ar1_coc_2hz.pkl', 'rule': 'R11 A1 RandomState(7)', 'pool_sizes': sizes, 'traces': meta},
              open(f'{OUTD}/coc_audit_meta.json', 'w'), indent=1)
    print(f'wrote {OUTD}/coc_audit.html ({os.path.getsize(f"{OUTD}/coc_audit.html") / 1e6:.1f} MB), {len(cards)} traces')
    for m in meta:
        print(f'  {m["id"]} {m["pool"]:16} {m["lon"]:32} {m["lat"]:26} {m["cause_type"]}')


HEAD = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoC Audit</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1f;--card:#fff;--line:#d8d8d4;--mut:#666;--acc:#2a5db0;--bev:#f1f1ee}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161618;--fg:#eee;--card:#222225;--line:#3a3a3e;--mut:#aaa;--acc:#7da7ff;--bev:#1b1b1e}}
:root[data-theme="dark"]{--bg:#161618;--fg:#eee;--card:#222225;--line:#3a3a3e;--mut:#aaa;--acc:#7da7ff;--bev:#1b1b1e}
body{background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,Segoe UI,sans-serif;margin:0;padding:0 16px 80px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:2;display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center}
h1{font-size:18px;margin:0} #prog{font-weight:600} .mut{color:var(--mut);font-size:13px}
button{font:inherit;padding:6px 14px;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:6px;cursor:pointer}
.how{max-width:1000px;font-size:14px;color:var(--mut);margin:10px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:14px 0}
.card.done{border-color:#3a9a5b}
.hdr{font-size:13px;color:var(--mut);overflow-wrap:anywhere} .hdr b{color:var(--fg);font-size:15px}
.imgs{display:flex;gap:6px;overflow-x:auto;padding:8px 0;align-items:flex-start}
figure{margin:0;flex:0 0 auto} .imgs img{display:block;height:190px;width:auto;border-radius:3px}
.fut img{outline:3px dashed #c77;outline-offset:-3px;opacity:.9} figcaption{font-size:12px;color:var(--mut);text-align:center}
.fut figcaption{color:#c55;font-weight:600} .na{font-size:12px;color:var(--mut);padding:40px 12px;border:1px dashed var(--line)}
.row{display:flex;flex-wrap:wrap;gap:16px} .txt{flex:1 1 380px;min-width:0}
.bev{width:340px;max-width:100%;height:auto} .leg{font-size:11px;color:var(--mut);max-width:340px}
.k{display:inline-block;width:10px;height:10px;margin:0 2px 0 6px;vertical-align:-1px;border:1px solid #0003}
.bev .bg{fill:var(--bev)} .bev .ln{fill:none;stroke:var(--mut);stroke-opacity:.3;stroke-width:1}
.int,.k.int{fill:#9b59b6;fill-opacity:.12;stroke:#9b59b6;stroke-opacity:.5} .k.int{background:#9b59b622}
.xw,.k.xw{fill:#3c8cff;fill-opacity:.35;stroke:#3c8cff} .k.xw{background:#3c8cff55}
.sl,.k.sl{fill:none;stroke:#e00;stroke-width:2;stroke-dasharray:4 2} .k.sl{background:none;border:2px dashed #e00}
.slp,.k.slp{fill:none;stroke:#3c8cff;stroke-width:2;stroke-dasharray:4 2} .k.slp{background:none;border:2px dashed #3c8cff}
.veh,.k.veh{fill:#ffb000;fill-opacity:.8;stroke:#000;stroke-width:.6} .k.veh{background:#ffb000}
.ped,.k.ped{fill:#e8336d;stroke:#000;stroke-width:.6} .k.ped{background:#e8336d}
.cyc,.k.cyc{fill:#20b2aa;stroke:#000;stroke-width:.6} .k.cyc{background:#20b2aa}
.obj,.k.obj{fill:#999;fill-opacity:.7;stroke:none} .k.obj{background:#999}
.bev .past{fill:none;stroke:#888;stroke-width:3} .bev .fut{fill:none;stroke:var(--acc);stroke-width:2.5}
.bev .futd{fill:var(--acc)} .bev .ego{fill:#2ecc71;stroke:#000;stroke-width:1}
.bev .lb{font-size:9px;fill:var(--fg)} .bev .tk{font-size:10px;fill:var(--mut)}
.lab{font-weight:700;margin-right:6px} .q{margin:6px 0;display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center}
.q span{min-width:72px;font-weight:600} label{cursor:pointer}
textarea{width:100%;box-sizing:border-box;min-height:48px;font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
</style></head><body>
<header><h1>CoC audit (2 Hz decisions)</h1><span id="prog">0 / __N__ done</span>
<button id="dl" type="button">Download CSV</button><span class="mut" id="save"></span></header>
<p class="how">For each trace, judge the <b>Decision</b> (ok / wrong) and the <b>Causes</b> (ok / wrong / missing =
a needed cause is absent) using the three t0 images and the BEV. The red-dashed FRONT image is t0+3 s:
use it only to check the decision. A trace counts as done when both questions are answered.
Answers are kept in this browser (if allowed); use Download CSV to save coc_audit_results.csv.</p>
'''

TAIL = '''<script>
(function(){
var KEY='coc_audit_r11', st={};
try{st=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){st={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st));document.getElementById('save').textContent='saved in this browser'}catch(e){document.getElementById('save').textContent='not saved (browser blocks storage): download the CSV'}}
function rec(id){return st[id]||(st[id]={decision:'',causes:'',notes:''})}
function prog(){var n=0;document.querySelectorAll('.card').forEach(function(c){var r=st[c.id];var d=r&&r.decision&&r.causes;c.classList.toggle('done',!!d);if(d)n++});
document.getElementById('prog').textContent=n+' / __N__ done'}
document.querySelectorAll('input[type=radio]').forEach(function(i){var r=st[i.dataset.id];if(r&&r[i.dataset.k]===i.value)i.checked=true;
i.addEventListener('change',function(){rec(i.dataset.id)[i.dataset.k]=i.value;save();prog()})});
document.querySelectorAll('textarea').forEach(function(t){var r=st[t.dataset.id];if(r)t.value=r.notes||'';
t.addEventListener('input',function(){rec(t.dataset.id).notes=t.value;save()})});
function q(s){s=String(s==null?'':s);return /[",\\n\\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
document.getElementById('dl').addEventListener('click',function(){var L=['id,decision,causes,notes'];
document.querySelectorAll('.card').forEach(function(c){var r=st[c.id]||{};L.push([c.id,r.decision||'',r.causes||'',r.notes||''].map(q).join(','))});
var b=new Blob([L.join('\\n')+'\\n'],{type:'text/csv'});var a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='coc_audit_results.csv';
document.body.appendChild(a);a.click();setTimeout(function(){URL.revokeObjectURL(a.href);a.remove()},500)});
prog();
})();
</script></body></html>
'''

if __name__ == '__main__':
    main()
