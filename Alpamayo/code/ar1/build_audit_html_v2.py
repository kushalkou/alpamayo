"""ar1/build_audit_html_v2.py -- CoC audit sheet v2 -> AR1_COC_AUDIT.html (replaces v1).

Same 100 val samples and order as AR1_COC_AUDIT.txt (stratified by lon decision). Rows are
recomputed with ar1/coc_template.py (v3 rules, identical decisions / causes / traces).
Per card:
 1 FILMSTRIP of CAM_FRONT keyframes in time order: t0-1.0 | t0-0.5 | t0 (NOW) | t0+1 |
   t0+2 | t0+3 | t0+6 s (keyframes are 2 Hz, so these are prev^2, prev^1, 0, next^2,
   next^4, next^6, next^12; the real offset is printed). On t0: 3D boxes of the agents
   named in the causes (lead, yield candidate), projected with the camera calibration,
   labelled; stop line / crosswalk polygons named in the causes drawn on the ground plane
   (z = ego ground) where they project into the image.
 2 SPEED CHART: CAN pose vel[0], t0-1.5 s .. t0+6 s, vertical line at t0 (SVG).
 3 TOP-DOWN (ego frame at t0, x forward = up): past path grey (keyframe LIDAR poses back to
   t0-2 s), future GT path coloured, ego box, named agents at t0, lane / connector
   polygons within 40 m (light), stop lines (red), crosswalks (blue) (SVG).
 4 RULE EVIDENCE: the quantities the decision rules used.
 5 TEXT: decision; causes one per line tagged [ego] / [map] / [boxes]; trace.
 6 TOGGLES: OK / wrong / unsure for decision, causes, no peeking; note. CSV download.
  python ar1/build_audit_html_v2.py
"""
import base64, io, json, math, os, pickle, re, datetime, html, bisect
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyquaternion import Quaternion
import sys
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import coc_template as CT

ROOT = '/home/dgx1user/Alpamayo-Kushal'
NUS = f'{ROOT}/Alpamayo/nuscenes'
DATA = f'{ROOT}/Alpamayo/data'
TXT = f'{ROOT}/AR1_COC_AUDIT.txt'
OUT = f'{ROOT}/AR1_COC_AUDIT.html'
STRIP = ((-2, 't0-1.0 s'), (-1, 't0-0.5 s'), (0, 't0 NOW'), (2, 't0+1 s'), (4, 't0+2 s'), (6, 't0+3 s'),
         (12, 't0+6 s'))


def getmap(scene):
    mn = CT._G['loc'][scene]
    if mn not in CT._G['maps']:
        CT._G['maps'][mn] = CT.Map(mn)
    return CT._G['maps'][mn]


def parse_ids(txt):
    out = []
    for line in open(txt):
        m = re.match(r'^(A\d{3})  (scene-\d+)  sample (\w+)$', line.strip('\n'))
        if m:
            out.append((m[1], m[3]))
    return out


def jpg(im, q=70):
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=q, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def corners(ann):
    """8 box corners (3 x 8) in global frame, devkit convention."""
    w, l, h = ann['size']
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1]); y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    c = Quaternion(ann['rotation']).rotation_matrix @ np.vstack([x, y, z])
    return c + np.array(ann['translation'])[:, None]


class Cam:
    def __init__(self, sd, EP, CS):
        ep, cs = EP[sd['ego_pose_token']], CS[sd['calibrated_sensor_token']]
        self.et, self.eq = np.array(ep['translation']), Quaternion(ep['rotation'])
        self.ct, self.cq = np.array(cs['translation']), Quaternion(cs['rotation'])
        self.K = np.array(cs['camera_intrinsic']); self.ground = ep['translation'][2]

    def project(self, P):
        """P 3 x N global -> (2 x N pixels, depth N)."""
        p = self.eq.inverse.rotation_matrix @ (P - self.et[:, None])
        p = self.cq.inverse.rotation_matrix @ (p - self.ct[:, None])
        d = p[2]; uv = self.K @ p
        return uv[:2] / np.maximum(uv[2:3], 1e-6), d


def film(cards_imgs, x, cam, im):
    """draw named agents / map items on the t0 image (1600 x 900), then downscale."""
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
    except Exception:
        font = ImageFont.load_default()
    drawn = []
    for role, o, col in (('lead', x['lead'], (255, 200, 0)), ('yield', x['yc'], (255, 60, 60))):
        if o is None:
            continue
        uv, d = cam.project(corners(o['ann']))
        lab = f'{o["kind"]} {role} {o["x"]:.0f} m'
        if (d > 0.5).all() and (uv[0] > -400).all() and (uv[0] < 2000).all():
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
                dr.line([tuple(uv[:, a]), tuple(uv[:, b])], fill=col, width=4)
            tx, ty = uv[0].min(), uv[1].min() - 36
            dr.rectangle([tx, ty, tx + 14 * len(lab) + 10, ty + 34], fill=(0, 0, 0))
            dr.text((tx + 5, ty + 2), lab, fill=col, font=font)
            drawn.append(lab)
        else:
            drawn.append(lab + ' (not in CAM_FRONT view)')
    for role, item, col in (('stop line', x['sl'], (230, 0, 0)), ('crosswalk', x['xw'], (60, 140, 255))):
        if item is None:
            continue
        rec = item[2]
        poly = getmap(x['scene']).m.extract_polygon(rec['polygon_token'])
        xy = np.array(poly.exterior.coords)
        P = np.vstack([xy.T, np.full(len(xy), cam.ground)])
        uv, d = cam.project(P)
        ok = d > 1.0
        if ok.sum() >= 2:
            pts = [tuple(uv[:, i]) for i in range(uv.shape[1]) if ok[i]]
            dr.line(pts + [pts[0]], fill=col, width=5)
            drawn.append(f'{role} {item[0]:.0f} m')
        else:
            drawn.append(f'{role} {item[0]:.0f} m (not in view)')
    return drawn


def speed_svg(t0, ut, v, W=420, H=150):
    sel = (ut >= t0 - 1.5e6) & (ut <= t0 + 6.0e6)
    if sel.sum() < 2:
        return '<div class="na">no CAN speed for this scene</div>'
    t = (ut[sel] - t0) / 1e6; s = v[sel]
    vmax = max(2.0, float(np.ceil(s.max() + 0.5)))
    X = lambda tt: 36 + (tt + 1.5) / 7.5 * (W - 46); Y = lambda vv: H - 22 - vv / vmax * (H - 34)
    pts = ' '.join(f'{X(a):.1f},{Y(b):.1f}' for a, b in zip(t, s))
    grid = ''.join(f'<line x1="{X(a):.1f}" y1="{H - 22}" x2="{X(a):.1f}" y2="10" class="g"/>'
                   f'<text x="{X(a):.1f}" y="{H - 6}" class="tk">{a:+d}</text>' for a in (-1, 0, 1, 2, 3, 4, 5, 6))
    yt = ''.join(f'<text x="30" y="{Y(b) + 4:.1f}" class="tk" text-anchor="end">{b:g}</text>'
                 f'<line x1="36" y1="{Y(b):.1f}" x2="{W - 10}" y2="{Y(b):.1f}" class="g"/>'
                 for b in np.linspace(0, vmax, 4).round(1))
    return (f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="ego speed">{grid}{yt}'
            f'<line x1="{X(0):.1f}" y1="8" x2="{X(0):.1f}" y2="{H - 22}" class="now"/>'
            f'<polyline points="{pts}" class="spd"/><text x="{X(0) + 4:.1f}" y="18" class="tk">t0</text>'
            f'<text x="{W - 10}" y="18" class="tk" text-anchor="end">m/s vs s</text></svg>')


def topdown_svg(x, rec, past, Mp, W=300, H=360):
    """ego frame: x forward -> up, y left -> left. 1 m = s px; view x in [-15, 55], y in [-25, 25]."""
    e = CT.ego_pose(rec); s = min(W / 50.0, H / 70.0)
    P = lambda gx, gy: CT.to_ego((gx, gy), e)
    def sxy(ex, ey):
        return W / 2 - ey * s, H - 15 * s - ex * s
    parts = []
    from shapely.geometry import box as sbox
    for L, cls in (('lane', 'ln'), ('lane_connector', 'ln'), ('ped_crossing', 'xw'), ('stop_line', 'sl')):
        for tok in Mp.in_radius(e[0], e[1], 60, (L,)):
            poly = Mp.poly[L][Mp.tok[L].index(tok)]
            pts = [sxy(*P(a, b)) for a, b in poly.exterior.coords]
            if all(not (0 <= u <= W and 0 <= v <= H) for u, v in pts):
                continue
            parts.append(f'<polygon points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in pts)}" class="{cls}"/>')
    if len(past) > 1:
        parts.append(f'<polyline points="{" ".join("%.1f,%.1f" % sxy(*P(a, b)) for a, b in past)}" class="past"/>')
    fp = np.asarray(rec['future_positions'])[:, :2]
    parts.append(f'<polyline points="{" ".join("%.1f,%.1f" % sxy(*P(a, b)) for a, b in [(e[0], e[1])] + list(fp))}" class="fut"/>')
    for a, b in fp:
        u, v = sxy(*P(a, b)); parts.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="2.2" class="futd"/>')
    ex = [sxy(2.6, 0.95), sxy(2.6, -0.95), sxy(-1.5, -0.95), sxy(-1.5, 0.95)]
    parts.append(f'<polygon points="{" ".join(f"{u:.1f},{v:.1f}" for u, v in ex)}" class="ego"/>')
    for role, o in (('lead', x['lead']), ('yield', x['yc'])):
        if o is None:
            continue
        u, v = sxy(o['x'], o['y'])
        parts.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="5" class="{role}"/>'
                     f'<text x="{u + 7:.1f}" y="{v + 4:.1f}" class="lb">{html.escape(o["kind"])} ({role})</text>')
    return (f'<svg viewBox="0 0 {W} {H}" class="td" role="img" aria-label="top-down path">'
            f'<rect x="0" y="0" width="{W}" height="{H}" class="bg"/>' + ''.join(parts) +
            f'<text x="6" y="14" class="tk">ego frame, up = forward; grid none; 10 m = {10 * s:.0f} px</text></svg>')


def causes_lines(x):
    e = x['ego']; out = [('ego', f'speed {e["v0"]:.1f} m/s now, {e["v_1s"]:.1f} m/s 1 s ago; turn signal {e["signal"]}'),
                         ('map', f'{x["nl"]} lane(s) in the ego direction; {"in" if x["inter0"] else "not in"} an intersection')]
    if x['sl']:
        out.append(('map', f'{CT.STOP_T.get(x["sl"][2]["stop_line_type"], "marked")} stop line {x["sl"][0]:.0f} m ahead'))
    if x['xw']:
        out.append(('map', f'crosswalk {x["xw"][0]:.0f} m ahead'))
    if x['lead']:
        out.append(('boxes', 'lead ' + CT.describe(x['lead'])))
    if x['yc']:
        out.append(('boxes', 'yield candidate ' + CT.describe(x['yc'])))
    if x['cons']:
        out.append(('boxes', 'construction zone (>= 3 cones / barriers within 30 m)'))
    out.append(('ego', f'route command [P]: {CT.CMD[x["command"]]}'))
    return out


def evidence(x, T):
    F = x['F']; t = T
    cur = np.abs(np.asarray(t['cur'][:12])); acc = np.asarray(t['acc'][:12])
    ev = [f'speed: now {x["ego"]["v0"]:.1f}, min next 6 s {F["vmin"]:.1f}, at +6 s {F["vend"]:.1f} m/s '
          f'(change {F["vend"] - x["ego"]["v0"]:+.1f})',
          f'decel ticks in first 3 s >= 30%: {"yes" if F["dec"] else "no"}; stops in next 6 s: '
          f'{"yes" if F["stop"] else "no"}; stays stopped > 50%: {"yes" if F["stay"] else "no"}; moves again '
          f'after stop: {"yes" if F["go_after"] else "no"}',
          f'GT controls: max |curvature| {cur.max():.3f} 1/m; min / max accel {acc.min():+.1f} / {acc.max():+.1f} m/s2',
          f'heading change +6 s {F["dpsi"]:+.0f} deg, of which inside intersections {F["dpsi_int"]:+.0f} deg '
          f'(turn rule: > 30); path through intersection: {"yes" if F["inter"] else "no"}',
          f'end offset x {F["x_end"]:.0f} m, y {F["y_end"]:+.1f} m; lane change (final lane not reachable): '
          f'{"yes" if F["lc"] else "no"}']
    for role, o in (('lead', x['lead']), ('yield candidate', x['yc'])):
        if o is not None:
            sp = f'{o["speed"]:.1f} m/s' if o['speed'] is not None else 'speed unknown'
            ev.append(f'{role}: {o["kind"]} at {o["x"]:.1f} m ahead, {o["y"]:+.1f} m lateral, {sp}, {CT.rel_dir(o)}')
    return ev


def main():
    ids = parse_ids(TXT); assert len(ids) == 100
    CT._init_globals()
    rows = CT.build_rows({t for _, t in ids}, procs=20, min_fut=6)
    J = lambda n: json.load(open(f'{NUS}/v1.0-trainval/{n}.json'))
    S = {s['token']: s for s in J('sample')}
    EP = {e['token']: e for e in J('ego_pose')}
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
    out = [HEAD]; lon_prev = None
    for aid, st in ids:
        x = rows[st]; rec = CT._G['W'][st]
        if x['lon'] != lon_prev:
            out.append(f'<h2>{html.escape(x["lon"])}</h2>'); lon_prev = x['lon']
        t0 = S[st]['timestamp']
        frames = []
        for k, lab in STRIP:
            tok = st
            for _ in range(abs(k)):
                tok = S[tok]['prev' if k < 0 else 'next'] if tok else ''
            if not tok:
                frames.append(f'<figure class="f{" now" if k == 0 else ""}"><div class="na">no frame</div>'
                              f'<figcaption>{lab}</figcaption></figure>'); continue
            sd = cam[tok]; im = Image.open(f'{NUS}/{sd["filename"]}').convert('RGB')
            if k == 0:
                drawn = film(None, x, Cam(sd, EP, CS), im)
            w = 480 if k == 0 else 300
            im = im.resize((w, round(900 * w / 1600)), Image.BILINEAR)
            dt = (sd['timestamp'] - t0) / 1e6
            cap = f'{lab} <span class="dt">({dt:+.2f} s)</span>'
            fut = ' fut' if k > 0 else ''
            frames.append(f'<figure class="f{" now" if k == 0 else ""}{fut}"><img src="{jpg(im)}" alt="{lab}">'
                          f'<figcaption>{cap}</figcaption></figure>')
        try:
            m = can.get_messages(x['scene'], 'pose')
            ut = np.array([q['utime'] for q in m]); v = np.array([q['vel'][0] for q in m])
        except Exception:
            ut = v = np.array([])
        spd = speed_svg(t0, ut, v) if len(ut) else '<div class="na">no CAN for this scene</div>'
        past, tok = [], st
        for _ in range(5):
            tok = S[tok]['prev']
            if not tok:
                break
            past.append(EP[lid[tok]['ego_pose_token']]['translation'][:2])
        past = past[::-1] + [CT.ego_pose(rec)[:2]]
        Mp = getmap(x['scene'])
        td = topdown_svg(x, rec, past, Mp)
        e = {k: html.escape(str(vv)) for k, vv in (('id', aid), ('token', st), ('scene', x['scene']))}
        when = datetime.datetime.utcfromtimestamp(t0 / 1e6).strftime('%Y-%m-%d %H:%M:%S UTC')
        cz = ''.join(f'<li><span class="tag t{tg}">[{tg}]</span> {html.escape(c)}</li>' for tg, c in causes_lines(x))
        evh = ''.join(f'<li>{html.escape(s_)}</li>' for s_ in evidence(x, T[st]))
        tr = CT.compose(x['lon'], x['lat'], x['cause'], x['ego'], x['F'], x['sl'], x['xw'], x['cons'])
        tog = ''.join(f'<div class="tog"><span>{lab}</span>' + ''.join(
            f'<label><input type="radio" name="{aid}_{k}" value="{val}" data-id="{aid}" data-k="{k}"> {val}</label>'
            for val in ('OK', 'wrong', 'unsure')) + '</div>'
            for k, lab in (('decision', 'Decision'), ('causes', 'Causes'), ('peeking', 'No peeking')))
        out.append(f'''<section class="card" id="{aid}" data-token="{e["token"]}">
<div class="hdr"><b>{aid}</b> &middot; {e["scene"]} &middot; {when} &middot; <code>{e["token"]}</code></div>
<div class="strip">{"".join(frames)}</div>
<p class="futnote">future frames (right of NOW): use only to check the decision. On NOW: {html.escape("; ".join(drawn)) or "no named agents or map items"}</p>
<div class="viz"><div><div class="vt">Ego speed (CAN), t0-1.5 s to t0+6 s</div>{spd}</div>
<div><div class="vt">Top-down, ego frame at t0 (grey past, blue future GT)</div>{td}</div></div>
<div class="vt">Rule evidence</div><ul class="ev">{evh}</ul>
<p><span class="lab">Decision</span> {html.escape(x["lon"])}; {html.escape(x["lat"])}</p>
<div class="lab">Causes</div><ul class="cz">{cz}</ul>
<p><span class="lab">Trace</span> {html.escape(tr)}</p>
<div class="togs">{tog}</div>
<textarea data-id="{aid}" placeholder="note (optional)"></textarea>
</section>''')
        print(aid, flush=True) if aid.endswith('0') else None
    out.append(TAIL)
    open(OUT, 'w').write('\n'.join(out))
    print(f'wrote {OUT}: {os.path.getsize(OUT) / 1e6:.1f} MB')


HEAD = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoC Audit Sheet</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1f;--card:#fff;--line:#ddd;--mut:#666;--acc:#2a5db0;--td:#f2f2ef}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161618;--fg:#eee;--card:#222225;
--line:#3a3a3e;--mut:#aaa;--acc:#7da7ff;--td:#1b1b1e}}
:root[data-theme="dark"]{--bg:#161618;--fg:#eee;--card:#222225;--line:#3a3a3e;--mut:#aaa;--acc:#7da7ff;--td:#1b1b1e}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:0 16px 80px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;z-index:2}
h1{font-size:19px;margin:4px 0} h2{margin:26px 0 8px;font-size:17px;color:var(--acc)}
.how{border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:12px 0;max-width:1000px;background:var(--card)}
.how b{display:inline-block;min-width:92px}
.counts{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:14px;color:var(--mut)} .counts b{color:var(--fg)}
button{font:inherit;padding:5px 14px;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:6px;cursor:pointer}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:14px 0}
.hdr{font-size:13px;color:var(--mut);overflow-wrap:anywhere} .hdr b{color:var(--fg)}
.strip{display:flex;gap:6px;overflow-x:auto;padding:8px 0;align-items:flex-start}
figure{margin:0;flex:0 0 auto} .strip img{display:block;height:169px;width:auto;border-radius:3px}
.strip .now img{height:270px;outline:4px solid #f0a500;outline-offset:-2px}
figcaption{font-size:12px;color:var(--mut);text-align:center} .now figcaption{color:var(--fg);font-weight:700}
.fut img{opacity:.92} .dt{color:var(--mut);font-weight:400} .na{font-size:12px;color:var(--mut);padding:30px 12px;border:1px dashed var(--line)}
.futnote{font-size:12px;color:var(--mut);margin:2px 0 8px}
.viz{display:flex;flex-wrap:wrap;gap:16px} .vt{font-size:13px;font-weight:600;margin:6px 0 2px}
.chart{width:420px;max-width:100%;height:auto} .td{width:300px;max-width:100%;height:auto}
.chart .g{stroke:var(--line);stroke-width:1} .chart .tk,.td .tk{font-size:10px;fill:var(--mut)}
.chart .now{stroke:#f0a500;stroke-width:2} .chart .spd{fill:none;stroke:var(--acc);stroke-width:2}
.td .bg{fill:var(--td)} .td .ln{fill:none;stroke:var(--mut);stroke-opacity:.35;stroke-width:1}
.td .xw{fill:#3c8cff;fill-opacity:.25;stroke:#3c8cff} .td .sl{fill:#e00;fill-opacity:.5;stroke:#e00}
.td .past{fill:none;stroke:#888;stroke-width:3} .td .fut{fill:none;stroke:var(--acc);stroke-width:2.5}
.td .futd{fill:var(--acc)} .td .ego{fill:#f0a500;stroke:#000;stroke-width:1}
.td .lead{fill:#ffc800;stroke:#000} .td .yield{fill:#ff3c3c;stroke:#000} .td .lb{font-size:10px;fill:var(--fg)}
ul{margin:4px 0 8px;padding-left:20px} .ev li,.cz li{margin:1px 0}
.lab{font-weight:700;min-width:68px;display:inline-block} .tag{font-family:monospace;font-size:12px}
.tego{color:#8a6d00} .tmap{color:#1d6fd6} .tboxes{color:#c2410c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .tego{color:#e0c060}}
.togs{display:flex;flex-wrap:wrap;gap:8px 22px;margin:8px 0} .tog span{font-weight:600;margin-right:6px}
.tog label{margin-right:8px;cursor:pointer} textarea{width:100%;box-sizing:border-box;min-height:44px;font:inherit;
background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px} code{font-size:12px}
</style></head><body>
<header><h1>CoC audit sheet v2 (100 val traces)</h1>
<div class="counts" id="counts"></div>
<p style="margin:6px 0"><button id="dl">Download results</button>
<span style="color:var(--mut);font-size:13px"> CSV: sample_id, decision, causes, peeking, note (OK / wrong / unsure). Kept in this browser.</span></p></header>
<div class="how"><b>How to audit</b><br>
<b>Decision:</b> does the label match the speed chart and path?<br>
<b>Causes:</b> is each named thing really there at t0 and relevant?<br>
<b>No peeking:</b> does any cause rely on something seen only in future frames?</div>'''

TAIL = '''<script>
(function(){
var KEY='coc_audit_v2', st={};
try{st=JSON.parse(localStorage.getItem(KEY)||'{}')||{};}catch(e){st={};}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st));}catch(e){}}
var cards=[].slice.call(document.querySelectorAll('.card')), ids=cards.map(function(c){return c.id;});
ids.forEach(function(id){st[id]=st[id]||{};});
document.querySelectorAll('input[type=radio]').forEach(function(r){
  if(st[r.dataset.id][r.dataset.k]===r.value) r.checked=true;
  r.addEventListener('change',function(){st[r.dataset.id][r.dataset.k]=r.value;save();counts();});});
document.querySelectorAll('textarea').forEach(function(t){
  t.value=st[t.dataset.id].note||'';
  t.addEventListener('input',function(){st[t.dataset.id].note=t.value;save();});});
function counts(){
  document.getElementById('counts').innerHTML=[['decision','Decision'],['causes','Causes'],['peeking','No peeking']].map(function(k){
    var n={OK:0,wrong:0,unsure:0}; ids.forEach(function(id){var v=st[id][k[0]]; if(n[v]!==undefined) n[v]++;});
    return k[1]+': <b>'+n.OK+'</b> OK, <b>'+n.wrong+'</b> wrong, <b>'+n.unsure+'</b> unsure, '+(ids.length-n.OK-n.wrong-n.unsure)+' open';
  }).join(' &middot; ');}
counts();
function q(v){v=(v||'').replace(/"/g,'""');return '"'+v+'"';}
document.getElementById('dl').addEventListener('click',function(){
  var rows=['sample_id,decision,causes,peeking,note'];
  cards.forEach(function(c){var s=st[c.id];
    rows.push([q(c.dataset.token),q(s.decision),q(s.causes),q(s.peeking),q(s.note)].join(','));});
  var b=new Blob([rows.join('\\n')+'\\n'],{type:'text/csv'}),a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='coc_audit_results.csv';document.body.appendChild(a);a.click();a.remove();});
})();
</script></body></html>'''

if __name__ == '__main__':
    main()
