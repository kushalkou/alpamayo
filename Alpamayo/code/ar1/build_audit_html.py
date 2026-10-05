"""ar1/build_audit_html.py -- CoC audit sheet: AR1_COC_AUDIT.txt -> AR1_COC_AUDIT.html.

Self-contained HTML (images embedded as base64 JPEG), one card per trace, in the order of
the .txt (stratified by lon decision, then lat). Per card: sample id, scene, time (UTC);
CAM_FRONT at t0 (640 px wide), at t0-1.0 s (keyframe 2 back; "past") and at t0+3.0 s
(keyframe 6 forward; "future, for checking the decision only"), 320 px; ego speed now and
3 s later (CAN, R1.2 ticks 0 and 30); route command [P]; decision, causes, trace; three
OK / wrong toggles (decision, causes, no peeking) and a note box. Running counts at the top;
"Download results" writes a CSV (sample_id, decision, causes, peeking, note) in the browser.
Answers persist in the browser's localStorage when available.
  python ar1/build_audit_html.py
"""
import base64, io, json, pickle, re, datetime, html
from PIL import Image

ROOT = '/home/dgx1user/Alpamayo-Kushal'
NUS = f'{ROOT}/Alpamayo/nuscenes'
DATA = f'{ROOT}/Alpamayo/data'
TXT = f'{ROOT}/AR1_COC_AUDIT.txt'
OUT = f'{ROOT}/AR1_COC_AUDIT.html'
CMD = {0: 'turn right', 1: 'turn left', 2: 'go straight'}


def parse(txt):
    cards, cur, key = [], None, None
    for line in open(txt).read().split('\n'):
        m = re.match(r'^(A\d{3})  (scene-\d+)  sample (\w+)$', line)
        if m:
            cur = {'id': m[1], 'scene': m[2], 'token': m[3], 'decision': '', 'causes': '', 'trace': ''}
            cards.append(cur); key = None; continue
        if cur is None:
            continue
        for k in ('decision', 'causes', 'trace'):
            if line.startswith(f'  {k}: '):
                cur[k] = line[len(k) + 4:].strip(); key = k; break
        else:
            if key in ('causes', 'trace') and line.startswith('         ') and line.strip():
                cur[key] += ' ' + line.strip()
            elif line.startswith('  ['):
                key = None
    return cards


def b64(path, width):
    im = Image.open(path).convert('RGB')
    im = im.resize((width, round(im.height * width / im.width)), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=72, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def main():
    cards = parse(TXT)
    assert len(cards) == 100, len(cards)
    J = lambda n: json.load(open(f'{NUS}/v1.0-trainval/{n}.json'))
    S = {s['token']: s for s in J('sample')}
    cam = {}
    for d in J('sample_data'):
        if d['is_key_frame'] and d['filename'].startswith('samples/CAM_FRONT/'):
            cam[d['sample_token']] = f'{NUS}/{d["filename"]}'
    Mt = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb'))
    W = {r['sample_token']: r for r in pickle.load(open(f'{DATA}/w1_data.pkl', 'rb'))['records']}
    for c in cards:
        st = c['token']
        prev = S[st]['prev'] and S[S[st]['prev']]['prev']
        nxt = st
        for _ in range(6):
            nxt = S[nxt]['next'] if nxt else ''
        c['img0'] = b64(cam[st], 640)
        c['imgp'] = b64(cam[prev], 320) if prev else ''
        c['imgf'] = b64(cam[nxt], 320) if nxt else ''
        c['time'] = datetime.datetime.utcfromtimestamp(S[st]['timestamp'] / 1e6).strftime('%Y-%m-%d %H:%M:%S UTC')
        v = Mt[st]['v']
        c['v0'] = f'{v[0]:.1f} m/s' if v[0] == v[0] else 'n/a'
        c['v3'] = f'{v[30]:.1f} m/s' if v[30] == v[30] else 'n/a'
        c['route'] = CMD[W[st]['command']]
        c['lon'] = c['decision'].split(';')[0].replace('lon = ', '').strip()
    out = [HEAD]
    lon_prev = None
    for c in cards:
        if c['lon'] != lon_prev:
            n = sum(x['lon'] == c['lon'] for x in cards)
            out.append(f'<h2>{html.escape(c["lon"])} ({n})</h2>')
            lon_prev = c['lon']
        e = {k: html.escape(str(v)) for k, v in c.items() if not k.startswith('img')}
        past = (f'<figure><img src="{c["imgp"]}" alt="past"><figcaption>past (t0 - 1.0 s)</figcaption></figure>'
                if c['imgp'] else '<figure><div class="na">no past frame</div></figure>')
        fut = (f'<figure><img src="{c["imgf"]}" alt="future"><figcaption>future (t0 + 3.0 s), for checking '
               f'the decision only</figcaption></figure>' if c['imgf'] else '<figure><div class="na">no future frame'
                                                                                '</div></figure>')
        tog = ''.join(f'<div class="tog"><span>{lab}</span><label><input type="radio" name="{e["id"]}_{k}" '
                      f'value="OK" data-id="{e["id"]}" data-k="{k}"> OK</label><label><input type="radio" '
                      f'name="{e["id"]}_{k}" value="wrong" data-id="{e["id"]}" data-k="{k}"> wrong</label></div>'
                      for k, lab in (('decision', 'Decision'), ('causes', 'Causes'), ('peeking', 'No peeking')))
        out.append(f'''<section class="card" id="{e["id"]}">
<div class="hdr"><b>{e["id"]}</b> &middot; {e["scene"]} &middot; {e["time"]} &middot; <code>{e["token"]}</code></div>
<div class="imgs"><figure class="big"><img src="{c["img0"]}" alt="t0"><figcaption>CAM_FRONT at t0</figcaption></figure>
<div class="small">{past}{fut}</div></div>
<p class="meta">ego speed now <b>{e["v0"]}</b>, 3 s later <b>{e["v3"]}</b> &middot; route command [P]: <b>{e["route"]}</b></p>
<p><span class="lab">Decision</span> {e["decision"]}</p>
<p><span class="lab">Causes</span> {e["causes"]}</p>
<p><span class="lab">Trace</span> {e["trace"]}</p>
<div class="togs">{tog}</div>
<textarea data-id="{e["id"]}" placeholder="note (optional)"></textarea>
</section>''')
    out.append(TAIL)
    open(OUT, 'w').write('\n'.join(out))
    import os
    print(f'wrote {OUT}: {len(cards)} cards, {os.path.getsize(OUT) / 1e6:.1f} MB')


HEAD = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoC Audit Sheet</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1f;--card:#fff;--line:#ddd;--mut:#666;--ok:#1b7f3b;--bad:#b3261e;--acc:#2a5db0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#161618;--fg:#eee;--card:#222225;
--line:#3a3a3e;--mut:#aaa;--ok:#5bc27a;--bad:#ef6b62;--acc:#7da7ff}}
:root[data-theme="dark"]{--bg:#161618;--fg:#eee;--card:#222225;--line:#3a3a3e;--mut:#aaa;--ok:#5bc27a;--bad:#ef6b62;--acc:#7da7ff}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:0 16px 80px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;z-index:2}
h1{font-size:20px;margin:6px 0} h2{margin:28px 0 8px;font-size:17px;color:var(--acc)}
.counts{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:14px;color:var(--mut)} .counts b{color:var(--fg)}
button{font:inherit;padding:6px 14px;border:1px solid var(--acc);background:var(--acc);color:#fff;border-radius:6px;cursor:pointer}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0;max-width:1000px}
.hdr{font-size:13px;color:var(--mut);overflow-wrap:anywhere} .hdr b{color:var(--fg)}
.imgs{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0} figure{margin:0} img{max-width:100%;height:auto;border-radius:4px;display:block}
.big{flex:1 1 640px;max-width:640px} .small{display:flex;flex-direction:column;gap:8px;flex:0 1 320px;max-width:320px}
figcaption{font-size:12px;color:var(--mut)} .na{font-size:12px;color:var(--mut);padding:20px;border:1px dashed var(--line)}
.lab{display:inline-block;min-width:68px;font-weight:600} p{margin:6px 0} .meta{color:var(--mut)}
.togs{display:flex;flex-wrap:wrap;gap:8px 22px;margin:8px 0} .tog span{font-weight:600;margin-right:6px}
.tog label{margin-right:8px;cursor:pointer} textarea{width:100%;box-sizing:border-box;min-height:44px;font:inherit;
background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
code{font-size:12px}
</style></head><body>
<header><h1>CoC audit sheet (100 val traces)</h1>
<div class="counts" id="counts"></div>
<p style="margin:6px 0"><button id="dl">Download results</button>
<span style="color:var(--mut);font-size:13px"> CSV: sample_id, decision, causes, peeking, note. Answers are kept in this browser.</span></p>
<p style="margin:4px 0;font-size:13px;color:var(--mut)">No peeking = the causes and trace use only what is visible at or
before t0 (plus the map). The future image is only for checking the decision.</p></header>'''

TAIL = '''<script>
(function(){
var KEY='coc_audit_v1', st={};
try{st=JSON.parse(localStorage.getItem(KEY)||'{}')||{};}catch(e){st={};}
function save(){try{localStorage.setItem(KEY,JSON.stringify(st));}catch(e){}}
var ids=[].map.call(document.querySelectorAll('.card'),function(c){return c.id;});
ids.forEach(function(id){st[id]=st[id]||{};});
document.querySelectorAll('input[type=radio]').forEach(function(r){
  var s=st[r.dataset.id]; if(s[r.dataset.k]===r.value) r.checked=true;
  r.addEventListener('change',function(){st[r.dataset.id][r.dataset.k]=r.value;save();counts();});});
document.querySelectorAll('textarea').forEach(function(t){
  t.value=st[t.dataset.id].note||'';
  t.addEventListener('input',function(){st[t.dataset.id].note=t.value;save();});});
function counts(){
  var h=[['decision','Decision'],['causes','Causes'],['peeking','No peeking']].map(function(k){
    var ok=0,bad=0; ids.forEach(function(id){if(st[id][k[0]]==='OK')ok++; else if(st[id][k[0]]==='wrong')bad++;});
    return k[1]+': <b>'+ok+'</b> OK, <b>'+bad+'</b> wrong, '+(ids.length-ok-bad)+' open';});
  document.getElementById('counts').innerHTML=h.join(' &middot; ');}
counts();
function q(v){v=(v||'').replace(/"/g,'""');return '"'+v+'"';}
document.getElementById('dl').addEventListener('click',function(){
  var rows=['sample_id,decision,causes,peeking,note'];
  ids.forEach(function(id){var c=document.getElementById(id).querySelector('code').textContent,s=st[id];
    rows.push([q(c),q(s.decision),q(s.causes),q(s.peeking),q(s.note)].join(','));});
  var b=new Blob([rows.join('\\n')+'\\n'],{type:'text/csv'}),a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='coc_audit_results.csv';document.body.appendChild(a);a.click();a.remove();});
})();
</script></body></html>'''

if __name__ == '__main__':
    main()
