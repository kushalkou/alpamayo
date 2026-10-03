"""ar1/test_hist.py -- R1.1 unit test: no image timestamp > t0; frames ordered.
python ar1/test_hist.py   (exit code 0 = pass)"""
import os, pickle, sys
H = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_hist.pkl', 'rb'))
bad = n = 0
for st, h in H.items():
    for c, seq in h['cams'].items():
        ts = [t for _, t in seq]
        n += len(ts)
        assert len(ts) == 4, (st, c)
        assert all(t <= h['t0'] for t in ts), f'image after t0: {st} {c} {ts} t0={h["t0"]}'
        assert ts == sorted(ts), f'not oldest->newest: {st} {c}'
for st in list(H)[::500]:
    for seq in H[st]['cams'].values():
        assert all(os.path.exists(p) for p, _ in seq), st
print(f'PASS: {len(H)} samples, {n} image slots, max(ts - t0) = '
      f'{max(t - h["t0"] for h in H.values() for s in h["cams"].values() for _, t in s) / 1e6:.4f} s')
