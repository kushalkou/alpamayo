"""w1/tok_floor.py -- WEEK1 item 5: tokenizer reconstruction floor, four variants.

Floor = ADE@6s of rollout(detokenize(tokenize(GT controls))) vs GT positions, seeded
with the true v0 / yaw0 (the model's own decode path; STOP -> (0,0)). No model.
  (i)   uniform bins (current TrajectoryTokenizer, unchanged)
  (ii)  percentile bins: 64 equal-mass bins per channel, edges at quantiles of the
        official-TRAIN non-STOP controls; centre = mean of the train values in the bin
  (iii) error-feedback: step by step, choose the (accel bin, curv bin) pair (64x64
        joint search) minimising the rolled-out position error AT THAT STEP, given
        the already-quantised earlier steps; STOP steps (speed < 0.1) stay STOP
  (iv)  (ii) + (iii)
Bins are fit on official train only. The decision is made on the 50-scene HOLDOUT;
official val is not touched. Reference: the frozen 1.348 is reproduced on the old
custom test split with (i).
Roundtrip exactness: detokenize(tokens) is recomputed twice from the stored tokens,
must be bit-identical, and the error-feedback tokens are regenerated from scratch
and must match exactly (determinism).
"""
import sys, math, pickle, contextlib, io
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from tokenizer import TrajectoryTokenizer, STOP_TOKEN, STOP_SPEED_THRESHOLD
from dataset import pose_to_xyyaw

DT, N = 0.5, 12
with contextlib.redirect_stdout(io.StringIO()):
    TOK = TrajectoryTokenizer()


def gt_of(t):
    x0, y0, yaw0 = pose_to_xyyaw(t['current_pose'])
    g = np.array(t['future_positions'])[:N] - np.array([x0, y0])
    return (np.array(t['future_accelerations'][:N]), np.array(t['future_curvatures'][:N]),
            np.array(t['future_speeds'][:N]), float(t['future_speeds'][0]), yaw0, g)


def roll_batch(acc, cur, v0, yaw0):
    """vectorised shrink_lib.rollout over candidates: acc/cur [..., n] -> positions."""
    n = acc.shape[-1]
    v = np.full(acc.shape[:-1], v0, float); yaw = np.full(acc.shape[:-1], yaw0, float)
    x = np.zeros_like(v); y = np.zeros_like(v); P = []
    for j in range(n):
        v = np.maximum(0.0, v + acc[..., j] * DT)
        yaw = yaw + v * cur[..., j] * DT
        x = x + v * np.cos(yaw) * DT; y = y + v * np.sin(yaw) * DT
        P.append(np.stack([x, y], -1))
    return np.stack(P, -2)


class Bins:
    def __init__(self, ac, kc, ae, ke):
        self.ac, self.kc, self.ae, self.ke = ac, kc, ae, ke     # centres, edges

    def tok(self, a, k, s):
        if s < STOP_SPEED_THRESHOLD: return STOP_TOKEN, STOP_TOKEN
        ai = int(np.clip(np.searchsorted(self.ae, a, 'right') - 1, 0, 63))
        ki = int(np.clip(np.searchsorted(self.ke, k, 'right') - 1, 0, 63))
        return ai, ki

    def detok(self, at, kt):
        if at == STOP_TOKEN: return 0.0, 0.0
        return float(self.ac[at]), float(self.kc[kt])


def uniform_bins():
    return Bins(TOK.accel_centers, TOK.curv_centers, TOK.accel_bins, TOK.curv_bins)


def percentile_bins(train):
    A, K = [], []
    for t in train:
        a, k, s, *_ = gt_of(t)
        m = s >= STOP_SPEED_THRESHOLD
        A.append(a[m]); K.append(k[m])
    A, K = np.concatenate(A), np.concatenate(K)
    out = []
    for X in (A, K):
        e = np.quantile(X, np.linspace(0, 1, 65)); e[0], e[-1] = -np.inf, np.inf
        e = np.maximum.accumulate(e)
        idx = np.clip(np.searchsorted(e, X, 'right') - 1, 0, 63)
        c = np.array([X[idx == b].mean() if (idx == b).any() else np.nan for b in range(64)])
        # empty bins (ties) inherit the neighbour centre
        for b in range(64):
            if np.isnan(c[b]): c[b] = c[b - 1] if b else np.nanmin(c)
        out += [c, e]
    return Bins(out[0], out[2], out[1], out[3])


def encode_plain(t, B):
    a, k, s, *_ = gt_of(t)
    return [B.tok(a[j], k[j], s[j]) for j in range(N)]


def encode_feedback(t, B):
    """greedy joint (a,k) search per step minimising position error at that step."""
    a, k, s, v0, yaw0, g = gt_of(t)
    AA, KK = np.meshgrid(np.arange(64), np.arange(64), indexing='ij')
    cand_a, cand_k = B.ac[AA.ravel()], B.kc[KK.ravel()]
    toks, qa, qk = [], [], []
    for j in range(N):
        if s[j] < STOP_SPEED_THRESHOLD:
            toks.append((STOP_TOKEN, STOP_TOKEN)); qa.append(0.0); qk.append(0.0); continue
        pa = np.tile(np.array(qa + [0.0]), (len(cand_a), 1)); pa[:, -1] = cand_a
        pk = np.tile(np.array(qk + [0.0]), (len(cand_k), 1)); pk[:, -1] = cand_k
        P = roll_batch(pa, pk, v0, yaw0)[:, -1]                   # position at step j
        best = int(np.argmin(np.linalg.norm(P - g[j], axis=1)))
        ai, ki = int(AA.ravel()[best]), int(KK.ravel()[best])
        toks.append((ai, ki)); qa.append(float(B.ac[ai])); qk.append(float(B.kc[ki]))
    return toks


def floor(trajs, B, enc):
    ades, toks_all = [], []
    for t in trajs:
        a, k, s, v0, yaw0, g = gt_of(t)
        toks = enc(t, B); toks_all.append(toks)
        da, dk = zip(*[B.detok(at, kt) for at, kt in toks])
        P = roll_batch(np.array(da)[None], np.array(dk)[None], v0, yaw0)[0]
        ades.append(float(np.linalg.norm(P - g, axis=1).mean()))
    return np.array(ades), toks_all


def exact_check(trajs, B, enc, toks_all):
    """detokenize twice bit-identical; tokens regenerate identically."""
    for t, tk in zip(trajs[:500], toks_all[:500]):
        d1 = [B.detok(*x) for x in tk]; d2 = [B.detok(*x) for x in tk]
        if d1 != d2 or enc(t, B) != tk:
            return False
    return True


def main():
    import os
    W = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_data.pkl', 'rb'))
    recs = [r for r in W['records'] if r['n_fut'] == 12]
    tr = [r for r in recs if r['split'] == 'train']
    ho = [r for r in recs if r['split'] == 'holdout']
    ref = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/leak_split.pkl', 'rb'))['test']
    U = uniform_bins(); P = percentile_bins(tr)
    e, _ = floor(ref, U, encode_plain)
    print(f'REFERENCE (i) uniform on old custom test n={len(ref)}: mean {e.mean():.3f} '
          f'median {np.median(e):.3f}   (frozen: 1.348 / 0.889)')
    print(f'\nofficial train n={len(tr)} (bins fit here), holdout n={len(ho)} (decision set)')
    print(f'{"variant":34} {"holdout mean":>12} {"median":>8} {"p95":>8} {"vs (i)":>8} {"exact":>6}')
    res = {}
    for name, B, enc in (('(i)   uniform', U, encode_plain), ('(ii)  percentile', P, encode_plain),
                         ('(iii) uniform + error-feedback', U, encode_feedback),
                         ('(iv)  percentile + error-feedback', P, encode_feedback)):
        e, tk = floor(ho, B, enc)
        ok = exact_check(ho, B, enc, tk)
        res[name] = e.mean()
        print(f'{name:34} {e.mean():12.3f} {np.median(e):8.3f} {np.percentile(e,95):8.3f} '
              f'{100*(e.mean()/res["(i)   uniform"]-1):+7.1f}% {str(ok):>6}')
    best = min(res, key=res.get); cut = 1 - res[best] / res['(i)   uniform']
    print(f'\nbest = {best.strip()}  cut = {100*cut:.1f}%  (rule: adopt only if >= 40% AND exact)')
    pickle.dump({'percentile_bins': (P.ac, P.kc, P.ae, P.ke)},
                open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_percentile_bins.pkl', 'wb'))


if __name__ == '__main__':
    main()
