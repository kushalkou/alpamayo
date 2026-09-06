"""shrink_lib.py — shared helpers for the Gate 3 offline sweeps."""
import pickle, math
import numpy as np

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
DT, N = 0.5, 12
HOR = {2: '1s', 4: '2s', 6: '3s', 12: '6s'}
CURV_T = 0.05
rng = np.random.RandomState(0)


def rollout(acc, cur, v0, yaw0):
    x = y = 0.0; yaw = yaw0; v = v0; out = np.empty((N, 2))
    for j in range(N):
        v = max(0.0, v + acc[j] * DT)
        yaw = yaw + v * cur[j] * DT
        x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
        out[j] = (x, y)
    return out


def cv_traj(v0, yaw0):
    k = np.arange(1, N + 1) * DT * v0
    return np.stack([k * math.cos(yaw0), k * math.sin(yaw0)], 1)


def ade(p, g, steps=N):
    return float(np.linalg.norm(p[:steps] - g[:steps], axis=1).mean())


def paired(a, b, n=10000, chunk=2000):
    d = np.asarray(a, float) - np.asarray(b, float); m = len(d)
    out = np.empty(n)
    for s in range(0, n, chunk):
        k = min(chunk, n - s); out[s:s+k] = d[rng.randint(0, m, (k, m))].mean(1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return d.mean(), lo, hi, min(1.0, 2 * min((out <= 0).mean(), (out >= 0).mean()))


def wilson(k, n, z=1.96):
    if n == 0: return float('nan'), float('nan')
    ph = k / n; den = 1 + z*z/n
    c = (ph + z*z/(2*n)) / den
    h = z*np.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    return c - h, c + h


def load(split):
    with open(f'{RES}/dump_{split}.pkl', 'rb') as f:
        D = pickle.load(f)
    meta = D['meta']; idx = sorted(meta)
    G = {i: np.array(meta[i]['gt']) for i in idx}
    CV = {i: cv_traj(meta[i]['v0'], meta[i]['yaw0']) for i in idx}
    return D, meta, idx, G, CV


def values(D, m, i, mode, tau=None):
    r = D['data'][m][i]
    a = np.array(r['argmax']); e = np.array(r['expect'])
    if mode == 'argmax': v = a
    elif mode == 'expect': v = e
    else:
        ps = np.array(r['p_stop']); v = np.where(ps > tau, a, e)
    return v[:12], v[12:]


def traj(D, meta, m, i, mode, tau=None):
    acc, cur = values(D, m, i, mode, tau)
    return rollout(acc, cur, meta[i]['v0'], meta[i]['yaw0'])


def maxpredcurv(D, m, i, mode='expect'):
    _, cur = values(D, m, i, mode)
    return float(np.abs(cur).max())


def strata(meta, idx, use_detector=False):
    out = {}
    for i in idx:
        if abs(meta[i]['v0']) < 0.5:
            out[i] = 'stationary'
        else:
            k = meta[i]['past_curv'] if use_detector else meta[i]['maxcurv']
            out[i] = 'turning' if k > CURV_T else 'straight'
    return out
