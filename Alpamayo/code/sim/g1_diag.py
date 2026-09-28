"""sim/g1_diag.py -- G1 leak diagnosis: is the blend's edge over CV+meanctl a future
leak, or legitimate v0 dependence (braking saturates at v=0; stopped cars can't brake)?
  (a) v0-conditional mean control (train, 10 v0-quantile bins) -- a Bayes-style prior
      that uses ONLY v0; if the MLP blend does not beat it, there is no leak.
  (b) permute the observation vector across test samples; at rho=0 it is independent
      of the future, so ADE must not change beyond noise.
  (c) per-stratum split of the blend - meanctl gap (stationary v0<0.5 vs moving).
"""
import sys, numpy as np, torch, contextlib, io
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/sim')
import shrink_lib as SL, model as M, gates as GT
from tokenizer import TrajectoryTokenizer

od = f'{M.SIMRES}/p0.00_rho0.00_s0_t0'
D = M.load_cell(0.0, 0.0, 0)
tr, te, va = D['train'], D['test'], D['val']
v_tr = np.array([tr['meta'][i]['v0'] for i in range(len(tr['meta']))])
edges = np.quantile(v_tr, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
ctl = np.concatenate([tr['acc'], tr['cur']], 1)
bmean = [ctl[(v_tr >= edges[b]) & (v_tr < edges[b+1])].mean(0) for b in range(10)]
bmean = [m if np.isfinite(m).all() else ctl.mean(0) for m in bmean]

def binned(S):
    out = []
    for i in range(len(S['meta'])):
        m = S['meta'][i]; b = min(max(int(np.searchsorted(edges, m['v0'], 'right') - 1), 0), 9)
        out.append(SL.rollout(bmean[b][:12], bmean[b][12:], m['v0'], m['yaw0']))
    return out

r = GT.eval_cell(od)
Gt = [np.array(te['meta'][i]['gt']) for i in range(len(te['meta']))]
Gv = [np.array(va['meta'][i]['gt']) for i in range(len(va['meta']))]
bt = np.array([SL.ade(t, g) for t, g in zip(binned(te), Gt)])
print('ADE@6s mean: CV %.3f  meanctl %.3f  v0-binned meanctl %.3f  mlp blend %.3f (a=%.2f)'
      % (r['cv'].mean(), r['meanctl_expect'].mean(), bt.mean(), r['blend'].mean(), r['alpha']))
print(GT.line('v0-binned meanctl - CV+meanctl', bt, r['meanctl_expect']))
print(GT.line('mlp blend - v0-binned meanctl', r['blend'], bt))
# blend the v0-binned predictor with CV too (alpha fit on val), for a like-for-like
CVv = [SL.cv_traj(va['meta'][i]['v0'], va['meta'][i]['yaw0']) for i in range(len(Gv))]
CVt = [SL.cv_traj(te['meta'][i]['v0'], te['meta'][i]['yaw0']) for i in range(len(Gt))]
Bv, Bt = binned(va), binned(te)
cur = [np.mean([SL.ade(a*b + (1-a)*c, g) for b, c, g in zip(Bv, CVv, Gv)]) for a in GT.ALPHAS]
ab = float(GT.ALPHAS[int(np.argmin(cur))])
bb = np.array([SL.ade(ab*b + (1-ab)*c, g) for b, c, g in zip(Bt, CVt, Gt)])
print('v0-binned blend alpha*=%.2f  ADE %.3f' % (ab, bb.mean()))
print(GT.line('mlp blend - v0-binned blend', r['blend'], bb))

# (c) strata
st = np.array([te['meta'][i]['v0'] < 0.5 for i in range(len(Gt))])
for nm, s in (('stationary', st), ('moving', ~st)):
    print(f'  [{nm:10} n={s.sum()}]' + GT.line('mlp blend - CV+meanctl', r['blend'][s], r['meanctl_expect'][s])[2:])
    print(f'  [{nm:10} n={s.sum()}]' + GT.line('mlp blend - v0-binned meanctl', r['blend'][s], bt[s])[2:])

# (b) permute obs on test, re-run the trained net? (net not saved) -> retrain same seed
with contextlib.redirect_stdout(io.StringIO()):
    tok = TrajectoryTokenizer()
net, (mu, sd), _ = M.train(D, 0, 'cuda:0')
X = M.feats(te); rs = np.random.RandomState(7)
Xp = X.copy(); Xp[:, 16:] = X[rs.permutation(len(X)), 16:]
def ade_of(X):
    with torch.no_grad():
        lg = net(torch.tensor((X - mu) / sd, device='cuda:0'))
    _, ev, _ = M.slot_values(lg, tok)
    return np.array([SL.ade(SL.rollout(ev[i, :12], ev[i, 12:], te['meta'][i]['v0'],
                     te['meta'][i]['yaw0']), Gt[i]) for i in range(len(Gt))])
a0, a1 = ade_of(X), ade_of(Xp)
print('mlp(expect) retrained same seed: %.3f (dump %.3f)' % (a0.mean(), r['mlp_expect'].mean()))
print(GT.line('mlp obs-permuted - mlp', a1, a0))
