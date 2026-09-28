"""sim/model.py -- small MLP planner for the synthetic benchmark, + dump writer.

[ego history 16 feats | observation 9] -> MLP -> 24 heads x 65 classes
(64 bins + STOP). NON-AUTOREGRESSIVE: all 24 slots are predicted independently from
the input. This is a SIMPLIFICATION relative to the nuScenes AR decoder.

Writes RES/sim/<cell>/dump_{val,test}.pkl in the exact dump_decode.py format:
  {'meta': {i: {gt, v0, yaw0, maxcurv, past_curv, n_hist, ...}},
   'data': {model_name: {i: {'argmax': [24], 'expect': [24], 'p_stop': [24]}}}}
Models in the dump:
  'mlp'     -- the trained MLP (expectation decode = STOP-aware mean, as dump_one)
  'meanctl' -- training-set mean control per slot (the zeroboth analogue);
               argmax == expect == mean control, p_stop = 0
  'v0mean'  -- training-set mean control per slot within the sample's v0 bin
Also writes ce.json: per-slot CE of the MLP and the train unigram on test.

Usage: python model.py --p 0.5 --rho 1.0 [--seed 0] [--train_seed 0] [--gpu 0]
"""
import os, sys, json, time, pickle, argparse, contextlib, io
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/sim')
import generate as GEN
from tokenizer import TrajectoryTokenizer, STOP_TOKEN, N_BINS

SIMRES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/sim'
NCLS = N_BINS + 1          # 0..63 bins, 64 = STOP
NSLOT = 24
# fixed v0 bins (m/s), shared by the sim and the nuScenes L2 locator
V0_EDGES = np.array([0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14])


def v0bin(v):
    return np.searchsorted(V0_EDGES, np.asarray(v), side='right')   # 0..14


def load_cell(p, rho, seed, arange=(GEN.A_LO, GEN.A_HI)):
    path = f'{GEN.OUT}/{GEN.cell_name(p, rho, seed, arange)}.pkl'
    if not os.path.exists(path):
        D = GEN.generate(p, rho, seed, arange)
        os.makedirs(GEN.OUT, exist_ok=True)
        with open(path + '.tmp', 'wb') as f: pickle.dump(D, f)
        os.replace(path + '.tmp', path)
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_cls(tokens):
    t = tokens.copy(); t[t == STOP_TOKEN] = N_BINS
    return t


class MLP(nn.Module):
    def __init__(self, d_in, h=512, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, NSLOT * NCLS))

    def forward(self, x):
        return self.net(x).view(-1, NSLOT, NCLS)


def feats(S, zero_obs=False):
    obs = np.zeros_like(S['obs']) if zero_obs else S['obs']
    return np.concatenate([S['ego'].reshape(len(S['ego']), -1), obs], 1).astype(np.float32)


def train(D, train_seed, device, epochs=80, bs=512, lr=1e-3, zero_obs=False):
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    Xtr, Xva = feats(D['train'], zero_obs), feats(D['val'], zero_obs)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda X: torch.tensor((X - mu) / sd, device=device)
    xtr, xva = norm(Xtr), norm(Xva)
    ytr = torch.tensor(to_cls(D['train']['tokens']), device=device)
    yva = torch.tensor(to_cls(D['val']['tokens']), device=device)
    net = MLP(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator(device='cpu').manual_seed(train_seed)
    best, best_state, best_ep = 1e9, None, -1
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr), generator=g).to(device)
        for s in range(0, len(xtr), bs):
            b = perm[s:s + bs]
            loss = F.cross_entropy(net(xtr[b]).reshape(-1, NCLS), ytr[b].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = F.cross_entropy(net(xva).reshape(-1, NCLS), yva.reshape(-1)).item()
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    return net, (mu, sd), dict(val_ce=best, best_epoch=best_ep, train_sec=time.time() - t0)


def slot_values(logits, tok):
    """logits [n,24,65] -> argmax_val, expect_val, p_stop, each [n,24] (dump_one rule).
    STOP-aware support {0..63} U {STOP}; STOP -> 0.0; argmax STOP -> 0.0."""
    q = torch.softmax(logits.float(), -1).cpu().numpy()
    cent = np.zeros((NSLOT, NCLS))
    cent[:12, :N_BINS] = tok.accel_centers; cent[12:, :N_BINS] = tok.curv_centers
    ev = (q * cent[None]).sum(-1)
    am = q.argmax(-1)
    av = np.take_along_axis(np.broadcast_to(cent, q.shape[:2] + (NCLS,)),
                            am[..., None], -1)[..., 0]
    return av, ev, q[..., N_BINS]


def run(p, rho, seed=0, train_seed=0, device='cuda:0', tag=None, zero_obs=False,
        arange=(GEN.A_LO, GEN.A_HI)):
    with contextlib.redirect_stdout(io.StringIO()):
        tok = TrajectoryTokenizer()
    D = load_cell(p, rho, seed, arange)
    net, (mu, sd), info = train(D, train_seed, device, zero_obs=zero_obs)
    cell = tag or (GEN.cell_name(p, rho, seed, arange) + f'_t{train_seed}'
                   + ('_noobs' if zero_obs else ''))
    od = f'{SIMRES}/{cell}'; os.makedirs(od, exist_ok=True)

    # training-set mean continuous control per slot (zeroboth analogue)
    mean_ctl = np.concatenate([D['train']['acc'].mean(0), D['train']['cur'].mean(0)])
    # v0-conditioned mean control (train, fixed V0_EDGES bins; empty bin -> global mean)
    ctl_tr = np.concatenate([D['train']['acc'], D['train']['cur']], 1)
    b_tr = v0bin([D['train']['meta'][i]['v0'] for i in range(len(ctl_tr))])
    v0mean = np.stack([ctl_tr[b_tr == b].mean(0) if (b_tr == b).any() else mean_ctl
                       for b in range(len(V0_EDGES) + 1)])

    # unigram over train tokens, pooled per half (accel slots / curv slots)
    ytr = to_cls(D['train']['tokens'])
    uni = {h: (np.bincount(ytr[:, sl].ravel(), minlength=NCLS) + 1.0)
           for h, sl in (('acc', slice(0, 12)), ('cur', slice(12, 24)))}
    uni = {h: u / u.sum() for h, u in uni.items()}
    # unigram conditioned on v0 bin, per half, add-one smoothing
    uni_v = {}
    for h, sl in (('acc', slice(0, 12)), ('cur', slice(12, 24))):
        c = np.ones((len(V0_EDGES) + 1, NCLS))
        for b in range(len(V0_EDGES) + 1):
            c[b] += np.bincount(ytr[b_tr == b][:, sl].ravel(), minlength=NCLS)
        uni_v[h] = c / c.sum(1, keepdims=True)

    ce = {}
    for split in ('val', 'test'):
        S = D[split]
        x = torch.tensor((feats(S, zero_obs) - mu) / sd, device=device)
        with torch.no_grad():
            lg = net(x)
        av, ev, ps = slot_values(lg, tok)
        y = to_cls(S['tokens'])
        lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
        ce_m = -np.take_along_axis(lp, y[..., None], -1)[..., 0].mean(0)        # [24]
        ce_u = np.array([-np.log(uni['acc' if j < 12 else 'cur'][y[:, j]]).mean()
                         for j in range(NSLOT)])
        bs_ = v0bin([S['meta'][i]['v0'] for i in range(len(y))])
        ce_uv = np.array([-np.log(uni_v['acc' if j < 12 else 'cur'][bs_, y[:, j]]).mean()
                          for j in range(NSLOT)])
        ce[split] = {'model': ce_m.tolist(), 'unigram': ce_u.tolist(), 'unigram_v0': ce_uv.tolist()}
        data = {'mlp': {}, 'meanctl': {}, 'v0mean': {}}
        for i in range(len(y)):
            data['mlp'][i] = {'argmax': av[i].tolist(), 'expect': ev[i].tolist(),
                              'p_stop': ps[i].tolist()}
            data['meanctl'][i] = {'argmax': mean_ctl.tolist(), 'expect': mean_ctl.tolist(),
                                  'p_stop': [0.0] * NSLOT}
            vm = v0mean[bs_[i]].tolist()
            data['v0mean'][i] = {'argmax': vm, 'expect': vm, 'p_stop': [0.0] * NSLOT}
        with open(f'{od}/dump_{split}.pkl', 'wb') as f:
            pickle.dump({'meta': S['meta'], 'data': data}, f)
    info.update(p=p, rho=rho, seed=seed, train_seed=train_seed, config=D['config'],
                zero_obs=zero_obs,
                turn_frac_test=float(np.mean([D['test']['meta'][i]['maxcurv'] > GEN.CURV_T
                                              for i in range(len(D['test']['meta']))])))
    with open(f'{od}/ce.json', 'w') as f:
        json.dump({'info': info, 'ce': ce}, f, indent=1, default=str)
    print(f'[model] {cell}: val_ce={info["val_ce"]:.4f} best_ep={info["best_epoch"]} '
          f'train {info["train_sec"]:.1f}s -> {od}', flush=True)
    return od


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p', type=float, required=True)
    ap.add_argument('--rho', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_seed', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    run(a.p, a.rho, a.seed, a.train_seed, f'cuda:{a.gpu}')


if __name__ == '__main__':
    main()
