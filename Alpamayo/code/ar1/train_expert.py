"""ar1/train_expert.py -- R2.3 step 2: train the flow-matching expert on frozen A3 (1 GPU).

Pre-registered recipe (fixed before the first run):
  data       A3 train set (official train minus holdout, n_fut = 12, 18,313), natural
             sampling; targets = 12 x (accel, curvature) chord controls (w1/targets.py),
             CLIPPED to accel [-6, 6] m/s^2 and curvature [-0.3, 0.3] 1/m (the chord
             curvature has a low-speed noise tail: train p0.5 / p99.5 = -43 / +0.88,
             sd 13.5; clipping 1.9% of curvature steps moves the GT rollout by 0.015 m
             mean ADE), then standardised per channel with train mean / sd. Fixed
             after a first launch showed sd 13.5, before any evaluation.
  context    the frozen A3 KV-cache (ar1/kv_a3.py), stop-grad by construction.
  optimiser  AdamW lr 1e-4, wd 0.01, betas (0.9, 0.95), warmup 500 steps, cosine to 0.1x,
             grad clip 1.0, batch 256, fp32, seed 42.
  length     max 100 epochs; every 5 epochs evaluate on HOLDOUT; patience 4 evaluations.
  selection  HOLDOUT median ADE@6s (n_fut = 12 subset) of ONE sample with fixed noise
             (seed 0), 10 Euler steps -> unicycle rollout from v0_can (targets.rollout,
             the same rollout as the token decoder). Never the flow-matching loss;
             official val is never used for any choice.
  output     6 samples per sample (noise seeds 0..5, fixed per split) for holdout and val
             -> results/ar1_expert_A3.pkl {split: {'tokens', 'traj' [N,6,12,2], 'ctrl'}};
             checkpoint models/checkpoints/_ar1_expert_A3/expert_best.pt.
  python ar1/train_expert.py
"""
import os, sys, time, math, pickle
import numpy as np, torch
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1'); sys.path.insert(0, f'{CODE}/ar1')
from expert import FlowExpert
from targets import rollout
from tok_floor import roll_batch

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_kv_A3_{}.pt'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/ar1_expert_A3.pkl'
CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints/_ar1_expert_A3'
DEV = 'cuda:0'


def roll_many(ctrl, v0):
    """ctrl [N,K,12,2] (physical units), v0 [N] -> [N,K,12,2] lidar-frame xy."""
    return np.stack([roll_batch(c[:, :, 0], c[:, :, 1], v, math.pi / 2) for c, v in zip(ctrl, v0)])


@torch.no_grad()
def predict(net, S, mu, sd, K, seed):
    net.eval()
    g = torch.Generator(device=DEV).manual_seed(seed)
    N = len(S['tokens']); out = np.zeros((N, K, 12, 2), np.float32)
    for i in range(0, N, 512):
        kv = S['kv'][i:i + 512].to(DEV); B = len(kv)
        for k in range(K):
            eps = torch.randn(B, 12, 2, device=DEV, generator=g)
            x = net.sample(kv, S['ctx_len'], eps)
            out[i:i + B, k] = (x.cpu().numpy() * sd + mu)
    net.train()
    return out


def holdout_score(net, H12, mu, sd, P12):
    """median ADE@6s of one sample (seed 0) on holdout n_fut = 12."""
    c = predict(net, H12, mu, sd, 1, 0)
    e = np.linalg.norm(roll_many(c, H12['v0'])[:, 0] - P12, axis=2).mean(1)
    return float(np.median(e)), float(e.mean())


def main():
    torch.manual_seed(42); np.random.seed(42)
    tr, ho, va = (torch.load(DATA.format(s), weights_only=False) for s in ('train', 'holdout', 'val'))
    A = np.stack([np.clip(tr['acc'], -6, 6), np.clip(tr['cur'], -0.3, 0.3)], -1).astype(np.float32)
    mu = A.reshape(-1, 2).mean(0); sd = A.reshape(-1, 2).std(0)
    print(f'[expert] train {len(A)}  mu {mu}  sd {sd}; holdout {len(ho["tokens"])} val {len(va["tokens"])}', flush=True)
    # GT for holdout selection: rollout of the continuous targets == lidar GT P (to float precision)
    okh = ~np.isnan(ho['acc'][:, 0])
    Ph = np.full((len(okh), 12, 2), np.nan)
    Ph[okh] = np.stack([rollout(a, c, v) for a, c, v in zip(ho['acc'][okh], ho['cur'][okh], ho['v0'][okh])])
    H12 = {'kv': ho['kv'][torch.from_numpy(okh)], 'tokens': [t for t, o in zip(ho['tokens'], okh) if o],
           'ctx_len': ho['ctx_len'], 'v0': ho['v0'][okh]}
    P12 = Ph[okh]
    An = torch.tensor((A - mu) / sd, device=DEV)
    KVtr = tr['kv'].to(DEV)
    net = FlowExpert().to(DEV)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95))
    B, E_MAX, EV, PAT = 256, 100, 5, 4
    spe = len(A) // B; total = spe * E_MAX
    lr_at = lambda s: 1e-4 * (s / 500 if s < 500 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 500) / (total - 500))))
    print(f'[expert] params {npar:,}; {spe} steps/epoch; max {E_MAX} epochs', flush=True)
    best, best_ep, bad, step, hist = 1e9, 0, 0, 0, []
    t_start = time.time()
    os.makedirs(CK, exist_ok=True)
    for ep in range(1, E_MAX + 1):
        perm = torch.randperm(len(A), device=DEV); tot = 0.0
        for i in range(spe):
            idx = perm[i * B:(i + 1) * B]
            loss = net.loss(An[idx], KVtr[idx], tr['ctx_len'])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            for g in opt.param_groups: g['lr'] = lr_at(step)
            opt.step(); step += 1; tot += loss.item()
        if ep % EV == 0:
            med, mean = holdout_score(net, H12, mu, sd, P12)
            hist.append({'epoch': ep, 'fm_loss': tot / spe, 'ho_ade_med': med, 'ho_ade_mean': mean,
                         'sec': time.time() - t_start})
            flag = ''
            if med < best:
                best, best_ep, bad, flag = med, ep, 0, ' *'
                torch.save({'state': net.state_dict(), 'mu': mu, 'sd': sd, 'epoch': ep}, f'{CK}/expert_best.pt')
            else:
                bad += 1
            print(f'[epoch {ep:3d}] fm_loss {tot / spe:.4f} | SELECT holdout ADE@6s median {med:.4f} '
                  f'(mean {mean:.4f}){flag} | {time.time() - t_start:.0f}s', flush=True)
            if bad >= PAT:
                break
    train_sec = time.time() - t_start
    ck = torch.load(f'{CK}/expert_best.pt', weights_only=False); net.load_state_dict(ck['state'])
    out = {'best_epoch': best_ep, 'ho_best_med': best, 'hist': hist, 'params': npar,
           'train_gpu_h': train_sec / 3600, 'mu': mu, 'sd': sd}
    t1 = time.time()
    for nm, S, seed in (('holdout', ho, 1000), ('val', va, 2000)):
        c = predict(net, S, mu, sd, 6, seed)
        out[nm] = {'tokens': S['tokens'], 'ctrl': c, 'traj': roll_many(c, S['v0']).astype(np.float32)}
    out['infer_gpu_h'] = (time.time() - t1) / 3600
    pickle.dump(out, open(RES, 'wb'))
    print(f'[expert] best epoch {best_ep} holdout median ADE@6s {best:.4f}; train {train_sec / 3600:.2f} GPU-h, '
          f'6-sample inference holdout+val {out["infer_gpu_h"] * 60:.1f} GPU-min; saved {RES}', flush=True)


if __name__ == '__main__':
    main()
