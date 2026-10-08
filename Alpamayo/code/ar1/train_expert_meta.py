"""ar1/train_expert_meta.py -- R3.3: flow-matching expert on the FROZEN M1 model (8 GPUs).

Same expert and recipe as ar1/train_expert.py (option A per-layer KV, 184M params, OT
path, t ~ U(0,1), fp32, AdamW 1e-4 / wd 0.01 / betas (0.9, 0.95), warmup 500, cosine to
0.1x, clip 1.0, effective batch 256 = 32 x 8 GPUs, targets clipped (accel +-6, curvature
+-0.3) and standardised, eval every 5 epochs, patience 4, max 100 epochs), except:
  context  the M1 sequence = 480 cam + 4 ego + 1 cmd + 18 meta-word tokens (ctx_len 503).
           Its per-layer KV is computed ONLINE by the frozen M1 model (no_grad = stop-grad;
           too large to cache: 28 MB per sample).
  words    TRAINING: the ground-truth meta words (teacher forcing, as AR1 trains the
           expert on labelled reasoning). SELECTION and INFERENCE: the words M1 generated
           itself (txt_gen in results/w1_dump_M1_<split>_f0.0.pkl).
  selection  HOLDOUT median ADE@6s (n_fut = 12) of one fixed-noise sample (seed 0).
Output: results/ar1_expert_M1.pkl (holdout / val: 6 samples, as ar1_expert_A3.pkl);
checkpoint models/checkpoints/_ar1_expert_M1/expert_best.pt.
  python -m torch.distributed.run --nproc_per_node=8 ar1/train_expert_meta.py --tag M1
"""
import os, sys, time, math, pickle, argparse
import numpy as np, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1'); sys.path.insert(0, f'{CODE}/ar1')
import records, vcache
import finetune_meta as FM
from expert import FlowExpert
from train_expert import roll_many
from targets import rollout
from kv_a3 import layer_kv

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
CKR = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'


@torch.no_grad()
def seq_kv(net, vt, ego, txt, dev):
    vla = net.vla
    ctx = vla._build_context(vt.to(dev, torch.float16), ego.to(dev, torch.float32))
    x = torch.cat([ctx, net.emb(txt.to(dev)).to(ctx.dtype)], 1)
    past = vla.cosmos.model.language_model(inputs_embeds=x, use_cache=True).past_key_values
    return torch.stack([torch.stack(layer_kv(past, l), 1) for l in range(28)], 1), x.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='M1')
    ap.add_argument('--bpg', type=int, default=32)
    ap.add_argument('--predict_only', action='store_true',
                    help='reuse expert_best.pt (training finished); history parsed from the train log')
    ap.add_argument('--labels', default='can', choices=('can', '2hz'),
                    help='R9: training words from the 10 Hz CAN labels or the 2 Hz labels (M1-v2a)')
    ap.add_argument('--train_log', default=None, help='R9: train log for --predict_only (default R3.3 name)')
    a = ap.parse_args()
    FM.LABELS['src'] = a.labels
    import datetime
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=3))   # uneven decode times across ranks
    r = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(r); dev = f'cuda:{r}'
    ws = dist.get_world_size(); m0 = r == 0
    log = lambda *x: print(*x, flush=True) if m0 else None
    torch.manual_seed(42)
    net = FM.build(dev, argparse.Namespace(dump=True))
    ck = torch.load(f'{CKR}/_w1_{a.tag}/alpamayo_best.pt', map_location='cpu')
    net.vla.load_state_dict(ck['model_state'], strict=False)
    net.vla.cosmos.model.language_model.gradient_checkpointing_disable(); net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb')); Mt = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb'))
    tr = records.build('train', ['cmd'], 0.0)
    st_ = {'missing': 0}
    TXT = torch.tensor([FM.text_ids(*FM.meta_labels(Mt, t['sample_token'], st_)) for t in tr])
    A = np.stack([np.stack([np.clip(t['acc'][:12], -6, 6), np.clip(t['cur'][:12], -0.3, 0.3)], -1) for t in tr]).astype(np.float32)
    mu = A.reshape(-1, 2).mean(0); sd = A.reshape(-1, 2).std(0)
    An = torch.tensor((A - mu) / sd)
    gen = {s: pickle.load(open(f'{RES}/w1_dump_{a.tag}_{s}_f0.0.pkl', 'rb'))['data'] for s in ('holdout', 'val')}
    ev = {s: records.build(s, ['cmd'], 0.0, n_fut=6) for s in ('holdout', 'val')}
    ho12 = [t for t in ev['holdout'] if t['n_fut'] == 12]
    P12 = np.stack([rollout(t['acc'][:12], t['cur'][:12], t['v0']) for t in ho12])
    exp = FlowExpert().to(dev)
    model = DDP(exp, device_ids=[r])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.95))
    B, E_MAX, EVAL, PAT = a.bpg, 100, 5, 4
    spe = len(tr) // (B * ws); total = spe * E_MAX
    lr_at = lambda s: 1e-4 * (s / 500 if s < 500 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 500) / (total - 500))))
    log(f'[expM] train {len(tr)} (missing tick labels {st_["missing"]}); mu {mu} sd {sd}; {spe} steps/epoch; '
        f'params {sum(p.numel() for p in exp.parameters()):,}')

    def batch_kv(Rs, idx, txt):
        vt = torch.stack([vcache.gather(H[Rs[i]['sample_token']], (3,)) for i in idx])
        ego = torch.stack([Rs[i]['w1_ego'] for i in idx])
        return seq_kv(net, vt, ego, txt, dev)

    @torch.no_grad()
    def predict(Rs, split, K, seed):
        exp.eval(); g = torch.Generator(device=dev).manual_seed(seed + r)
        out = {}
        mine = list(range(r, len(Rs), ws))
        for j in range(0, len(mine), B):
            idx = mine[j:j + B]
            txt = torch.tensor([gen[split][Rs[i]['sample_token']]['txt_gen'] for i in idx])
            kv, C = batch_kv(Rs, idx, txt)
            c = np.zeros((len(idx), K, 12, 2), np.float32)
            for k in range(K):
                x = exp.sample(kv, C, torch.randn(len(idx), 12, 2, device=dev, generator=g))
                c[:, k] = x.cpu().numpy() * sd + mu
            for q, i in enumerate(idx):
                out[i] = c[q]
        parts = [None] * ws
        dist.all_gather_object(parts, out)
        exp.train()
        M = {}
        for p in parts:
            M.update(p)
        return np.stack([M[i] for i in range(len(Rs))])

    best, best_ep, bad, step, hist = 1e9, 0, 0, 0, []
    t_start = time.time(); os.makedirs(f'{CKR}/_ar1_expert_{a.tag}', exist_ok=True)
    if a.predict_only:
        import re
        for l in open(a.train_log or f'/home/dgx1user/Alpamayo-Kushal/Alpamayo/ar1_r33_{a.tag}_expert.log'):
            mm = re.match(r'\[epoch +(\d+)\] fm_loss ([\d.]+) \| SELECT holdout ADE@6s median ([\d.]+) \(mean ([\d.]+)\).* (\d+)s', l)
            if mm:
                hist.append({'epoch': int(mm[1]), 'fm_loss': float(mm[2]), 'ho_ade_med': float(mm[3]),
                             'ho_ade_mean': float(mm[4]), 'sec': float(mm[5])})
        best_h = min(hist, key=lambda h: h['ho_ade_med']); best, best_ep = best_h['ho_ade_med'], best_h['epoch']
        log(f'[expM] predict-only: {len(hist)} evaluations parsed, best epoch {best_ep} ({best:.4f})')
    for ep in range(1, 0 if a.predict_only else E_MAX + 1):
        g = torch.Generator().manual_seed(1000 + ep)
        perm = torch.randperm(len(tr), generator=g)
        tot = 0.0; t_ep = time.time()
        for s in range(spe):
            idx = perm[(s * ws + r) * B:(s * ws + r + 1) * B].tolist()
            kv, C = batch_kv(tr, idx, TXT[idx])
            a0 = An[idx].to(dev)
            eps = torch.randn_like(a0); t = torch.rand(len(idx), device=dev)
            xt = t[:, None, None] * a0 + (1 - t[:, None, None]) * eps
            loss = torch.nn.functional.mse_loss(model(xt, t, kv, C), a0 - eps)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for pg in opt.param_groups: pg['lr'] = lr_at(step)
            opt.step(); step += 1; tot += loss.item()
            if m0 and step == 20:
                log(f'  step 20: {(time.time() - t_ep) / 20:.2f} s/step, peak {torch.cuda.max_memory_allocated(dev) / 1e9:.1f} GB')
        if ep % EVAL == 0:
            c = predict(ho12, 'holdout', 1, 0)
            e = np.linalg.norm(roll_many(c, np.array([t['v0'] for t in ho12]))[:, 0] - P12, axis=2).mean(1)
            med = float(np.median(e)); hist.append({'epoch': ep, 'fm_loss': tot / spe, 'ho_ade_med': med,
                                                    'ho_ade_mean': float(e.mean()), 'sec': time.time() - t_start})
            flag = ''
            if med < best:
                best, best_ep, bad, flag = med, ep, 0, ' *'
                if m0:
                    torch.save({'state': exp.state_dict(), 'mu': mu, 'sd': sd, 'epoch': ep},
                               f'{CKR}/_ar1_expert_{a.tag}/expert_best.pt')
            else:
                bad += 1
            log(f'[epoch {ep:3d}] fm_loss {tot / spe:.4f} | SELECT holdout ADE@6s median {med:.4f} '
                f'(mean {e.mean():.4f}){flag} | {time.time() - t_start:.0f}s')
            if bad >= PAT:
                break
        dist.barrier()
    train_s = hist[-1]['sec'] if a.predict_only else time.time() - t_start
    dist.barrier()
    ck = torch.load(f'{CKR}/_ar1_expert_{a.tag}/expert_best.pt', weights_only=False, map_location='cpu')
    exp.load_state_dict(ck['state'])           # map_location: all ranks loaded onto cuda:0 -> OOM
    assert ck['epoch'] == best_ep, (ck['epoch'], best_ep)
    t1 = time.time()
    out = {'best_epoch': best_ep, 'ho_best_med': best, 'hist': hist, 'params': sum(p.numel() for p in exp.parameters()),
           'train_gpu_h': train_s * ws / 3600, 'mu': mu, 'sd': sd}
    for s, seed in (('holdout', 1000), ('val', 2000)):
        c = predict(ev[s], s, 6, seed)
        out[s] = {'tokens': [t['sample_token'] for t in ev[s]], 'ctrl': c,
                  'traj': roll_many(c, np.array([t['v0'] for t in ev[s]])).astype(np.float32)}
    out['infer_gpu_h'] = (time.time() - t1) * ws / 3600
    if m0:
        pickle.dump(out, open(f'{RES}/ar1_expert_{a.tag}.pkl', 'wb'))
    log(f'[expM] best epoch {best_ep} holdout {best:.4f}; train {out["train_gpu_h"]:.2f} GPU-h; '
        f'inference {out["infer_gpu_h"]:.2f} GPU-h; EXPERT_META_DONE')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
