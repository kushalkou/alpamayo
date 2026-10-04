"""ar1/finetune_meta.py -- R3.3 "META-ACTION + TRAJ" arm (AR1 Table 6 analog) on the B1 base.

Context  = B1: 3 front cams at t0 (cached native-order tokens, 480) + 4 ego + cmd [P].
Target   = 18 TEXT tokens from the frozen LLM vocabulary, then the 24 trajectory tokens:
  6 lon words, one per second t0+1 .. t0+6 s (R1.2 10 Hz labels at ticks 10, 20, .., 60):
    gentle_acc ' accelerate' | strong_acc ' surge' | gentle_dec ' slow' | strong_dec ' brake'
    | maintain ' maintain' | stop ' stop' | reverse ' reverse'                (1 token each)
  6 lat words, same ticks, 2 tokens each (fixed format, so every class has one form):
    steer_L ' gentle left' | steer_R ' gentle right' | sharp_L ' sharp left' | sharp_R
    ' sharp right' | reverse_L ' back left' | reverse_R ' back right' | straight
    ' keep straight'
  A missing tick label (-1) takes the previous valid tick of the same sample, else
  maintain / straight (counted in the log).
Training: teacher forcing; plain CE on ALL 42 target tokens, mean per token. Text logits
  come from the frozen Cosmos lm_head (full vocabulary, fp32 CE), trajectory logits from the
  trained output_head; text inputs embedded by the frozen embed_tokens. Everything else is
  the A3 recipe exactly as finetune_w1 --fix --cmd --turn_weighted: LoRA r16 q/v/o, fix ego
  (standardise + LayerNorm + gain, ego-MLP lr x10 via LrMultAdamW; cmd token at base lr), lr 5e-5 warmup 100
  cosine to 0.1x, wd 0.05, clip 1.0, GradScaler, batch 3 x 8 GPUs, turn-weighted sampler
  (target 0.40), 10 epochs, patience 5.
Selection: AR median ADE@6s on the fixed 400 holdout samples (fixed_val_indices, seed 1234;
  n_fut = 12 holdout, as finetune_w1), where the model FIRST generates its meta words
  (constrained decoding) and THEN the trajectory (argmax), STOP -> (0, 0), rollout from
  v0_can. Never the teacher-forced loss; official val never.
Inference (dump): constrained greedy decoding of the words (lon slot: argmax over the 7 lon
  words; lat slot: first token argmax over {gentle, sharp, back, keep}, second over the
  allowed directions); the unconstrained argmax is recorded too. Then the trajectory as in
  dump_w1 (argmax / STOP-aware expectation / p_stop per slot) -> results/w1_dump_<tag>_<split>
  _f0.0.pkl, readable by gate_a.load / preds, plus 'meta_gen', 'meta_gt', 'meta_free_ok'.
  python -m torch.distributed.run --nproc_per_node=8 ar1/finetune_meta.py --tag M1 --seed 42
  python -m torch.distributed.run --nproc_per_node=8 ar1/finetune_meta.py --tag M1 --dump holdout,val
"""
import os, sys, time, math, pickle, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1'); sys.path.insert(0, f'{CODE}/ar1')
import records, extra_tokens, fixrun, vcache
from model import load_model, TRAJ_LEN
from finetune import CFG, get_lr, build_turn_weights, DistributedWeightedSampler
from ar_eval import fixed_val_indices, unicycle_rollout
from inference import slot_centers, N_ACT, STOP_ID
from w1tok import W1Tokenizer

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
LON_W = [42780, 21781, 6301, 34618, 10306, 2936, 9931]          # accelerate surge slow brake maintain stop reverse
LAT_A = {'gentle': 21700, 'sharp': 17232, 'back': 1182, 'keep': 2506}
LAT_B = {'left': 2115, 'right': 1290, 'straight': 7678}
LAT_W = [('gentle', 'left'), ('gentle', 'right'), ('sharp', 'left'), ('sharp', 'right'),
         ('back', 'left'), ('back', 'right'), ('keep', 'straight')]   # R1.2 LAT order
ALLOWED_B = {'gentle': ('left', 'right'), 'sharp': ('left', 'right'), 'back': ('left', 'right'),
             'keep': ('straight',)}
N_TXT = 6 + 12
TICKS = (10, 20, 30, 40, 50, 60)


def meta_labels(Mt, st, stats):
    out = []
    for key, dflt in (('lon', 4), ('lat', 6)):
        seq, last = [], None
        for j in TICKS:
            v = int(Mt[st][key][j])
            if v < 0:
                stats['missing'] += 1
                v = last if last is not None else dflt
            seq.append(v); last = v
        out.append(seq)
    return out                                              # [lon x6], [lat x6]


def text_ids(lon, lat):
    ids = [LON_W[c] for c in lon]
    for c in lat:
        a, b = LAT_W[c]; ids += [LAT_A[a], LAT_B[b]]
    return ids


class DS(torch.utils.data.Dataset):
    def __init__(self, R, H, Mt, stats):
        self.R, self.H = R, H
        self.meta = [meta_labels(Mt, t['sample_token'], stats) for t in R]

    def __len__(self):
        return len(self.R)

    def __getitem__(self, i):
        t = self.R[i]
        lon, lat = self.meta[i]
        tok = [p[0] for p in t['w1_tokens']] + [p[1] for p in t['w1_tokens']] if len(t['w1_tokens']) == 12 \
            else [0] * 24
        return {'vt': vcache.gather(self.H[t['sample_token']], (3,)), 'ego': t['w1_ego'],
                'txt': torch.tensor(text_ids(lon, lat)), 'tok': torch.tensor(tok), 'i': i,
                'lon': torch.tensor(lon), 'lat': torch.tensor(lat)}


class Meta(nn.Module):
    """wraps the VLA; forward = teacher-forced logits for the 18 text + 24 traj targets."""
    def __init__(self, vla):
        super().__init__()
        self.vla = vla
        self.emb = vla.cosmos.model.language_model.embed_tokens
        self.lm_head = vla.cosmos.lm_head

    def forward(self, vt, ego, txt, tok):
        ctx = self.vla._build_context(vt.to(torch.float16), ego)
        C = ctx.shape[1]
        te = self.emb(txt).to(ctx.dtype)
        tr = self.vla.traj_embed(tok[:, :-1]).to(ctx.dtype)
        h = self.vla.cosmos.model.language_model(inputs_embeds=torch.cat([ctx, te, tr], 1),
                                                 use_cache=False).last_hidden_state
        lt = self.lm_head(h[:, C - 1:C - 1 + N_TXT]).float()                  # [B,18,V]
        lj = self.vla.output_head(h[:, C - 1 + N_TXT:C - 1 + N_TXT + TRAJ_LEN].float())  # [B,24,129]
        return lt, lj


def build(dev, a):
    vla = load_model(lora_rank=CFG['lora_rank'], lora_alpha=CFG['lora_alpha'],
                     lora_dropout=CFG['lora_dropout'], device=dev)
    vla = extra_tokens.install(vla, ['cmd'], records.N_CLS)
    if a.dump:
        vla = fixrun.install_fix(vla, torch.zeros(4), torch.ones(4))
    else:
        mean, sd = fixrun.train_stats()
        vla = fixrun.install_fix(vla, mean, sd)
    return Meta(vla)


@torch.no_grad()
def decode(net, vt, ego, tk, dev, gt=None, full=False):
    """constrained word decoding, then trajectory; returns dict (dump_w1 fields if full)."""
    vla = net.vla; lm = vla.cosmos.model.language_model
    ctx = vla._build_context(vt.to(dev, torch.float16).unsqueeze(0), ego.to(dev, torch.float32).unsqueeze(0))
    o = lm(inputs_embeds=ctx, use_cache=True); past = o.past_key_values; h = o.last_hidden_state[:, -1]
    words, free_ok, txt = [], [], []

    def step(token_id):
        nonlocal past, h
        o = lm(inputs_embeds=net.emb(torch.tensor([[token_id]], device=dev)).to(ctx.dtype),
               past_key_values=past, use_cache=True)
        past = o.past_key_values; h = o.last_hidden_state[:, -1]

    for s in range(6):                                       # lon words
        lg = net.lm_head(h).float()[0]
        c = int(torch.argmax(lg[LON_W]).item())
        free_ok.append(int(lg.argmax().item()) == LON_W[c]); words.append(c); txt.append(LON_W[c]); step(LON_W[c])
    for s in range(6):                                       # lat words (2 tokens)
        lg = net.lm_head(h).float()[0]
        ka = list(LAT_A); ia = [LAT_A[k] for k in ka]
        A = ka[int(torch.argmax(lg[ia]).item())]
        f1 = int(lg.argmax().item()) == LAT_A[A]; step(LAT_A[A]); txt.append(LAT_A[A])
        lg = net.lm_head(h).float()[0]
        kb = ALLOWED_B[A]; ib = [LAT_B[k] for k in kb]
        Bw = kb[int(torch.argmax(lg[ib]).item())]
        free_ok.append(f1 and int(lg.argmax().item()) == LAT_B[Bw]); step(LAT_B[Bw]); txt.append(LAT_B[Bw])
        words.append(LAT_W.index((A, Bw)))
    logits = vla.output_head(h.float())[0]
    av, ev, ps, at, lp = [], [], [], [], []
    for stp in range(TRAJ_LEN):
        cen = torch.as_tensor(slot_centers(tk, stp), dtype=torch.float32, device=dev)
        tid = int(logits.argmax(-1).item())
        vlog = torch.cat([logits[:N_ACT], logits[STOP_ID:STOP_ID + 1]])
        q = torch.softmax(vlog, -1)
        ev.append(float((q * torch.cat([cen, torch.zeros(1, device=dev)])).sum()))
        ps.append(float(q[-1]))
        av.append(0.0 if tid == STOP_ID else float(cen[min(max(tid, 0), N_ACT - 1)]))
        at.append(tid)
        if gt is not None:
            g = gt[stp]; lp.append(float(torch.log_softmax(vlog, -1)[N_ACT if g == STOP_ID else g]))
        if stp == TRAJ_LEN - 1:
            break
        o = lm(inputs_embeds=vla.traj_embed(torch.tensor([tid], device=dev)).unsqueeze(1).to(ctx.dtype),
               past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = vla.output_head(o.last_hidden_state[:, -1].float())[0]
    return {'argmax': av, 'expect': ev, 'p_stop': ps, 'tok': at, 'lp65': lp, 'gt_tok': gt,
            'meta_gen': words, 'meta_free_ok': free_ok, 'txt_gen': txt}


def ade6_select(net, R, H, idx, dev, tk, rank, ws):
    net.eval()
    out = {}
    for i in idx[rank::ws]:
        t = R[int(i)]
        if len(t['future_positions']) < 12:
            continue
        d = decode(net, vcache.gather(H[t['sample_token']], (3,)), t['w1_ego'], tk, dev)
        a = np.array(d['argmax'])
        acc, cur = a[:12], a[12:]
        P, _ = unicycle_rollout(acc, cur, float(t['future_speeds'][0]), float(t['w1_ego'][3, 1]))
        out[int(i)] = float(np.linalg.norm(P - np.asarray(t['future_positions'])[:12], axis=1).mean())
    parts = [None] * ws
    dist.all_gather_object(parts, out)
    net.train()
    v = [x for p in parts for x in p.values()]
    return float(np.median(v)), float(np.mean(v)), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=3)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--dump', default='')
    a = ap.parse_args()
    dist.init_process_group('nccl')
    r = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(r); dev = f'cuda:{r}'
    ws = dist.get_world_size(); m0 = r == 0
    log = lambda *x: print(*x, flush=True) if m0 else None
    torch.manual_seed(a.seed + r)
    net = build(dev, a)
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb'))
    Mt = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb'))
    tk = W1Tokenizer()
    ckdir = f'{CK}/_w1_{a.tag}'
    if a.dump:
        ck = torch.load(f'{ckdir}/alpamayo_best.pt', map_location='cpu')
        miss = [k for k in ck['model_state'] if k not in net.vla.state_dict()]
        assert not miss, miss[:5]
        net.vla.load_state_dict(ck['model_state'], strict=False)
        net.vla.cosmos.model.language_model.gradient_checkpointing_disable(); net.eval()
        for split in a.dump.split(','):
            R = records.build(split, ['cmd'], 0.0, n_fut=6)
            stats = {'missing': 0}
            ds = DS(R, H, Mt, stats)
            out = {}; t0 = time.time()
            for c, i in enumerate(range(r, len(R), ws)):
                t = R[i]
                gt = [x for x, _ in t['w1_tokens']] + [k for _, k in t['w1_tokens']] if len(t['w1_tokens']) == 12 else None
                d = decode(net, vcache.gather(H[t['sample_token']], (3,)), t['w1_ego'], tk, dev, gt)
                d['meta_gt'] = ds.meta[i][0] + ds.meta[i][1]
                out[t['sample_token']] = d
                if m0 and c % 100 == 0:
                    log(f'  [{split}] {c}/{len(R) // ws} {time.time() - t0:.0f}s')
            parts = [None] * ws
            dist.all_gather_object(parts, out)
            if m0:
                M = {}
                for p in parts:
                    M.update(p)
                op = f'{RES}/w1_dump_{a.tag}_{split}_f0.0.pkl'
                pickle.dump({'order': [t['sample_token'] for t in R], 'data': M, 'ckpt_epoch': ck.get('epoch'),
                             'extra': ['cmd'], 'meta_missing_ticks': stats['missing']}, open(op, 'wb'))
                log(f'[meta] saved {op} n={len(M)}')
            dist.barrier()
        log('META_DUMP_DONE'); dist.destroy_process_group(); return

    tr = records.build('train', ['cmd'], 0.0)
    ho = records.build('holdout', ['cmd'], 0.0)
    stats = {'missing': 0}
    dtr = DS(tr, H, Mt, stats)
    weights, f0, wt = build_turn_weights(tr, CFG['turn_thresh'], CFG['turn_target_frac'])
    samp = DistributedWeightedSampler(weights, ws, r, seed=a.seed)
    dl = torch.utils.data.DataLoader(dtr, batch_size=a.batch_size, sampler=samp, num_workers=4,
                                     pin_memory=True, drop_last=True)
    model = DDP(net, device_ids=[r])
    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = fixrun.LrMultAdamW(trainable, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    spe = len(dl); total = spe * a.epochs
    idx = fixed_val_indices(len(ho), k=CFG['val_ade_k'], seed=CFG['val_ade_seed'])
    log(f'[meta] {a.tag} seed {a.seed}: train {len(tr)} holdout {len(ho)}; missing tick labels {stats["missing"]}; '
        f'turn frac {f0:.3f} -> 0.40 (w {wt:.2f}); {spe} steps/epoch; trainable '
        f'{sum(q.numel() for q in trainable):,}; select on {len(idx)} holdout')
    best, bad, step = 1e9, 0, 0
    os.makedirs(ckdir, exist_ok=True)
    for ep in range(a.epochs):
        samp.set_epoch(ep); model.train(); tot = tt = tj = 0.0; t_s = time.time(); t_ep = time.time()
        torch.cuda.reset_peak_memory_stats(dev)
        for bi, b in enumerate(dl):
            lt, lj = model(b['vt'].to(dev), b['ego'].to(dev, torch.float32), b['txt'].to(dev), b['tok'].to(dev))
            ct = F.cross_entropy(lt.reshape(-1, lt.shape[-1]), b['txt'].to(dev).reshape(-1), reduction='sum')
            cj = F.cross_entropy(lj.reshape(-1, lj.shape[-1]), b['tok'].to(dev).reshape(-1), reduction='sum')
            n = b['txt'].numel() + b['tok'].numel()
            loss = (ct + cj) / n
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            for g in opt.param_groups:
                g['lr'] = get_lr(step, total, CFG)
            scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
            tot += loss.item(); tt += ct.item() / b['txt'].numel(); tj += cj.item() / b['tok'].numel()
            if m0 and (step % 50 == 0 or a.max_steps):
                log(f'  step {step:6d} ep {ep + 1} | loss {loss.item():.4f} avg {tot / (bi + 1):.4f} '
                    f'(text {tt / (bi + 1):.4f} traj {tj / (bi + 1):.4f}) | lr {get_lr(step, total, CFG):.2e} | '
                    f'scale {scaler.get_scale():.0f} | {(time.time() - t_s) / (50 if not a.max_steps else 1):.2f}s/step | '
                    f'peak {torch.cuda.max_memory_allocated(dev) / 1e9:.1f}GB')
                t_s = time.time()
            if a.max_steps and step >= a.max_steps:
                break
        raw = model.module
        raw.vla.cosmos.model.language_model.gradient_checkpointing_disable()
        med, mean, n = ade6_select(raw, ho, H, idx[:40] if a.max_steps else idx, dev, tk, r, ws)
        raw.vla.cosmos.model.language_model.gradient_checkpointing_enable()
        log(f'[epoch {ep + 1}] SELECT holdout AR ADE@6s median={med:.4f} (mean={mean:.4f}, n={n}) '
            f'best={min(best, med):.4f} | train loss {tot / (bi + 1):.4f} (text {tt / (bi + 1):.4f} '
            f'traj {tj / (bi + 1):.4f}) | {time.time() - t_ep:.0f}s')
        if med < best:
            best, bad = med, 0
            if m0:
                st = {k: v for k, v in raw.vla.state_dict().items()
                      if 'lora_' in k or any(k.startswith(p) for p in ('ego_encoder', 'traj_embed', 'output_head'))}
                torch.save({'epoch': ep + 1, 'val_ade6': med, 'model_state': st, 'seed': a.seed},
                           f'{ckdir}/alpamayo_best.pt')
                log(f'[ckpt] saved best epoch {ep + 1}')
        else:
            bad += 1
        if bad >= a.patience or a.max_steps:
            break
    log(f'[meta] done; best holdout median ADE@6s {best:.4f}')
    log('META_TRAIN_DONE')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
