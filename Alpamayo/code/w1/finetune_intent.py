"""w1/finetune_intent.py -- QUEUE v2 G8: VLA INTENT PREDICTOR (decision bottleneck, stage 1).

Same backbone (Cosmos-Reason1-7B, frozen fp16, LoRA r16 on q/v/o) + the FIX recipe
(standardised ego, post-MLP LayerNorm + gain, ego-MLP lr x10; w1/fixrun.py) + the cmd
token (w1/extra_tokens.py). Output = lon meta-action (4 classes: stop / accelerate /
decelerate / maintain; records.meta_lon, the A2 rule, unchanged) from a NEW linear head
(3584 -> 4, fp32, in the x10 lr group like the ego MLP, as it is also a fresh module) on
the LM hidden state at the LAST CONTEXT POSITION (the cmd token). No trajectory tokens
are fed; traj_embed / output_head are unused and frozen. Plain CE, natural sampling.
  V8a  ego + cmd, NO cameras (visual tokens removed, --no_vision)   learner control
  V8b  6 cameras + ego + cmd (live-encoded, 1536 visual tokens)     the claim
Recipe otherwise as the trajectory runs: lr 5e-5 (warmup 100, cosine to 0.1x), AdamW wd
0.05, clip 1.0, GradScaler, batch 3 x 8 GPUs, lora_dropout 0.1, seed 42.
Data: train = official train minus the 50 holdout scenes (n_fut = 12, as the trajectory
runs; 18,313); selection = macro-F1 on the FULL holdout (n_fut >= 6; true labels) after
every epoch; patience 5. Official val is only predicted once, with the selected weights.
Output: results/w1_intent_<tag>.pkl  per-class probabilities for holdout and val
(order = records.build(split, n_fut=6)), labels, per-epoch log; checkpoint
models/checkpoints/_w1_<tag>/intent_best.pt.
  python -m torch.distributed.run --nproc_per_node=8 w1/finetune_intent.py --tag V8a --no_vision
"""
import os, sys, time, math, pickle, argparse
import numpy as np, torch, torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1')
import records, extra_tokens, fixrun
from model import load_model
from finetune import CFG, get_lr, encode_live
from vision_live import preprocess_image, CAMERAS

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
NCLS = 4


class DS(torch.utils.data.Dataset):
    def __init__(self, R, vision):
        self.R, self.vision = R, vision

    def __len__(self):
        return len(self.R)

    def __getitem__(self, i):
        t = self.R[i]
        img = (torch.stack([preprocess_image(t['cam_paths'][c], augment=False) for c in CAMERAS])
               if self.vision else torch.zeros(1))
        return {'images': img, 'ego': t['w1_ego'], 'y': t['meta_lon'], 'i': i}


class Intent(nn.Module):
    def __init__(self, vla):
        super().__init__()
        self.vla = vla
        self.head = nn.Linear(vla.output_head.in_features, NCLS)

    def forward(self, vt, ego):
        ctx = self.vla._build_context(vt, ego)
        h = self.vla.cosmos.model.language_model(inputs_embeds=ctx, use_cache=False).last_hidden_state
        return self.head(h[:, -1, :].float())


def macro_f1(y, p):
    f = []
    for c in range(NCLS):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--no_vision', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=3)
    ap.add_argument('--max_steps', type=int, default=0, help='smoke: stop after N steps')
    a = ap.parse_args()
    vision = not a.no_vision

    dist.init_process_group(backend='nccl')
    lr_ = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr_)
    dev = f'cuda:{lr_}'; ws = dist.get_world_size(); m0 = lr_ == 0
    torch.manual_seed(a.seed + lr_)
    log = lambda *x: print(*x, flush=True) if m0 else None

    vla = load_model(lora_rank=CFG['lora_rank'], lora_alpha=CFG['lora_alpha'],
                     lora_dropout=CFG['lora_dropout'], device=dev)
    vla = extra_tokens.install(vla, ['cmd'], records.N_CLS)
    mean, sd = fixrun.train_stats()
    vla = fixrun.install_fix(vla, mean, sd)
    for q in list(vla.traj_embed.parameters()) + list(vla.output_head.parameters()):
        q.requires_grad_(False)
    torch.manual_seed(a.seed)                       # identical head init on every rank
    net = Intent(vla)
    nn.init.normal_(net.head.weight, std=0.02); nn.init.zeros_(net.head.bias)
    net.head.to(dev, torch.float32)
    for q in net.head.parameters():
        fixrun._EGO_PARAM_IDS.add(id(q))
    torch.manual_seed(a.seed + lr_)
    visual = vla.cosmos.model.visual
    model = DDP(net, device_ids=[lr_])
    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = fixrun.LrMultAdamW(trainable, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    log(f'[intent] {a.tag} vision={vision} trainable {sum(q.numel() for q in trainable):,} '
        f'(head + ego lr x{fixrun.EGO_LR_MULT})')

    tr = records.build('train', ['cmd'], 0.0)
    ho = records.build('holdout', ['cmd'], 0.0, n_fut=6)
    va = records.build('val', ['cmd'], 0.0, n_fut=6)
    ytr = np.array([t['meta_lon'] for t in tr])
    log(f'[intent] train {len(tr)} holdout {len(ho)} val {len(va)}; train class freq '
        + ' '.join(f'{records.LON[c]}={np.mean(ytr == c):.3f}' for c in range(NCLS)))
    samp = torch.utils.data.distributed.DistributedSampler(DS(tr, vision), shuffle=True, seed=a.seed)
    dl = torch.utils.data.DataLoader(DS(tr, vision), batch_size=a.batch_size, sampler=samp,
                                     num_workers=4, pin_memory=True, drop_last=True)
    spe = len(dl); total = spe * a.epochs

    def ctx_vis(images, n):
        if not vision:
            return torch.zeros(n, 0, 3584, dtype=torch.float16, device=dev)
        return encode_live(visual, images, dev)

    @torch.no_grad()
    def predict(R):
        model.eval()
        mine = list(range(len(R)))[lr_::ws]
        sub = torch.utils.data.Subset(DS(R, vision), mine)
        out = {}
        for b in torch.utils.data.DataLoader(sub, batch_size=6, num_workers=4):
            with torch.autocast('cuda', enabled=False):
                lg = net(ctx_vis(b['images'], len(b['i'])), b['ego'].to(dev, torch.float32))
            for i, p in zip(b['i'].tolist(), torch.softmax(lg.float(), -1).cpu().numpy()):
                out[i] = p
        parts = [None] * ws
        dist.all_gather_object(parts, out)
        model.train()
        M = {}
        for p in parts:
            M.update(p)
        return np.stack([M[i] for i in range(len(R))])

    yho = np.array([t['meta_lon'] for t in ho])
    best, best_ep, bad, best_state, hist = -1.0, 0, 0, None, []
    step = 0
    model.train()
    for ep in range(a.epochs):
        samp.set_epoch(ep); t_ep = time.time(); t_s = time.time(); tot = 0.0
        for bi, b in enumerate(dl):
            vt = ctx_vis(b['images'], a.batch_size)
            lg = model(vt, b['ego'].to(dev, torch.float32))
            loss = nn.functional.cross_entropy(lg.float(), b['y'].to(dev))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            for g in opt.param_groups:
                g['lr'] = get_lr(step, total, CFG)
            scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
            tot += loss.item()
            if step % 50 == 0:
                sps = (time.time() - t_s) / 50; t_s = time.time()
                log(f'  step {step:6d} ep {ep+1} | loss {loss.item():.4f} avg {tot/(bi+1):.4f} | '
                    f'lr {get_lr(step, total, CFG):.2e} | scale {scaler.get_scale():.0f} | {sps:.2f}s/step | '
                    f'peak {torch.cuda.max_memory_allocated(dev)/1e9:.1f}GB')
                if step == 100:
                    log(f'[intent] ESTIMATE: {sps:.2f} s/step x {spe} steps/epoch = '
                        f'{sps*spe/3600:.2f} h/epoch wall = {8*sps*spe/3600:.1f} GPU-h/epoch; '
                        f'{a.epochs} epochs max = {a.epochs*sps*spe/3600:.1f} h wall, '
                        f'{8*a.epochs*sps*spe/3600:.0f} GPU-h (+ holdout eval per epoch)')
            if a.max_steps and step >= a.max_steps:
                break
        ph = predict(ho); pr = ph.argmax(1)
        f1, acc = macro_f1(yho, pr), float((pr == yho).mean())
        hist.append({'epoch': ep + 1, 'train_loss': tot / (bi + 1), 'ho_macro_f1': f1, 'ho_acc': acc,
                     'sec': time.time() - t_ep})
        log(f'[epoch {ep+1}] SELECT holdout macro-F1 {f1:.4f} (acc {acc:.4f}) train_loss {tot/(bi+1):.4f} '
            f'best {max(best, f1):.4f}  {time.time()-t_ep:.0f}s')
        if f1 > best:
            best, best_ep, bad = f1, ep + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.named_parameters() if v.requires_grad}
            best_state.update({k: v.cpu().clone() for k, v in net.named_buffers() if 'ego_encoder' in k})
            if m0:
                os.makedirs(f'{CK}/_w1_{a.tag}', exist_ok=True)
                torch.save({'epoch': ep + 1, 'ho_macro_f1': f1, 'state': best_state},
                           f'{CK}/_w1_{a.tag}/intent_best.pt')
        else:
            bad += 1
            log(f'[epoch {ep+1}] no macro-F1 improvement ({bad}/{a.patience})')
        if bad >= a.patience or a.max_steps:
            break
    net.load_state_dict(best_state, strict=False)
    log(f'[intent] best epoch {best_ep} holdout macro-F1 {best:.4f}; predicting holdout + val')
    out = {'tag': a.tag, 'vision': vision, 'best_epoch': best_ep, 'hist': hist}
    for nm, R in (('holdout', ho), ('val', va)):
        P = predict(R)
        out[nm] = {'order': [t['sample_token'] for t in R], 'probs': P,
                   'y': np.array([t['meta_lon'] for t in R])}
        log(f'[intent] {nm}: macro-F1 {macro_f1(out[nm]["y"], P.argmax(1)):.4f} acc '
            f'{float((P.argmax(1) == out[nm]["y"]).mean()):.4f}')
    if m0 and not a.max_steps:
        pickle.dump(out, open(f'{RES}/w1_intent_{a.tag}.pkl', 'wb'))
        log(f'[intent] saved {RES}/w1_intent_{a.tag}.pkl')
    log('INTENT_DONE')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
