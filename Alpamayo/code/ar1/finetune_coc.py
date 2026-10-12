"""ar1/finetune_coc.py -- R11 Part B / R7B: "CoC + meta + traj" arm (AR1 Table 6 "CoC & Traj").

M1-v2a recipe (ar1/finetune_meta.py, --labels 2hz, cmd [P], B1 context) with the CoC text
of data/ar1_coc_2hz.pkl inserted BEFORE the meta-action words (AR1 order: reasoning, then
words, then trajectory). Sequence per sample:
  [480 cam + 4 ego + 1 cmd] [CoC tokens + <|im_end|>] [18 word tokens] [24 traj tokens]
CoC = Qwen tokenizer ids of the pickle's 'coc' string ("Driving decision: <lon>; <lat>.
Critical components: <causes>. Reasoning: <trace>") + <|im_end|> (151645) as terminator.
Variable length: each sequence is RIGHT-padded after its trajectory, so causal attention
and positions of every real token equal the unpadded sequence (no attention mask needed).
Loss: plain CE, mean over ALL target tokens (CoC + terminator + 18 words + 24 traj), as
M1 (mean per token). Everything else as M1-v2a (LoRA, fix ego, lr, sampler, 10 epochs,
patience 5, selection = holdout AR median ADE@6s on the fixed 400, the model GENERATING its
CoC (greedy, <= 200 tokens) then its words then the trajectory).
Inference modes (--mode): gen (i) CoC generated greedily; none (ii) no CoC, words straight
after the context; shuf (iii) teacher-forced CoC of a donor val sample with the same GT
(lon, lat) decision (RandomState(11) derangement within class; singleton classes keep their
own); gt (iv) GT CoC teacher-forced [ORACLE]. Words (constrained) and trajectory are always
generated. Dumps: results/w1_dump_<tag>[_<mode>]_<split>_f0.0.pkl (gen keeps the plain tag)
with gate_a fields + meta_gen, txt_gen, coc_gen (decoded text), coc_len.
  python -m torch.distributed.run --nproc_per_node=8 ar1/finetune_coc.py --tag COC --seed 42
  python -m torch.distributed.run --nproc_per_node=8 ar1/finetune_coc.py --tag COC --dump holdout,val --mode gen
"""
import os, sys, time, pickle, argparse, datetime, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1'); sys.path.insert(0, f'{CODE}/ar1')
import records, fixrun
import finetune_meta as FM
from model import TRAJ_LEN
from finetune import CFG, get_lr, build_turn_weights, DistributedWeightedSampler
from ar_eval import fixed_val_indices, unicycle_rollout
from inference import slot_centers, N_ACT, STOP_ID
from w1tok import W1Tokenizer

DATA, RES, CK = FM.DATA, FM.RES, FM.CK
END = 151645                       # <|im_end|>
MAXGEN = 200
TOKP = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'


def coc_ids(tk_text, coc):
    return {st: tk_text(x['coc'])['input_ids'] + [END] for st, x in coc.items()}


def shuffled_text(coc, toks):
    """(iii): donor of the same GT (lon, lat), RandomState(11) derangement within class."""
    cls = collections.defaultdict(list)
    for t in sorted(toks):
        cls[(coc[t]['lon'], coc[t]['lat'])].append(t)
    rng = np.random.RandomState(11); out = {}; single = 0
    for k in sorted(cls):
        m = cls[k]
        if len(m) == 1:
            out[m[0]] = coc[m[0]]['coc']; single += 1; continue
        p = rng.permutation(len(m))
        while (p == np.arange(len(m))).any():
            p = rng.permutation(len(m))
        for a, b in zip(m, p):
            d = m[b]; head = f'Driving decision: {coc[a]["lon"]}; {coc[a]["lat"]}. '
            assert coc[d]['coc'].startswith(head) and coc[a]['coc'].startswith(head)
            out[a] = head + coc[d]['coc'][len(head):]
    return out, single


class DSC(FM.DS):
    def __init__(self, R, H, Mt, stats, ids, keep=None):
        super().__init__(R, H, Mt, stats); self.ids = ids; self.keep = keep   # R13 AUX: per-sample CoC mask

    def __getitem__(self, i):
        d = super().__getitem__(i)
        if self.keep is not None and not self.keep[i]:
            d['coc'] = torch.zeros(0, dtype=torch.long)                         # words follow the context
        else:
            d['coc'] = torch.tensor(self.ids[self.R[i]['sample_token']])
        return d


def collate(B):
    L = torch.tensor([len(b['coc']) for b in B]); C = torch.full((len(B), int(L.max())), END, dtype=torch.long)
    for j, b in enumerate(B):
        C[j, :len(b['coc'])] = b['coc']
    return {'vt': torch.stack([b['vt'] for b in B]), 'ego': torch.stack([b['ego'] for b in B]),
            'txt': torch.stack([b['txt'] for b in B]), 'tok': torch.stack([b['tok'] for b in B]), 'coc': C, 'L': L}


class CoC(FM.Meta):
    def forward(self, vt, ego, coc, L, txt, tok):
        ctx = self.vla._build_context(vt.to(torch.float16), ego)
        B, C, D = ctx.shape
        ce = self.emb(coc).to(ctx.dtype); we = self.emb(txt).to(ctx.dtype); tr = self.vla.traj_embed(tok[:, :-1]).to(ctx.dtype)
        n = C + int(L.max()) + FM.N_TXT + TRAJ_LEN - 1
        x = ctx.new_zeros(B, n, D)
        for b in range(B):
            l = int(L[b]); s = torch.cat([ctx[b], ce[b, :l], we[b], tr[b]], 0); x[b, :len(s)] = s
        h = self.vla.cosmos.model.language_model(inputs_embeds=x, use_cache=False).last_hidden_state
        lt, lj = [], []
        for b in range(B):
            l = int(L[b])
            lt.append(self.lm_head(h[b, C - 1:C - 1 + l + FM.N_TXT]).float())
            lj.append(self.vla.output_head(h[b, C - 1 + l + FM.N_TXT:C - 1 + l + FM.N_TXT + TRAJ_LEN].float()))
        return lt, torch.stack(lj)


def build(dev, a):
    return CoC(FM.build(dev, a).vla)


@torch.no_grad()
def decode(net, vt, ego, tk, dev, mode='gen', forced=None, gt=None):
    vla = net.vla; lm = vla.cosmos.model.language_model
    ctx = vla._build_context(vt.to(dev, torch.float16).unsqueeze(0), ego.to(dev, torch.float32).unsqueeze(0))
    o = lm(inputs_embeds=ctx, use_cache=True); past = o.past_key_values; h = o.last_hidden_state[:, -1]
    cg = []
    if mode == 'gen':
        for _ in range(MAXGEN):
            t = int(net.lm_head(h).float()[0].argmax().item()); cg.append(t)
            o = lm(inputs_embeds=net.emb(torch.tensor([[t]], device=dev)).to(ctx.dtype), past_key_values=past, use_cache=True)
            past = o.past_key_values; h = o.last_hidden_state[:, -1]
            if t == END:
                break
        if cg[-1] != END:                    # length cap: close the reasoning so words follow the terminator
            o = lm(inputs_embeds=net.emb(torch.tensor([[END]], device=dev)).to(ctx.dtype), past_key_values=past, use_cache=True)
            past = o.past_key_values; h = o.last_hidden_state[:, -1]
    elif mode in ('shuf', 'gt'):
        o = lm(inputs_embeds=net.emb(torch.tensor([forced], device=dev)).to(ctx.dtype), past_key_values=past, use_cache=True)
        past = o.past_key_values; h = o.last_hidden_state[:, -1]
    words, txt = [], []

    def step(t):
        nonlocal past, h
        o = lm(inputs_embeds=net.emb(torch.tensor([[t]], device=dev)).to(ctx.dtype), past_key_values=past, use_cache=True)
        past = o.past_key_values; h = o.last_hidden_state[:, -1]
    for s in range(6):
        lg = net.lm_head(h).float()[0]; c = int(torch.argmax(lg[FM.LON_W]).item())
        words.append(c); txt.append(FM.LON_W[c]); step(FM.LON_W[c])
    for s in range(6):
        lg = net.lm_head(h).float()[0]; ka = list(FM.LAT_A); A = ka[int(torch.argmax(lg[[FM.LAT_A[k] for k in ka]]).item())]
        step(FM.LAT_A[A]); txt.append(FM.LAT_A[A])
        lg = net.lm_head(h).float()[0]; kb = FM.ALLOWED_B[A]; Bw = kb[int(torch.argmax(lg[[FM.LAT_B[k] for k in kb]]).item())]
        step(FM.LAT_B[Bw]); txt.append(FM.LAT_B[Bw]); words.append(FM.LAT_W.index((A, Bw)))
    logits = vla.output_head(h.float())[0]
    av, ev, ps, at = [], [], [], []
    for stp in range(TRAJ_LEN):
        cen = torch.as_tensor(slot_centers(tk, stp), dtype=torch.float32, device=dev)
        tid = int(logits.argmax(-1).item())
        q = torch.softmax(torch.cat([logits[:N_ACT], logits[STOP_ID:STOP_ID + 1]]), -1)
        ev.append(float((q * torch.cat([cen, torch.zeros(1, device=dev)])).sum())); ps.append(float(q[-1]))
        av.append(0.0 if tid == STOP_ID else float(cen[min(max(tid, 0), N_ACT - 1)])); at.append(tid)
        if stp == TRAJ_LEN - 1:
            break
        o = lm(inputs_embeds=vla.traj_embed(torch.tensor([tid], device=dev)).unsqueeze(1).to(ctx.dtype),
               past_key_values=past, use_cache=True)
        past = o.past_key_values; logits = vla.output_head(o.last_hidden_state[:, -1].float())[0]
    return {'argmax': av, 'expect': ev, 'p_stop': ps, 'tok': at, 'gt_tok': gt, 'meta_gen': words, 'txt_gen': txt,
            'coc_ids': cg, 'coc_len': len(cg)}


def ade6_select(net, R, H, idx, dev, tk, rank, ws, mode='gen'):
    net.eval(); out = {}
    for i in idx[rank::ws]:
        t = R[int(i)]
        if len(t['future_positions']) < 12:
            continue
        d = decode(net, FM.vis_tokens(H, t['sample_token']), t['w1_ego'], tk, dev, mode)
        a = np.array(d['argmax'])
        P, _ = unicycle_rollout(a[:12], a[12:], float(t['future_speeds'][0]), float(t['w1_ego'][3, 1]))
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
    ap.add_argument('--mode', default='gen', choices=('gen', 'none', 'shuf', 'gt'))
    ap.add_argument('--coc', default=f'{DATA}/ar1_coc_2hz.pkl', help='R12: CoC pickle (v3: ar1_coc_2hz_v3.pkl)')
    ap.add_argument('--aux', type=float, default=0.0, help='R13: CoC kept per sample with this p (0 = always kept)')
    ap.add_argument('--select_mode', default='gen', choices=('gen', 'none'), help='R13: holdout selection decode mode')
    ap.add_argument('--smoke_decode', type=int, default=0, help='smoke test: decode N holdout samples after training')
    a = ap.parse_args()
    FM.LABELS['src'] = '2hz'
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=3))
    r = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(r); dev = f'cuda:{r}'
    ws = dist.get_world_size(); m0 = r == 0
    log = lambda *x: print(*x, flush=True) if m0 else None
    torch.manual_seed(a.seed + r)
    net = build(dev, a)
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb')); Mt = None
    coc = pickle.load(open(a.coc, 'rb'))
    log(f'[coc] traces {a.coc}')
    from transformers import AutoTokenizer
    tkt = AutoTokenizer.from_pretrained(TOKP)
    IDS = coc_ids(tkt, coc)
    tk = W1Tokenizer(); ckdir = f'{CK}/_w1_{a.tag}'
    if a.dump:
        ck = torch.load(f'{ckdir}/alpamayo_best.pt', map_location='cpu')
        assert not [k for k in ck['model_state'] if k not in net.vla.state_dict()]
        net.vla.load_state_dict(ck['model_state'], strict=False)
        net.vla.cosmos.model.language_model.gradient_checkpointing_disable(); net.eval()
        for split in a.dump.split(','):
            R = records.build(split, ['cmd'], 0.0, n_fut=6)
            forced = {}
            if a.mode == 'gt':
                forced = {t['sample_token']: IDS[t['sample_token']] for t in R}
            if a.mode == 'shuf':
                sh, single = shuffled_text(coc, [t['sample_token'] for t in R])
                forced = {k: tkt(v)['input_ids'] + [END] for k, v in sh.items()}
                log(f'[coc] {split} shuffled causes: {single} singleton classes keep their own CoC')
            ds = FM.DS(R, H, Mt, {'missing': 0})
            out = {}; t0 = time.time()
            for c, i in enumerate(range(r, len(R), ws)):
                t = R[i]
                gt = [x for x, _ in t['w1_tokens']] + [k for _, k in t['w1_tokens']] if len(t['w1_tokens']) == 12 else None
                d = decode(net, FM.vis_tokens(H, t['sample_token']), t['w1_ego'], tk, dev, a.mode,
                           forced.get(t['sample_token']), gt)
                d['coc_gen'] = tkt.decode([x for x in d.pop('coc_ids') if x != END])
                d['meta_gt'] = ds.meta[i][0] + ds.meta[i][1]
                out[t['sample_token']] = d
                if m0 and c % 100 == 0:
                    log(f'  [{split} {a.mode}] {c}/{len(R) // ws} {time.time() - t0:.0f}s')
            parts = [None] * ws
            dist.all_gather_object(parts, out)
            if m0:
                M = {}
                for p in parts:
                    M.update(p)
                name = a.tag if a.mode == 'gen' else f'{a.tag}_{a.mode}'
                op = f'{RES}/w1_dump_{name}_{split}_f0.0.pkl'
                pickle.dump({'order': [t['sample_token'] for t in R], 'data': M, 'ckpt_epoch': ck.get('epoch'),
                             'extra': ['cmd'], 'mode': a.mode}, open(op, 'wb'))
                log(f'[coc] saved {op} n={len(M)} ({time.time() - t0:.0f}s)')
            dist.barrier()
        log('COC_DUMP_DONE'); dist.destroy_process_group(); return

    tr = records.build('train', ['cmd'], 0.0); ho = records.build('holdout', ['cmd'], 0.0)
    L = np.array([len(IDS[t['sample_token']]) for t in tr])
    keep = (np.random.RandomState(42).rand(len(tr)) < a.aux) if a.aux > 0 else None
    dtr = DSC(tr, H, Mt, {'missing': 0}, IDS, keep)
    if keep is not None:
        log(f'[coc] AUX: CoC kept on {keep.sum()} of {len(tr)} train samples (p {a.aux}); selection mode {a.select_mode}')
    weights, f0, wt = build_turn_weights(tr, CFG['turn_thresh'], CFG['turn_target_frac'])
    samp = DistributedWeightedSampler(weights, ws, r, seed=a.seed)
    dl = torch.utils.data.DataLoader(dtr, batch_size=a.batch_size, sampler=samp, num_workers=4, pin_memory=True,
                                     drop_last=True, collate_fn=collate)
    model = DDP(net, device_ids=[r])
    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = fixrun.LrMultAdamW(trainable, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    spe = len(dl); total = spe * a.epochs
    idx = fixed_val_indices(len(ho), k=CFG['val_ade_k'], seed=CFG['val_ade_seed'])
    log(f'[coc] {a.tag} seed {a.seed}: train {len(tr)} holdout {len(ho)}; CoC+END tokens mean {L.mean():.1f} max {L.max()}; '
        f'turn frac {f0:.3f} -> 0.40; {spe} steps/epoch; trainable {sum(q.numel() for q in trainable):,}')
    best, bad, step = 1e9, 0, 0; hist = []
    os.makedirs(ckdir, exist_ok=True)
    for ep in range(a.epochs):
        samp.set_epoch(ep); model.train(); tot = tc = tj = 0.0; t_s = time.time(); t_ep = time.time()
        torch.cuda.reset_peak_memory_stats(dev)
        for bi, b in enumerate(dl):
            cocb, Lb = b['coc'].to(dev), b['L']
            lt, lj = model(b['vt'].to(dev), b['ego'].to(dev, torch.float32), cocb, Lb, b['txt'].to(dev), b['tok'].to(dev))
            tgt = [torch.cat([cocb[j, :int(Lb[j])], b['txt'][j].to(dev)]) for j in range(len(Lb))]
            ct = sum(F.cross_entropy(lt[j], tgt[j], reduction='sum') for j in range(len(Lb)))
            cj = F.cross_entropy(lj.reshape(-1, lj.shape[-1]), b['tok'].to(dev).reshape(-1), reduction='sum')
            nt = sum(len(x) for x in tgt); n = nt + b['tok'].numel()
            loss = (ct + cj) / n
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
            for g in opt.param_groups:
                g['lr'] = get_lr(step, total, CFG)
            scaler.step(opt); scaler.update(); opt.zero_grad(); step += 1
            tot += loss.item(); tc += ct.item() / nt; tj += cj.item() / b['tok'].numel(); hist.append(loss.item())
            if m0 and (step % 50 == 0 or (a.max_steps and step % 10 == 0)):
                log(f'  step {step:6d} ep {ep + 1} | loss {loss.item():.4f} avg {tot / (bi + 1):.4f} (text {tc / (bi + 1):.4f} '
                    f'traj {tj / (bi + 1):.4f}) | lr {get_lr(step, total, CFG):.2e} | scale {scaler.get_scale():.0f} | '
                    f'{(time.time() - t_s) / (10 if a.max_steps else 50):.2f}s/step | peak {torch.cuda.max_memory_allocated(dev) / 1e9:.1f}GB')
                t_s = time.time()
            if a.max_steps and step >= a.max_steps:
                break
        raw = model.module
        raw.vla.cosmos.model.language_model.gradient_checkpointing_disable()
        if a.smoke_decode:
            import re
            raw.eval(); res = []
            for i in idx[r:a.smoke_decode:ws]:
                t = ho[int(i)]
                d = decode(raw, FM.vis_tokens(H, t['sample_token']), t['w1_ego'], tk, dev, 'gen')
                txt = tkt.decode([x for x in d['coc_ids'] if x != END])
                ok = re.match(r'^Driving decision: ([^;]+); ([^.]+)\. Critical components: (.+)\. Reasoning: (.+)$', txt)
                res.append((bool(ok), d['coc_ids'][-1] == END, len(d['tok']) == 24 and np.isfinite(d['expect']).all(), txt))
            parts = [None] * ws; dist.all_gather_object(parts, res)
            if m0:
                R_ = [x for p in parts for x in p]; h = np.array(hist)
                log(f'[smoke] loss first 20 steps {h[:20].mean():.4f} -> last 20 {h[-20:].mean():.4f}; '
                    f'decoded {len(R_)}: CoC parses {sum(x[0] for x in R_)}, ends with terminator {sum(x[1] for x in R_)}, '
                    f'24 traj tokens finite {sum(x[2] for x in R_)}')
                for x in R_[:4]:
                    log(f'  [smoke text] {x[3][:300]}')
            break
        med, mean, n = ade6_select(raw, ho, H, idx, dev, tk, r, ws, a.select_mode)
        raw.vla.cosmos.model.language_model.gradient_checkpointing_enable()
        log(f'[epoch {ep + 1}] SELECT holdout AR ADE@6s median={med:.4f} (mean={mean:.4f}, n={n}) best={min(best, med):.4f} '
            f'| train loss {tot / (bi + 1):.4f} (text {tc / (bi + 1):.4f} traj {tj / (bi + 1):.4f}) | {time.time() - t_ep:.0f}s')
        if med < best:
            best, bad = med, 0
            if m0:
                st = {k: v for k, v in raw.vla.state_dict().items()
                      if 'lora_' in k or any(k.startswith(p) for p in ('ego_encoder', 'traj_embed', 'output_head'))}
                torch.save({'epoch': ep + 1, 'val_ade6': med, 'model_state': st, 'seed': a.seed}, f'{ckdir}/alpamayo_best.pt')
                log(f'[ckpt] saved best epoch {ep + 1}')
        else:
            bad += 1
        if bad >= a.patience:
            break
    log(f'[coc] done; best holdout median ADE@6s {best:.4f}')
    log('COC_TRAIN_DONE')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
