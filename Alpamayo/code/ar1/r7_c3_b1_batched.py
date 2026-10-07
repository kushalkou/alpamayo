"""ar1/r7_c3_b1_batched.py -- R7 C3: batched B1 (traj-only) decoder for the blank-image control.

Same decode as w1/dump_w1.py (prefill, then 24 trajectory tokens: argmax / STOP-aware
expectation / p_stop), B samples at a time (all contexts 485 tokens, no padding).
Variants (one model load, sharded over ranks):
  real    cached tokens of the sample itself (batched reference)
  cams    cached tokens of another sample, permutation RandomState(99) over val (as R6 V1)
  blank   every visual token = per-dim train mean (data/r7_vis_train_mean.npy)
Output results/w1_dump_<tag>_<variant>_val_f0.0.pkl (gate_a format).
  python -m torch.distributed.run --nproc_per_node=8 ar1/r7_c3_b1_batched.py --tag B1 \
      --variants real,cams,blank [--check 200]
"""
import os, sys, time, pickle, argparse, datetime
import numpy as np, torch
import torch.distributed as dist
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1'); sys.path.insert(0, f'{CODE}/ar1')
import records, extra_tokens, fixrun, vcache
from model import load_model, TRAJ_LEN
from inference import slot_centers, N_ACT, STOP_ID
from w1tok import W1Tokenizer

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'


@torch.no_grad()
def decode_batch(model, vt, ego, tk, dev):
    lm = model.cosmos.model.language_model; B = vt.shape[0]
    ctx = model._build_context(vt.to(dev, torch.float16), ego.to(dev, torch.float32))
    o = lm(inputs_embeds=ctx, use_cache=True); past = o.past_key_values
    logits = model.output_head(o.last_hidden_state[:, -1].float())
    av, ev, ps, at = [], [], [], []
    for stp in range(TRAJ_LEN):
        cen = torch.as_tensor(slot_centers(tk, stp), dtype=torch.float32, device=dev)
        tid = logits.argmax(-1)
        q = torch.softmax(torch.cat([logits[:, :N_ACT], logits[:, STOP_ID:STOP_ID + 1]], 1), -1)
        ev.append((q * torch.cat([cen, torch.zeros(1, device=dev)])[None]).sum(1)); ps.append(q[:, -1])
        av.append(torch.where(tid == STOP_ID, torch.zeros_like(cen[0:1]).expand(B), cen[tid.clamp(0, N_ACT - 1)]))
        at.append(tid)
        if stp == TRAJ_LEN - 1:
            break
        o = lm(inputs_embeds=model.traj_embed(tid)[:, None].to(ctx.dtype), past_key_values=past, use_cache=True)
        past = o.past_key_values; logits = model.output_head(o.last_hidden_state[:, -1].float())
    cat = lambda L: torch.stack(L, 1).cpu().numpy()
    return cat(av), cat(ev), cat(ps), cat(at)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='B1')
    ap.add_argument('--variants', default='real')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--check', type=int, default=0)
    a = ap.parse_args()
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=3))
    r = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(r); dev = f'cuda:{r}'; ws = dist.get_world_size()
    model = load_model(device=dev)
    model = extra_tokens.install(model, ['cmd'], records.N_CLS)
    model = fixrun.install_fix(model, torch.zeros(4), torch.ones(4))     # in_mean / in_sd from the ckpt
    ck = torch.load(f'{CK}/_w1_{a.tag}/alpamayo_best.pt', map_location='cpu')
    assert not [k for k in ck['model_state'] if k not in model.state_dict()]
    model.load_state_dict(ck['model_state'], strict=False)
    model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb')); tk = W1Tokenizer()
    BLANK = torch.from_numpy(np.load(f'{DATA}/r7_vis_train_mean.npy')).to(torch.float16)
    R = records.build('val', ['cmd'], 0.0, n_fut=6)
    if a.check:
        R = R[:a.check]
    perm = np.random.RandomState(99).permutation(len(R))
    for var in a.variants.split(','):
        mine = list(range(len(R)))[r::ws]; out = {}; t0 = time.time()
        for j in range(0, len(mine), a.bs):
            b = mine[j:j + a.bs]
            vt = torch.stack([vcache.gather(H[R[perm[i] if var == 'cams' else i]['sample_token']], (3,)) for i in b])
            if var == 'blank':
                vt = BLANK[None, None].expand_as(vt).clone()
            ego = torch.stack([R[i]['w1_ego'] for i in b])
            av, ev, ps, at = decode_batch(model, vt, ego, tk, dev)
            for q, i in enumerate(b):
                t = R[i]
                gt = [x for x, _ in t['w1_tokens']] + [k for _, k in t['w1_tokens']] if len(t['w1_tokens']) == 12 else None
                out[t['sample_token']] = {'argmax': av[q].tolist(), 'expect': ev[q].tolist(), 'p_stop': ps[q].tolist(),
                                          'tok': at[q].tolist(), 'gt_tok': gt}
        parts = [None] * ws
        dist.all_gather_object(parts, out)
        if r == 0:
            M = {}
            for p in parts:
                M.update(p)
            name = f'{a.tag}_{var}' + (f'_check{a.check}' if a.check else '')
            pickle.dump({'order': [t['sample_token'] for t in R], 'data': M, 'variant': var},
                        open(f'{RES}/w1_dump_{name}_val_f0.0.pkl', 'wb'))
            print(f'[b1 batched] {name}: {len(M)} samples, {time.time() - t0:.0f}s', flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
