"""ar1/meta_decode_batched.py -- R6 Part 1: batched M1 / M1-v2a decoder for verification runs.

Same decoding as finetune_meta.decode (constrained greedy words, then 24 trajectory tokens
with argmax / STOP-aware expectation / p_stop), but B samples at a time (all contexts have
the same length, so no padding). Variants (one model load, sharded over ranks):
  gen        own words (constrained greedy)                       -> check vs saved dump
  cams / ego / cmd   input shuffle, permutation RandomState(99) over all val records
  force_gt   words teacher-forced to the GT 2 Hz labels [ORACLE]
  force_maj  words forced to maintain x6 + keep straight x6
  force_s1   GT-stopped stationary samples only: slot-1 lon word forced to gentle_acc,
             the other 11 words generated
Output: results/w1_dump_<tag>_<variant>_val_f0.0.pkl (gate_a format + 'meta_gen' = the 12
word classes actually in the sequence).
  python -m torch.distributed.run --nproc_per_node=8 ar1/meta_decode_batched.py --tag M1v2a \
      --variants gen,cams,ego,cmd,force_gt,force_maj,force_s1 [--check 200]
"""
import os, sys, time, pickle, argparse, datetime
import numpy as np, torch
import torch.distributed as dist
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1'); sys.path.insert(0, f'{CODE}/ar1')
import records, vcache
import finetune_meta as FM
from model import TRAJ_LEN
from inference import slot_centers, N_ACT, STOP_ID
from w1tok import W1Tokenizer

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
LONW = torch.tensor(FM.LON_W)
A_KEYS = list(FM.LAT_A); A_IDS = torch.tensor([FM.LAT_A[k] for k in A_KEYS])
B_KEYS = ['left', 'right', 'straight']; B_IDS = torch.tensor([FM.LAT_B[k] for k in B_KEYS])


@torch.no_grad()
def decode_batch(net, vt, ego, tk, dev, force=None, force_s1=False):
    """vt [B,N,3584] fp16, ego [B,5,4]; force: LongTensor [B,18] token ids or None."""
    vla = net.vla; lm = vla.cosmos.model.language_model; B = vt.shape[0]
    ctx = vla._build_context(vt.to(dev, torch.float16), ego.to(dev, torch.float32))
    o = lm(inputs_embeds=ctx, use_cache=True); past = o.past_key_values; h = o.last_hidden_state[:, -1]
    ids = []

    def step(tok):
        nonlocal past, h
        o = lm(inputs_embeds=net.emb(tok[:, None]).to(ctx.dtype), past_key_values=past, use_cache=True)
        past = o.past_key_values; h = o.last_hidden_state[:, -1]

    for s in range(6):
        if force is not None:
            t = force[:, s].to(dev)
        else:
            lg = net.lm_head(h).float()
            t = LONW.to(dev)[lg[:, LONW.to(dev)].argmax(1)]
            if force_s1 and s == 0:
                t = torch.full_like(t, FM.LON_W[0])                 # gentle_acc
        ids.append(t); step(t)
    for s in range(6):
        if force is not None:
            ta = force[:, 6 + 2 * s].to(dev)
        else:
            lg = net.lm_head(h).float(); ta = A_IDS.to(dev)[lg[:, A_IDS.to(dev)].argmax(1)]
        ids.append(ta); step(ta)
        if force is not None:
            tb = force[:, 7 + 2 * s].to(dev)
        else:
            lg = net.lm_head(h).float()[:, B_IDS.to(dev)]
            keep = ta == FM.LAT_A['keep']
            lg[keep, 0] = -1e9; lg[keep, 1] = -1e9; lg[~keep, 2] = -1e9
            tb = B_IDS.to(dev)[lg.argmax(1)]
        ids.append(tb); step(tb)
    logits = vla.output_head(h.float())
    av, ev, ps, at = [], [], [], []
    for stp in range(TRAJ_LEN):
        cen = torch.as_tensor(slot_centers(tk, stp), dtype=torch.float32, device=dev)
        tid = logits.argmax(-1)
        vlog = torch.cat([logits[:, :N_ACT], logits[:, STOP_ID:STOP_ID + 1]], 1)
        q = torch.softmax(vlog, -1)
        ev.append((q * torch.cat([cen, torch.zeros(1, device=dev)])[None]).sum(1))
        ps.append(q[:, -1])
        av.append(torch.where(tid == STOP_ID, torch.zeros_like(cen[0:1]).expand(B), cen[tid.clamp(0, N_ACT - 1)]))
        at.append(tid)
        if stp == TRAJ_LEN - 1:
            break
        o = lm(inputs_embeds=vla.traj_embed(tid)[:, None].to(ctx.dtype), past_key_values=past, use_cache=True)
        past = o.past_key_values; logits = vla.output_head(o.last_hidden_state[:, -1].float())
    ids = torch.stack(ids, 1).cpu().numpy()
    words = []
    for row in ids:
        lon = [FM.LON_W.index(int(x)) for x in row[:6]]
        lat = [FM.LAT_W.index((A_KEYS[int((A_IDS == int(row[6 + 2 * s])).nonzero()[0])],
                               B_KEYS[int((B_IDS == int(row[7 + 2 * s])).nonzero()[0])])) for s in range(6)]
        words.append(lon + lat)
    cat = lambda L: torch.stack(L, 1).cpu().numpy()
    return cat(av), cat(ev), cat(ps), cat(at), words, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='M1v2a')
    ap.add_argument('--variants', default='gen')
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--check', type=int, default=0, help='only the first N val records (decoder check)')
    a = ap.parse_args()
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=3))
    r = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(r); dev = f'cuda:{r}'; ws = dist.get_world_size()
    net = FM.build(dev, argparse.Namespace(dump=True))
    ck = torch.load(f'{CK}/_w1_{a.tag}/alpamayo_best.pt', map_location='cpu')
    net.vla.load_state_dict(ck['model_state'], strict=False)
    net.vla.cosmos.model.language_model.gradient_checkpointing_disable(); net.eval()
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb')); lab = pickle.load(open(f'{DATA}/ar1_meta2hz.pkl', 'rb'))
    V0 = pickle.load(open(f'{DATA}/w1_v0.pkl', 'rb')); T = pickle.load(open(f'{DATA}/w1_targets.pkl', 'rb'))['targets']
    tk = W1Tokenizer()
    R = records.build('val', ['cmd'], 0.0, n_fut=6)
    if a.check:
        R = R[:a.check]
    perm = np.random.RandomState(99).permutation(len(R))
    for var in a.variants.split(','):
        idx = list(range(len(R)))
        if var == 'force_s1':
            idx = [i for i, t in enumerate(R) if V0[t['sample_token']]['v0_can'] < 0.5 and
                   np.linalg.norm(np.asarray(T[t['sample_token']]['P'])[:6], axis=1).max() < 0.5]
        mine = idx[r::ws]; out = {}; t0 = time.time()
        for j in range(0, len(mine), a.bs):
            b = mine[j:j + a.bs]
            src_c = [perm[i] if var == 'cams' else i for i in b]
            vt = torch.stack([vcache.gather(H[R[i]['sample_token']], (3,)) for i in src_c])
            ego = torch.stack([R[i]['w1_ego'].clone() for i in b])
            if var == 'ego':
                ego[:, :4] = torch.stack([R[perm[i]]['w1_ego'][:4] for i in b])
            if var == 'cmd':
                ego[:, 4] = torch.stack([R[perm[i]]['w1_ego'][4] for i in b])
            force = None
            if var == 'force_gt':
                force = torch.tensor([FM.text_ids(list(lab[R[i]['sample_token']][:6]), list(lab[R[i]['sample_token']][6:])) for i in b])
            if var == 'force_maj':
                force = torch.tensor([FM.text_ids([4] * 6, [6] * 6)] * len(b))
            av, ev, ps, at, words, ids = decode_batch(net, vt, ego, tk, dev, force, force_s1=(var == 'force_s1'))
            for q, i in enumerate(b):
                t = R[i]
                gt = [x for x, _ in t['w1_tokens']] + [k for _, k in t['w1_tokens']] if len(t['w1_tokens']) == 12 else None
                out[t['sample_token']] = {'argmax': av[q].tolist(), 'expect': ev[q].tolist(), 'p_stop': ps[q].tolist(),
                                          'tok': at[q].tolist(), 'gt_tok': gt, 'meta_gen': words[q], 'txt': ids[q].tolist()}
        parts = [None] * ws
        dist.all_gather_object(parts, out)
        if r == 0:
            M = {}
            for p in parts:
                M.update(p)
            name = f'{a.tag}_{var}' + (f'_check{a.check}' if a.check else '')
            pickle.dump({'order': [R[i]['sample_token'] for i in idx], 'data': M, 'variant': var},
                        open(f'{RES}/w1_dump_{name}_val_f0.0.pkl', 'wb'))
            print(f'[batched] {name}: {len(M)} samples, {time.time() - t0:.0f}s', flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
