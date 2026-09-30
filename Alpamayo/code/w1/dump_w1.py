"""w1/dump_w1.py -- free-running AR dump of a WEEK1 checkpoint (argmax fed back; the
identical decode rule to dump_decode.dump_one / leak/dump_ce.py).

Per sample, per slot: argmax value, STOP-aware expectation value, p(STOP), argmax token
id, and the log-prob of the GT token (65-way). Samples = records.build(split, n_fut=6):
official val (all 5,119) and holdout (1,717, for blend alpha). Test-time meta-action
flip rates (A2) are passed as --flips 0,0.1,0.2,0.4 (seed 777, independent of the
training flips). Output: results/w1_dump_<tag>_<split>_f<rate>.pkl
  python -m torch.distributed.run --nproc_per_node=8 w1/dump_w1.py --tag A1 --cmd
      --no_vision --splits val,holdout
"""
import os, sys, time, pickle, argparse
import numpy as np, torch
import torch.distributed as dist
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1')
import records, extra_tokens
from w1tok import W1Tokenizer
from model import load_model, TRAJ_LEN
from ar_eval import encode_live_one
from inference import prefill_context, slot_centers, N_ACT, STOP_ID
from dataset import NuScenesVLADataset
import dataset

CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--cmd', action='store_true')
    ap.add_argument('--meta', action='store_true')
    ap.add_argument('--no_vision', action='store_true')
    ap.add_argument('--zero_vision', action='store_true')
    ap.add_argument('--splits', default='val,holdout')
    ap.add_argument('--flips', default='0')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    extra = ['lat', 'lon'] if a.meta else (['cmd'] if a.cmd else [])
    dataset.compute_ego_state = lambda traj: traj['w1_ego']
    dataset.TrajectoryTokenizer = W1Tokenizer

    dist.init_process_group(backend='nccl')
    lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device = f'cuda:{lr}'; ws = dist.get_world_size(); m0 = lr == 0
    tok = W1Tokenizer()
    model = load_model(device=device)
    if extra:
        model = extra_tokens.install(model, extra, records.N_CLS)
    ck = torch.load(a.ckpt or f'{CK}/_w1_{a.tag}/alpamayo_best.pt', map_location='cpu')
    missing = [k for k in ck['model_state'] if k not in model.state_dict()]
    assert not missing, missing[:5]
    model.load_state_dict(ck['model_state'], strict=False)
    model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual = model.cosmos.model.visual; lm = model.cosmos.model.language_model
    torch.set_grad_enabled(False)
    if m0: print(f'[dumpw1] {a.tag} ckpt epoch {ck.get("epoch")} val_ade6 {ck.get("val_ade6")} '
                 f'extra={extra}', flush=True)

    for split in a.splits.split(','):
        for f in [float(x) for x in a.flips.split(',')]:
            R = records.build(split, extra, f, flip_seed=777, n_fut=6)
            if a.limit: R = R[:a.limit]
            ds = NuScenesVLADataset(R, split=split, augment=False)
            my = list(range(len(R)))[lr::ws]
            out = {}; t0 = time.time()
            for c, i in enumerate(my):
                t = R[i]; ego = t['w1_ego']
                if a.no_vision:
                    vt = torch.zeros(1, 0, 3584, dtype=torch.float16, device=device)
                else:
                    vt = encode_live_one(visual, ds[i]['images'], device)
                    if a.zero_vision: vt = torch.zeros_like(vt)
                gt = [x for x, _ in t['w1_tokens']] + [k for _, k in t['w1_tokens']] \
                    if len(t['w1_tokens']) == 12 else None
                pre = prefill_context(model, vt, ego, device=device)
                past, ctx_dtype = pre['past'], pre['ctx_dtype']; logits = pre['logits0']
                av, ev, ps, at, lp = [], [], [], [], []
                for step in range(TRAJ_LEN):
                    cen = torch.as_tensor(slot_centers(tok, step), dtype=torch.float32, device=logits.device)
                    tid = int(logits.argmax(-1).item())
                    vlog = torch.cat([logits[:N_ACT], logits[STOP_ID:STOP_ID + 1]])
                    q = torch.softmax(vlog, -1)
                    ev.append(float((q * torch.cat([cen, torch.zeros(1, device=cen.device)])).sum()))
                    ps.append(float(q[-1]))
                    av.append(0.0 if tid == STOP_ID else float(cen[min(max(tid, 0), N_ACT - 1)]))
                    at.append(tid)
                    if gt is not None:
                        g = gt[step]; lp.append(float(torch.log_softmax(vlog, -1)[N_ACT if g == STOP_ID else g]))
                    if step == TRAJ_LEN - 1: break
                    emb = model.traj_embed(torch.tensor([tid], device=device)).unsqueeze(1).to(ctx_dtype)
                    o = lm(inputs_embeds=emb, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    logits = model.output_head(o.last_hidden_state[:, -1, :].float())[0]
                out[t['sample_token']] = {'argmax': av, 'expect': ev, 'p_stop': ps, 'tok': at,
                                          'lp65': lp, 'gt_tok': gt, 'shown': t['shown']}
                if m0 and c % 100 == 0:
                    print(f'  [{split} f={f}] {c}/{len(my)} {time.time()-t0:.0f}s', flush=True)
            shard = f'/tmp/claude-1000/dumpw1/{a.tag}_{split}_{f}'
            os.makedirs(shard, exist_ok=True)
            pickle.dump(out, open(f'{shard}/s_{lr}.pkl.tmp', 'wb'))
            os.replace(f'{shard}/s_{lr}.pkl.tmp', f'{shard}/s_{lr}.pkl')
            dist.barrier()
            if m0:
                M = {}
                for r in range(ws):
                    M.update(pickle.load(open(f'{shard}/s_{r}.pkl', 'rb')))
                    os.remove(f'{shard}/s_{r}.pkl')
                op = f'{RES}/w1_dump_{a.tag}_{split}_f{f}.pkl'
                pickle.dump({'order': [t['sample_token'] for t in R], 'data': M,
                             'ckpt_epoch': ck.get('epoch'), 'extra': extra}, open(op, 'wb'))
                print(f'[dumpw1] saved {op} n={len(M)}', flush=True)
            dist.barrier()
    if m0: print('DUMPW1_DONE', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
