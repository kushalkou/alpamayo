"""leak/dump_ce.py -- AR dump (dump_decode.py format) + FREE-RUNNING CE of the GT token.

Per sample, per model, one argmax-fed-back AR pass (identical rule to dump_decode.dump_one)
storing argmax / expect / p_stop, plus the log-prob of the GROUND-TRUTH token at each
slot under the model's free-running distribution (never teacher-forced):
  lp65  : over the STOP-aware support {0..63} U {STOP}, renormalised
  lp129 : over the full 129-way softmax
Output: results/dump_{split}_{tag}.pkl = {'meta', 'data', 'ce': {model: {i: {lp65, lp129}}}}
meta is copied from the frozen dump_{split}.pkl for the same indices.

--causal : ego features from leak.causal_ego (past poses only) and the >=2-past-pose
           subset; indices keep their positions in the full split (paired with dumps).
Launch:
  python -m torch.distributed.run --nproc_per_node=8 leak/dump_ce.py --split test \
      --tag ce --models y1_ego,y1_full
"""
import os, sys, time, pickle, argparse
import numpy as np, torch
import torch.distributed as dist
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/leak')
import causal_ego

CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
# name: (ckpt, zero_vision, zero_ego, causal_features)
ALL = {
    'y1_ego':  (f'{CK}/_y1_egoonly_turnw/alpamayo_best.pt', True, False, False),
    'y1_full': (f'{CK}/_y1_full_turnw/alpamayo_best.pt', False, False, False),
    'causal_ego':  (f'{CK}/_causal_ego_s42/alpamayo_best.pt', True, False, True),
    'causal_full': (f'{CK}/_causal_full_s42/alpamayo_best.pt', False, False, True),
    'A0_causal_ego_novis': (f'{CK}/_causal_A0_ego_novis_s42/alpamayo_best.pt', 'remove', False, True),
    'A0_causal_ego_novis_s123': (f'{CK}/_causal_A0_ego_novis_s123/alpamayo_best.pt', 'remove', False, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--causal', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--probs', action='store_true',
                    help='also store the 65-way free-running distribution per slot')
    a = ap.parse_args()
    names = a.models.split(',')
    if a.causal:
        causal_ego.patch()
    import dataset
    from dataset import NuScenesVLADataset
    from model import load_model, TRAJ_LEN
    from tokenizer import TrajectoryTokenizer
    from ar_eval import encode_live_one
    from inference import prefill_context, slot_centers, N_ACT, STOP_ID

    dist.init_process_group(backend='nccl')
    lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device = f'cuda:{lr}'; ws = dist.get_world_size(); m0 = lr == 0
    shard = f'/tmp/claude-1000/dumpce_shards/{a.split}_{a.tag}'
    os.makedirs(shard, exist_ok=True)

    trajs = pickle.load(open(f'{RES}/leak_split.pkl', 'rb'))[a.split]
    meta0 = pickle.load(open(f'{RES}/dump_{a.split}.pkl', 'rb'))['meta']
    idx = [i for i in range(len(trajs)) if (not a.causal or causal_ego.keep(trajs[i]))]
    if a.limit: idx = idx[:a.limit]
    ds = NuScenesVLADataset(trajs, split=a.split, augment=False)
    tok = TrajectoryTokenizer()
    model = load_model(device=device)
    model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual = model.cosmos.model.visual
    torch.set_grad_enabled(False)
    lm = model.cosmos.model.language_model
    my = idx[lr::ws]
    if m0: print(f'[dumpce] split={a.split} n={len(idx)} models={names} causal={a.causal}', flush=True)

    def gt_tokens(t):
        tk = tok.tokenize(t)
        return [x for x, _ in tk] + [k for _, k in tk]

    data, ce = {}, {}
    t0 = time.time()
    for name in names:
        path, zv, ze, cf = ALL[name]
        assert cf == a.causal, f'{name} needs --causal={cf}'
        ck = torch.load(path, map_location='cpu')
        model.load_state_dict(ck['model_state'], strict=False); model.zero_ego = ze
        d, c = {}, {}
        for n_, i in enumerate(my):
            t = trajs[i]
            if zv:
                ego = dataset.compute_ego_state(t)
                vt = torch.zeros(1, 0 if zv == 'remove' else 1536, 3584,
                                 dtype=torch.float16, device=device)
            else:
                item = ds[i]; ego = item['ego_state']
                vt = encode_live_one(visual, item['images'], device)
            gt = gt_tokens(t)
            pre = prefill_context(model, vt, ego, device=device)
            past, ctx_len, ctx_dtype = pre['past'], pre['ctx_len'], pre['ctx_dtype']
            logits = pre['logits0']
            av, ev, ps, l65, l129, q65 = [], [], [], [], [], []
            for step in range(TRAJ_LEN):
                centers = torch.as_tensor(slot_centers(tok, step), dtype=torch.float32,
                                          device=logits.device)
                tokid = int(logits.argmax(-1).item())
                vlog = torch.cat([logits[:N_ACT], logits[STOP_ID:STOP_ID + 1]])
                vcent = torch.cat([centers, torch.zeros(1, device=logits.device)])
                q = torch.softmax(vlog, -1)
                ev.append(float((q * vcent).sum())); ps.append(float(q[-1]))
                if a.probs: q65.append(q.float().cpu().numpy().astype('float32'))
                av.append(0.0 if tokid == STOP_ID else float(centers[min(max(tokid, 0), N_ACT - 1)]))
                g = gt[step]; gi = N_ACT if g == STOP_ID else g
                l65.append(float(torch.log_softmax(vlog, -1)[gi]))
                l129.append(float(torch.log_softmax(logits, -1)[g]))
                if step == TRAJ_LEN - 1: break
                emb = model.traj_embed(torch.tensor([tokid], device=device)).unsqueeze(1).to(ctx_dtype)
                out = lm(inputs_embeds=emb, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = model.output_head(out.last_hidden_state[:, -1, :].float())[0]
            d[i] = {'argmax': av, 'expect': ev, 'p_stop': ps}
            c[i] = {'lp65': l65, 'lp129': l129, 'gt': gt}
            if a.probs: c[i]['q65'] = np.stack(q65)
            if m0 and n_ % 50 == 0:
                print(f'  [{name}] {n_}/{len(my)}  {time.time()-t0:.0f}s', flush=True)
        data[name], ce[name] = d, c

    tmp = f'{shard}/s_{lr}.pkl.tmp'
    pickle.dump({'data': data, 'ce': ce}, open(tmp, 'wb')); os.replace(tmp, f'{shard}/s_{lr}.pkl')
    if not m0:
        dist.destroy_process_group(); return
    while sum(os.path.exists(f'{shard}/s_{r}.pkl') for r in range(ws)) < ws:
        time.sleep(5)
    DATA = {m: {} for m in names}; CE = {m: {} for m in names}
    for r in range(ws):
        g = pickle.load(open(f'{shard}/s_{r}.pkl', 'rb'))
        for m in names:
            DATA[m].update(g['data'][m]); CE[m].update(g['ce'][m])
    META = {i: meta0[i] for i in sorted(DATA[names[0]])}
    op = f'{RES}/dump_{a.split}_{a.tag}.pkl'
    pickle.dump({'meta': META, 'data': DATA, 'ce': CE}, open(op, 'wb'))
    print(f'[dumpce] saved -> {op} n={len(META)}', flush=True)
    print('DUMPCE_DONE', flush=True)
    for r in range(ws):
        os.remove(f'{shard}/s_{r}.pkl')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
