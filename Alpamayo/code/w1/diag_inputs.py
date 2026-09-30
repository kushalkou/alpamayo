"""w1/diag_inputs.py -- planner item 2 cheap diagnostics (a, b, d), CPU, no training.
Run only if the 3.5a retry fails.
  a. L2 norms of the ego tokens (EgoEncoderMLP output, eval mode) and the cmd token,
     for a given checkpoint on the 256 overfit samples, vs the mean L2 norm of the
     Cosmos text-token embeddings (model.embed_tokens.weight).
  b. per-feature mean/sd of the 16 ego features on official train (and the 256);
     standardisation status; which features carry absolute heading / position.
  d. where slot 0 reads from (code path) -- printed from the source, plus a mask check.
Usage: python w1/diag_inputs.py <ckpt.pt>
"""
import sys, inspect, pickle
import numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records
from model import EgoEncoderMLP, AlpamayoVLA
from safetensors import safe_open

COS = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'


def main():
    ck = torch.load(sys.argv[1], map_location='cpu')['model_state']
    tr = records.build('train', ('cmd',))
    rs = np.random.RandomState(0)
    ov = [tr[i] for i in sorted(rs.choice(len(tr), 256, replace=False))]
    # ---- a ----
    with safe_open(f'{COS}/model-00001-of-00004.safetensors', 'pt') as f:
        E = f.get_tensor('model.embed_tokens.weight').float()
    tn = E.norm(dim=1)
    enc = EgoEncoderMLP(); enc.load_state_dict({k[len('ego_encoder.'):]: v for k, v in ck.items()
                                                if k.startswith('ego_encoder.') and '.xtok.' not in k})
    enc.eval()
    with torch.no_grad():
        X = torch.stack([t['w1_ego'][:4] for t in ov])
        en = enc(X).norm(dim=-1)                               # [256,4]
    cmd = ck.get('ego_encoder.xtok.cmd.weight')
    print('a. token L2 norms')
    print(f'   Cosmos text embeddings: mean {tn.mean():.3f}  median {tn.median():.3f}  '
          f'p5/p95 {np.percentile(tn,5):.3f}/{np.percentile(tn,95):.3f}  (n={len(tn)})')
    print(f'   ego tokens (rows t-3..t): mean ' + ' '.join(f'{x:.3f}' for x in en.mean(0).tolist())
          + f'  overall {en.mean():.3f}  (ratio to text mean {en.mean()/tn.mean():.1f}x)')
    if cmd is not None:
        print(f'   cmd tokens (right/left/straight): ' + ' '.join(f'{x:.3f}' for x in cmd.norm(dim=1).tolist())
              + f'  (ratio {cmd.norm(dim=1).mean()/tn.mean():.2f}x)')
    # pairwise distinctness of the first-token context: ego-token spread across samples
    with torch.no_grad():
        cur = enc(X)[:, 3]                                     # current-row token
    d = torch.cdist(cur, cur); iu = torch.triu_indices(256, 256, 1)
    print(f'   current-row ego token: median pairwise L2 {d[iu[0], iu[1]].median():.3f} vs its norm '
          f'{cur.norm(dim=1).median():.3f}')
    # ---- b ----
    names = ['speed', 'rel_yaw+pi/2', 'yaw_rate', 'accel']
    F = torch.stack([t['w1_ego'][:4] for t in tr]).numpy()      # [N,4,4]
    print('\nb. ego features (records.ego_state_w1), official train n=%d; rows t-3..t' % len(F))
    for j, n in enumerate(names):
        print(f'   {n:13} mean ' + ' '.join(f'{F[:, r, j].mean():+8.3f}' for r in range(4))
              + '   sd ' + ' '.join(f'{F[:, r, j].std():7.3f}' for r in range(4)))
    print('   standardised before the ego MLP: NO (raw units: m/s, rad, rad/s, m/s^2)')
    print('   absolute heading / position: NONE. Yaw is relative to t0 (+pi/2 constant, so the')
    print('   current row is always exactly pi/2); no x/y position enters. The old')
    print('   compute_ego_state fed the GLOBAL yaw; ego_state_w1 does not.')
    print('   definition:\n' + inspect.getsource(records.ego_state_w1))
    # ---- d ----
    print('d. slot-0 read path (model.AlpamayoVLA.forward, training) and decode (prefill):')
    src = inspect.getsource(AlpamayoVLA.forward)
    for line in src.splitlines():
        if 'ctx_len' in line or 'last_hidden_state' in line or 'output_head' in line or 'lm_in' in line:
            print('   | ' + line.rstrip())
    print('   -> slot 0 = output_head(hidden state at position ctx_len-1), i.e. the LAST context')
    print('      token = the cmd token (after the 4 ego tokens; 0 visual tokens). The LM is')
    print('      called with inputs_embeds and NO attention_mask -> the default causal mask, so')
    print('      position ctx_len-1 attends to all 5 context tokens (4 ego + cmd).')


if __name__ == '__main__':
    main()
