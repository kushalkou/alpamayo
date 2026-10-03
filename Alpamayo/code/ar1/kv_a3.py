"""ar1/kv_a3.py -- R2.3 step 1: per-layer KV-cache of the FROZEN A3 VLM (one GPU).

A3 = ego + cmd [P], fix recipe, turn-weighted, no visual tokens (context = 4 ego tokens +
1 cmd token = 5 positions). The model is in eval mode (LoRA dropout off) and no_grad, so
the cache is exactly the stop-gradient input the expert would see online; precomputing
it once is equivalent and far cheaper (5 tokens x 28 layers x 2 x 4 x 128 fp16 =
286 KB per sample).
Also checks the expert's RoPE: layer-0 keys recomputed as rope(k_proj(input_norm(ctx)))
must equal the cached layer-0 keys.
Output: Alpamayo/data/ar1_kv_A3_<split>.pt {'kv': fp16 [N,28,2,4,5,128], 'tokens',
'v0', 'acc', 'cur' (n_fut=12 only, else nan), 'ctx_len'}; splits train (n_fut = 12,
18,313; the A3 training set), holdout and val (n_fut >= 6; evaluation order).
"""
import sys, time, pickle
import numpy as np, torch
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1'); sys.path.insert(0, f'{CODE}/ar1')
import records, extra_tokens, fixrun
from model import load_model
from expert import rope

CK = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints/_w1_A3/alpamayo_best.pt'
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_kv_A3_{}.pt'


def layer_kv(past, l):
    if hasattr(past, 'layers'):
        return past.layers[l].keys, past.layers[l].values
    return past.key_cache[l], past.value_cache[l]


@torch.no_grad()
def main():
    dev = 'cuda:0'
    m = load_model(device=dev)
    m = extra_tokens.install(m, ['cmd'], records.N_CLS)
    m = fixrun.install_fix(m, torch.zeros(4), torch.ones(4))
    ck = torch.load(CK, map_location='cpu')
    miss = [k for k in ck['model_state'] if k not in m.state_dict()]
    assert not miss, miss[:5]
    m.load_state_dict(ck['model_state'], strict=False)
    lm = m.cosmos.model.language_model
    lm.gradient_checkpointing_disable(); m.eval()
    print(f'[kv] A3 epoch {ck.get("epoch")}', flush=True)
    t0 = time.time(); checked = False
    for split, nf in (('train', 12), ('holdout', 6), ('val', 6)):
        R = records.build(split, ['cmd'], 0.0, n_fut=nf)
        KV = []
        for i in range(0, len(R), 256):
            ego = torch.stack([t['w1_ego'] for t in R[i:i + 256]]).to(dev)
            ctx = m._build_context(torch.zeros(len(ego), 0, 3584, dtype=torch.float16, device=dev), ego)
            past = lm(inputs_embeds=ctx, use_cache=True).past_key_values
            kv = torch.stack([torch.stack(layer_kv(past, l), 1) for l in range(28)], 1)   # B,28,2,4,C,128
            if not checked:
                L0 = lm.layers[0]
                k = L0.self_attn.k_proj(L0.input_layernorm(ctx))
                k = rope(k.view(len(ego), ctx.shape[1], 4, 128).transpose(1, 2).float(),
                         torch.arange(ctx.shape[1], device=dev))
                err = (k - kv[:, 0, 0].float()).abs().max().item()
                print(f'[kv] RoPE check: layer-0 keys recomputed vs cache, max |diff| {err:.3e} '
                      f'(|k| max {kv[:, 0, 0].float().abs().max().item():.1f})', flush=True)
                assert err < 0.05, err
                checked = True
            KV.append(kv.cpu())
        T = [(t['acc'][:12], t['cur'][:12]) if len(t['acc']) >= 12 else (np.full(12, np.nan),) * 2 for t in R]
        torch.save({'kv': torch.cat(KV), 'tokens': [t['sample_token'] for t in R],
                    'v0': np.array([t['v0'] for t in R]), 'acc': np.stack([a for a, _ in T]),
                    'cur': np.stack([c for _, c in T]), 'ctx_len': int(ctx.shape[1])}, OUT.format(split))
        print(f'[kv] {split}: {len(R)} samples, ctx_len {ctx.shape[1]}, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
