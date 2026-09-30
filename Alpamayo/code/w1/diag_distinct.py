"""w1/diag_distinct.py -- planner amendment item e (CPU, no training): are the context
tokens distinguishable across samples?

For the 256 overfit samples and a given checkpoint:
  ego tokens  = EgoEncoderMLP(w1_ego rows 0-3) in eval mode (+ the post-MLP norm layer
                if the checkpoint has one, i.e. the fix run) -> [256, 4, 3584]
  cmd token   = cmd embedding of each sample's command -> [256, 3584]
Per token position:
  - mean pairwise cosine similarity across samples (off-diagonal), RAW and after the
    layer-0 input RMSNorm of the LM (model.layers.0.input_layernorm, eps 1e-6);
  - share of the norm explained by the across-sample mean vector:
        ||mean_i x_i||^2 / mean_i ||x_i||^2.
Usage: python w1/diag_distinct.py <ckpt.pt>
"""
import sys
import numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records
from model import EgoEncoderMLP
from safetensors import safe_open

COS = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'


def rmsnorm(x, w, eps=1e-6):
    return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w


def pair_cos(X):
    Xn = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    G = Xn @ Xn.T; n = len(X)
    return float((G.sum() - G.diagonal().sum()) / (n * (n - 1)))


def mean_share(X):
    return float(X.mean(0).norm() ** 2 / (X.norm(dim=-1) ** 2).mean())


def main():
    ck = torch.load(sys.argv[1], map_location='cpu')['model_state']
    tr = records.build('train', ('cmd',))
    rs = np.random.RandomState(0)
    ov = [tr[i] for i in sorted(rs.choice(len(tr), 256, replace=False))]
    with safe_open(f'{COS}/model-00001-of-00004.safetensors', 'pt') as f:
        w0 = f.get_tensor('model.layers.0.input_layernorm.weight').float()
    enc = EgoEncoderMLP()
    sd = {k[len('ego_encoder.'):]: v for k, v in ck.items() if k.startswith('ego_encoder.net.')}
    enc.load_state_dict(sd); enc.eval()
    post = {k: v for k, v in ck.items() if k.startswith('ego_encoder.post')}
    X = torch.stack([t['w1_ego'][:4] for t in ov])
    with torch.no_grad():
        ego = enc(X).float()
        if post:                                       # fix run: LayerNorm + scalar gain
            ln = torch.nn.LayerNorm(ego.shape[-1])
            ln.load_state_dict({'weight': post['ego_encoder.post_ln.weight'],
                                'bias': post['ego_encoder.post_ln.bias']})
            ego = ln(ego) * post['ego_encoder.post_gain']
    cmdw = ck.get('ego_encoder.xtok.cmd.weight')
    print(f'checkpoint {sys.argv[1]}  (post-MLP norm layer: {"yes" if post else "no"})')
    print(f'{"token":12} {"norm":>8} {"cos raw":>9} {"cos RMSNorm":>12} {"mean-share raw":>15} {"mean-share RMS":>15}')
    rows = [(f'ego t-{3-p}' if p < 3 else 'ego t', ego[:, p]) for p in range(4)]
    if cmdw is not None:
        rows.append(('cmd', cmdw[[t['command'] for t in ov]].float()))
    for name, T in rows:
        R = rmsnorm(T, w0)
        print(f'{name:12} {T.norm(dim=-1).mean():8.3f} {pair_cos(T):9.4f} {pair_cos(R):12.4f} '
              f'{mean_share(T):15.4f} {mean_share(R):15.4f}')
    c = np.bincount([t['command'] for t in ov], minlength=3)
    print(f'(cmd has only 3 distinct vectors; the 256 samples split right/left/straight = {list(c)})')


if __name__ == '__main__':
    main()
