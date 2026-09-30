"""w1/test_fixrun.py -- fix-run plumbing on a real EgoEncoderMLP (CPU)."""
import sys, math, torch, torch.nn as nn
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import fixrun, extra_tokens
from model import EgoEncoderMLP


class M(nn.Module):
    def __init__(s): super().__init__(); s.ego_encoder = EgoEncoderMLP()
    def _build_context(s, vis, ego): return torch.cat([vis, s.ego_encoder(ego.float())], 1)


def test_fix():
    m = extra_tokens.install(M(), ['cmd'], {'cmd': 3})
    mean, sd = torch.tensor([5., 1.57, 0., 0.]), torch.tensor([3.6, .08, .09, 1.])
    m = fixrun.install_fix(m, mean, sd); m.eval()
    x = torch.randn(64, 4, 4) * sd + mean
    out = m.ego_encoder(x)
    assert abs(out.norm(dim=-1).mean().item() - fixrun.TEXT_NORM) < 0.02, out.norm(dim=-1).mean()
    sdict = m.state_dict()
    for k in ('ego_encoder.in_mean', 'ego_encoder.in_sd', 'ego_encoder.post_ln.weight', 'ego_encoder.post_gain'):
        assert k in sdict, k
    ego_ids = {id(p) for n, p in m.ego_encoder.named_parameters() if not n.startswith('xtok.')}
    assert fixrun._EGO_PARAM_IDS == ego_ids and id(m.ego_encoder.xtok['cmd'].weight) not in ego_ids
    # LR multiplier: identical grads -> first Adam step is 10x larger for ego params
    a = nn.Parameter(torch.zeros(3)); b = m.ego_encoder.post_gain
    fixrun._EGO_PARAM_IDS.add(id(b))
    opt = fixrun.LrMultAdamW([a, b], lr=1e-3, weight_decay=0.0)
    g0 = b.detach().clone(); a.grad = torch.ones(3); b.grad = torch.ones(())
    opt.step()
    ra, rb = a.detach().abs().mean().item(), (b.detach() - g0).abs().item()
    assert abs(rb / ra - 10.0) < 1e-3, (ra, rb)
    assert opt.param_groups[0]['lr'] == 1e-3 and opt.param_groups[1]['lr'] == 1e-3   # restored
    # context still built end to end
    ego = torch.cat([x[:2], torch.tensor([[[2., 0, 0, 0]], [[0., 0, 0, 0]]])], 1)
    assert m._build_context(torch.zeros(2, 0, 3584), ego).shape == (2, 5, 3584)


if __name__ == '__main__':
    test_fix(); print('PASS test_fix')
