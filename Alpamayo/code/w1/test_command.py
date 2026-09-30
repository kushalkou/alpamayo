"""w1/test_command.py -- command token plumbing, on a tiny stand-in model (CPU)."""
import sys, torch, torch.nn as nn
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import command as C


class Enc(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.Linear(4, 8)
    def forward(s, x): return s.l(x)


class Tiny(nn.Module):
    def __init__(s): super().__init__(); s.ego_encoder = Enc()
    def _build_context(s, vis, ego): return torch.cat([vis, s.ego_encoder(ego.float())], 1)


def test_plumbing():
    m = C.install(Tiny(), dim=8)
    vis = torch.zeros(2, 3, 8)
    e = torch.stack([C.ego_rows(torch.randn(4, 4), c) for c in (0, 2)])
    ctx = m._build_context(vis, e)
    assert ctx.shape == (2, 3 + 4 + 1, 8)
    assert torch.equal(ctx[0, -1], m.ego_encoder.cmd_embed.weight[0])
    assert torch.equal(ctx[1, -1], m.ego_encoder.cmd_embed.weight[2])
    assert torch.equal(ctx[:, :7], m.__class__._build_context(m, vis, e[:, :4]))
    assert 'ego_encoder.cmd_embed.weight' in m.state_dict()      # saved by finetune
    ctx.sum().backward()
    assert m.ego_encoder.cmd_embed.weight.grad is not None       # trains


if __name__ == '__main__':
    test_plumbing(); print('PASS test_plumbing')
