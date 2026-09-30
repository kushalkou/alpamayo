"""w1/test_w1.py -- STEP 3 unit tests for the official-split training wrapper.
Run: python w1/test_w1.py (CPU; loads the w1 data pickles)."""
import sys, math, copy, types, pickle
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records, extra_tokens
from ar_eval import unicycle_rollout

D = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
W = pickle.load(open(f'{D}/w1_data.pkl', 'rb'))
V0 = pickle.load(open(f'{D}/w1_v0.pkl', 'rb'))
TT = pickle.load(open(f'{D}/w1_targets.pkl', 'rb'))


class Guard(dict):
    def __init__(self, d, allowed): super().__init__(d); self.allowed = allowed
    def __getitem__(self, k):
        assert k in self.allowed, f'forbidden key {k!r}'; return dict.__getitem__(self, k)
    def get(self, k, default=None):
        assert k in self.allowed, f'forbidden key {k!r}'; return dict.get(self, k, default)


def test_ego_causal():
    """ego_state_w1 reads only past/current poses and the t0-causal CAN fields."""
    n = 0
    for r in W['records'][:400]:
        can = Guard(V0[r['sample_token']], {'can_ok', 'v0_can', 'yr_can', 'a_can'})
        e = records.ego_state_w1(r['past_poses'], r['current_pose'], can)
        assert e.shape == (4, 4) and torch.isfinite(e).all()
        assert abs(float(e[3, 1]) - math.pi / 2) < 1e-6          # rollout heading row
        n += 1
    assert n == 400


def test_record_ignores_future():
    """scrambling every future_* field of a record leaves its w1_ego unchanged."""
    r = [x for x in W['records'] if len(x['past_poses']) == 4][10]
    a = records.ego_state_w1(r['past_poses'], r['current_pose'], V0[r['sample_token']])
    r2 = copy.deepcopy(r)
    for k in list(r2):
        if k.startswith('future'): r2[k] = np.random.randn(*np.shape(r2[k])) * 100
    b = records.ego_state_w1(r2['past_poses'], r2['current_pose'], V0[r['sample_token']])
    assert torch.equal(a, b)


def test_tokenizer_and_rollout_path():
    """GT tokens through the UNCHANGED ar_eval rollout reproduce the step-2a floor, in
    the lidar frame, with the patched fields (v0 = future_speeds[0], yaw0 = ego[3,1])."""
    import finetune_w1 as FW
    tok = FW.W1Tokenizer()
    R = records.build('holdout', ('cmd',))[:200]
    for t in R:
        ak = tok.tokenize(t)
        acc, cur = zip(*[tok.detokenize_step(a, k) for a, k in ak])
        pred, _ = unicycle_rollout(acc, cur, float(t['future_speeds'][0]), float(t['w1_ego'][3, 1]))
        from targets import detok_roll
        from tok_floor import Bins
        ref = detok_roll(Bins(*TT['bins']), TT['targets'][t['sample_token']]['tokens'], t['v0'])
        assert np.abs(pred - ref).max() < 1e-4, np.abs(pred - ref).max()   # float32 pi/2 heading row
        gtl = np.array(t['future_positions']) - np.array(t['current_pose']['translation'][:2])
        assert np.allclose(gtl, TT['targets'][t['sample_token']]['P'][:12])
        assert tok.tokenize(t) == [tuple(x) for x in TT['targets'][t['sample_token']]['tokens'][:12]]


class Enc(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.Linear(4, 8)
    def forward(s, x): return s.l(x)


class Tiny(nn.Module):
    def __init__(s): super().__init__(); s.ego_encoder = Enc()
    def _build_context(s, vis, ego): return torch.cat([vis, s.ego_encoder(ego.float())], 1)


def test_extra_tokens_and_no_vision():
    m = extra_tokens.install(Tiny(), ['cmd', 'lon'], {'cmd': 3, 'lon': 4}, dim=8)
    ego = torch.randn(2, 6, 4); ego[:, 4, 0] = torch.tensor([2., 0.]); ego[:, 5, 0] = torch.tensor([3., 1.])
    for nvis in (5, 0):                                      # present vs REMOVED
        ctx = m._build_context(torch.zeros(2, nvis, 8), ego)
        assert ctx.shape == (2, nvis + 4 + 2, 8)
        assert torch.equal(ctx[1, -2], m.ego_encoder.xtok['cmd'].weight[0])
        assert torch.equal(ctx[0, -1], m.ego_encoder.xtok['lon'].weight[3])
    sd = m.state_dict()
    assert 'ego_encoder.xtok.cmd.weight' in sd and 'ego_encoder.xtok.lon.weight' in sd
    ctx.sum().backward(); assert m.ego_encoder.xtok['lon'].weight.grad is not None


def test_flips():
    R = records.build('train', ('cmd', 'lon'), 0.10, flip_seed=42)
    lon_flip = np.mean([t['shown']['lon'] != t['meta_lon'] for t in R])
    lat_flip = np.mean([t['shown']['lat'] != t['command'] for t in R])
    cmd_flip = np.mean([t['shown']['cmd'] != t['command'] for t in R])
    assert 0.08 < lon_flip < 0.12 and 0.08 < lat_flip < 0.12 and cmd_flip == 0, (lon_flip, lat_flip)
    assert all(float(t['w1_ego'][5, 0]) == t['shown']['lon'] for t in R[:500])
    R0 = records.build('train', ('cmd', 'lon'), 0.0)
    assert all(t['shown']['lon'] == t['meta_lon'] for t in R0)


if __name__ == '__main__':
    for f in (test_ego_causal, test_record_ignores_future, test_tokenizer_and_rollout_path,
              test_extra_tokens_and_no_vision, test_flips):
        f(); print(f'PASS {f.__name__}')
