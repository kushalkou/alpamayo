"""w1/fixrun.py -- planner item 2 FIX RUN (only if the 3.5a retry fails).

install_fix(model, mean, sd) changes ONLY the ego pathway:
  1. standardise the 4 ego features with OFFICIAL-TRAIN statistics (per feature, pooled
     over the 4 rows); stored as buffers ego_encoder.in_mean / in_sd, so checkpoints
     and dumps carry them (save_checkpoint keeps every 'ego_encoder*' key);
  2. no absolute heading / position to drop: ego_state_w1 already has none (yaw is
     relative to t0; no x/y) -- verified in diag_inputs.py (b);
  3. after the ego MLP: LayerNorm(3584) then a learnable scalar gain, initialised so
     the token L2 norm matches the Cosmos text-embedding mean norm (0.870):
     gain0 = 0.870 / sqrt(3584) (unit-weight LayerNorm output has norm ~sqrt(D));
  4. ego-MLP LR x10 (ego_encoder.net + post_ln + post_gain; NOT the cmd embedding)
     via LrMultAdamW: finetune.py resets every group's lr each step and Adam is
     gradient-scale invariant, so the multiplier is applied inside step().
"""
import math, types
import numpy as np, torch, torch.nn as nn

TEXT_NORM = 0.870
EGO_LR_MULT = 10.0
_EGO_PARAM_IDS = set()


def train_stats():
    import records
    tr = records.build('train', ())
    F = torch.stack([t['w1_ego'][:4] for t in tr]).reshape(-1, 4).double()
    mean, sd = F.mean(0), F.std(0)
    sd[sd < 1e-6] = 1.0
    return mean.float(), sd.float()


def install_fix(model, mean, sd):
    enc = model.ego_encoder
    p = next(enc.net.parameters())
    D = enc.net[-1].out_features
    enc.register_buffer('in_mean', mean.to(p.device, torch.float32))
    enc.register_buffer('in_sd', sd.to(p.device, torch.float32))
    enc.post_ln = nn.LayerNorm(D).to(p.device, torch.float32)
    enc.post_gain = nn.Parameter(torch.tensor(TEXT_NORM / math.sqrt(D), device=p.device))

    def forward(self, x):
        h = self.net((x - self.in_mean) / self.in_sd)
        return self.post_ln(h) * self.post_gain

    enc.forward = types.MethodType(forward, enc)
    _EGO_PARAM_IDS.clear()
    for n, q in enc.named_parameters():
        if not n.startswith('xtok.'):
            _EGO_PARAM_IDS.add(id(q))
    return model


class LrMultAdamW(torch.optim.AdamW):
    """AdamW whose ego-MLP params get lr x EGO_LR_MULT at every step (group-level)."""
    def __init__(self, params, **kw):
        params = list(params)
        ego = [q for q in params if id(q) in _EGO_PARAM_IDS]
        rest = [q for q in params if id(q) not in _EGO_PARAM_IDS]
        groups = [{'params': rest, 'lr_mult': 1.0}]
        if ego:
            groups.append({'params': ego, 'lr_mult': EGO_LR_MULT})
        super().__init__(groups, **kw)

    @torch.no_grad()
    def step(self, closure=None):
        saved = [g['lr'] for g in self.param_groups]
        for g in self.param_groups:
            g['lr'] = g['lr'] * g.get('lr_mult', 1.0)
        out = super().step(closure)
        for g, lr in zip(self.param_groups, saved):
            g['lr'] = lr
        return out
