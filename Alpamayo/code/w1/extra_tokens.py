"""w1/extra_tokens.py -- learned context tokens for command / meta-action (generalises
w1/command.py). ego_state rows 4.. carry [class,0,0,0]; each named row gets its own
nn.Embedding(n_classes, 3584), stored at ego_encoder.xtok.<name> so
finetune.save_checkpoint keeps it (prefix 'ego_encoder'), fp32 master, trained.
Tokens are appended after the 4 ego tokens, in row order."""
import types
import torch, torch.nn as nn


def install(model, names, n_cls, dim=3584, std=0.02):
    enc = model.ego_encoder
    p = next(enc.parameters())
    enc.xtok = nn.ModuleDict({k: nn.Embedding(n_cls[k], dim) for k in names}).to(
        device=p.device, dtype=torch.float32)
    for e in enc.xtok.values():
        nn.init.normal_(e.weight, std=std)
    orig = model._build_context
    names = list(names)

    def _build_context(self, visual_tokens, ego_state):
        assert ego_state.shape[1] == 4 + len(names), (ego_state.shape, names)
        ctx = orig(visual_tokens, ego_state[:, :4])
        toks = [self.ego_encoder.xtok[k](ego_state[:, 4 + j, 0].round().long())
                .to(ctx.dtype).unsqueeze(1) for j, k in enumerate(names)]
        return torch.cat([ctx] + toks, dim=1)

    model._build_context = types.MethodType(_build_context, model)
    return model
