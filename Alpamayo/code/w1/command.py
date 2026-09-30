"""w1/command.py -- WEEK1 item 4: 3-way navigation command as ONE learned context token.

Rule (VAD tools/data_converter/vad_nuscenes_converter.py, computed in w1/build_data.py):
    # drive command according to final fut step offset from lcf
    if ego_fut_trajs[-1][0] >= 2:    command = Turn Right   (0)
    elif ego_fut_trajs[-1][0] <= -2: command = Turn Left    (1)
    else:                            command = Go Straight  (2)
  ego_fut_trajs[-1] = LIDAR_TOP position at future step 6 (3 s) in the t0 LIDAR frame
  (x = right), i.e. the command is DERIVED FROM THE GT FUTURE -- the literature
  convention (VAD / UniAD / ST-P3 style), not a causal input.

Plumbing (no frozen file edited): the command rides in a 5th ego_state row
[cmd, 0, 0, 0] so it flows through training, AR validation and dumps unchanged; the
patched _build_context splits it off and appends ONE token = cmd_embed(cmd) after the
4 ego tokens (context 1540 -> 1541). cmd_embed lives INSIDE ego_encoder
('ego_encoder.cmd_embed.weight') so finetune.save_checkpoint keeps it, it is fp32 master
like the other adapters, and it trains. Without install(), a 5-row ego_state is an error.
"""
import types
import torch, torch.nn as nn

N_CMD = 3


def ego_rows(ego4, command):
    """[4,4] ego tensor + int command -> [5,4] (command in row 4, col 0)."""
    row = torch.zeros(1, ego4.shape[1], dtype=ego4.dtype); row[0, 0] = float(command)
    return torch.cat([ego4, row], 0)


def install(model, dim=3584, std=0.02):
    enc = model.ego_encoder
    p = next(enc.parameters())
    enc.cmd_embed = nn.Embedding(N_CMD, dim).to(device=p.device, dtype=torch.float32)
    nn.init.normal_(enc.cmd_embed.weight, std=std)
    orig = model._build_context

    def _build_context(self, visual_tokens, ego_state):
        assert ego_state.shape[1] == 5, 'command model expects a 5-row ego_state'
        cmd = ego_state[:, 4, 0].round().long()
        ctx = orig(visual_tokens, ego_state[:, :4])
        tok = self.ego_encoder.cmd_embed(cmd).to(ctx.dtype).unsqueeze(1)
        return torch.cat([ctx, tok], dim=1)

    model._build_context = types.MethodType(_build_context, model)
    return model
