"""ar1/bench_inputs.py -- R1.1 cost benchmark (NOT training; no checkpoint, no eval).

NOT YET RUN (launch needs user approval; see AR1_R1_REPORT.md).

Stage-1 trajectory-token step (finetune_w1 --fix --cmd recipe: LoRA r16 q/v/o, fix ego
MLP x10 lr, cmd token, plain CE in fp32, GradScaler, grad checkpointing) with the AR1-
style visual context: 3 cams x 4 frames x 160 = 1,920 visual tokens, live-encoded by
the frozen tower (vision_ar1.encode). Context = 1,920 vis + 4 ego + 1 cmd = 1,925, plus
23 teacher-forced trajectory tokens. batch 2 per GPU, 8 GPUs (DDP), --steps N; timing
over steps 11..N (data loading included). Prints peak memory and s/step.
  python -m torch.distributed.run --nproc_per_node=8 ar1/bench_inputs.py --steps 60
"""
import os, sys, time, pickle, argparse
import numpy as np, torch, torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1'); sys.path.insert(0, f'{CODE}/ar1')
import records, extra_tokens, fixrun
import vision_ar1 as V
from model import load_model
from finetune import CFG

HIST = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_hist.pkl'


class DS(torch.utils.data.Dataset):
    def __init__(self, R, H):
        self.R, self.H = R, H

    def __len__(self):
        return len(self.R)

    def __getitem__(self, i):
        t = self.R[i]; h = self.H[t['sample_token']]['cams']
        img = torch.stack([V.preprocess(p) for c in h for p, _ in h[c]])     # [12,3,280,448]
        tok = torch.tensor([p[0] for p in t['w1_tokens']] + [p[1] for p in t['w1_tokens']])
        return {'images': img, 'ego': t['w1_ego'], 'tok': tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--batch_size', type=int, default=2)
    a = ap.parse_args()
    dist.init_process_group(backend='nccl')
    lr_ = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr_)
    dev = f'cuda:{lr_}'; m0 = lr_ == 0
    torch.manual_seed(42 + lr_)
    vla = load_model(lora_rank=CFG['lora_rank'], lora_alpha=CFG['lora_alpha'],
                     lora_dropout=CFG['lora_dropout'], device=dev)
    vla = extra_tokens.install(vla, ['cmd'], records.N_CLS)
    mean, sd = fixrun.train_stats()
    vla = fixrun.install_fix(vla, mean, sd)
    visual = vla.cosmos.model.visual
    model = DDP(vla, device_ids=[lr_])
    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = fixrun.LrMultAdamW(trainable, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda')
    tr = records.build('train', ['cmd'], 0.0)
    H = pickle.load(open(HIST, 'rb'))
    samp = torch.utils.data.distributed.DistributedSampler(DS(tr, H), shuffle=True, seed=42)
    dl = torch.utils.data.DataLoader(DS(tr, H), batch_size=a.batch_size, sampler=samp,
                                     num_workers=6, pin_memory=True, drop_last=True)
    spe = len(dl)
    ce = nn.CrossEntropyLoss()
    model.train(); torch.cuda.reset_peak_memory_stats(dev)
    times = []; t = time.time(); ctx_len = None
    for step, b in enumerate(dl, 1):
        B = b['images'].shape[0]
        vt = V.encode(visual, b['images'].to(dev).flatten(0, 1)).reshape(B, 12 * V.TOK_PER_IMG, -1)
        ctx_len = vt.shape[1] + b['ego'].shape[1]
        logits = model(vt, b['ego'].to(dev, torch.float32), b['tok'].to(dev))
        loss = ce(logits.float().reshape(-1, logits.shape[-1]), b['tok'].to(dev).reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(trainable, CFG['grad_clip'])
        scaler.step(opt); scaler.update(); opt.zero_grad()
        torch.cuda.synchronize(dev)
        times.append(time.time() - t); t = time.time()
        if m0 and step % 10 == 0:
            print(f'  step {step} loss {loss.item():.3f} {times[-1]:.2f}s', flush=True)
        if step >= a.steps:
            break
    peak = torch.tensor([torch.cuda.max_memory_allocated(dev) / 1e9,
                         torch.cuda.max_memory_reserved(dev) / 1e9], device=dev)
    allp = [torch.zeros_like(peak) for _ in range(dist.get_world_size())]
    dist.all_gather(allp, peak)
    if m0:
        sps = float(np.mean(times[10:])); med = float(np.median(times[10:]))
        pk = torch.stack(allp).max(0).values.tolist()
        print(f'[bench] visual tokens/sample {12 * V.TOK_PER_IMG}; context {ctx_len} '
              f'(+23 traj in) = {ctx_len + 23} LM tokens/sample; batch {a.batch_size} x 8 GPUs')
        print(f'[bench] peak allocated {pk[0]:.1f} GB, reserved {pk[1]:.1f} GB (max over ranks)')
        print(f'[bench] s/step mean {sps:.2f} median {med:.2f} (steps 11..{a.steps}); '
              f'steps/epoch {spe}; 10 epochs = {10 * spe * sps / 3600:.1f} h wall = '
              f'{80 * spe * sps / 3600:.0f} GPU-h (training only)')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
