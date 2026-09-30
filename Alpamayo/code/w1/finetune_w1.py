"""w1/finetune_w1.py -- WEEK1 STEP 3: training wrapper for the OFFICIAL split.

Runs the UNCHANGED finetune.py loop (its own __main__ argument parsing, exec'd
verbatim) with these patches:
  data      train = official train minus 50 holdout scenes (n_fut=12);
            "val" = the 50-scene HOLDOUT (n_fut=12), used only for AR-val-ADE
            selection on finetune's fixed 400-sample subset (seed 1234).
            Official val is never loaded here.
  ego       dataset.compute_ego_state -> precomputed causal w1_ego (+ extra rows)
  tokens    tokenizer (ii) via W1Tokenizer (precomputed tokens, percentile centres)
  rollout   ar_eval sees future_speeds[0] = v0_can, heading row = pi/2, GT = the
            lidar-frame trajectory (records.py), so its ADE is in VAD's frame
  loss      plain CE (STEP 2c) unless --weighted
  extra     --cmd: command token; --meta: the two meta-action tokens lat (= command,
            flippable) + lon, INSTEAD of the cmd token; --meta_flip f: training flip
            rate (applies to lat and lon)
  vision    --no_vision: visual tokens REMOVED (context has 0 visual tokens; no image
            IO). The flag is independent of --zero_vision (zeroed but present).
  sampling  natural by default; --turn_weighted = old weights (old CAM curvature > .05)
  --overfit N: train AND select on the same N train samples (STEP 3.5a)
  checkpoints -> models/checkpoints/_w1_<tag>/

  python -m torch.distributed.run --nproc_per_node=8 w1/finetune_w1.py --tag A1 \
      --cmd --no_vision --zero_vision --seed 42 --epochs 10 --patience 5 \
      --batch_size 3 --grad_accum_steps 1
"""
import os, sys, textwrap
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/w1')


def pop_flag(name, has_value=False, default=None):
    if name not in sys.argv:
        return default
    k = sys.argv.index(name)
    if has_value:
        v = sys.argv[k + 1]; del sys.argv[k:k + 2]; return v
    del sys.argv[k]; return True


TAG = pop_flag('--tag', True, 'w1')
CMD = pop_flag('--cmd', default=False)
META = pop_flag('--meta', default=False)
FLIP = float(pop_flag('--meta_flip', True, '0'))
NOVIS = pop_flag('--no_vision', default=False)
WEIGHTED = pop_flag('--weighted', default=False)
OVERFIT = int(pop_flag('--overfit', True, '0'))
LORA_DROP = pop_flag('--lora_dropout', True, None)      # 3.5a retry: 0
MIN_LR_RATIO = pop_flag('--min_lr_ratio', True, None)   # 3.5a retry: 1.0 = constant LR after warmup
FIX = pop_flag('--fix', default=False)                  # 3.5a fix run: w1/fixrun.py
PROBE = pop_flag('--probe_grad', default=False)         # diagnostic c: grad norms per group
# A2 (--meta): the two meta-action tokens lat (= command, FLIPPABLE) and lon; no separate
# (unflippable) cmd token, which would contradict a flipped lat label.
EXTRA = ['lat', 'lon'] if META else (['cmd'] if CMD else [])

import pickle, numpy as np, torch
import dataset, finetune, ar_eval, tokenizer as TKZ
import records, extra_tokens
from model import TRAJ_VOCAB


from w1tok import W1Tokenizer


def build_split_w1(*a, **kw):
    tr = records.build('train', EXTRA, FLIP, flip_seed=42)
    ho = records.build('holdout', EXTRA, 0.0)          # selection sees TRUE labels
    if OVERFIT:
        rs = np.random.RandomState(0)
        tr = [tr[i] for i in sorted(rs.choice(len(tr), OVERFIT, replace=False))]
        ho = tr
        finetune.CFG['val_ade_k'] = OVERFIT
    print(f'[w1] train {len(tr)}  select(holdout) {len(ho)}  extra={EXTRA} flip={FLIP} '
          f'no_vision={bool(NOVIS)} overfit={OVERFIT}', flush=True)
    return tr, ho, []


dataset.compute_ego_state = lambda traj: traj['w1_ego']
dataset.TrajectoryTokenizer = W1Tokenizer
finetune.TrajectoryTokenizer = W1Tokenizer
ar_eval.TrajectoryTokenizer = W1Tokenizer
dataset.build_scene_split = build_split_w1
finetune.build_scene_split = build_split_w1
if not WEIGHTED:
    finetune.get_class_weights = lambda ds, device='cuda:0': torch.ones(TRAJ_VOCAB, device=device)
_GROUP = {}
if EXTRA or FIX or PROBE:
    _load = finetune.load_model

    def load_model_x(*a, **kw):
        m = _load(*a, **kw)
        if EXTRA:
            m = extra_tokens.install(m, EXTRA, records.N_CLS)
        if FIX:
            import fixrun
            mean, sd = fixrun.train_stats()
            m = fixrun.install_fix(m, mean, sd)
            print(f'[w1] FIX: standardise mean {mean.tolist()} sd {sd.tolist()}; post LayerNorm + '
                  f'gain {float(m.ego_encoder.post_gain):.5f}; ego-MLP lr x{fixrun.EGO_LR_MULT}', flush=True)
        for n, q in m.named_parameters():
            if q.requires_grad:
                _GROUP[id(q)] = ('lora' if 'lora_' in n else 'ego_xtok' if '.xtok.' in n else
                                 'ego_mlp' if n.startswith('ego_encoder') else n.split('.')[0])
        return m
    finetune.load_model = load_model_x
if FIX:
    import fixrun
    torch.optim.AdamW = fixrun.LrMultAdamW
if PROBE:
    # diagnostic c: per-group grad L2 norm at every optimizer step (after unscale,
    # before clipping); rank 0 prints; run with --max_steps 200.
    _clip = torch.nn.utils.clip_grad_norm_
    _step = [0]

    def clip_probe(params, max_norm, *a, **kw):
        params = list(params)
        acc = {}
        for q in params:
            if q.grad is not None and id(q) in _GROUP:
                g = _GROUP[id(q)]
                acc[g] = acc.get(g, 0.0) + float(q.grad.detach().float().pow(2).sum())
        _step[0] += 1
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print('[probe] step %d ' % _step[0] + ' '.join(f'{k}={v ** 0.5:.4e}' for k, v in sorted(acc.items())),
                  flush=True)
        return _clip(params, max_norm, *a, **kw)
    torch.nn.utils.clip_grad_norm_ = clip_probe
if NOVIS:
    dataset.preprocess_image = lambda path, augment=False: torch.zeros(3, 1, 1)
    finetune.encode_live = lambda visual, images, device: torch.zeros(
        images.shape[0], 0, 3584, dtype=torch.float16, device=device)
    ar_eval.encode_live_one = lambda visual, images, device, dtype=torch.float16: torch.zeros(
        1, 0, 3584, dtype=dtype, device=device)
if LORA_DROP is not None:
    finetune.CFG['lora_dropout'] = float(LORA_DROP)
if MIN_LR_RATIO is not None:
    finetune.CFG['min_lr_ratio'] = float(MIN_LR_RATIO)
finetune.CFG['checkpoint_dir'] = f'{finetune.CFG["checkpoint_dir"]}/_w1_{TAG}'
os.makedirs(finetune.CFG['checkpoint_dir'], exist_ok=True)
print(f'[w1] ckpt dir {finetune.CFG["checkpoint_dir"]}  plain_CE={not WEIGHTED}  '
      f'lora_dropout={finetune.CFG["lora_dropout"]}  min_lr_ratio={finetune.CFG["min_lr_ratio"]}', flush=True)

if __name__ == '__main__':
    src = open(f'{CODE}/finetune.py').read()
    exec(textwrap.dedent(src.split("if __name__ == '__main__':", 1)[1]), finetune.__dict__)
