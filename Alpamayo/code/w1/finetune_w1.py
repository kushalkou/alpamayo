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
  extra     --cmd: command token; --meta: + meta-action lon token (lat = command
            token, which is then flippable); --meta_flip f: training flip rate
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
EXTRA = (['cmd'] if CMD else []) + (['lon'] if META else [])

import pickle, numpy as np, torch
import dataset, finetune, ar_eval, tokenizer as TKZ
import records, extra_tokens
from model import TRAJ_VOCAB


class W1Tokenizer(TKZ.TrajectoryTokenizer):
    """Tokenizer (ii): percentile bins (median centres) on the lidar-frame targets.
    tokenize() returns the precomputed tokens; detokenize_step() is inherited
    (STOP -> (0,0), else the bin centres below)."""
    def __init__(self):
        b = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_targets.pkl', 'rb'))['bins']
        self.accel_centers, self.curv_centers = np.asarray(b[0]), np.asarray(b[1])
        self.accel_bins, self.curv_bins = np.asarray(b[2]), np.asarray(b[3])

    def tokenize(self, traj):
        return list(traj['w1_tokens'])


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
if EXTRA:
    _load = finetune.load_model

    def load_model_x(*a, **kw):
        return extra_tokens.install(_load(*a, **kw), EXTRA, records.N_CLS | {'cmd': 3})
    finetune.load_model = load_model_x
if NOVIS:
    dataset.preprocess_image = lambda path, augment=False: torch.zeros(3, 1, 1)
    finetune.encode_live = lambda visual, images, device: torch.zeros(
        images.shape[0], 0, 3584, dtype=torch.float16, device=device)
    ar_eval.encode_live_one = lambda visual, images, device, dtype=torch.float16: torch.zeros(
        1, 0, 3584, dtype=dtype, device=device)
finetune.CFG['checkpoint_dir'] = f'{finetune.CFG["checkpoint_dir"]}/_w1_{TAG}'
os.makedirs(finetune.CFG['checkpoint_dir'], exist_ok=True)
print(f'[w1] ckpt dir {finetune.CFG["checkpoint_dir"]}  plain_CE={not WEIGHTED}', flush=True)

if __name__ == '__main__':
    src = open(f'{CODE}/finetune.py').read()
    exec(textwrap.dedent(src.split("if __name__ == '__main__':", 1)[1]), finetune.__dict__)
