"""leak/finetune_causal.py -- OVERNIGHT2 item 6: retrain with CAUSAL ego features.

Runs the UNCHANGED finetune.py training loop, with three changes applied by patching:
  1. dataset.compute_ego_state -> leak.causal_ego.causal_ego_state (past poses only)
  2. every split filtered to samples with >= 2 past poses (causal features defined)
  3. checkpoint_dir -> models/checkpoints/_causal_<tag> (never touches existing ckpts)
Arguments are parsed by finetune.py's own __main__ block (exec'd verbatim).

Y1 ego recipe:
  python -m torch.distributed.run --nproc_per_node=8 leak/finetune_causal.py \
      --tag ego_s42 --zero_vision --turn_weighted --seed 42 --epochs 10 --patience 5 \
      --batch_size 3 --grad_accum_steps 1   (Y1: 698 steps/GPU/epoch = bs 3, accum 1)
"""
import os, sys
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE); sys.path.insert(0, f'{CODE}/leak')

tag = 'causal'
if '--tag' in sys.argv:
    k = sys.argv.index('--tag'); tag = sys.argv[k + 1]; del sys.argv[k:k + 2]
NOVIS = '--no_vision' in sys.argv          # WEEK1 A0: visual tokens REMOVED (not zeroed)
if NOVIS:
    sys.argv.remove('--no_vision')

import causal_ego
causal_ego.patch()
import dataset
_orig_split = dataset.build_scene_split


def build_scene_split_causal(*a, **kw):
    tr, va, te = _orig_split(*a, **kw)
    f = lambda L: [t for t in L if causal_ego.keep(t)]
    out = f(tr), f(va), f(te)
    print(f'[causal] filtered to >=2 past poses: train {len(out[0])}/{len(tr)}  '
          f'val {len(out[1])}/{len(va)}  test {len(out[2])}/{len(te)}', flush=True)
    return out


dataset.build_scene_split = build_scene_split_causal
import finetune
finetune.build_scene_split = build_scene_split_causal
finetune.CFG['checkpoint_dir'] = f'{finetune.CFG["checkpoint_dir"]}/_causal_{tag}'
os.makedirs(finetune.CFG['checkpoint_dir'], exist_ok=True)
print(f'[causal] ckpt dir {finetune.CFG["checkpoint_dir"]}', flush=True)

# Ego-only speedup, EXACTLY equivalent: with --zero_vision the recipe computes the visual
# tokens and then replaces them with zeros_like(). Skip the image load + frozen encoder and
# hand back those zeros directly (same shape [B,1536,3584], same fp16 dtype). No RNG is
# involved in either path (augment is off), so training is numerically identical.
if NOVIS:
    import torch, ar_eval
    dataset.preprocess_image = lambda path, augment=False: torch.zeros(3, 1, 1)
    finetune.encode_live = lambda visual, images, device: torch.zeros(
        images.shape[0], 0, 3584, dtype=torch.float16, device=device)
    ar_eval.encode_live_one = lambda visual, images, device, dtype=torch.float16: torch.zeros(
        1, 0, 3584, dtype=dtype, device=device)
    print('[causal] no_vision: visual tokens REMOVED (context = 4 ego tokens)', flush=True)
elif '--zero_vision' in sys.argv:
    import torch, ar_eval
    dataset.preprocess_image = lambda path, augment=False: torch.zeros(3, 1, 1)
    finetune.encode_live = lambda visual, images, device: torch.zeros(
        images.shape[0], 1536, 3584, dtype=torch.float16, device=device)
    ar_eval.encode_live_one = lambda visual, images, device, dtype=torch.float16: torch.zeros(
        1, 1536, 3584, dtype=dtype, device=device)
    print('[causal] zero_vision: image load + encoder skipped (outputs are zeroed anyway)',
          flush=True)

src = open(f'{CODE}/finetune.py').read()
main_block = src.split("if __name__ == '__main__':", 1)[1]
import textwrap
exec(textwrap.dedent(main_block), finetune.__dict__)
