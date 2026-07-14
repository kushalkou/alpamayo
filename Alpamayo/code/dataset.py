"""
dataset.py — NuScenes VLA Dataset

Key fixes vs previous version:
  - Split by SCENE (not sample) to prevent data leakage
  - Compute class weights for weighted loss (fixes modal token collapse)
  - Load precomputed visual tokens from disk
  - No nuScenes API needed at __getitem__ time

Usage:
    from dataset import get_dataloaders, get_class_weights
    train_loader, val_loader, _ = get_dataloaders(...)
    weights = get_class_weights(train_loader.dataset)
"""

import os
import sys
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from tokenizer import TrajectoryTokenizer
from vision_live import preprocess_image, CAMERAS

# ── Constants ─────────────────────────────────────────────────────────────────

NUSCENES_ROOT    = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
TRAJ_PATH        = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/trajectories_full.pkl'
TRAJ_VOCAB       = 129
TRAJ_LEN         = 24

SPLIT_SEED  = 42
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# test = remaining 0.15

# ── Ego state ─────────────────────────────────────────────────────────────────

def quat_to_yaw(rotation):
    w, x, y, z = rotation
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

def pose_to_xyyaw(pose):
    x   = float(pose['translation'][0])
    y   = float(pose['translation'][1])
    yaw = quat_to_yaw(pose['rotation'])
    return x, y, yaw

def compute_ego_state(traj):
    """Per-timestep ego kinematics for the 4 ego tokens: rows = [t-3, t-2, t-1, t].

    W1 FIX. The previous version used a BACKWARD difference over `past_poses`, which
    (a) returned speed=0 whenever history was padded (many samples have <4 past poses;
    some have 0), and (b) systematically under-estimated speed at accelerations. The model
    therefore never received the car's true speed — the root cause behind "no model beats
    the constant-velocity baseline" and "more inputs → worse".

    New scheme: FORWARD differences over the chronological sequence [past..., current,
    future[0]]. The current row's speed becomes dist(current→future[0])/dt, which equals
    `future_speeds[0]` exactly (verified), i.e. the true current velocity — the same speed
    the unicycle rollout is seeded with. All quantities are current *instantaneous*
    kinematics (speed / yaw / yaw-rate / accel a real vehicle reads from its own sensors),
    never a padded zero.
    """
    dt = 0.5

    past = list(traj.get('past_poses', []))
    seq_poses = past + [traj['current_pose']]          # chronological, ends at current
    xyy = [pose_to_xyyaw(p) for p in seq_poses]        # [(x,y,yaw), ...]

    # Forward anchor from the immediate future so the CURRENT row gets a real forward diff.
    fx = float(traj['future_positions'][0][0])
    fy = float(traj['future_positions'][0][1])
    fyaws = traj.get('future_yaws', None)
    fyaw = float(fyaws[0]) if fyaws is not None and len(fyaws) > 0 else xyy[-1][2]

    ext_x   = [p[0] for p in xyy] + [fx]
    ext_y   = [p[1] for p in xyy] + [fy]
    ext_yaw = [p[2] for p in xyy] + [fyaw]
    m = len(xyy)                                        # real poses; current index = m-1

    def ang(a):
        return float(np.arctan2(np.sin(a), np.cos(a)))

    fwd_speed, fwd_yawrate = [], []
    for i in range(m):
        d = float(np.hypot(ext_x[i+1]-ext_x[i], ext_y[i+1]-ext_y[i]))
        fwd_speed.append(d / dt)                        # forward diff => true velocity at i
        fwd_yawrate.append(ang(ext_yaw[i+1]-ext_yaw[i]) / dt)

    fs = traj.get('future_speeds', None)
    states = []
    for i in range(m):
        if i < m - 1:
            accel = (fwd_speed[i+1] - fwd_speed[i]) / dt
        else:                                           # current row: use next future speed
            accel = ((float(fs[1]) - fwd_speed[i]) / dt) if (fs is not None and len(fs) > 1) else 0.0
        states.append([fwd_speed[i], ext_yaw[i], fwd_yawrate[i], accel])

    # Left-pad by repeating the EARLIEST real state (never a zero row) to get 4 rows.
    while len(states) < 4:
        states = [states[0]] + states
    states = states[-4:]

    return torch.tensor(states, dtype=torch.float32)   # [4, 4]

# ── Scene-based split ─────────────────────────────────────────────────────────

def build_scene_split(trajectories, nuscenes_root=NUSCENES_ROOT, seed=SPLIT_SEED):
    """
    Split trajectories by SCENE to prevent data leakage.
    Consecutive frames from the same scene must all go to the same split.

    Returns (train_trajs, val_trajs, test_trajs)
    """
    from nuscenes.nuscenes import NuScenes
    print("[dataset] Loading nuScenes to build scene-based split...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_root, verbose=False)

    # sample_token -> scene_token
    sample_to_scene = {}
    for scene in nusc.scene:
        token = scene['first_sample_token']
        while token:
            sample_to_scene[token] = scene['token']
            token = nusc.get('sample', token)['next']

    # Group trajectories by scene; resolve the 6 camera image paths ONCE here
    # (live vision path — no precomputed token cache anymore) so __getitem__
    # needs no nuScenes handle.
    scene_to_trajs = {}
    missing = 0
    for traj in trajectories:
        st = traj['sample_token']
        sc = sample_to_scene.get(st)
        if sc is None:
            missing += 1
            continue
        sample = nusc.get('sample', st)
        traj['cam_paths'] = {c: nusc.get_sample_data_path(sample['data'][c])
                             for c in CAMERAS}
        scene_to_trajs.setdefault(sc, []).append(traj)

    if missing:
        print(f"[dataset] WARNING: {missing} trajectories had no scene mapping (skipped)")

    # Shuffle scenes deterministically
    scene_tokens = sorted(scene_to_trajs.keys())
    rng = random.Random(seed)
    rng.shuffle(scene_tokens)

    n        = len(scene_tokens)
    n_train  = int(n * TRAIN_FRAC)
    n_val    = int(n * VAL_FRAC)

    train_scenes = scene_tokens[:n_train]
    val_scenes   = scene_tokens[n_train:n_train + n_val]
    test_scenes  = scene_tokens[n_train + n_val:]

    train = [t for sc in train_scenes for t in scene_to_trajs[sc]]
    val   = [t for sc in val_scenes   for t in scene_to_trajs[sc]]
    test  = [t for sc in test_scenes  for t in scene_to_trajs[sc]]

    print(f"[dataset] Scene split: {len(train_scenes)} train / {len(val_scenes)} val / {len(test_scenes)} test scenes")
    print(f"[dataset] Sample split: {len(train)} train / {len(val)} val / {len(test)} test trajectories")
    return train, val, test

# ── Dataset ───────────────────────────────────────────────────────────────────

class NuScenesVLADataset(Dataset):
    """
    Live vision: CPU workers load the 6 camera images, apply optional image-space
    photometric augmentation, and normalize. The frozen encoder runs on the GPU in
    the train loop (see finetune.py / vision_live.encode_normalized_images).

    Returns per sample:
        images       : float32 [6, 3, 448, 448]  (normalized, CAMERAS order)
        ego_state    : float32 [4, 4]
        traj_tokens  : int64   [24]
        sample_token : str
    """

    def __init__(self, trajectories, split='train', augment=False):
        self.trajs     = trajectories
        self.split     = split
        self.augment   = augment
        self.tokenizer = TrajectoryTokenizer()
        print(f"[dataset] {split}: {len(trajectories)} samples  (augment={augment})")

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj         = self.trajs[idx]
        sample_token = traj['sample_token']

        # 6 camera images -> [6, 3, 448, 448] normalized; photometric aug if enabled
        cam_paths = traj['cam_paths']
        images = torch.stack([preprocess_image(cam_paths[c], augment=self.augment)
                              for c in CAMERAS], dim=0)

        # Ego state
        ego_state = compute_ego_state(traj)   # [4, 4]

        # Trajectory tokens
        token_pairs  = self.tokenizer.tokenize(traj)
        accel_tokens = [p[0] for p in token_pairs]
        curv_tokens  = [p[1] for p in token_pairs]
        traj_tokens  = torch.tensor(accel_tokens + curv_tokens, dtype=torch.long)  # [24]

        return {
            'images':       images,
            'ego_state':    ego_state,
            'traj_tokens':  traj_tokens,
            'sample_token': sample_token,
        }

# ── Class weights ─────────────────────────────────────────────────────────────

def get_class_weights(dataset, device='cuda:0'):
    """
    Compute inverse-frequency class weights for the trajectory token vocabulary.
    Samples up to 5000 trajectories for efficiency.

    Returns: float32 tensor [TRAJ_VOCAB] on device
    """
    print("[dataset] Computing class weights from training set...")
    counts = torch.zeros(TRAJ_VOCAB, dtype=torch.float32)

    n_sample = min(len(dataset), 5000)
    indices  = random.sample(range(len(dataset)), n_sample)

    # Tokenize directly from trajs (NOT dataset[i]) so we don't trigger image IO.
    for i in indices:
        token_pairs = dataset.tokenizer.tokenize(dataset.trajs[i])
        for a, c in token_pairs:
            counts[a] += 1
            counts[c] += 1

    # Sqrt inverse frequency — softer than raw inverse frequency
    # Raw inv-freq makes modal token weight ~0, which is too aggressive
    # Sqrt gives a gentler rebalancing: common tokens get lower weight,
    # rare tokens get higher weight, but neither extreme is zeroed out
    counts  = counts + 1.0                        # smoothing
    weights = 1.0 / torch.sqrt(counts)
    weights = weights / weights.sum() * TRAJ_VOCAB  # normalize so mean weight ≈ 1

    print(f"[dataset] Weight range: {weights.min():.3f} — {weights.max():.3f}")
    print(f"[dataset] Modal token weight: {weights[35]:.3f}  (was over-represented)")
    return weights.to(device)

# ── Public API ────────────────────────────────────────────────────────────────

def get_dataloaders(
    trajectories_path = TRAJ_PATH,
    nuscenes_root     = NUSCENES_ROOT,
    batch_size        = 1,
    num_workers       = 4,
    pin_memory        = True,
    seed              = SPLIT_SEED,
    load_test         = False,
    augment           = False,
):
    print(f"[dataset] Loading trajectories from {trajectories_path}")
    with open(trajectories_path, 'rb') as f:
        all_trajs = pickle.load(f)
    print(f"[dataset] Loaded {len(all_trajs)} trajectories.")

    train_trajs, val_trajs, test_trajs = build_scene_split(all_trajs, nuscenes_root, seed)

    train_dataset = NuScenesVLADataset(train_trajs, split='train', augment=augment)
    val_dataset   = NuScenesVLADataset(val_trajs,   split='val',   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    test_loader = None
    if load_test:
        test_dataset = NuScenesVLADataset(test_trajs, split='test')
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory)
        print("[dataset] WARNING: Test loader created. Only for final eval.")

    return train_loader, val_loader, test_loader


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train_loader, val_loader, _ = get_dataloaders(batch_size=2, num_workers=2)

    print(f"\n[smoke test] Train batches: {len(train_loader)}")
    print(f"[smoke test] Val   batches: {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"\n[smoke test] images        : {batch['images'].shape}")
    print(f"[smoke test] ego_state     : {batch['ego_state'].shape}")
    print(f"[smoke test] traj_tokens   : {batch['traj_tokens'].shape}")
    print(f"[smoke test] traj_tokens[0]: {batch['traj_tokens'][0]}")

    weights = get_class_weights(train_loader.dataset)
    print(f"\n[smoke test] class weights shape: {weights.shape}")
    print("[smoke test] PASSED")