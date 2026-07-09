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

sys.path.insert(0, '/home/drive1/Alpamayo/tokenization')
from tokenizer import TrajectoryTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

NUSCENES_ROOT    = '/home/drive1/Alpamayo/nuscenes_full'
VISUAL_TOKENS_DIR = Path('/home/drive1/Alpamayo/data/visual_tokens')
TRAJ_PATH        = '/home/drive1/Alpamayo/data/trajectories_full.pkl'
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
    dt = 0.5
    raw_poses = list(traj.get('past_poses', []))
    raw_poses.append(traj['current_pose'])
    poses = [pose_to_xyyaw(p) for p in raw_poses]
    while len(poses) < 4:
        poses = [poses[0]] + poses
    poses = poses[-4:]

    states, speeds = [], []
    for i in range(4):
        x_cur, y_cur, yaw_cur = poses[i]
        if i == 0:
            speed = yaw_rate = accel = 0.0
        else:
            x_prev, y_prev, yaw_prev = poses[i - 1]
            speed    = float(np.sqrt((x_cur-x_prev)**2 + (y_cur-y_prev)**2)) / dt
            dyaw     = float(np.arctan2(np.sin(yaw_cur - yaw_prev), np.cos(yaw_cur - yaw_prev)))
            yaw_rate = dyaw / dt
            accel    = (speed - speeds[-1]) / dt if speeds else 0.0
        speeds.append(speed)
        states.append([speed, yaw_cur, yaw_rate, accel])

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

    # Group trajectories by scene
    scene_to_trajs = {}
    missing = 0
    for traj in trajectories:
        st = traj['sample_token']
        sc = sample_to_scene.get(st)
        if sc is None:
            missing += 1
            continue
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
    Returns per sample:
        visual_tokens : float16 [1536, 3584]
        ego_state     : float32 [4, 4]
        traj_tokens   : int64   [24]
        sample_token  : str
    """

    def __init__(self, trajectories, split='train'):
        self.trajs     = trajectories
        self.split     = split
        self.tokenizer = TrajectoryTokenizer()
        print(f"[dataset] {split}: {len(trajectories)} samples")

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj         = self.trajs[idx]
        sample_token = traj['sample_token']

        # Visual tokens
        token_path = VISUAL_TOKENS_DIR / f"{sample_token}.pt"
        if token_path.exists():
            visual_tokens = torch.load(token_path, map_location='cpu', weights_only=True)
        else:
            visual_tokens = torch.zeros(1536, 3584, dtype=torch.float16)
            print(f"[dataset] WARNING: missing visual tokens for {sample_token}")

        # Ego state
        ego_state = compute_ego_state(traj)   # [4, 4]

        # Trajectory tokens
        token_pairs  = self.tokenizer.tokenize(traj)
        accel_tokens = [p[0] for p in token_pairs]
        curv_tokens  = [p[1] for p in token_pairs]
        traj_tokens  = torch.tensor(accel_tokens + curv_tokens, dtype=torch.long)  # [24]

        return {
            'visual_tokens': visual_tokens,
            'ego_state':     ego_state,
            'traj_tokens':   traj_tokens,
            'sample_token':  sample_token,
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

    for i in indices:
        item = dataset[i]
        for tok in item['traj_tokens'].tolist():
            counts[tok] += 1

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
):
    print(f"[dataset] Loading trajectories from {trajectories_path}")
    with open(trajectories_path, 'rb') as f:
        all_trajs = pickle.load(f)
    print(f"[dataset] Loaded {len(all_trajs)} trajectories.")

    train_trajs, val_trajs, test_trajs = build_scene_split(all_trajs, nuscenes_root, seed)

    train_dataset = NuScenesVLADataset(train_trajs, split='train')
    val_dataset   = NuScenesVLADataset(val_trajs,   split='val')

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
    print(f"\n[smoke test] visual_tokens : {batch['visual_tokens'].shape}")
    print(f"[smoke test] ego_state     : {batch['ego_state'].shape}")
    print(f"[smoke test] traj_tokens   : {batch['traj_tokens'].shape}")
    print(f"[smoke test] traj_tokens[0]: {batch['traj_tokens'][0]}")

    weights = get_class_weights(train_loader.dataset)
    print(f"\n[smoke test] class weights shape: {weights.shape}")
    print("[smoke test] PASSED")