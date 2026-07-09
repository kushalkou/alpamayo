"""
extract_trajectories.py
Extracts ego trajectories from nuScenes full dataset.
For each sample, extracts:
- past: 2 second history (4 poses)
- future: 6 second future (12 poses at 2Hz)
Computes speed, acceleration, curvature at each future step.
"""

import numpy as np
import pickle
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from ego_encoder import quaternion_to_yaw

DT = 0.5  # seconds between samples

def get_ego_pose(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    return {
        'translation': np.array(ego_pose['translation']),
        'rotation': np.array(ego_pose['rotation']),
        'timestamp': cam_data['timestamp']
    }

def compute_curvature(positions, yaws):
    """
    Curvature κ = dyaw / ds  (rad/m)
    where ds is distance traveled.
    """
    n = len(positions)
    curvatures = np.zeros(n)
    for i in range(1, n):
        ds = np.linalg.norm(positions[i] - positions[i-1])
        dyaw = yaws[i] - yaws[i-1]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        if ds > 0.01:  # avoid division by zero for stopped vehicles
            curvatures[i] = dyaw / ds
        else:
            curvatures[i] = 0.0
    curvatures[0] = curvatures[1]
    return curvatures

def extract_trajectory(nusc, sample_token, future_steps=12, past_steps=4):
    """
    For a given sample, walk forward future_steps and backward past_steps.
    Returns None if not enough future frames exist.
    """
    # Collect future tokens
    future_tokens = []
    token = sample_token
    for _ in range(future_steps):
        sample = nusc.get('sample', token)
        if sample['next'] == '':
            return None  # not enough future
        token = sample['next']
        future_tokens.append(token)

    # Collect past tokens
    past_tokens = []
    token = sample_token
    for _ in range(past_steps):
        sample = nusc.get('sample', token)
        if sample['prev'] == '':
            break
        token = sample['prev']
        past_tokens.append(token)
    past_tokens.reverse()

    # Get ego poses for future
    future_poses = [get_ego_pose(nusc, t) for t in future_tokens]
    current_pose = get_ego_pose(nusc, sample_token)
    all_future = [current_pose] + future_poses

    positions = np.array([p['translation'][:2] for p in all_future])
    yaws = np.array([quaternion_to_yaw(p['rotation']) for p in all_future])

    # Compute speeds
    speeds = np.zeros(len(all_future))
    for i in range(1, len(all_future)):
        dist = np.linalg.norm(positions[i] - positions[i-1])
        speeds[i] = dist / DT
    speeds[0] = speeds[1]

    # Compute accelerations
    accels = np.zeros(len(all_future))
    for i in range(1, len(all_future)):
        accels[i] = (speeds[i] - speeds[i-1]) / DT
    accels[0] = accels[1]

    # Compute curvatures
    curvatures = compute_curvature(positions, yaws)

    # Get past poses
    past_poses = [get_ego_pose(nusc, t) for t in past_tokens]

    return {
        'sample_token': sample_token,
        'current_pose': current_pose,
        'future_positions': positions[1:],       # [12, 2]
        'future_yaws': yaws[1:],                 # [12]
        'future_speeds': speeds[1:],             # [12]
        'future_accelerations': accels[1:],      # [12]
        'future_curvatures': curvatures[1:],     # [12]
        'past_poses': past_poses,
    }


if __name__ == '__main__':
    print("Loading NuScenes...")
    nusc = NuScenes(
        version='v1.0-trainval',
        dataroot='/home/drive1/Alpamayo/nuscenes_full',
        verbose=False
    )

    trajectories = []
    skipped = 0

    for sample in tqdm(nusc.sample, desc='Extracting trajectories'):
        result = extract_trajectory(nusc, sample['token'])
        if result is not None:
            trajectories.append(result)
        else:
            skipped += 1

    print(f"\nExtracted: {len(trajectories)} trajectories")
    print(f"Skipped (not enough future): {skipped}")

    # Quick stats
    accels = np.concatenate([t['future_accelerations'] for t in trajectories])
    curvs = np.concatenate([t['future_curvatures'] for t in trajectories])
    speeds = np.concatenate([t['future_speeds'] for t in trajectories])

    print(f"\nAcceleration stats:")
    print(f"  min={accels.min():.3f}, max={accels.max():.3f}, mean={accels.mean():.3f} m/s²")
    print(f"\nCurvature stats:")
    print(f"  min={curvs.min():.3f}, max={curvs.max():.3f}, mean={curvs.mean():.3f} rad/m")
    print(f"\nSpeed stats:")
    print(f"  min={speeds.min():.3f}, max={speeds.max():.3f}, mean={speeds.mean():.3f} m/s")

    # Save
    save_path = '/home/drive1/Alpamayo/week2_tokenization/trajectories_full.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"\nSaved to {save_path}")