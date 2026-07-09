"""
ego_encoder.py
Derives speed, yaw, acceleration from nuScenes ego poses
and projects them into ego tokens via a lightweight MLP.
"""

import torch
import torch.nn as nn
import numpy as np
from camera_loader import CameraLoader

DEVICE = torch.device('cuda:0')
DT = 0.5  # seconds between nuScenes samples (2Hz)

def quaternion_to_yaw(q):
    """
    Convert quaternion [w, x, y, z] to yaw angle (rotation around Z axis).
    """
    w, x, y, z = q
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw

def compute_ego_signals(ego_history):
    """
    Given a list of ego poses (chronological), compute:
    speed, yaw, yaw_rate, acceleration at each timestep.
    
    ego_history: list of dicts with 'translation', 'rotation', 'timestamp'
    Returns: dict of numpy arrays, each of length len(ego_history)
    """
    n = len(ego_history)
    positions = np.array([e['translation'][:2] for e in ego_history])  # [n, 2] x,y only
    yaws = np.array([quaternion_to_yaw(e['rotation']) for e in ego_history])  # [n]

    # Speed: distance / dt
    speeds = np.zeros(n)
    for i in range(1, n):
        dist = np.linalg.norm(positions[i] - positions[i-1])
        speeds[i] = dist / DT
    if len(speeds) > 1:
        speeds[0] = speeds[1]  # fill first with second

    # Yaw rate: change in yaw / dt
    yaw_rates = np.zeros(n)
    for i in range(1, n):
        dyaw = yaws[i] - yaws[i-1]
        # Wrap to [-pi, pi]
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        yaw_rates[i] = dyaw / DT
    if len(yaw_rates) > 1:
        yaw_rates[0] = yaw_rates[1]

    # Acceleration: change in speed / dt
    accels = np.zeros(n)
    for i in range(1, n):
        accels[i] = (speeds[i] - speeds[i-1]) / DT
    if len(accels) > 1:
        accels[0] = accels[1]

    return {
        'speed': speeds,
        'yaw': yaws,
        'yaw_rate': yaw_rates,
        'acceleration': accels,
        'position': positions
    }


class EgoMLP(nn.Module):
    """
    Lightweight MLP that projects ego signals into token embeddings.
    Input: [speed, yaw, yaw_rate, acceleration] = 4 features
    Output: ego token of dimension 768 (same as ViT)
    """
    def __init__(self, input_dim=4, output_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class EgoEncoder:
    def __init__(self):
        self.mlp = EgoMLP(input_dim=4, output_dim=768).to(DEVICE)
        self.mlp.eval()
        print(f"Ego MLP loaded on {DEVICE}")
        total_params = sum(p.numel() for p in self.mlp.parameters())
        print(f"Trainable parameters: {total_params:,}")

    def encode(self, ego_history):
        """
        ego_history: list of ego pose dicts (from camera_loader.get_2s_history)
        Returns: tensor of shape [n_timesteps, 768]
        """
        signals = compute_ego_signals(ego_history)

        # Stack into feature matrix [n, 4]
        features = np.stack([
            signals['speed'],
            signals['yaw'],
            signals['yaw_rate'],
            signals['acceleration']
        ], axis=1).astype(np.float32)

        x = torch.tensor(features).to(DEVICE)

        with torch.no_grad():
            ego_tokens = self.mlp(x)  # [n, 768]

        return ego_tokens, signals


if __name__ == '__main__':
    loader = CameraLoader()
    scene_samples = loader.get_scene_samples(scene_index=0)

    # Get 2-second history for sample 5 (has enough history)
    sample_token = scene_samples[5]
    history = loader.get_2s_history(sample_token)

    print(f"\n2-second history: {len(history)} poses")

    # Compute signals
    signals = compute_ego_signals(history)
    print(f"\nDerived ego signals:")
    for i in range(len(history)):
        print(f"  t-{len(history)-1-i}: "
              f"speed={signals['speed'][i]:.2f} m/s  "
              f"yaw={np.degrees(signals['yaw'][i]):.1f} deg  "
              f"yaw_rate={signals['yaw_rate'][i]:.3f} rad/s  "
              f"accel={signals['acceleration'][i]:.3f} m/s²")

    # Encode
    encoder = EgoEncoder()
    ego_tokens, _ = encoder.encode(history)

    print(f"\nEgo tokens shape: {ego_tokens.shape}")
    print(f"Each timestep → 768-dim token (matches ViT dimension)")