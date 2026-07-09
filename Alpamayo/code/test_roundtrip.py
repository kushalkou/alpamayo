"""
test_roundtrip.py
Validates the full roundtrip:
trajectory → tokens → detokenize → unicycle integration → (x, y, yaw)
Measures ADE against ground truth positions.
Target: ADE < 1.0 meter
"""

import numpy as np
import pickle
from tokenizer import TrajectoryTokenizer
from ego_encoder import quaternion_to_yaw

DT = 0.5  # seconds

def unicycle_integrate(v0, yaw0, x0, y0, accelerations, curvatures, dt=DT):
    """
    Integrate unicycle model forward given control sequence.
    
    v(t+1)   = v(t) + a * dt
    yaw(t+1) = yaw(t) + v * κ * dt
    x(t+1)   = x(t) + v * cos(yaw) * dt
    y(t+1)   = y(t) + v * sin(yaw) * dt
    
    Returns: positions [n, 2], yaws [n]
    """
    v   = v0
    yaw = yaw0
    x   = x0
    y   = y0

    positions = []
    yaws      = []

    for a, k in zip(accelerations, curvatures):
        # Update velocity
        v = max(0.0, v + a * dt)  # clamp to non-negative speed
        # Update yaw
        yaw = yaw + v * k * dt
        # Update position
        x = x + v * np.cos(yaw) * dt
        y = y + v * np.sin(yaw) * dt

        positions.append([x, y])
        yaws.append(yaw)

    return np.array(positions), np.array(yaws)


def compute_ade(pred_positions, gt_positions):
    """Average Displacement Error — mean L2 distance across all timesteps."""
    errors = np.linalg.norm(pred_positions - gt_positions, axis=1)
    return errors.mean()

def compute_fde(pred_positions, gt_positions):
    """Final Displacement Error — L2 distance at last timestep."""
    return np.linalg.norm(pred_positions[-1] - gt_positions[-1])


if __name__ == '__main__':
    # Load trajectories and tokenizer
    with open('trajectories_full.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    tokenizer = TrajectoryTokenizer()

    print(f"\nRunning roundtrip validation on 100 trajectories...")
    print(f"Pipeline: GT trajectory → tokenize → detokenize → unicycle → compare\n")

    ades = []
    fdes = []

    for i, traj in enumerate(trajectories[:100]):
        # Step 1: Tokenize
        tokens = tokenizer.tokenize(traj)

        # Step 2: Detokenize back to continuous controls
        rec_accels, rec_curvs = tokenizer.detokenize(tokens)

        # Step 3: Get initial conditions from current pose
        pose   = traj['current_pose']
        x0     = pose['translation'][0]
        y0     = pose['translation'][1]
        yaw0   = quaternion_to_yaw(pose['rotation'])

        # Initial speed from first future step
        v0 = traj['future_speeds'][0] if traj['future_speeds'][0] > 0 else 0.0

        # Step 4: Integrate unicycle model
        pred_positions, _ = unicycle_integrate(v0, yaw0, x0, y0, rec_accels, rec_curvs)

        # Step 5: Compare against ground truth
        gt_positions = traj['future_positions']  # [12, 2]

        ade = compute_ade(pred_positions, gt_positions)
        fde = compute_fde(pred_positions, gt_positions)

        ades.append(ade)
        fdes.append(fde)

    # Results
    ades = np.array(ades)
    fdes = np.array(fdes)

    print(f"Results over 100 trajectories:")
    print(f"  ADE — mean: {ades.mean():.3f}m | median: {np.median(ades):.3f}m | max: {ades.max():.3f}m")
    print(f"  FDE — mean: {fdes.mean():.3f}m | median: {np.median(fdes):.3f}m | max: {fdes.max():.3f}m")
    print(f"\nTarget: ADE < 1.0m")

    if ades.mean() < 1.0:
        print(f"  ✅ PASSED — ADE {ades.mean():.3f}m < 1.0m")
    else:
        print(f"  ❌ FAILED — ADE {ades.mean():.3f}m >= 1.0m — bins need refinement")