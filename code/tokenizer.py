"""
tokenizer.py
Converts continuous acceleration + curvature sequences
into discrete tokens and back.
64 bins each = 128 total action tokens, matching Alpamayo spec.
"""

import numpy as np
import pickle

# Bin design based on full dataset statistics
ACCEL_MIN = -11.0
ACCEL_MAX =  10.0
CURV_MIN  = -0.80
CURV_MAX  =  0.65
N_BINS    = 64

# Special tokens
STOP_SPEED_THRESHOLD = 0.1  # m/s — below this, vehicle is stopped
STOP_TOKEN = N_BINS * 2     # token id 128 = STOP

class TrajectoryTokenizer:
    def __init__(self):
        # Uniform bins for now — can switch to non-uniform later
        self.accel_bins = np.linspace(ACCEL_MIN, ACCEL_MAX, N_BINS + 1)
        self.curv_bins  = np.linspace(CURV_MIN,  CURV_MAX,  N_BINS + 1)

        # Bin centers for detokenization
        self.accel_centers = (self.accel_bins[:-1] + self.accel_bins[1:]) / 2
        self.curv_centers  = (self.curv_bins[:-1]  + self.curv_bins[1:])  / 2

        print(f"Tokenizer initialized:")
        print(f"  Acceleration bins: {N_BINS} bins from {ACCEL_MIN} to {ACCEL_MAX} m/s²")
        print(f"  Curvature bins:    {N_BINS} bins from {CURV_MIN} to {CURV_MAX} rad/m")
        print(f"  Special tokens:    STOP={STOP_TOKEN}")
        print(f"  Total vocab size:  {N_BINS * 2 + 1}")

    def tokenize_step(self, accel, curvature, speed):
        """
        Tokenize a single timestep.
        Returns (accel_token, curv_token) or STOP_TOKEN if stopped.
        """
        if speed < STOP_SPEED_THRESHOLD:
            return (STOP_TOKEN, STOP_TOKEN)

        # Clip to valid range
        accel = np.clip(accel, ACCEL_MIN, ACCEL_MAX - 1e-6)
        curv  = np.clip(curvature, CURV_MIN, CURV_MAX - 1e-6)

        # Digitize: find which bin
        accel_token = int(np.digitize(accel, self.accel_bins) - 1)
        curv_token  = int(np.digitize(curv,  self.curv_bins)  - 1)

        # Clamp to valid range
        accel_token = np.clip(accel_token, 0, N_BINS - 1)
        curv_token  = np.clip(curv_token,  0, N_BINS - 1)

        return (accel_token, curv_token)

    def tokenize(self, trajectory):
        """
        Tokenize a full trajectory.
        trajectory: dict with 'future_accelerations', 'future_curvatures', 'future_speeds'
        Returns: list of (accel_token, curv_token) tuples, length = future_steps
        """
        accels = trajectory['future_accelerations']
        curvs  = trajectory['future_curvatures']
        speeds = trajectory['future_speeds']

        tokens = []
        for a, k, s in zip(accels, curvs, speeds):
            tokens.append(self.tokenize_step(a, k, s))
        return tokens

    def detokenize_step(self, accel_token, curv_token):
        """
        Map token ids back to continuous acceleration and curvature.
        """
        if accel_token == STOP_TOKEN:
            return 0.0, 0.0

        accel = self.accel_centers[accel_token]
        curv  = self.curv_centers[curv_token]
        return accel, curv

    def detokenize(self, tokens):
        """
        Map list of (accel_token, curv_token) back to continuous values.
        Returns: (accelerations, curvatures) as numpy arrays
        """
        accels = []
        curvs  = []
        for at, kt in tokens:
            a, k = self.detokenize_step(at, kt)
            accels.append(a)
            curvs.append(k)
        return np.array(accels), np.array(curvs)


if __name__ == '__main__':
    # Load trajectories
    with open('trajectories_full.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    tokenizer = TrajectoryTokenizer()

    # Test on first 5 trajectories
    print(f"\nTokenizing first 5 trajectories:")
    for i in range(5):
        traj = trajectories[i]
        tokens = tokenizer.tokenize(traj)
        accel_tokens = [t[0] for t in tokens]
        curv_tokens  = [t[1] for t in tokens]
        print(f"\n  Trajectory {i}:")
        print(f"    Accel tokens: {accel_tokens}")
        print(f"    Curv tokens:  {curv_tokens}")

        # Detokenize back
        rec_accels, rec_curvs = tokenizer.detokenize(tokens)
        orig_accels = traj['future_accelerations']
        orig_curvs  = traj['future_curvatures']

        accel_err = np.abs(rec_accels - orig_accels).mean()
        curv_err  = np.abs(rec_curvs  - orig_curvs).mean()
        print(f"    Accel reconstruction error: {accel_err:.4f} m/s²")
        print(f"    Curv  reconstruction error: {curv_err:.4f} rad/m")