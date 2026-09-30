"""w1/w1tok.py -- tokenizer (ii): percentile bins (median centres) on the lidar-frame
targets (w1/targets.py). tokenize() returns the precomputed tokens; detokenize_step()
is inherited from TrajectoryTokenizer (STOP -> (0,0), else the bin centres)."""
import pickle, numpy as np
import tokenizer as TKZ

BINS = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_targets.pkl'


class W1Tokenizer(TKZ.TrajectoryTokenizer):
    def __init__(self):
        b = pickle.load(open(BINS, 'rb'))['bins']
        self.accel_centers, self.curv_centers = np.asarray(b[0]), np.asarray(b[1])
        self.accel_bins, self.curv_bins = np.asarray(b[2]), np.asarray(b[3])

    def tokenize(self, traj):
        return list(traj['w1_tokens'])
