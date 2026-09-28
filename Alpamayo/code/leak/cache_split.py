"""leak/cache_split.py -- cache val/test trajectories (same order as dump_{val,test}.pkl)."""
import sys, pickle, numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
import inference as INF
from dataset import build_scene_split
with open(INF.TRAJECTORIES_PATH, 'rb') as f: allt = pickle.load(f)
tr, va, te = build_scene_split(allt, INF.NUSCENES_ROOT)
out = {'val': va, 'test': te, 'n_train': len(tr)}
pickle.dump(out, open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/leak_split.pkl', 'wb'))
# verify alignment with the dumps
for s, T in (('val', va), ('test', te)):
    M = pickle.load(open(f'/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/dump_{s}.pkl', 'rb'))['meta']
    err = max(np.abs(np.array(M[i]['gt']) - (np.array(T[i]['future_positions'])[:12]
              - np.array(T[i]['current_pose']['translation'][:2]))).max() for i in M)
    print(s, len(T), len(M), 'max gt mismatch', err)
