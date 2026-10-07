"""ar1/r7_blank_mean.py -- R7 C3: per-dimension mean of the cached visual tokens over ALL
train samples (finetune train split), all 3 front cams at t0 and all 480 positions -> one
3,584-d vector, data/r7_vis_train_mean.npy (float32). CPU.
  python ar1/r7_blank_mean.py"""
import sys, pickle, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'
H = None


def part(toks):
    import vcache
    s = np.zeros(3584); n = 0
    for t in toks:
        x = vcache.gather(H[t], (3,)).numpy().astype(np.float64); s += x.sum(0); n += len(x)
    return s, n


def main():
    global H
    import records
    H = pickle.load(open(f'{DATA}/ar1_hist.pkl', 'rb'))
    toks = [r['sample_token'] for r in records.build('train', [], 0.0)]
    t0 = time.time()
    with Pool(8) as p:
        out = p.map(part, [toks[i::64] for i in range(64)])
    s = sum(o[0] for o in out); n = sum(o[1] for o in out)
    m = (s / n).astype(np.float32)
    np.save(f'{DATA}/r7_vis_train_mean.npy', m)
    print(f'train samples {len(toks)}, tokens {n} (= {n // len(toks)} per sample); mean |m| {np.abs(m).mean():.4f}, '
          f'norm {np.linalg.norm(m):.2f}; {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
