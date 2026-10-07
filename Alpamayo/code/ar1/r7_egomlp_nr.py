"""ar1/r7_egomlp_nr.py -- R7 A3: Ego-MLP without the command (16 causal ego features), seeds
42 / 123 / 2024, w1/ego_mlp.train_one unchanged (holdout ADE@6s selection). CPU.
Output results/w1_egomlp_nr.pkl (same format as results/w1_egomlp.pkl).
  python ar1/r7_egomlp_nr.py"""
import sys, pickle, time
import numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records
from ego_mlp import train_one

OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/w1_egomlp_nr.pkl'


def feats(R):
    return np.stack([t['w1_ego'][:4].numpy().ravel() for t in R]).astype(np.float32)


def main():
    tr = records.build('train', ()); ho = records.build('holdout', (), n_fut=6); va = records.build('val', (), n_fut=6)
    Xtr = feats(tr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: (X - mu) / sd
    Ytr = np.stack([t['future_positions'] for t in tr]).reshape(len(tr), -1).astype(np.float32)
    Yho = np.stack([np.pad(t['future_positions'], ((0, 12 - len(t['future_positions'])), (0, 0)))
                    for t in ho]).reshape(len(ho), -1)
    mho = np.array([t['n_fut'] == 12 for t in ho])
    preds = {}
    for seed in (42, 123, 2024):
        t0 = time.time()
        net, best, bep = train_one(seed, z(Xtr), Ytr, z(feats(ho)), Yho, mho, 'cpu')
        with torch.no_grad():
            pv = net(torch.tensor(z(feats(va)))).view(-1, 12, 2).numpy()
            ph = net(torch.tensor(z(feats(ho)))).view(-1, 12, 2).numpy()
        preds[seed] = {'val': pv, 'holdout': ph, 'holdout_ade': best, 'epoch': bep}
        print(f'[egomlp NR] seed {seed}: {Xtr.shape[1]} features, holdout ADE@6s {best:.3f} (epoch {bep}), '
              f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump({'preds': preds, 'val_tokens': [t['sample_token'] for t in va]}, open(OUT, 'wb'))


if __name__ == '__main__':
    main()
