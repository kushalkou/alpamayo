"""ar1/relabel_2hz.py -- R5 B1: AR1 Table 5 meta-actions from the 2 Hz GT target controls
(rules in R5_REPORT.md header). Output data/ar1_meta2hz.pkl {sample_token: [lon x6, lat x6]}
for every w1 record (slots beyond n_fut repeat the last valid slot); report to stdout."""
import sys, pickle, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E
from r33_eval import derive
from meta_actions import LON, LAT
import finetune_meta as FM
from w1tok import W1Tokenizer

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'


def labels(t):
    n = len(t['acc'])
    a = np.zeros(12); k = np.zeros(12); a[:n] = t['acc'][:12]; k[:n] = t['cur'][:12]
    d = derive(a, k, t['v0'])
    last = max(1, n // 2)                            # slots with step 2t-1 < n are valid
    lon, lat = d[:6], d[6:]
    lon = lon[:last] + [lon[last - 1]] * (6 - last); lat = lat[:last] + [lat[last - 1]] * (6 - last)
    return lon + lat


def main():
    W, V0, T = E.data()
    out = {r['sample_token']: labels(T[r['sample_token']]) for r in W['records']}
    pickle.dump(out, open(f'{DATA}/ar1_meta2hz.pkl', 'wb'))
    Mt = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb')); st = {'missing': 0}
    tk = W1Tokenizer(); maj = np.array([4] * 6 + [6] * 6)
    for split in ('train', 'holdout', 'val'):
        R = [r for r in W['records'] if r['split'] == split and r['n_fut'] == 12]
        new = np.array([out[r['sample_token']] for r in R])
        can = np.array([sum(FM.meta_labels(Mt, r['sample_token'], st), []) for r in R])
        gt = np.array([derive(T[r['sample_token']]['acc'][:12], T[r['sample_token']]['cur'][:12],
                              T[r['sample_token']]['v0']) for r in R])
        tok = []
        for r in R:
            pr = [tk.detokenize_step(a, k) if a != 128 and k != 128 else (0.0, 0.0)
                  for a, k in T[r['sample_token']]['tokens'][:12]]
            tok.append(derive(np.array([p[0] for p in pr]), np.array([p[1] for p in pr]), T[r['sample_token']]['v0']))
        tok = np.array(tok)
        ag = (new == can).mean(0)
        print(f'\n{split} n_fut = 12, n {len(R)}')
        for h, o, nm in (('lon', 0, LON), ('lat', 6, LAT)):
            sh = np.bincount(new[:, o:o + 6].ravel(), minlength=7) / new[:, o:o + 6].size
            print(f'  {h} 2 Hz class shares: ' + ' '.join(f'{nm[c]} {sh[c]:.3f}' for c in range(7)))
        print(f'  all-majority share 2 Hz {(new == maj).all(1).mean():.3f} (10 Hz CAN {(can == maj).all(1).mean():.3f})')
        print(f'  GT-vs-GT consistency (GT-trajectory-derived vs 2 Hz labels): {(gt == new).all(1).mean():.3f}')
        print(f'  side: GT-TOKEN rollout derived vs 2 Hz labels: all 12 {(tok == new).all(1).mean():.3f}; slot mean '
              f'lon {(tok == new)[:, :6].mean():.3f} lat {(tok == new)[:, 6:].mean():.3f}')
        print('  per-slot agreement with old 10 Hz CAN labels: lon ' + ' '.join(f'{x:.3f}' for x in ag[:6])
              + ' | lat ' + ' '.join(f'{x:.3f}' for x in ag[6:]))


if __name__ == '__main__':
    main()
