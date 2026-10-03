"""ar1/hist_index.py -- R1.1: AR1-style camera history index (CPU).

For every w1 record (29,049; train / holdout / val), t0 = sample['timestamp'] (LIDAR_TOP
keyframe time; the same t0 as w1/v0_sources.py). Cameras CAM_FRONT, CAM_FRONT_LEFT,
CAM_FRONT_RIGHT; 4 frames per camera at nominal t0, t0-0.5, t0-1.0, t0-1.5 s.
Only keyframe images are on disk (samples/; sweeps/ was never downloaded), so frame k
is the camera's keyframe image of the k-th previous sample (sample.prev^k). Rule:
  frame k must have timestamp <= t0 (asserted in test_hist.py); if the keyframe
  camera image of the CURRENT sample is later than t0, the previous keyframe image is
  used for every k instead (the nearest earlier image) -- counted in the report.
  First frames of a scene (no prev^k): repeat the oldest available frame (flag pad).
Output: Alpamayo/data/ar1_hist.pkl {sample_token: {'t0', 'cams': {cam: [(path, ts) x4
oldest -> newest]}, 'n_pad': int}}; stats to stdout.
"""
import json, pickle, collections
import numpy as np

ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/ar1_hist.pkl'
CAMS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')
NF = 4
NOMINAL = (1.5, 1.0, 0.5, 0.0)          # seconds before t0, oldest -> newest


def main():
    W = pickle.load(open(f'{ROOT}/../data/w1_data.pkl', 'rb'))['records']
    S = {s['token']: s for s in json.load(open(f'{ROOT}/v1.0-trainval/sample.json'))}
    print('loading sample_data.json ...', flush=True)
    kf = {}                                 # (sample_token, cam) -> (filename, ts)
    for d in json.load(open(f'{ROOT}/v1.0-trainval/sample_data.json')):
        if not d['is_key_frame']:
            continue
        f = d['filename']
        for c in CAMS:
            if f.startswith(f'samples/{c}/'):
                kf[(d['sample_token'], c)] = (f'{ROOT}/{f}', d['timestamp'])
    out, after, npad, off = {}, collections.Counter(), collections.Counter(), {c: [] for c in CAMS}
    for r in W:
        st = r['sample_token']; t0 = S[st]['timestamp']
        chain = [st]
        while len(chain) < NF + 1 and S[chain[-1]]['prev']:
            chain.append(S[chain[-1]]['prev'])
        cams = {}
        for c in CAMS:
            seq = [kf[(s, c)] for s in chain]           # newest -> oldest
            if seq[0][1] > t0:                          # current image after t0
                after[c] += 1
                seq = seq[1:]
            seq = seq[:NF]
            pad = NF - len(seq)
            seq = seq + [seq[-1]] * pad                 # repeat oldest
            cams[c] = seq[::-1]                         # oldest -> newest
            npad[(r['split'], pad)] += 1 if c == 'CAM_FRONT' else 0
            if pad == 0:
                off[c].append([(t0 - ts) / 1e6 for _, ts in cams[c]])
        out[st] = {'t0': t0, 'cams': cams, 'n_pad': NF - min(len(chain), NF), 'split': r['split']}
    pickle.dump(out, open(OUT, 'wb'))
    print(f'records {len(out)}; current keyframe image later than t0 (replaced): {dict(after)}')
    print('padded frames per sample (CAM_FRONT), by split:',
          {k: v for k, v in sorted(npad.items())})
    print('age of each frame t0 - ts [s], unpadded samples, median [p5, p95] per slot '
          f'(nominal {NOMINAL}):')
    for c in CAMS:
        a = np.array(off[c])
        print(f'  {c:16} ' + '  '.join(f'{np.median(a[:, k]):.3f} [{np.percentile(a[:, k], 5):.3f},'
                                       f'{np.percentile(a[:, k], 95):.3f}]' for k in range(NF))
              + f'   min age {a.min():.3f}')


if __name__ == '__main__':
    main()
