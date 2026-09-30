"""w1/gate35.py -- STEP 3.5a gate (redefined by the planner 2026-09-30).

PASS requires BOTH, on the FINAL (epoch-150) checkpoint, on the 256 overfit samples:
  (1) AR median ADE@6s <= 1.0 m  (argmax, free-running; the selection decode)
  (2) slot-0 TF token accuracy >= 90%. Slot 0 has no token prefix, so its
      teacher-forced prediction IS the first free-running token (argmax of logits0).
INPUT-SHUFFLE test: ego+cmd rows permuted across the 256 (seed 99), everything else
unchanged; report slot-0 accuracy and AR median ADE shuffled vs unshuffled.
Usage: python w1/gate35.py <tag_latest> <tag_latest_shuf> [<tag_best>] ...
"""
import sys, pickle
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
from targets import rollout as roll12
from w1tok import W1Tokenizer

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
T = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_targets.pkl', 'rb'))['targets']


def stats(tag):
    D = pickle.load(open(f'{RES}/w1_dump_{tag}_train_f0.0.pkl', 'rb'))
    ade, s0, ex, fl = [], [], [], []
    tok = W1Tokenizer()
    for st in D['order']:
        d, t = D['data'][st], T[st]
        v = np.array(d['argmax'])
        ade.append(np.linalg.norm(roll12(v[:12], v[12:], t['v0']) - t['P'][:12], axis=1).mean())
        s0.append(d['tok'][0] == d['gt_tok'][0])
        ex.append(all(p == g for p, g in zip(d['tok'], d['gt_tok'])))
        g = d['gt_tok']
        a = [0.0 if x == 128 else tok.accel_centers[x] for x in g[:12]]
        k = [0.0 if x == 128 else tok.curv_centers[x] for x in g[12:]]
        fl.append(np.linalg.norm(roll12(a, k, t['v0']) - t['P'][:12], axis=1).mean())
    ade = np.array(ade)
    return dict(epoch=D['ckpt_epoch'], med=float(np.median(ade)), mean=float(ade.mean()),
                s0=float(np.mean(s0)), exact=float(np.mean(ex)), floor=float(np.mean(fl)),
                floor_med=float(np.median(fl)), n=len(ade))


def main():
    tags = sys.argv[1:]
    S = {t: stats(t) for t in tags}
    for t, s in S.items():
        print(f'  {t:28} epoch {s["epoch"]}  AR median ADE@6s {s["med"]:.3f} (mean {s["mean"]:.3f})  '
              f'slot-0 acc {s["s0"]:.3f}  exact-seq {s["exact"]:.3f}  floor {s["floor"]:.3f}/{s["floor_med"]:.3f}  n={s["n"]}')
    fin, shuf = S[tags[0]], S[tags[1]]
    ok1, ok2 = fin['med'] <= 1.0, fin['s0'] >= 0.90
    print(f'\n  GATE 3.5a on {tags[0]}: median {fin["med"]:.3f} <= 1.0 -> {ok1};  slot-0 {fin["s0"]:.3f} '
          f'>= 0.90 -> {ok2};  => {"PASS" if ok1 and ok2 else "FAIL"}')
    print(f'  INPUT-SHUFFLE: slot-0 acc {fin["s0"]:.3f} -> {shuf["s0"]:.3f};  AR median ADE '
          f'{fin["med"]:.3f} -> {shuf["med"]:.3f} (mean {fin["mean"]:.3f} -> {shuf["mean"]:.3f})')


if __name__ == '__main__':
    main()
