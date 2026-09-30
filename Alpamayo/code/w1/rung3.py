"""w1/rung3.py -- QUEUE C4/C5: mini rung 3 on the Ego-MLP (small-model demonstration).

C4a  meta-action classifier: MLP (19-512-512-4, ReLU) on causal ego (16) + cmd one-hot
     (3) -> lon label (stop/accelerate/decelerate/maintain); CE loss, AdamW 1e-3, batch 256,
     100 epochs, cosine; selection = best HOLDOUT accuracy; 3 seeds; CPU.
     Accuracy + confusion matrix on official val (all 5,119 and excl. 140 first frames).
C4b  feed its PREDICTED lon (lat = cmd) into the L5 and L5-noise Ego-MLPs of C1 (retrained
     here with the identical recipe/seed; reproduction of C1's predictions is checked).
     classifier seed s -> L5 / L5n seed s. ADE@6s, L2 TemAvg 1/2/3 s, gap closed
     (X - L2)/(L5 - L2) against C1's seed-mean L2 and L5.
C4c  the classifier's accuracy is placed on the flip curve (f7) at x = 100 * (1 - acc).
C5   confusion-shaped noise: at error rate e in {10,20,40}%, each val sample's TRUE lon is
     replaced with probability e by a class drawn from the classifier's val confusion row
     for that true class, off-diagonal only (renormalised); seed 777. L5-noise evaluated.
"""
import sys, pickle, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records, evalw1 as E, plan_metrics as PMX
from ladder import feats, fit, SEEDS, EXTRA, OUT as LADDER

torch.set_num_threads(24)
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/w1_rung3.pkl'
RATES = (0.1, 0.2, 0.4)


def cls_feats(R):
    ego = np.stack([t['w1_ego'][:4].numpy().ravel() for t in R])
    return np.concatenate([ego, np.eye(3)[[t['command'] for t in R]]], 1).astype(np.float32)


def train_cls(seed, Xtr, ytr, Xho, yho):
    torch.manual_seed(seed); np.random.seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, 4))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    EP = 100; sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    xt, yt, xh = torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xho)
    g = torch.Generator().manual_seed(seed); best, state = -1, None
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(xt), generator=g)
        for s in range(0, len(xt), 256):
            b = perm[s:s + 256]
            loss = nn.functional.cross_entropy(net(xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            acc = float((net(xh).argmax(1).numpy() == yho).mean())
        if acc > best:
            best, state, bep = acc, {k: v.clone() for k, v in net.state_dict().items()}, ep
    net.load_state_dict(state); net.eval()
    return net, best, bep


def with_lon(R, lon):
    """copy of val records with 'shown' lon replaced (lat/cmd stay true)."""
    return [dict(t, shown=dict(t['shown'], lon=int(l))) for t, l in zip(R, lon)]


def confusion_noise(ytrue, C, e, seed=777):
    rs = np.random.RandomState(seed); out = ytrue.copy()
    for i in range(len(ytrue)):
        if rs.rand() < e:
            row = C[ytrue[i]].astype(float).copy(); row[ytrue[i]] = 0
            if row.sum() == 0: row[:] = 1; row[ytrue[i]] = 0
            out[i] = rs.choice(4, p=row / row.sum())
    return out


def main():
    t0 = time.time()
    tr0 = records.build('train', EXTRA, 0.0)
    tr10 = records.build('train', EXTRA, 0.1, flip_seed=42)
    ho = records.build('holdout', EXTRA, 0.0, n_fut=6)
    va = records.build('val', EXTRA, 0.0, n_fut=6)
    L = pickle.load(open(LADDER, 'rb'))
    Yho = np.stack([np.pad(t['future_positions'], ((0, 12 - len(t['future_positions'])), (0, 0)))
                    for t in ho]).reshape(len(ho), -1)
    mho = np.array([t['n_fut'] == 12 for t in ho])
    ytr = np.array([t['meta_lon'] for t in tr0]); yho = np.array([t['meta_lon'] for t in ho])
    yva = np.array([t['meta_lon'] for t in va])
    Xtr = cls_feats(tr0); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: (X - mu) / sd
    recs = E.split_records('val')
    ff = np.array([not t['can_ok'] and len(r['past_poses']) == 0 for t, r in zip(va, recs)])
    CLS, PRED, CONF, P = {}, {}, {}, {}
    for s in SEEDS:
        net, acc_ho, bep = train_cls(s, z(Xtr), ytr, z(cls_feats(ho)), yho)
        with torch.no_grad():
            PRED[s] = net(torch.tensor(z(cls_feats(va)))).argmax(1).numpy()
        CONF[s] = np.array([[((yva == i) & (PRED[s] == j)).sum() for j in range(4)] for i in range(4)])
        print(f'[rung3] classifier seed {s}: holdout acc {acc_ho:.4f} (ep {bep}); val acc '
              f'{(PRED[s] == yva).mean():.4f}  {time.time()-t0:.0f}s', flush=True)
        # retrain L5 / L5n with the C1 recipe, check reproduction, then predict
        for lab, trainset in (('L5', tr0), ('L5n', tr10)):
            net5, z5, _, _ = fit('L5', None, s, trainset, ho, Yho, mho)
            with torch.no_grad():
                p_true = net5(torch.tensor(z5(feats(va, 'L5')))).view(-1, 12, 2).numpy()
                dev = np.abs(p_true - L['P'][(lab, s, 0.0)]).max()
                P[(lab, s, 'pred')] = net5(torch.tensor(z5(feats(with_lon(va, PRED[s]), 'L5')))).view(-1, 12, 2).numpy()
                if lab == 'L5n':
                    C0 = CONF[s]
                    for e in RATES:
                        yn = confusion_noise(yva, C0, e)
                        P[(lab, s, f'conf{int(e*100)}')] = net5(torch.tensor(z5(feats(with_lon(va, yn), 'L5')))).view(-1, 12, 2).numpy()
                        P[(lab, s, f'conf{int(e*100)}_err')] = float((yn != yva).mean())
            print(f'[rung3]   {lab} seed {s}: reproduces C1 predictions, max |dev| {dev:.2e} m', flush=True)
    occ = [PMX.occupancies(r) for r in recs]
    PER = {k: E.per_sample(recs, v, occ) for k, v in P.items() if not k[2].endswith('_err')}
    pickle.dump({'PRED': PRED, 'CONF': CONF, 'yva': yva, 'ff': ff, 'P': P, 'PER': PER}, open(OUT, 'wb'))
    report(PRED, CONF, yva, ff, PER, P, L)


def tem(p, m):
    return [float(p['l2'][m, :k].mean()) for k in (2, 4, 6)]


def report(PRED, CONF, yva, ff, PER, P, L):
    names = ['stop', 'accel', 'decel', 'maint']
    for blk, m in (('ALL 5,119', np.ones(len(yva), bool)), (f'EXCL. FIRST FRAMES ({int((~ff).sum())})', ~ff)):
        print(f'\n===== {blk} =====')
        print('C4a classifier accuracy per seed: ' + '  '.join(
            f's{s} {(PRED[s][m] == yva[m]).mean():.4f}' for s in SEEDS)
              + f'   (majority class "maintain" = {(yva[m] == 3).mean():.4f})')
        C = sum(np.array([[((yva[m] == i) & (PRED[s][m] == j)).sum() for j in range(4)] for i in range(4)])
                for s in SEEDS)
        print('  confusion, summed over 3 seeds (rows = true, cols = predicted):')
        print('           ' + ''.join(f'{n:>8}' for n in names) + '   recall')
        for i in range(4):
            print(f'    {names[i]:6} ' + ''.join(f'{C[i, j]:8d}' for j in range(4)) + f'   {C[i, i]/C[i].sum():.3f}')
        LP = L['PER']
        mean = lambda lab, key: np.mean([np.nanmean(LP[(lab, s, 0.0)]['ade'][m]) if key == 'ade'
                                         else tem(LP[(lab, s, 0.0)], m)[2] for s in SEEDS])
        a2, a5 = mean('L2', 'ade'), mean('L5', 'ade'); t2, t5 = mean('L2', 't'), mean('L5', 't')
        print(f'\n  C4b/C5 seed means (3 seeds); gap closed vs C1 L2 (ADE {a2:.3f}, TemAvg3s {t2:.3f}) and '
              f'L5 (ADE {a5:.3f}, TemAvg3s {t5:.3f}):')
        print(f'    {"model / lon source":34} {"ADE@6s":>7} {"L2 TemAvg 1/2/3s":>20}  gap(ADE) gap(TemAvg3s)')
        rows = [('L5 oracle lon (C1)', [LP[('L5', s, 0.0)] for s in SEEDS]),
                ('L5 PREDICTED lon', [PER[('L5', s, 'pred')] for s in SEEDS]),
                ('L5-noise oracle lon (C1)', [LP[('L5n', s, 0.0)] for s in SEEDS]),
                ('L5-noise PREDICTED lon', [PER[('L5n', s, 'pred')] for s in SEEDS])]
        rows += [(f'L5-noise conf-shaped {e}%', [PER[('L5n', s, f'conf{e}')] for s in SEEDS]) for e in (10, 20, 40)]
        rows += [(f'L5-noise uniform flip {e}% (C1)', [LP[('L5n', s, e / 100)] for s in SEEDS]) for e in (10, 20, 40)]
        rows += [('L2 ego + cmd (C1)', [LP[('L2', s, 0.0)] for s in SEEDS])]
        for lab, ps in rows:
            ade = np.mean([np.nanmean(p['ade'][m]) for p in ps]); t = np.mean([tem(p, m) for p in ps], 0)
            print(f'    {lab:34} {ade:7.3f} {t[0]:6.3f} {t[1]:6.3f} {t[2]:6.3f}  '
                  f'{(ade - a2)/(a5 - a2):+8.3f} {(t[2] - t2)/(t5 - t2):+13.3f}')
    print('\n  realised confusion-shaped error rates (seed 42/123/2024): ' + '; '.join(
        f'{e}%: ' + ' '.join(f'{P[("L5n", s, f"conf{e}_err")]:.3f}' for s in SEEDS) for e in (10, 20, 40)))


if __name__ == '__main__':
    main()
