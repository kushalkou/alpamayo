"""w1/c7_bottleneck.py -- QUEUE v2 C7: FIXED-HEAD evaluation of the decision bottleneck.

One head architecture for every meta-action source: the Ego-MLP of C1 rung L5 (ego 16 +
lon 4 + lat 3 [= cmd] -> 12 waypoints; 512-512 ReLU, L1, AdamW 1e-3, batch 256, 100
epochs, cosine; ego_mlp.train_one), 3 head seeds (42, 123, 2024), CPU. The lon slot takes
either a one-hot (HARD = argmax) or the class probabilities (SOFT). lat = cmd throughout.
Labels: records.meta_lon (the A2 rule), frozen.

Meta-action sources (val inputs):
  none        the L2 head (ego + cmd, no meta-action)           baseline   (C1 L2)
  oracle      true lon/lat                                       ceiling    (C1 L5) [privileged]
  oracle fN   true labels with N% uniform flips (lat AND lon, as A2 / C1; train flips
              seed 42 at the SAME rate, test flips seed 777)   curve      [privileged]
  MLP         MLP classifier on ego + cmd (rung3 recipe, selection = HOLDOUT MACRO-F1 as
              for the VLA), 3 seeds; head seed s uses classifier seed s   non-visual control
  V8a         VLA intent predictor, no cameras (seed 42)         same VLA, no cameras
  V8b         VLA intent predictor, 6 cameras (seed 42)          VLA with cameras
Head training variants per predictor:
  h1  trained on ORACLE lon replaced by draws that match the predictor's HOLDOUT confusion:
      a train sample of true class c gets the output of a random holdout sample of true
      class c (seed = head seed + 500). HARD = that sample's argmax (exactly a draw from
      the holdout confusion row); SOFT = its probability vector.
  h2  MLP only: 5-fold scene-grouped cross-fitted out-of-fold predictions on train (fold
      classifiers: same recipe and seed); val inputs come from the full-train classifier
      in both h1 and h2, so h1 vs h2 differ only in how the head was trained.
Head checkpoint selection: holdout ADE@6s with the predictor's own HOLDOUT outputs as the
lon input (oracle: true labels; oracle fN: true labels flipped at N%, seed 777).

PRE-REGISTERED (written before any V8 result): primary variant = h1 HARD. Downstream
statistic = per-sample metric averaged over the 3 head seeds, paired scene bootstrap
(10,000). "Cameras add decision information" only if V8b beats V8a with the 95% CI
excluding 0 on L2@3s (NoAvg) or ADE@6s in the primary variant; h1 SOFT is reported
alongside. Classifier comparisons: macro-F1 difference, scene bootstrap (2,000 draws),
V8b vs V8a and V8b vs MLP seed 42.
Usage: python w1/c7_bottleneck.py [report]   (cached in results/w1_c7.pkl; recomputes
only what is missing, e.g. after V8b's predictions appear)
"""
import os, sys, pickle, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records, evalw1 as E, plan_metrics as PMX
from ego_mlp import train_one
from ladder import SEEDS, EXTRA, OUT as LADDER

torch.set_num_threads(24)
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
OUT = f'{RES}/w1_c7.pkl'
RATES = (0.1, 0.2, 0.4)
NAMES = ('stop', 'accel', 'decel', 'maint')


# ---------------------------------------------------------------- classifier (MLP)
def cls_feats(R):
    ego = np.stack([t['w1_ego'][:4].numpy().ravel() for t in R])
    return np.concatenate([ego, np.eye(3)[[t['command'] for t in R]]], 1).astype(np.float32)


def macro_f1(y, p):
    f = []
    for c in range(4):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def train_cls(seed, Xtr, ytr, Xho, yho):
    """rung3 recipe (19-512-512-4, CE, AdamW 1e-3, bs 256, 100 ep, cosine); selection =
    holdout macro-F1. Returns a function X -> probabilities."""
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
            f1 = macro_f1(yho, net(xh).argmax(1).numpy())
        if f1 > best:
            best, state = f1, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()

    def prob(X):
        with torch.no_grad():
            return torch.softmax(net(torch.tensor(X)), -1).numpy()
    return prob


# ---------------------------------------------------------------- head
def head_feats(R, lon):
    ego = np.stack([t['w1_ego'][:4].numpy().ravel() for t in R])
    lat = np.eye(3)[[t['shown']['lat'] for t in R]]
    return np.concatenate([ego, lon, lat], 1).astype(np.float32)


def fit_head(seed, tr, lon_tr, ho, lon_ho, Yho, mho, va, lon_va):
    Xtr = head_feats(tr, lon_tr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ytr = np.stack([t['future_positions'] for t in tr]).reshape(len(tr), -1).astype(np.float32)
    net, best, bep = train_one(seed, (Xtr - mu) / sd, Ytr, (head_feats(ho, lon_ho) - mu) / sd, Yho, mho, 'cpu')
    with torch.no_grad():
        return net(torch.tensor((head_feats(va, lon_va) - mu) / sd)).view(-1, 12, 2).numpy(), best, bep


def onehot(y):
    return np.eye(4)[np.asarray(y)]


def matched_draw(ytr, yho, Pho, seed):
    """per train sample of true class c: the output of a random holdout sample of class c."""
    rs = np.random.RandomState(seed + 500)
    idx = np.empty(len(ytr), int)
    for c in range(4):
        pool = np.where(yho == c)[0]; m = ytr == c
        idx[m] = rs.choice(pool, m.sum(), replace=True)
    return Pho[idx]


def scene_folds(R, k=5, seed=0):
    sc = sorted({t['scene_name'] for t in R})
    rs = np.random.RandomState(seed); sc = [sc[i] for i in rs.permutation(len(sc))]
    f = {s: i % k for i, s in enumerate(sc)}
    return np.array([f[t['scene_name']] for t in R])


# ---------------------------------------------------------------- main
def main():
    C = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {'P': {}, 'CLS': {}, 'info': {}}
    P, CLS = C['P'], C['CLS']
    t0 = time.time()
    save = lambda: pickle.dump(C, open(OUT, 'wb'))
    tr = records.build('train', EXTRA, 0.0)
    ho = records.build('holdout', EXTRA, 0.0, n_fut=6)
    va = records.build('val', EXTRA, 0.0, n_fut=6)
    Yho = np.stack([np.pad(t['future_positions'], ((0, 12 - len(t['future_positions'])), (0, 0)))
                    for t in ho]).reshape(len(ho), -1)
    mho = np.array([t['n_fut'] == 12 for t in ho])
    ytr, yho, yva = (np.array([t['meta_lon'] for t in R]) for R in (tr, ho, va))
    C['info'].update(ytr=ytr, yho=yho, yva=yva, order=[t['sample_token'] for t in va],
                     scene=np.array([t['scene_name'] for t in va]),
                     ho_scene=np.array([t['scene_name'] for t in ho]))
    log = lambda s: print(f'[c7 {time.time()-t0:5.0f}s] {s}', flush=True)

    # --- oracle flip curve (matched train rate, test seed 777), hard, lat + lon flipped
    for e in RATES:
        if all(('oracle', f'f{int(e*100)}', 'h1', 'hard', s) in P for s in SEEDS):
            continue
        trf = records.build('train', EXTRA, e, flip_seed=42)
        hof = records.build('holdout', EXTRA, e, flip_seed=777, n_fut=6)
        vaf = records.build('val', EXTRA, e, flip_seed=777, n_fut=6)
        for s in SEEDS:
            lon = lambda R: onehot([t['shown']['lon'] for t in R])
            p, b, ep = fit_head(s, trf, lon(trf), hof, lon(hof), Yho, mho, vaf, lon(vaf))
            P[('oracle', f'f{int(e*100)}', 'h1', 'hard', s)] = p
            log(f'oracle flip {e}: head seed {s} holdout ADE {b:.3f} (ep {ep})')
        save()

    # --- MLP classifier: full-train (3 seeds) + 5-fold OOF on train
    Xtr, Xho, Xva = cls_feats(tr), cls_feats(ho), cls_feats(va)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: (X - mu) / sd
    folds = scene_folds(tr)
    for s in SEEDS:
        if ('MLP', s) in CLS:
            continue
        prob = train_cls(s, z(Xtr), ytr, z(Xho), yho)
        oof = np.zeros((len(tr), 4))
        for k in range(5):
            m = folds == k
            pk = train_cls(s, z(Xtr[~m]), ytr[~m], z(Xho), yho)
            oof[m] = pk(z(Xtr[m]))
        CLS[('MLP', s)] = {'holdout': prob(z(Xho)), 'val': prob(z(Xva)), 'oof': oof}
        log(f'MLP classifier seed {s}: holdout macro-F1 {macro_f1(yho, CLS[("MLP", s)]["holdout"].argmax(1)):.4f}'
            f' val macro-F1 {macro_f1(yva, CLS[("MLP", s)]["val"].argmax(1)):.4f}'
            f' OOF train macro-F1 {macro_f1(ytr, oof.argmax(1)):.4f}')
        save()

    # --- VLA predictors
    for v in ('V8a', 'V8b'):
        f = f'{RES}/w1_intent_{v}.pkl'
        if os.path.exists(f) and (v, 42) not in CLS:
            I = pickle.load(open(f, 'rb'))
            assert I['val']['order'] == C['info']['order']
            assert I['holdout']['order'] == [t['sample_token'] for t in ho]
            CLS[(v, 42)] = {'holdout': I['holdout']['probs'], 'val': I['val']['probs'],
                            'best_epoch': I['best_epoch']}
            log(f'{v}: loaded (best epoch {I["best_epoch"]})')

    # --- heads per predictor
    for (src, cs), D in list(CLS.items()):
        hseeds = [cs] if src == 'MLP' else list(SEEDS)
        for s in hseeds:
            for form in ('hard', 'soft'):
                cvt = (lambda Q: onehot(Q.argmax(1))) if form == 'hard' else (lambda Q: Q)
                key = (src, 'pred', 'h1', form, s)
                if key not in P:
                    lon_tr = cvt(matched_draw(ytr, yho, D['holdout'], s))
                    P[key], b, ep = fit_head(s, tr, lon_tr, ho, cvt(D['holdout']), Yho, mho, va, cvt(D['val']))
                    log(f'{key}: holdout ADE {b:.3f} (ep {ep})'); save()
                key2 = (src, 'pred', 'h2', form, s)
                if src == 'MLP' and key2 not in P:
                    P[key2], b, ep = fit_head(s, tr, cvt(D['oof']), ho, cvt(D['holdout']), Yho, mho, va, cvt(D['val']))
                    log(f'{key2}: holdout ADE {b:.3f} (ep {ep})'); save()

    # --- per-sample metrics
    recs = E.split_records('val')
    assert [r['sample_token'] for r in recs] == C['info']['order']
    PER = C.setdefault('PER', {})
    todo = [k for k in P if k not in PER]
    if todo:
        occ = [PMX.occupancies(r) for r in recs]
        for k in todo:
            PER[k] = E.per_sample(recs, P[k], occ)
        save()
    log('compute done')


# ---------------------------------------------------------------- report
def avg(pers):
    return {k: np.mean([p[k] for p in pers], 0) for k in pers[0]}


def boot_f1(scene, y, pa, pb, n=2000, seed=0):
    us = np.unique(scene); idx = [np.where(scene == s)[0] for s in us]
    rs = np.random.RandomState(seed); d = []
    for _ in range(n):
        ii = np.concatenate([idx[j] for j in rs.randint(0, len(us), len(us))])
        d.append(macro_f1(y[ii], pa[ii]) - macro_f1(y[ii], pb[ii]))
    lo, hi = np.percentile(d, [2.5, 97.5]); d = np.array(d)
    return macro_f1(y, pa) - macro_f1(y, pb), lo, hi, min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def report():
    C = pickle.load(open(OUT, 'rb')); L = pickle.load(open(LADDER, 'rb'))
    P, CLS, PER, I = C['P'], C['CLS'], C['PER'], C['info']
    yva, yho, scene = I['yva'], I['yho'], I['scene']
    recs = E.split_records('val')
    W, V0, _ = E.data()
    ff = np.array([not V0[r['sample_token']]['can_ok'] and len(r['past_poses']) == 0 for r in recs])
    preds = [k for k in (('MLP', 42), ('MLP', 123), ('MLP', 2024), ('V8a', 42), ('V8b', 42)) if k in CLS]

    print('C7 CLASSIFIERS -- lon meta-action (4 classes); val = official val n_fut >= 6')
    for blk, m in (('ALL 5,119', np.ones(len(yva), bool)), (f'EXCL. FIRST FRAMES ({int((~ff).sum())})', ~ff)):
        print(f'\n===== {blk} =====')
        print(f'  {"predictor":14} {"split":8} {"acc":>6} {"macroF1":>8}   recall ' + ' '.join(f'{n:>6}' for n in NAMES))
        for nm, yy, mm in (('holdout', yho, np.ones(len(yho), bool)), ('val', yva, m)):
            print(f'  {"majority":14} {nm:8} {np.mean(yy[mm] == 3):6.3f} {macro_f1(yy[mm], np.full(mm.sum(), 3)):8.3f}')
            for k in preds:
                p = CLS[k][nm].argmax(1)[mm]; y = yy[mm]
                rec = [((p == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in range(4)]
                print(f'  {k[0]+" s"+str(k[1]):14} {nm:8} {np.mean(p == y):6.3f} {macro_f1(y, p):8.3f}          '
                      + ' '.join(f'{r:6.3f}' for r in rec))
        mlp = [k for k in preds if k[0] == 'MLP']
        if mlp:
            f = [macro_f1(yva[m], CLS[k]['val'].argmax(1)[m]) for k in mlp]
            a = [np.mean(CLS[k]['val'].argmax(1)[m] == yva[m]) for k in mlp]
            print(f'  MLP 3-seed mean +- sd: acc {np.mean(a):.3f} +- {np.std(a):.3f}  macro-F1 {np.mean(f):.3f} +- {np.std(f):.3f}')
        for k in preds:
            if k[0] == 'MLP' and k[1] != 42:
                continue
            p = CLS[k]['val'].argmax(1)[m]; y = yva[m]
            print(f'  confusion {k[0]} s{k[1]} (rows true, cols pred): ' + ' | '.join(
                f'{NAMES[i]}: ' + ' '.join(f'{((y == i) & (p == j)).sum():4d}' for j in range(4)) for i in range(4)))
        print('  macro-F1 differences (scene bootstrap, 2,000):')
        for a_, b_ in ((('V8b', 42), ('V8a', 42)), (('V8b', 42), ('MLP', 42)), (('V8a', 42), ('MLP', 42))):
            if a_ in CLS and b_ in CLS:
                d = boot_f1(scene[m], yva[m], CLS[a_]['val'].argmax(1)[m], CLS[b_]['val'].argmax(1)[m])
                print(f'    {a_[0]} - {b_[0]}: {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] p={d[3]:.4f}')

    # downstream
    LP = L['PER']
    S = {'none (ego + cmd) [C1 L2]': [LP[('L2', s, 0.0)] for s in SEEDS],
         'oracle [C1 L5] [P]': [LP[('L5', s, 0.0)] for s in SEEDS]}
    for e in (10, 20, 40):
        k = [('oracle', f'f{e}', 'h1', 'hard', s) for s in SEEDS]
        if all(x in PER for x in k):
            S[f'oracle flip {e}% (matched) [P]'] = [PER[x] for x in k]
    for src in ('MLP', 'V8a', 'V8b'):
        for h in ('h1', 'h2'):
            for form in ('hard', 'soft'):
                k = [x for x in PER if x[:4] == (src, 'pred', h, form)]
                if len(k) == 3:
                    S[f'{src} {h} {form}'] = [PER[x] for x in sorted(k, key=lambda x: x[4])]
    C['SUMMARY'] = {}
    for blk, m in (('ALL 5,119', np.ones(len(yva), bool)), (f'EXCL. FIRST FRAMES ({int((~ff).sum())})', ~ff)):
        sub = [r for r, k in zip(recs, m) if k]
        cut = lambda p: {k: v[m] for k, v in p.items()}
        print(f'\n===== DOWNSTREAM, fixed head (L5 architecture), {blk}; rows = per-sample metrics averaged '
              f'over 3 head seeds =====')
        print(E.HDR)
        A = {k: avg(v) for k, v in S.items()}
        for k, v in A.items():
            print(E.row(k[:22], cut(v)) + f'   {k}')
        a0 = np.nanmean(cut(A['none (ego + cmd) [C1 L2]'])['ade']); a1 = np.nanmean(cut(A['oracle [C1 L5] [P]'])['ade'])
        l0 = cut(A['none (ego + cmd) [C1 L2]'])['l2'][:, 5].mean(); l1 = cut(A['oracle [C1 L5] [P]'])['l2'][:, 5].mean()
        print('\n  seed mean +- sd and gap closed (X - none)/(oracle - none):')
        for k, v in S.items():
            ad = [np.nanmean(p['ade'][m]) for p in v]; l2 = [p['l2'][m, 5].mean() for p in v]
            C['SUMMARY'][(blk[:3], k)] = (np.mean(ad), np.std(ad), np.mean(l2), np.std(l2))
            print(f'    {k:34} ADE6 {np.mean(ad):.3f} +- {np.std(ad):.3f}  L2@3s {np.mean(l2):.3f} +- {np.std(l2):.3f}'
                  f'  gap(ADE) {(np.mean(ad)-a0)/(a1-a0):+.3f}  gap(L2@3s) {(np.mean(l2)-l0)/(l1-l0):+.3f}')
        print('\n  PAIRED SCENE BOOTSTRAP (negative => first better)')
        for form in ('hard', 'soft'):
            for a_, b_ in (('V8b', 'V8a'), ('V8b', 'MLP'), ('V8a', 'MLP'), ('V8b', 'none'), ('MLP', 'none'),
                           ('V8a', 'none')):
                ka = f'{a_} h1 {form}'; kb = 'none (ego + cmd) [C1 L2]' if b_ == 'none' else f'{b_} h1 {form}'
                if ka in A and kb in A:
                    print(E.compare(sub, cut(A[ka]), cut(A[kb]), f'{a_} - {b_} (h1 {form})'))
        for form in ('hard', 'soft'):
            if f'MLP h2 {form}' in A:
                print(E.compare(sub, cut(A[f'MLP h1 {form}']), cut(A[f'MLP h2 {form}']), f'MLP h1 - h2 ({form})'))
        if 'V8b h1 hard' in A and 'V8a h1 hard' in A:
            d1 = E.scene_boot(sub, cut(A['V8b h1 hard'])['l2'][:, 5], cut(A['V8a h1 hard'])['l2'][:, 5])
            d2 = E.scene_boot(sub, cut(A['V8b h1 hard'])['ade'], cut(A['V8a h1 hard'])['ade'])
            yes = d1[2] < 0 or d2[2] < 0
            print(f'\n  PRE-REGISTERED VERDICT (h1 hard): V8b - V8a L2@3s {E.fmt_ci(d1)}, ADE6 {E.fmt_ci(d2)} -> '
                  + ('CAMERAS ADD DECISION INFORMATION' if yes else 'cameras do NOT add decision information '
                     '(no CI excludes 0 in V8b\'s favour)'))
    pickle.dump(C, open(OUT, 'wb'))


if __name__ == '__main__':
    if sys.argv[1:] != ['report']:
        main()
    report()
