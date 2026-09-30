"""w1/ego_mlp.py -- WEEK1 STEP 3.5b: ego-status MLP baseline (AD-MLP style, Li et al.).

Input: the causal ego state used by the VLA (records.ego_state_w1: 4 rows x [speed,
relative yaw, yaw rate, accel], current row from CAN) = 16 features, + one-hot command
(3) = 19, standardised on train. Output: 12 waypoints (x,y) in the t0 LIDAR frame, by
DIRECT REGRESSION (no tokenizer, no rollout). MLP 19-512-512-24, ReLU, L1 loss on the
waypoints, AdamW 1e-3, batch 256, 100 epochs, cosine LR. Checkpoint selection = best
HOLDOUT mean ADE@6s. Official split; 3 seeds. Evaluated on official val (all 5,119) with
CV, KIN and ORACLE-KIN via evalw1.
"""
import sys, pickle, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records, evalw1 as E

OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/w1_egomlp.pkl'


def feats(R):
    X = np.stack([t['w1_ego'][:4].numpy().ravel() for t in R])
    C = np.eye(3)[[t['command'] for t in R]]
    return np.concatenate([X, C], 1).astype(np.float32)


def train_one(seed, Xtr, Ytr, Xho, Yho, mho, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                        nn.Linear(512, 24)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    EP = 100; sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    xh = torch.tensor(Xho, device=dev)
    best, state = 1e9, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(EP):
        net.train(); perm = torch.randperm(len(xt), generator=g).to(dev)
        for s in range(0, len(xt), 256):
            b = perm[s:s + 256]
            loss = (net(xt[b]) - yt[b]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            ph = net(xh).view(-1, 12, 2).cpu().numpy()
        ade = np.linalg.norm(ph[mho] - Yho[mho].reshape(-1, 12, 2), axis=2).mean()
        if ade < best:
            best, state, bep = ade, {k: v.clone() for k, v in net.state_dict().items()}, ep
    net.load_state_dict(state); net.eval()
    return net, best, bep


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tr = records.build('train', ('cmd',)); ho = records.build('holdout', ('cmd',), n_fut=6)
    va = records.build('val', ('cmd',), n_fut=6)
    Xtr = feats(tr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: (X - mu) / sd
    Ytr = np.stack([t['future_positions'] for t in tr]).reshape(len(tr), -1).astype(np.float32)
    Yho = np.stack([np.pad(t['future_positions'], ((0, 12 - len(t['future_positions'])), (0, 0)))
                    for t in ho]).reshape(len(ho), -1)
    mho = np.array([t['n_fut'] == 12 for t in ho])
    preds = {}
    for seed in (42, 123, 2024):
        t0 = time.time()
        net, best, bep = train_one(seed, z(Xtr), Ytr, z(feats(ho)), Yho, mho, dev)
        with torch.no_grad():
            pv = net(torch.tensor(z(feats(va)), device=dev)).view(-1, 12, 2).cpu().numpy()
            ph = net(torch.tensor(z(feats(ho)), device=dev)).view(-1, 12, 2).cpu().numpy()
        preds[seed] = {'val': pv, 'holdout': ph, 'holdout_ade': best, 'epoch': bep}
        print(f'[egomlp] seed {seed}: holdout ADE@6s {best:.3f} (epoch {bep}), {time.time()-t0:.0f}s',
              flush=True)
    pickle.dump({'preds': preds, 'val_tokens': [t['sample_token'] for t in va]}, open(OUT, 'wb'))

    # ---- evaluation on official val ----
    recs = E.split_records('val')
    assert [r['sample_token'] for r in recs] == [t['sample_token'] for t in va]
    occ = [__import__('plan_metrics').occupancies(r) for r in recs]
    tab = E.oracle_kin_table()
    base = {'CV': E.cv(recs), 'KIN': E.kin(recs), 'ORACLE-KIN': E.oracle_kin(recs, tab)}
    PER = {k: E.per_sample(recs, v, occ) for k, v in base.items()}
    for s in preds:
        PER[f'egoMLP s{s}'] = E.per_sample(recs, preds[s]['val'], occ)
    print(f'\nOFFICIAL VAL n={len(recs)} (ADE/FDE@6s on n_fut=12: '
          f'{int((~np.isnan(PER["CV"]["ade"])).sum())})')
    print(f'oracle-kin table: lon accel {tab[0]}  lat curvature {tab[1]}')
    print(E.HDR)
    for k, p in PER.items():
        print(E.row(k, p))
    print('\npaired scene-level bootstrap (negative => first better)')
    for s in preds:
        for b in ('CV', 'KIN', 'ORACLE-KIN'):
            print(E.compare(recs, PER[f'egoMLP s{s}'], PER[b], f'egoMLP s{s} - {b}'))
    print(E.compare(recs, PER['KIN'], PER['CV'], 'KIN - CV'))
    st = E.strata(recs)
    print('\nstrata L2@3s NoAvg mean (n):')
    for sname in ('straight', 'turning', 'stationary'):
        m = st == sname
        print(f'  {sname:10} n={m.sum():5d} ' + '  '.join(
            f'{k} {PER[k]["l2"][m, 5].mean():.3f}' for k in PER))


if __name__ == '__main__':
    main()
