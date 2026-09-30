"""w1/egomlp_ab.py -- planner item 2: quantify the VAD-converter ego-status leak.

Ego-MLP (w1/ego_mlp.py recipe, unchanged: 512-512, L1, 100 epochs, holdout selection),
3 seeds, CPU, official split, two feature sources + one-hot command:
  (a) OUR causal features: records.ego_state_w1 (16; CAN at the last message <= t0)
  (b) VAD's gt_ego_lcf_feat exactly as its converter computes it (9; w1/vad_feats.py,
      incl. all four future reads)
Also the kinematic rule under (b): v0 = VAD v0, accel = VAD ax, yaw rate = VAD ego_w.
Tables: all 5,119 and excluding the 140 first-of-scene samples without causal speed.
"""
import sys, pickle
import numpy as np, torch
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import records, evalw1 as E, plan_metrics as PMX
from ego_mlp import train_one
from targets import rollout as roll12

torch.set_num_threads(16)
VF = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_vadfeat.pkl', 'rb'))


def fa(R):
    return np.concatenate([np.stack([t['w1_ego'][:4].numpy().ravel() for t in R]),
                           np.eye(3)[[t['command'] for t in R]]], 1).astype(np.float32)


def fb(R):
    return np.concatenate([np.stack([VF[t['sample_token']] for t in R]),
                           np.eye(3)[[t['command'] for t in R]]], 1).astype(np.float32)


def kin_b(recs):
    out = []
    for r in recs:
        f = VF[r['sample_token']]; v0, a, yr = f[7], f[2], f[4]
        acc = np.zeros(12); cur = np.zeros(12); acc[0] = a
        v1 = max(0.0, v0 + a * 0.5)
        if v1 > 0.5: cur[0] = yr / v1
        out.append(roll12(acc, cur, v0))
    return np.stack(out)


def main():
    tr = records.build('train', ('cmd',)); ho = records.build('holdout', ('cmd',), n_fut=6)
    va = records.build('val', ('cmd',), n_fut=6)
    Ytr = np.stack([t['future_positions'] for t in tr]).reshape(len(tr), -1).astype(np.float32)
    Yho = np.stack([np.pad(t['future_positions'], ((0, 12 - len(t['future_positions'])), (0, 0)))
                    for t in ho]).reshape(len(ho), -1)
    mho = np.array([t['n_fut'] == 12 for t in ho])
    P = {}
    for src, F in (('a', fa), ('b', fb)):
        Xtr = F(tr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        z = lambda X: (X - mu) / sd
        for seed in (42, 123, 2024):
            net, best, bep = train_one(seed, z(Xtr), Ytr, z(F(ho)), Yho, mho, 'cpu')
            with torch.no_grad():
                P[(src, seed)] = net(torch.tensor(z(F(va)))).view(-1, 12, 2).numpy()
            print(f'[mlp {src}] seed {seed}: holdout ADE@6s {best:.3f} (epoch {bep})', flush=True)
    recs = E.split_records('val')
    assert [r['sample_token'] for r in recs] == [t['sample_token'] for t in va]
    occ = [PMX.occupancies(r) for r in recs]
    PER = {'CV': E.per_sample(recs, E.cv(recs), occ), 'KIN (ours)': E.per_sample(recs, E.kin(recs), occ),
           'KIN (VAD feats)': E.per_sample(recs, kin_b(recs), occ)}
    for k, v in P.items():
        PER[f'MLP ({k[0]}) s{k[1]}'] = E.per_sample(recs, v, occ)
    ff = np.array([not t['can_ok'] and len(r['past_poses']) == 0 for t, r in zip(va, recs)])
    print(f'first-of-scene without causal speed: n={ff.sum()}')
    for lab, m in (('ALL 5,119', np.ones(len(recs), bool)), (f'EXCL. FIRST FRAME (n={m_n})'
                   if (m_n := int((~ff).sum())) else '', ~ff)):
        sub = [r for r, k in zip(recs, m) if k]
        print(f'\n=== {lab} ===\n' + E.HDR)
        for k, p in PER.items():
            print(E.row(k, {kk: vv[m] for kk, vv in p.items()}))
        print('  paired scene-level bootstrap (b) - (a), per seed:')
        for s in (42, 123, 2024):
            pa = {kk: vv[m] for kk, vv in PER[f'MLP (a) s{s}'].items()}
            pb = {kk: vv[m] for kk, vv in PER[f'MLP (b) s{s}'].items()}
            print(E.compare(sub, pb, pa, f'MLP (b)-(a) s{s}'))
            cb = E.scene_boot(sub, pb['col_vp'][:, :6].mean(1), pa['col_vp'][:, :6].mean(1))
            cn = E.scene_boot(sub, pb['col_veh'][:, 5], pa['col_veh'][:, 5])
            print(f'      collision (b)-(a): TemAvg@3s {100*cb[0]:+.3f}% [{100*cb[1]:+.3f},{100*cb[2]:+.3f}]'
                  f'  NoAvg@3s {100*cn[0]:+.3f}% [{100*cn[1]:+.3f},{100*cn[2]:+.3f}]')
        pk = {kk: vv[m] for kk, vv in PER['KIN (VAD feats)'].items()}
        po = {kk: vv[m] for kk, vv in PER['KIN (ours)'].items()}
        print(E.compare(sub, pk, po, 'KIN (VAD feats) - KIN (ours)'))


if __name__ == '__main__':
    main()
