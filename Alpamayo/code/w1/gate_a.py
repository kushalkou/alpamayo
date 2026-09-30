"""w1/gate_a.py -- WEEK1 GATE A evaluation (official val, ALL 5,119; nothing tuned on val).

Decode: STOP-aware hybrid (argmax where p(STOP) > tau, else expectation), tau chosen on
HOLDOUT from {0.3,0.5,0.7,0.9} by mean ADE@6s; rollout in the lidar frame from v0_can.
Tables carry CV, KIN (kinematic reference), ORACLE-KIN (amendment 4) and the ego-MLP
(seed 42), with mean/median/p95, L2 both conventions, collision both conventions,
ADE/FDE@6s, strata. Comparisons: paired SCENE-level bootstrap (evalw1.scene_boot).
Blend: alpha fit on HOLDOUT, applied to val; + the three nulls (gate41 definitions,
alpha fit on holdout, 20 seeds). Per-slot argmax accuracy (leak fingerprint).
Usage: python w1/gate_a.py A1 [A2] [A3]
"""
import sys, pickle, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
import evalw1 as E, plan_metrics as PMX, records
from targets import rollout as roll12
from tokenizer import STOP_TOKEN

RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
TAUS = (0.3, 0.5, 0.7, 0.9)


def load(tag, split, f=0.0):
    return pickle.load(open(f'{RES}/w1_dump_{tag}_{split}_f{f}.pkl', 'rb'))


def preds(dump, recs, mode, tau=None):
    W, V0, T = E.data()
    out = []
    for r in recs:
        d = dump['data'][r['sample_token']]
        a, e, p = np.array(d['argmax']), np.array(d['expect']), np.array(d['p_stop'])
        v = a if mode == 'argmax' else (e if mode == 'expect' else np.where(p > tau, a, e))
        out.append(roll12(v[:12], v[12:], T[r['sample_token']]['v0']))
    return np.stack(out)


def ade12(recs, P):
    W, V0, T = E.data()
    m = np.array([r['n_fut'] == 12 for r in recs])
    G = np.stack([T[r['sample_token']]['P'][:12] for r, k in zip(recs, m) if k])
    return float(np.linalg.norm(P[m] - G, axis=2).mean())


def nulls(Pho, Cho, rho, Pva, Cva, rva, occ, cv_per, a_real):
    """gate41 nulls in this frame; alpha fit on holdout; mean gain vs CV on val L2@3s/ADE6."""
    Dh, Dv = Pho - Cho, Pva - Cva
    res = {}
    mD = Dh.mean(0, keepdims=True)
    a = E.fit_alpha(Cho + mD, Cho, rho)
    res['null_mean'] = (a, E.per_sample(rva, Cva + a * mD, occ),
                        E.per_sample(rva, Cva + a_real * mD, occ))
    for nm in ('null_gauss', 'null_perm'):
        A, G, F = [], [], []
        for sd in range(5):
            r = np.random.RandomState(1000 + sd)
            if nm == 'null_gauss':
                sh = np.sqrt((Dh ** 2).sum(2).mean(0) / 2)[None, :, None]
                sv = np.sqrt((Dv ** 2).sum(2).mean(0) / 2)[None, :, None]
                Nh, Nv = r.randn(*Dh.shape) * sh, r.randn(*Dv.shape) * sv
            else:
                Nh, Nv = Dh[r.permutation(len(Dh))], Dv[r.permutation(len(Dv))]
            a = E.fit_alpha(Cho + Nh, Cho, rho); A.append(a)
            G.append(E.per_sample(rva, Cva + a * Nv, occ)); F.append(E.per_sample(rva, Cva + a_real * Nv, occ))
        res[nm] = (float(np.mean(A)), G[0], F[0])
    return res


def main():
    tags = sys.argv[1:] or ['A1']
    rva = E.split_records('val'); rho = E.split_records('holdout')
    occ = [PMX.occupancies(r) for r in rva]
    tab = E.oracle_kin_table()
    Cva, Cho = E.cv(rva), E.cv(rho)
    base = {'CV': Cva, 'KIN': E.kin(rva), 'ORACLE-KIN': E.oracle_kin(rva, tab)}
    mlp = pickle.load(open(f'{RES}/w1_egomlp.pkl', 'rb'))
    base['egoMLP s42'] = mlp['preds'][42]['val']
    PER = {k: E.per_sample(rva, v, occ) for k, v in base.items()}
    st = E.strata(rva)
    PRED = {}
    for tag in tags:
        dh, dv = load(tag, 'holdout'), load(tag, 'val')
        best = min(TAUS, key=lambda t: ade12(rho, preds(dh, rho, 'hybrid', t)))
        for mode in ('argmax', 'expect', 'hybrid'):
            P = preds(dv, rva, mode, best)
            PER[f'{tag} {mode}' + (f' t{best}' if mode == 'hybrid' else '')] = E.per_sample(rva, P, occ)
            if mode == 'hybrid':
                PRED[tag] = (P, preds(dh, rho, 'hybrid', best), best)
    H = {tag: [k for k in PER if k.startswith(f'{tag} hybrid')][0] for tag in tags}
    W, V0, _ = E.data()
    ff = np.array([not V0[r['sample_token']]['can_ok'] and len(r['past_poses']) == 0 for r in rva])
    for blk, m in (('ALL 5,119', np.ones(len(rva), bool)),
                   (f'EXCL. FIRST FRAMES (n={int((~ff).sum())})', ~ff)):
        sub = [r for r, k in zip(rva, m) if k]
        cut = lambda p: {k: v[m] for k, v in p.items()}
        print(f'\n===== GATE A -- official val, {blk} (ADE/FDE@6s on the n_fut=12 part); '
              f'model = free-running AR, lidar frame =====')
        print(E.HDR)
        for k, p in PER.items():
            print(E.row(k, cut(p)))
        print('\nPAIRED SCENE-LEVEL BOOTSTRAP (negative => first better); models = hybrid decode')
        for tag in tags:
            for b in ('CV', 'KIN', 'egoMLP s42') + (('ORACLE-KIN',) if tag == 'A2' else ()):
                print(E.compare(sub, cut(PER[H[tag]]), cut(PER[b]), f'{tag} - {b}'))
            print(E.compare(sub, cut(PER[f'{tag} expect']), cut(PER[f'{tag} argmax']), f'{tag} expect - argmax'))
        for a_, b_ in (('A2', 'A1'), ('A3', 'A1')):
            if a_ in tags and b_ in tags:
                print(E.compare(sub, cut(PER[H[a_]]), cut(PER[H[b_]]), f'{a_} - {b_}'))

    print('\nSLOT-0 ACCURACY + INPUT-SHUFFLE TEST (val; ego + extra rows permuted across samples, '
          'seed 99; hybrid decode, same tau)')
    for tag in tags:
        dv = load(tag, 'val')
        s0 = np.mean([dv['data'][r['sample_token']]['tok'][0] == dv['data'][r['sample_token']]['gt_tok'][0]
                      for r in rva if r['n_fut'] == 12])
        line = f'  {tag}: slot-0 acc {s0:.3f}  ADE6 {E.summary(PER[H[tag]])["ADE6"]:.3f}'
        try:
            ds = load(f'{tag}_shuf', 'val')
            ps = E.per_sample(rva, preds(ds, rva, 'hybrid', PRED[tag][2]), occ)
            s0s = np.mean([ds['data'][r['sample_token']]['tok'][0] == ds['data'][r['sample_token']]['gt_tok'][0]
                           for r in rva if r['n_fut'] == 12])
            a, b = E.summary(PER[H[tag]])['ADE6'], E.summary(ps)['ADE6']
            line += (f'  | SHUFFLED: slot-0 acc {s0s:.3f}  ADE6 {b:.3f}  (x{b/a:.3f}; within 5% -> '
                     f'{"FAILS shuffle test" if abs(b/a - 1) <= 0.05 else "passes"})')
        except FileNotFoundError:
            line += '  | shuffle dump missing'
        print(line)

    print('\nBLEND alpha*model + (1-alpha)*CV, alpha fit on HOLDOUT (ADE@6s), + nulls')
    for tag in tags:
        Pv, Ph, tau = PRED[tag]
        a = E.fit_alpha(Ph, Cho, rho)
        bl = E.per_sample(rva, a * Pv + (1 - a) * Cva, occ)
        print(E.row(f'{tag} blend a={a:.2f}', bl))
        print(E.compare(rva, bl, PER['CV'], f'{tag} blend - CV'))
        print(E.compare(rva, bl, PER['KIN'], f'{tag} blend - KIN'))
        for nm, (an, pn, pf) in nulls(Ph, Cho, rho, Pv, Cva, rva, occ, PER['CV'], a).items():
            g = PER['CV']['l2'][:, 5].mean() - pn['l2'][:, 5].mean()
            gf = PER['CV']['l2'][:, 5].mean() - pf['l2'][:, 5].mean()
            print(f'    {nm:11} alpha* {an:.2f}  L2@3s gain vs CV {g:+.4f}   at forced real alpha {gf:+.4f}')

    print('\nSTRATA (L2@3s NoAvg mean; L2 TemAvg@3s mean)')
    for s in ('straight', 'turning', 'stationary'):
        m = st == s
        print(f'  {s:10} n={m.sum():5d} ' + '  '.join(
            f'{k.split(" t0")[0]} {PER[k]["l2"][m, 5].mean():.3f}/{PER[k]["l2"][m].mean():.3f}'
            for k in ['CV', 'KIN', 'ORACLE-KIN', 'egoMLP s42'] + [H[t] for t in tags]))
        for tag in tags:
            sub = [r for r, k in zip(rva, m) if k]
            print('   ', E.compare(sub, {k: v[m] for k, v in PER[H[tag]].items()},
                                   {k: v[m] for k, v in PER['KIN'].items()}, f'{tag} - KIN [{s}]'))

    print('\nPER-SLOT ARGMAX TOKEN ACCURACY (val, n_fut=12)')
    for tag in tags:
        dv = load(tag, 'val')
        rows = [dv['data'][r['sample_token']] for r in rva if r['n_fut'] == 12]
        acc = np.mean([[d['tok'][s] == d['gt_tok'][s] for s in range(24)] for d in rows], 0)
        ce = -np.mean([d['lp65'][1:12] for d in rows])
        print(f'  {tag}: accel ' + ' '.join(f'{x:.2f}' for x in acc[:12]) + ' | curv '
              + ' '.join(f'{x:.2f}' for x in acc[12:]) + f' | free-running CE a1-11 {ce:.3f}')

    if 'A2' in tags:
        print('\nA2 FLIP-RATE SWEEP (test-time meta-action flips, seed 777; model trained at 10%)')
        _, _, tau = PRED['A2']
        curve = {}
        for f in (0.0, 0.1, 0.2, 0.4):
            try:
                dv = load('A2', 'val', f)
            except FileNotFoundError:
                continue
            per = E.per_sample(rva, preds(dv, rva, 'hybrid', tau), occ)
            lab = [dv['data'][r['sample_token']]['shown'] for r in rva]
            ok = E.per_sample(rva, E.oracle_kin(rva, tab, [x['lon'] for x in lab], [x['lat'] for x in lab]), occ)
            s, so = E.summary(per), E.summary(ok)
            curve[f] = s['ADE6']
            pickle.dump(curve, open(f'{RES}/w1_a2_flip_ade.pkl', 'wb'))
            print(f'  flip {f:.1f}: A2 ADE6 {s["ADE6"]:.3f} L2@3s {s["L2_NoAvg_3s"]:.3f} L2T@3s '
                  f'{s["L2_TemAvg_3s"]:.3f} | ORACLE-KIN with the same flipped labels: ADE6 '
                  f'{so["ADE6"]:.3f} L2@3s {so["L2_NoAvg_3s"]:.3f}')


if __name__ == '__main__':
    main()
