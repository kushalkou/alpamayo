"""ar1/r33_amend.py -- R3.3 amendment A1 / A2 (CPU, from dumps). [A3 rows when present]

A1 Consistency metric fix. NEW labels = R1.2 rules applied to the GT 2 Hz trajectory
   controls (w1/targets acc / cur, v0_can; same derivation as r33_eval.derive). Reported:
   GT self-consistency (GT-trajectory-derived vs NEW labels), M1 generated words vs NEW
   labels (acc / macro-F1 per slot), M1 consistency (predicted-trajectory-derived vs
   generated words), predicted-trajectory-derived vs NEW labels. CAN labels (what M1 was
   trained on) as the side number. Val n_fut = 12 (the 6 s labels exist).
A2 Collapse diagnosis. Per slot: distribution of generated words vs the GT distribution
   (CAN labels = training targets; NEW labels as well). Share of samples whose 12 generated
   words equal (a) the per-slot majority sequence (maintain x6, straight x6) and (b) the
   most frequent full 12-word sequence among TRAIN labels; the GT share of the same
   sequences for reference.
  python ar1/r33_amend.py [TAG ...]   (default M1; extra tags e.g. M1_T07 are added as rows)
"""
import sys, pickle, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, records
import q2_traj as Q
from gate_a import load
from meta_actions import LON, LAT
from r33_eval import derive, controls, f1
import finetune_meta as FM

DATA = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data'


def slot_line(name, y, p):
    acc = [(p[:, s] == y[:, s]).mean() for s in range(6)]
    mf = [f1(y[:, s], p[:, s], 7) for s in range(6)]
    return (f'    {name:34} acc ' + ' '.join(f'{x:.3f}' for x in acc) + f' (mean {np.mean(acc):.3f}) | mF1 '
            + ' '.join(f'{x:.3f}' for x in mf) + f' (mean {np.mean(mf):.3f})')


def main(tags):
    C = Q.ctx(); W, V0, T = E.data()
    rva = [r for r in C['rva'] if r['n_fut'] == 12]
    Mt = pickle.load(open(f'{DATA}/ar1_meta.pkl', 'rb'))
    st = {'missing': 0}
    can = np.array([sum(FM.meta_labels(Mt, r['sample_token'], st), []) for r in rva])
    new = np.array([derive(T[r['sample_token']]['acc'][:12], T[r['sample_token']]['cur'][:12],
                           T[r['sample_token']]['v0']) for r in rva])
    print(f'A1 / A2 on val n_fut = 12 (n = {len(rva)}); NEW = labels from GT 2 Hz controls, '
          f'CAN = R1.2 10 Hz labels (M1 training targets)')
    print(f'\nA1. GT self-consistency (GT-trajectory-derived vs NEW labels): all 12 '
          f'{(new == new).all(1).mean():.3f} (identical by construction: both are the same '
          f'derivation of the same controls)')
    ok = new == can
    print(f'    NEW vs CAN labels: all 12 {ok.all(1).mean():.3f}; lon 6/6 {ok[:, :6].all(1).mean():.3f}; '
          f'lat 6/6 {ok[:, 6:].all(1).mean():.3f}; per-slot lon {ok[:, :6].mean():.3f} lat {ok[:, 6:].mean():.3f}')
    G = {}
    for tag in tags:
        dv = load(tag, 'val')['data']
        gen = np.array([dv[r['sample_token']]['meta_gen'] for r in rva])
        der = np.array([derive(a, k, T[r['sample_token']]['v0'])
                        for (a, k), r in zip(controls(load(tag, 'val'), rva, Q.run(tag)['tau']), rva)])
        G[tag] = gen
        print(f'\n  {tag}: generated words and predicted trajectory (hybrid decode, tau {Q.run(tag)["tau"]})')
        for lab, y in (('NEW', new), ('CAN (side)', can)):
            print(slot_line(f'words vs {lab} labels, lon', y[:, :6], gen[:, :6]))
            print(slot_line(f'words vs {lab} labels, lat', y[:, 6:], gen[:, 6:]))
        c = der == gen
        print(f'    CONSISTENCY (pred-traj-derived == generated): all 12 {c.all(1).mean():.3f}; lon 6/6 '
              f'{c[:, :6].all(1).mean():.3f}; lat 6/6 {c[:, 6:].all(1).mean():.3f}; slot mean lon {c[:, :6].mean():.3f} '
              f'lat {c[:, 6:].mean():.3f}  (label-free; same with either label set)')
        for lab, y in (('NEW', new), ('CAN', can)):
            d = der == y
            print(f'    pred-traj-derived vs {lab} labels: all 12 {d.all(1).mean():.3f}; slot mean lon '
                  f'{d[:, :6].mean():.3f} lat {d[:, 6:].mean():.3f}')
    # A2
    tr = records.build('train', ['cmd'], 0.0)
    trl = [tuple(sum(FM.meta_labels(Mt, t['sample_token'], st), [])) for t in tr]
    top, ntop = collections.Counter(trl).most_common(1)[0]
    maj = tuple([4] * 6 + [6] * 6)
    print(f'\nA2. COLLAPSE. per-slot class shares (val n_fut = 12); classes lon {LON}, lat {LAT}')
    for tag, gen in G.items():
        for half, off, names in (('lon', 0, LON), ('lat', 6, LAT)):
            print(f'  {tag} {half}:')
            for s in range(6):
                g = np.bincount(gen[:, off + s], minlength=7) / len(gen)
                y = np.bincount(can[:, off + s], minlength=7) / len(can)
                n = np.bincount(new[:, off + s], minlength=7) / len(new)
                print(f'    t+{s + 1}s gen ' + ' '.join(f'{x:.3f}' for x in g) + ' | CAN ' + ' '.join(f'{x:.3f}' for x in y)
                      + ' | NEW ' + ' '.join(f'{x:.3f}' for x in n))
        print(f'  {tag}: distinct generated 12-word sequences {len(set(map(tuple, gen)))} (CAN labels '
              f'{len(set(map(tuple, can)))}, NEW {len(set(map(tuple, new)))})')
        for nm, seq in (('majority (maintain x6, straight x6)', maj), ('most frequent train sequence', top)):
            print(f'  {tag}: share == {nm}: generated {(gen == np.array(seq)).all(1).mean():.3f} | CAN '
                  f'{(can == np.array(seq)).all(1).mean():.3f} | NEW {(new == np.array(seq)).all(1).mean():.3f}')
    print(f'  most frequent train sequence: lon {[LON[c] for c in top[:6]]} lat {[LAT[c] for c in top[6:]]} '
          f'({ntop} of {len(trl)} train)')


if __name__ == '__main__':
    main(sys.argv[1:] or ['M1'])
