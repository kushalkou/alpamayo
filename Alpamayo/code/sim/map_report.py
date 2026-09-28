"""sim/map_report.py -- pilot table, full map CSV + phase map PNG, sensitivity summary.

Reads results/sim/{map_rows,sens_rows}.json written by gates.py (row caches).
Writes Alpamayo/viz/sim_map.csv and Alpamayo/viz/sim_phase_map.png.
Cell winner over 3 seeds = majority of per-seed winners (rule in gates.py); the count
k/3 is printed. Model is NON-AUTOREGRESSIVE (simplification).
"""
import sys, json, csv, collections
import numpy as np
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/sim')
import model as M
from gates import PS, RHOS, table

VIZ = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/viz'
COLORS = {'MODEL': '#2a78d6', 'BLEND': '#eb6834', 'CV': '#1baf7a'}   # palette slots 1-3


def majority(rows):
    c = collections.Counter(r['winner'] for r in rows)
    w, k = c.most_common(1)[0]
    return w, k


def main():
    rows = json.load(open(f'{M.SIMRES}/map_rows.json'))
    # ---- pilot = seed 0 on the pilot subgrid ----
    pil = [r for r in rows if r['seed'] == 0 and r['p'] in (0, .2, .5) and r['rho'] in (0, .5, 1)]
    pil.sort(key=lambda r: (r['p'], r['rho']))
    print('PILOT (seed 0, p in {0,.2,.5} x rho in {0,.5,1})')
    print(table(pil))

    # ---- CSV: every (cell, seed) row ----
    cols = ['p', 'rho', 'seed', 'turn_frac', 'cv', 'v0mean', 'model', 'model_argmax', 'blend',
            'alpha', 'model_cv_d', 'model_cv_lo', 'model_cv_hi', 'blend_cv_d', 'blend_cv_lo',
            'blend_cv_hi', 'model_v0m_d', 'model_v0m_lo', 'model_v0m_hi', 'blend_v0m_d',
            'blend_v0m_lo', 'blend_v0m_hi', 'v0m_cv_d', 'v0m_cv_lo', 'v0m_cv_hi', 'winner',
            'ig_unigram', 'ig_v0unigram']
    with open(f'{VIZ}/sim_map.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r['p'], r['rho'], r['seed'])):
            w.writerow([r['p'], r['rho'], r['seed'], round(r['info']['turn_frac_test'], 4)] +
                       [round(r[k], 4) for k in ('cv', 'v0mean_expect', 'mlp_expect',
                                                 'mlp_argmax', 'blend', 'alpha')] +
                       [round(x, 4) for k in ('mlp_expect_vs_cv', 'blend_vs_cv',
                                              'mlp_expect_vs_v0mean', 'blend_vs_v0mean',
                                              'v0mean_expect_vs_cv') for x in r[k][:3]] +
                       [r['winner'], round(r['ig'], 4), round(r['ig_v0'], 4)])

    # ---- aggregated map ----
    print('\nFULL MAP (3 seeds): majority winner (k/3), mean alpha*, mean ADE@6s')
    print(f'  {"p":>4} {"rho":>4}  {"CV":>6} {"v0mean":>6} {"model":>6} {"blend":>6} '
          f'{"a*":>5}  {"winner":>9}  per-seed')
    agg = {}
    for p in PS:
        for rho in RHOS:
            R = [r for r in rows if r['p'] == p and r['rho'] == rho]
            wn, k = majority(R)
            a = np.mean([r['alpha'] for r in R])
            agg[(p, rho)] = (wn, k, a, np.mean([r['cv'] - r['mlp_expect'] for r in R]))
            print(f'  {p:4.2f} {rho:4.2f}  {np.mean([r["cv"] for r in R]):6.3f} '
                  f'{np.mean([r["v0mean_expect"] for r in R]):6.3f} '
                  f'{np.mean([r["mlp_expect"] for r in R]):6.3f} '
                  f'{np.mean([r["blend"] for r in R]):6.3f} {a:5.2f}  {wn:>6} {k}/3  '
                  + ' '.join(r['winner'][0] for r in sorted(R, key=lambda r: r['seed'])))

    # ---- phase map PNG ----
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=150)
    fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
    for yi, p in enumerate(PS):
        for xi, rho in enumerate(RHOS):
            wn, k, a, g = agg[(p, rho)]
            ax.add_patch(plt.Rectangle((xi - .5 + .02, yi - .5 + .02), .96, .96,
                                       color=COLORS[wn], alpha=1.0 if k == 3 else 0.55, lw=0))
            ax.text(xi, yi + .17, f'{wn} {k}/3', ha='center', va='center', fontsize=7.5,
                    color='#ffffff', weight='bold')
            ax.text(xi, yi - .15, f'a*={a:.2f}', ha='center', va='center', fontsize=7.5,
                    color='#ffffff')
    ax.set_xlim(-.5, len(RHOS) - .5); ax.set_ylim(-.5, len(PS) - .5)
    ax.set_xticks(range(len(RHOS))); ax.set_xticklabels([f'{r:g}' for r in RHOS])
    ax.set_yticks(range(len(PS))); ax.set_yticklabels([f'{p:g}' for p in PS])
    ax.set_xlabel('rho  (fraction of scenes where the future is visible)', color='#3d3d3a')
    ax.set_ylabel('p  (turn fraction)', color='#3d3d3a')
    ax.set_title('Sim winner by cell (majority of 3 seeds; faded = 2/3)\n'
                 'MODEL: model < CV, CI excl. 0;  BLEND: blend < CV, CI excl. 0;  else CV.'
                 '  Non-AR MLP.', fontsize=9, color='#1a1a19')
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0, colors='#3d3d3a')
    ax.legend(handles=[Patch(color=c, label=n) for n, c in COLORS.items()],
              loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f'{VIZ}/sim_phase_map.png', facecolor=fig.get_facecolor())
    print(f'\nwrote {VIZ}/sim_map.csv, {VIZ}/sim_phase_map.png')


if __name__ == '__main__':
    main()
