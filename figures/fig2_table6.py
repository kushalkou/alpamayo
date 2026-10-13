"""fig2: Table 6 analog, L2@3s NoAvg, 3-seed mean (sd). Sources: R9_REPORT.md A1 (traj, +meta,
with / without route), R13_REPORT.md B1 (+CoC with route; mean / sd of reported seeds), paired
deltas R9 A1 and R13 GC1-3. CoC without route: not run."""
import numpy as np
from _common import plt, save, C

coc = [2.048, 2.064, 2.099]
grp = {'with route [P]': [(2.091, .064), (1.942, .056), (np.mean(coc), np.std(coc, ddof=1))],
       'without route': [(2.144, .028), (2.082, .012), (None, None)]}
names = ['traj-only (B1)', '+meta, 2 Hz words (M1-v2a)', '+CoC v3 (CoC)']
cols = [C['cam'], C['meta'], C['coc']]
fig, ax = plt.subplots(figsize=(16, 8)); w = 0.26
for gi, (g, vals) in enumerate(grp.items()):
    for k, (m, sd) in enumerate(vals):
        x = gi + (k - 1) * w
        if m is None:
            ax.text(x, 1.86, 'not run', ha='center', rotation=90, color='#666'); continue
        ax.bar(x, m, w * 0.9, yerr=sd, capsize=5, color=cols[k], label=names[k] if gi == 0 else None)
        ax.text(x, m + sd + 0.006, f'{m:.3f}', ha='center', fontsize=11)
ax.set_xticks([0, 1]); ax.set_xticklabels(list(grp)); ax.set_ylim(1.8, 2.25)
ax.set_ylabel('L2 @ 3 s NoAvg (m), 3-seed mean, bar = sd')
txt = ('paired scene bootstrap, 3-seed means (95% CI):\n'
       '+meta - traj, with route   -0.150 [-0.232, -0.073]\n'
       '+meta - traj, without route -0.063 [-0.114, -0.008]\n'
       '+CoC - +meta, with route   +0.129 [+0.059, +0.215]\n'
       'no route - route: traj +0.053 [+0.012, +0.091]; +meta +0.140 [+0.043, +0.253]')
ax.text(1.35, 2.22, txt, va='top', fontsize=11, family='monospace')
ax.set_xlim(-0.5, 2.6)
ax.set_title('Table 6 analog (nuScenes val, L2@3s); AR1 Table 6 uses minADE_6 on its CoC test set: compare direction only')
ax.legend(loc='upper left'); fig.tight_layout(); save(fig, 'fig2_table6')
