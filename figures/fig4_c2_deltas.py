"""fig4: C2 strata paired deltas, L2@3s, 3-seed means, 95% scene-bootstrap CI. Sources: R7A_REPORT.md
C2 (M1-v2a - B1, M1-v2a - no-cam VLA), R8_REPORT.md Part B (M1-v2a no route - M1-v2a), R13_REPORT.md
B1 (CoC - M1-v2a)."""
import numpy as np
from _common import plt, save, C

D = {'M1-v2a - B1': ([-0.528, 0.358, -0.117], [(-0.984, -0.072), (0.221, 0.515), (-0.170, -0.065)], 'meta'),
     'M1-v2a - no-cam VLA': ([0.490, -0.590, -0.062], [(0.218, 0.798), (-0.822, -0.386), (-0.114, -0.008)], 'nocam'),
     'M1-v2a no route - M1-v2a': ([0.806, 0.022, 0.039], [(0.143, 1.507), (-0.224, 0.285), (0.011, 0.068)], 'nr'),
     'CoC v3 - M1-v2a': ([0.036, 0.129, 0.144], [(-0.410, 0.626), (-0.051, 0.310), (0.106, 0.182)], 'coc')}
S = ['STOPPED (n 679)', 'START (n 231)', 'MOVING (n 4,209)']
fig, ax = plt.subplots(figsize=(16, 8)); w = 0.2
for k, (lab, (m, ci, c)) in enumerate(D.items()):
    x = np.arange(3) + (k - 1.5) * w
    err = np.array([[mm - lo for mm, (lo, hi) in zip(m, ci)], [hi - mm for mm, (lo, hi) in zip(m, ci)]])
    ax.bar(x, m, w * 0.9, color=C[c], label=lab, yerr=err, capsize=4)
ax.axhline(0, color='k', lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(S)
ax.set_ylabel('paired difference in L2 @ 3 s (m); bars = 95% CI')
ax.set_title('C2 strata (rule-defined, adopted after R4): where each change helps (< 0) or hurts (> 0)')
ax.legend(); fig.tight_layout(); save(fig, 'fig4_c2_deltas')
