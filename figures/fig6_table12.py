"""fig6: Table 12 analog, M1-v2a token decoder vs flow expert (mean of 6), 3 seeds. Source:
R10_REPORT.md B2 (3-seed rows; r10_b2.txt)."""
import numpy as np
from _common import plt, save, C

S = ['C2 STOPPED L2@3s', 'C2 START L2@3s', 'C2 MOVING L2@3s', 'ADE@6s (n_fut 12)']
tok = [0.606, 9.273, 1.755, 2.891]; exp = [0.906, 8.682, 1.611, 2.732]
fig, axs = plt.subplots(1, 4, figsize=(16, 6))
for ax, s, a, b in zip(axs, S, tok, exp):
    ax.bar([0, 1], [a, b], color=[C['meta'], C['expert']])
    for x, v in ((0, a), (1, b)):
        ax.text(x, v * 1.01, f'{v:.3f}', ha='center', fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['token decoder', 'expert mean of 6']); ax.set_title(s)
    ax.set_ylim(0, max(a, b) * 1.15)
fig.suptitle('Table 12 analog, M1-v2a, 3 seeds: expert - token ADE@6s -0.159 [-0.217, -0.099]; '
             'STOPPED L2@3s +0.300 [+0.141, +0.481]; hold 0.871 vs 0.863')
fig.tight_layout(); save(fig, 'fig6_table12')
