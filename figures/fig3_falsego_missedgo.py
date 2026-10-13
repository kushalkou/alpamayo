"""fig3: false-go vs missed-go per model, C2 definitions (ADOPTED AFTER R4): FG = pred max disp. 3 s
> 1.0 m on STOPPED (n 679), MG = pred <= 1.0 m on START (n 231). Sources: R7A_REPORT.md C2 (all rows
except CoC), R13_REPORT.md B1 per-seed C2 values for CoC (r13b_coc3seeds.txt; mean of 3 seeds)."""
import numpy as np
from _common import plt, save, C

pts = [('CV', 0.000, 1.000, 'base'), ('Kinematic', 0.028, 0.896, 'base'), ('Ego-MLP + cmd', 0.033, 0.771, 'base'),
       ('Ego-MLP no cmd', 0.034, 0.870, 'base'), ('no-cam VLA', 0.011, 0.817, 'nocam'), ('B1', 0.299, 0.519, 'cam'),
       ('B2 (1)', 0.312, 0.489, 'cam'), ('M1 10 Hz', 0.309, 0.506, 'meta'), ('M1-v2a', 0.112, 0.633, 'meta'),
       ('no-cam M1-v2a (1)', 0.027, 0.784, 'meta'), ('B1 no route (1)', 0.321, 0.411, 'nr'),
       ('M1-v2a no route (1)', 0.234, 0.645, 'nr'),
       ('CoC v3', float(np.mean([0.125, 0.158, 0.150])), float(np.mean([0.645, 0.658, 0.762])), 'coc')]
fig, ax = plt.subplots(figsize=(16, 9))
for lab, fg, mg, c in pts:
    ax.scatter(fg, mg, s=140, color=C[c], edgecolor='k', zorder=3)
    ax.annotate(lab, (fg, mg), xytext=(8, 4), textcoords='offset points', fontsize=12)
ax.set_xlabel('false-go: share of C2 STOPPED samples (n 679) where the model moves > 1 m in 3 s')
ax.set_ylabel('missed-go: share of C2 START samples (n 231)\nwhere the model moves <= 1 m in 3 s')
ax.set_xlim(-0.02, 0.36); ax.set_ylim(0.35, 1.03)
ax.set_title('The standstill trade: cameras cut missed starts but add false starts; 2 Hz words move B1 back toward no-cam\n'
             '(3-seed means unless (1); C2 strata adopted after R4)')
fig.tight_layout(); save(fig, 'fig3_falsego_missedgo')
