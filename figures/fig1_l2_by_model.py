"""fig1: L2@3s NoAvg (official val, all 5,119) by model. Values copied from the reports:
CV, KIN, Ego-MLP + cmd: REVIEW_RESULTS.md T6; Ego-MLP VAD-converter features: REVIEW_RESULTS.md T4
(leaky inputs); Ego-MLP-NR: R7A_REPORT.md; no-cam VLA, B1, B2, M1: PLANNER_HANDOFF.md section 2;
M1-v2a 3 seeds and no-cam M1-v2a: R6_REPORT.md; B1-NR, M1-v2a-NR 3 seeds: R8_REPORT.md; CoC
(v3) per seed: R13_REPORT.md B1 (mean / sd of the three reported values); CoC-AUX: R13 B2;
GT CoC forced [ORACLE] per seed: R13 B1."""
import numpy as np
from _common import plt, save, C

rows = [  # label, mean, sd (None = single run), colour, note
    ('CV (no learning)', 2.762, None, 'base', ''), ('Kinematic rule', 2.178, None, 'base', ''),
    ('Ego-MLP + cmd [P] (3)', 1.637, None, 'base', ''), ('Ego-MLP no cmd (3)', 1.740, None, 'base', ''),
    ('Ego-MLP + cmd, VAD-converter feats (3)', 1.345, None, 'base', 'leaky inputs'),
    ('VLA no camera (3)', 1.954, 0.054, 'nocam', ''), ('VLA B1 3 cams (3)', 2.091, 0.064, 'cam', ''),
    ('VLA B2 3 cams x 4 kf (1)', 2.130, None, 'cam', ''), ('M1 10 Hz words (3)', 2.178, 0.049, 'meta', ''),
    ('M1-v2a 2 Hz words (3)', 1.942, 0.056, 'meta', ''), ('no-cam M1-v2a (1)', 1.860, None, 'meta', ''),
    ('B1 no route (3)', 2.144, 0.028, 'nr', ''), ('M1-v2a no route (3)', 2.082, 0.012, 'nr', ''),
    ('CoC v3 (3)', float(np.mean([2.048, 2.064, 2.099])), float(np.std([2.048, 2.064, 2.099], ddof=1)), 'coc', ''),
    ('CoC-AUX, reasoning off (1)', 1.937, None, 'coc', ''),
    ('CoC v3, GT CoC forced [ORACLE] (3)', float(np.mean([1.827, 1.797, 1.833])), float(np.std([1.827, 1.797, 1.833], ddof=1)), 'oracle', 'oracle'),
]
fig, ax = plt.subplots(figsize=(16, 9))
y = np.arange(len(rows))[::-1]
for yi, (lab, m, sd, c, note) in zip(y, rows):
    ax.barh(yi, m, color=C[c], xerr=sd if sd else None, capsize=4, alpha=0.9 if not note else 0.5,
            hatch='//' if note else None)
    ax.text(m + (sd or 0) + 0.03, yi, f'{m:.3f}' + (f' (sd {sd:.3f})' if sd else '') + (f'  [{note}]' if note else ''),
            va='center', fontsize=11)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel('L2 @ 3 s, NoAvg (m), official nuScenes val, all 5,119 samples; (n) = seeds; bars = sd over seeds')
ax.set_xlim(0, 3.2); ax.set_title('L2@3s by model (lower is better); [P] privileged VAD command in every VLA row unless "no route"')
fig.tight_layout(); save(fig, 'fig1_l2_by_model')
