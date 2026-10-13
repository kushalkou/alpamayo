"""fig5: CoC v3 seed 42, generated decision recall by GT class (mode i). Source: R13_REPORT.md A2
(r13_partA.txt confusion matrices)."""
from _common import plt, save, C

lon = [('Set speed tracking', 2928, 0.954), ('Lead obstacle following', 1304, 0.103),
       ('Stop for static constraints', 696, 0.707), ('Yield (agent right-of-way)', 97, 0.000),
       ('Speed adaptation (road events)', 94, 0.000)]
lat = [('Lane keeping & centering', 3940, 0.926), ('None', 608, 0.832), ('Turn left', 209, 0.522),
       ('Turn right', 237, 0.561), ('Lane change left', 51, 0.000), ('Lane change right', 68, 0.000),
       ('In-lane nudge (L + R)', 6, 0.000)]
fig, axs = plt.subplots(1, 2, figsize=(16, 7))
for ax, rows, t in ((axs[0], lon, 'longitudinal decision'), (axs[1], lat, 'lateral decision')):
    y = range(len(rows))[::-1]
    ax.barh(list(y), [r[2] for r in rows], color=C['coc'])
    for yi, r in zip(y, rows):
        ax.text(r[2] + 0.02, yi, f'{r[2]:.3f}  (n {r[1]:,})', va='center', fontsize=11)
    ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, 1.25)
    ax.set_xlabel('recall (generated decision = GT)'); ax.set_title(t)
fig.suptitle('CoC v3 seed 42: generated decision recall by GT class (val 5,119); decisions correct (lon and lat) 0.584')
fig.tight_layout(); save(fig, 'fig5_coc_recall')
