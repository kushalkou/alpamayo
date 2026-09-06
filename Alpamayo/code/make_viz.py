"""make_viz.py — GATE 3 item 8: slide figures for Wednesday.

Palette: the documented reference instance, categorical slots 1-3
(blue #2a78d6, orange #eb6834, aqua #1baf7a) -- the palette doc records these
three as validated on the ALL-PAIRS gate in both modes (worst CVD dE 9.2,
normal-vision 24.0), which is the gate that applies when every series can sit
beside every other. Aqua is below 3:1 on the light surface, so the relief rule
applies: every series carries a visible direct label, never colour alone.
Light surface only (these are slide PNGs).

Panel (d) uses the project's established BEV convention instead:
GT green, model red, CV grey dashed, blended blue.
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shrink_lib import load, traj, ade, CURV_T, N

VIZ = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/viz'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'      # categorical slots 1-3
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e3e2df'
ALPHAS = np.round(np.arange(0, 1.0001, 0.05), 2)
MODELS = [('y1_full', 'full vision+ego', S1),
          ('y1_ego', 'ego-only', S2),
          ('zeroboth_jul12', 'zero-input', S3)]
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.edgecolor': GRID, 'font.size': 11,
    'axes.titlesize': 13, 'axes.spines.top': False, 'axes.spines.right': False,
})


def stack(T, idx): return np.stack([T[i] for i in idx])
def ade_arr(P, G): return np.linalg.norm(P - G, axis=2).mean(1)


def main():
    os.makedirs(VIZ, exist_ok=True)
    Dv, mv, iv, Gv, CVv = load('val')
    Dt, mt, it, Gt, CVt = load('test')
    rng = np.random.RandomState(0)

    curves, astars, cis, Ttest = {}, {}, {}, {}
    for key, lab, col in MODELS:
        Mv = stack({i: traj(Dv, mv, key, i, 'expect') for i in iv}, iv)
        Cv = stack(CVv, iv); GV = stack(Gv, iv)
        per = np.stack([ade_arr(a*Mv + (1-a)*Cv, GV) for a in ALPHAS])   # [21,n]
        curves[key] = per.mean(1)
        astars[key] = float(ALPHAS[int(per.mean(1).argmin())])
        bs = [ALPHAS[int(per[:, rng.randint(0, len(iv), len(iv))].mean(1).argmin())]
              for _ in range(2000)]
        cis[key] = tuple(np.percentile(bs, [2.5, 97.5]))
        Ttest[key] = {i: traj(Dt, mt, key, i, 'expect') for i in it}

    # ---- (a) val alpha curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    cvv = float(np.mean([ade(CVv[i], Gv[i]) for i in iv]))
    ax.axhline(cvv, color=INK2, ls=(0, (5, 4)), lw=1.4, zorder=2)
    for key, lab, col in MODELS:
        a_s = astars[key]; lo, hi = cis[key]
        ax.axvspan(lo, hi, color=col, alpha=0.10, lw=0, zorder=1)
        ax.plot(ALPHAS, curves[key], color=col, lw=2, zorder=3)
        j = int(np.argmin(np.abs(ALPHAS - a_s)))
        ax.plot([a_s], [curves[key][j]], 'o', ms=9, color=col,
                markeredgecolor=SURF, markeredgewidth=2, zorder=4)
        # direct label at the RIGHT end, where the curves are well separated
        ax.annotate(lab, (ALPHAS[-1], curves[key][-1]), textcoords='offset points',
                    xytext=(6, -3), color=col, fontsize=10.5, zorder=5, va='center')
    ax.annotate('CV baseline', (0.0, cvv), textcoords='offset points',
                xytext=(2, 6), color=INK2, fontsize=10)
    # alpha* summary block, lower right, no collisions with the curves
    lines = [f"alpha*  (95% CI)"] + [
        f"{lab:<16} {astars[k]:.2f}  [{cis[k][0]:.2f}, {cis[k][1]:.2f}]"
        for k, lab, _ in MODELS]
    ax.text(0.015, 0.975, "\n".join(lines), transform=ax.transAxes, fontsize=9.5,
            family='monospace', color=INK, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=SURF, edgecolor=GRID))
    ax.set_xlim(-0.02, 1.30)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('alpha   (0 = pure CV,  1 = pure model)')
    ax.set_ylabel('VAL ADE@6s (m)')
    ax.set_title('Shrinkage curve: blending the model toward constant velocity\n'
                 'alpha fit on VAL (n=3,572); shaded band = bootstrap 95% CI on alpha*',
                 loc='left')
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f'{VIZ}/a_alpha_curve.png'); plt.close(fig)

    # ---- (b) CDF on TURNING -------------------------------------------------
    turn = [i for i in it if mt[i]['maxcurv'] > CURV_T]
    a_s = astars['y1_full']
    ser = [('CV', np.array([ade(CVt[i], Gt[i]) for i in turn]), INK2, (0, (5, 4))),
           ('model (V1s)', np.array([ade(Ttest['y1_full'][i], Gt[i]) for i in turn]), S1, '-'),
           (f'blended (alpha={a_s:.2f})',
            np.array([ade(a_s*Ttest['y1_full'][i] + (1-a_s)*CVt[i], Gt[i]) for i in turn]),
            S2, '-')]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    xmax = max(np.percentile(v, 99) for _, v, _, _ in ser)
    # stagger the direct labels so they never collide (relief rule: never colour alone)
    anchors = [(0.22, (8, -14)), (0.88, (8, -2)), (0.55, (10, -14))]
    for (lab, v, col, ls), (q, off) in zip(ser, anchors):
        xs = np.sort(v); ys = np.arange(1, len(xs)+1) / len(xs)
        ax.plot(xs, ys, color=col, lw=2, ls=ls, zorder=3)
        ax.annotate(lab, (np.percentile(v, 100*q), q), color=col, fontsize=10.5,
                    textcoords='offset points', xytext=off, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=SURF,
                              edgecolor='none', alpha=0.85))
    ax.axhline(0.5, color=GRID, lw=1, zorder=1)
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1)
    ax.set_xlabel('per-sample ADE@6s (m)   -- lower is better')
    ax.set_ylabel('fraction of samples <= x')
    ax.set_title('TURNING subset (n=%d): the model wins at the median and loses in the tail\n'
                 'curves left of CV = better; crossing right of CV = worse' % len(turn),
                 loc='left')
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f'{VIZ}/b_turning_cdf.png'); plt.close(fig)

    # ---- (c) decode fix vs vision effect ------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
    labels = ['Decode fix\n(argmax -> expectation)', 'Vision effect\n(full vs ego-only)']
    vals = [0.333, 0.045]
    los = [0.295, -0.048]; his = [0.371, 0.137]
    cols = [S1, S2]
    ypos = [1, 0]
    for y, v, lo, hi, c, lab in zip(ypos, vals, los, his, cols, labels):
        ax.barh(y, v, height=0.42, color=c, zorder=3)
        ax.plot([lo, hi], [y, y], color=INK, lw=2, zorder=4)
        ax.plot([lo, lo], [y-0.09, y+0.09], color=INK, lw=2, zorder=4)
        ax.plot([hi, hi], [y-0.09, y+0.09], color=INK, lw=2, zorder=4)
    ax.text(0.392, 1, '-0.333 m,  p < 1e-4', va='center', color=INK, fontsize=11)
    ax.text(0.158, 0, '-0.045 m,  p = 0.35  (not significant)', va='center',
            color=INK, fontsize=11)
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(-0.1, 0.78)
    ax.set_xlabel('ADE@6s improvement (m), full test set n=3,614   [95% CI]')
    ax.set_title('The estimator is worth more than the cameras', loc='left')
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f'{VIZ}/c_decode_vs_vision.png'); plt.close(fig)

    # ---- (d) qualitative BEV ------------------------------------------------
    em = np.array([ade(Ttest['y1_full'][i], Gt[i]) for i in turn])
    ec = np.array([ade(CVt[i], Gt[i]) for i in turn])
    good = turn[int(np.argmax((ec - em) * (em < np.percentile(em, 50))))]
    bad = turn[int(np.argmax(em))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150)
    for ax, i, ttl in ((axes[0], good, 'model BEATS CV (median regime)'),
                       (axes[1], bad, 'model BLOWS UP (p99 tail)')):
        g = Gt[i]; mm = Ttest['y1_full'][i]; cc = CVt[i]
        bl = a_s*mm + (1-a_s)*cc
        ax.plot(g[:, 0], g[:, 1], color='#008300', lw=2.6, marker='o', ms=4, zorder=5)
        ax.plot(mm[:, 0], mm[:, 1], color='#e34948', lw=2, marker='o', ms=3.5, zorder=4)
        ax.plot(cc[:, 0], cc[:, 1], color='#8a8983', lw=1.8, ls=(0, (5, 4)), zorder=3)
        ax.plot(bl[:, 0], bl[:, 1], color=S1, lw=2, zorder=4)
        ax.plot([0], [0], marker='s', ms=8, color=INK, zorder=6)
        ax.set_title(f"{ttl}\nmodel {ade(mm,g):.2f} m | CV {ade(cc,g):.2f} m | "
                     f"blend {ade(bl,g):.2f} m", loc='left', fontsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    handles = [Line2D([], [], color='#008300', lw=2.6, label='ground truth'),
               Line2D([], [], color='#e34948', lw=2, label='model (V1s)'),
               Line2D([], [], color='#8a8983', lw=1.8, ls=(0, (5, 4)), label='constant velocity'),
               Line2D([], [], color=S1, lw=2, label=f'blended (alpha={a_s:.2f})'),
               Line2D([], [], color=INK, marker='s', ls='', ms=8, label='ego at t=0')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.suptitle('Turning scenes: 6-second futures', x=0.011, ha='left', fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(f'{VIZ}/d_bev_turning.png'); plt.close(fig)

    print("alpha*:", {k: (astars[k], cis[k]) for k, _, _ in MODELS})
    print("wrote:", sorted(os.listdir(VIZ)))


if __name__ == '__main__':
    main()
