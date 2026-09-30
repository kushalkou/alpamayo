"""w1/figs.py -- QUEUE C2: review-deck figures -> Alpamayo/viz/review/{f*.png,f*.pdf}.
White background, slide-size fonts, no em dashes. Each figure is computed from result
files where they exist; numbers copied from committed report tables carry their commit.
Usage: python w1/figs.py [f1 f2 ...]   (default: all whose data exist)
"""
import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')

OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/viz/review'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#8a8a85'
INK, INK2 = '#1a1a19', '#52514e'
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False})


def save(fig, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}', dpi=200, facecolor='white')
    plt.close(fig)
    print('wrote', name)


def v2t(v, s, tok):
    if v == 0.0: return 128
    c = tok.accel_centers if s < 12 else tok.curv_centers
    return int(np.argmin(np.abs(c - v)))


def f1():
    """leak fingerprint: per-slot argmax accuracy, leaky y1_ego vs causal_ego (custom test,
    causal subset n=3358, free-running AR)."""
    import contextlib, io
    from tokenizer import TrajectoryTokenizer
    with contextlib.redirect_stdout(io.StringIO()):
        tok = TrajectoryTokenizer()
    F = pickle.load(open(f'{RES}/dump_test.pkl', 'rb'))
    C = pickle.load(open(f'{RES}/dump_test_causal_ego.pkl', 'rb'))
    idx = sorted(C['data']['causal_ego']); gt = C['ce']['causal_ego']
    acc = {}
    for lab, D, m in (('leaky y1_ego', F, 'y1_ego'), ('causal_ego (leak removed)', C, 'causal_ego')):
        acc[lab] = np.mean([[v2t(D['data'][m][i]['argmax'][s], s, tok) == gt[i]['gt'][s] for s in range(24)]
                            for i in idx], 0)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(24)
    for (lab, a), c in zip(acc.items(), (ORANGE, BLUE)):
        ax.plot(x, a, marker='o', ms=7, lw=2, color=c, label=lab)
    ax.annotate(f'accel slot 1: {acc["leaky y1_ego"][1]:.2f}\n(input = its own target)',
                xy=(1, acc['leaky y1_ego'][1]), xytext=(4, 0.9), fontsize=13, color=INK,
                arrowprops=dict(arrowstyle='->', color=INK2))
    ax.axvline(11.5, color=GREY, lw=1, ls=':')
    ax.text(5.5, 0.03, 'acceleration slots', ha='center', color=INK2)
    ax.text(17.5, 0.03, 'curvature slots', ha='center', color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f'a{i}' for i in range(12)] + [f'k{i}' for i in range(12)], fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel('argmax token accuracy')
    ax.set_title(f'Leak fingerprint: the leaky model gets accel slot 1 right far more often (n={len(idx)})')
    ax.legend(loc='upper right')
    save(fig, 'f1_leak_fingerprint')


def f2():
    """blend gain vs CV before / after removing the leak (custom split test, causal subset
    n=3358). Numbers: OVERNIGHT2_REPORT.md 1c/6/6b (commits 495cb86, 7f0e24e, 2e45c04)."""
    rows = [('kinematic rule\n(no learning)', 0.184, 0.138, 0.228, GREY),
            ('leaky ego-only\n(frozen headline)', 0.164, 0.137, 0.191, ORANGE),
            ('causal ego-only', 0.014, 0.003, 0.024, BLUE),
            ('causal full vision', 0.014, 0.002, 0.025, BLUE),
            ('zero-input model', 0.008, 0.006, 0.010, AQUA)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (lab, g, lo, hi, c) in enumerate(rows):
        ax.bar(i, g, color=c, width=0.6)
        ax.errorbar(i, g, yerr=[[g - lo], [hi - g]], color=INK, capsize=6, lw=1.5)
        ax.text(i, hi + 0.006, f'{g:.3f} m', ha='center', color=INK)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('ADE@6s gain over CV (m), 95% CI')
    ax.set_title('Removing the ego-state leak takes the blend gain from 0.164 m to 0.014 m')
    ax.set_ylim(0, 0.26)
    save(fig, 'f2_before_after_leak')


def f3():
    """our causal baselines (measured, official val n=5,119, 3.5b table, commit 950101a)
    vs literature (reported; planner-verified, GaussianAD table / UniAD paper)."""
    t = [1, 2, 3]
    ours_tem = {'CV (measured)': ([0.376, 0.784, 1.326], GREY, 'o'),
                'kinematic rule (measured)': ([0.278, 0.562, 0.998], AQUA, 's'),
                'Ego-MLP + cmd (measured)': ([0.241, 0.461, 0.777], BLUE, 'D')}
    lit_tem = {'VAD-Base (reported)': ([0.41, 0.70, 1.05], ORANGE, '^')}
    ours_no = {'CV (measured)': ([0.527, 1.448, 2.762], GREY, 'o'),
               'kinematic rule (measured)': ([0.362, 1.049, 2.178], AQUA, 's'),
               'Ego-MLP + cmd (measured)': ([0.324, 0.822, 1.633], BLUE, 'D')}
    lit_no = {'VAD-Base (reported)': ([0.54, 1.15, 1.98], ORANGE, '^'),
              'UniAD (reported)': ([0.48, 0.96, 1.65], VIOLET, 'v')}
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, ours, lit, ttl in ((axs[0], ours_tem, lit_tem, 'L2, averaged over horizon (ST-P3 / VAD)'),
                               (axs[1], ours_no, lit_no, 'L2 at the horizon (UniAD)')):
        for lab, (v, c, mk) in ours.items():
            ax.plot(t, v, marker=mk, ms=8, lw=2, color=c, label=lab)
        for lab, (v, c, mk) in lit.items():
            ax.plot(t, v, marker=mk, ms=9, lw=2, ls='--', color=c, label=lab)
        ax.set_xticks(t); ax.set_xticklabels(['1 s', '2 s', '3 s']); ax.set_ylabel('L2 (m)')
        ax.set_title(ttl); ax.legend(fontsize=11)
    fig.suptitle('Causal ego-status baselines against published planners (nuScenes val)', fontsize=16)
    save(fig, 'f3_baselines_vs_literature')


def f4():
    """Ego-MLP with our causal features vs VAD-converter features, all 5,119 vs excluding
    the 140 first frames; seed values from the P2 tables (commit 762c5af)."""
    d = {('ours', 'all'): [1.636, 1.639, 1.646], ('VAD', 'all'): [1.345, 1.346, 1.344],
         ('ours', 'excl'): [1.309, 1.312, 1.320], ('VAD', 'excl'): [1.344, 1.347, 1.343]}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, (grp, lab) in enumerate((('all', 'all 5,119 samples'), ('excl', 'excluding 140 first frames'))):
        for k, (src, c) in enumerate((('ours', BLUE), ('VAD', ORANGE))):
            v = np.array(d[(src, grp)]); x = j * 3 + k
            ax.bar(x, v.mean(), color=c, width=0.8, label=(f'{src} features' if j == 0 else None))
            ax.scatter([x] * 3, v, color=INK, s=18, zorder=3)
            ax.text(x, v.mean() + 0.03, f'{v.mean():.3f}', ha='center', color=INK)
    ax.set_xticks([0.5, 3.5]); ax.set_xticklabels(['all 5,119 samples', 'excluding 140 first frames'])
    ax.set_ylabel('L2 at 3 s (m), UniAD convention'); ax.set_ylim(0, 1.9)
    ax.set_title("VAD's ego features help only on first frames, where they read the future")
    ax.legend(loc='upper right')
    save(fig, 'f4_vad_converter_effect')


def f5(tags=('overfit256_latest',)):
    """slot-0 memorisation on the 256 overfit samples: AR per-slot accuracy, and ADE when
    slot 0 is right vs wrong."""
    from targets import rollout as roll12
    T = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_targets.pkl', 'rb'))['targets']
    tag = [t for t in tags if os.path.exists(f'{RES}/w1_dump_{t}_train_f0.0.pkl')][-1]
    D = pickle.load(open(f'{RES}/w1_dump_{tag}_train_f0.0.pkl', 'rb'))
    ok, ade, s0 = [], [], []
    for st in D['order']:
        d = D['data'][st]; v = np.array(d['argmax'])
        ok.append([p == g for p, g in zip(d['tok'], d['gt_tok'])])
        ade.append(np.linalg.norm(roll12(v[:12], v[12:], T[st]['v0']) - T[st]['P'][:12], axis=1).mean())
        s0.append(d['tok'][0] == d['gt_tok'][0])
    ok, ade, s0 = np.array(ok), np.array(ade), np.array(s0)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2.2, 1]})
    ax = axs[0]
    ax.bar(np.arange(24), ok.mean(0), color=BLUE, width=0.75)
    ax.set_xticks(range(24)); ax.set_xticklabels([f'a{i}' for i in range(12)] + [f'k{i}' for i in range(12)], fontsize=10)
    ax.set_ylim(0, 1); ax.set_ylabel('free-running token accuracy')
    ax.set_title(f'Every slot is only as good as slot 0 ({s0.mean():.0%})')
    ax = axs[1]
    vals = [ade[s0], ade[~s0]]
    ax.bar([0, 1], [np.median(v) for v in vals], color=[AQUA, ORANGE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, np.median(v) + 0.1, f'median {np.median(v):.2f} m\nn={len(v)}', ha='center', color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['slot 0 right', 'slot 0 wrong'])
    ax.set_ylabel('ADE@6s (m)'); ax.set_ylim(0, max(np.median(vals[1]) * 1.5, 1))
    ax.set_title('One wrong first token')
    fig.suptitle(f'Overfit test on 256 train samples ({tag}, epoch {D["ckpt_epoch"]})', fontsize=16)
    save(fig, 'f5_slot0_memorisation')


def f6():
    """mini information ladder (C1): seed-mean ADE@6s, all 5,119; privileged bars hatched."""
    L = pickle.load(open(f'{RES}/w1_ladder.pkl', 'rb'))
    PER = L['PER']; seeds = (42, 123, 2024)
    rows = [('CV', PER['CV'], False, GREY), ('kinematic\nrule', PER['KIN'], False, GREY),
            ('L1 ego', [PER[('L1', s, 0.0)] for s in seeds], False, BLUE),
            ('L2b cmd\nonly', [PER[('L2b', s, 0.0)] for s in seeds], True, BLUE),
            ('L2 ego\n+ cmd', [PER[('L2', s, 0.0)] for s in seeds], True, BLUE),
            ('oracle\nkinematic', PER[('OKIN', 0.0)], True, AQUA),
            ('L5 ego\n+ oracle', [PER[('L5', s, 0.0)] for s in seeds], True, ORANGE)]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (lab, p, priv, c) in enumerate(rows):
        v = [np.nanmean(q['ade']) for q in (p if isinstance(p, list) else [p])]
        ax.bar(i, np.mean(v), color=c, hatch='//' if priv else None, edgecolor='white', width=0.7)
        if len(v) > 1: ax.scatter([i] * len(v), v, color=INK, s=16, zorder=3)
        ax.text(i, np.mean(v) + 0.25, f'{np.mean(v):.2f}', ha='center', color=INK)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel('ADE@6s (m), official val'); ax.set_title(
        'Mini information ladder on a small Ego-MLP (hatched = privileged input)')
    save(fig, 'f6_mini_ladder')


def f7():
    """flip-rate curve: L5-noise Ego-MLP (C1) and oracle-kinematic; A2 if its dumps exist."""
    L = pickle.load(open(f'{RES}/w1_ladder.pkl', 'rb'))
    PER = L['PER']; F = (0.0, 0.1, 0.2, 0.4); seeds = (42, 123, 2024)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = [np.mean([np.nanmean(PER[('L5n', s, f)]['ade']) for s in seeds]) for f in F]
    ax.plot([100 * f for f in F], y, marker='D', ms=8, lw=2, color=BLUE, label='Ego-MLP + oracle (trained at 10% flips)')
    y = [np.nanmean(PER[('OKIN', f)]['ade']) for f in F]
    ax.plot([100 * f for f in F], y, marker='s', ms=8, lw=2, color=AQUA, label='oracle-kinematic rule')
    if os.path.exists(f'{RES}/w1_rung3.pkl'):             # C4c / C5
        R3 = pickle.load(open(f'{RES}/w1_rung3.pkl', 'rb'))
        y0 = np.mean([np.nanmean(PER[('L5n', s, 0.0)]['ade']) for s in seeds])
        yc = [y0] + [np.mean([np.nanmean(R3['PER'][('L5n', s, f'conf{e}')]['ade']) for s in seeds])
                     for e in (10, 20, 40)]
        ax.plot([0, 10, 20, 40], yc, marker='D', ms=8, lw=2, ls='--', color=VIOLET,
                label='Ego-MLP + oracle, confusion-shaped noise (lon only)')
        acc = np.mean([(R3['PRED'][s] == R3['yva']).mean() for s in seeds])
        yp = np.mean([np.nanmean(R3['PER'][('L5n', s, 'pred')]['ade']) for s in seeds])
        ax.scatter([100 * (1 - acc)], [yp], s=160, marker='*', color=ORANGE, zorder=5,
                   label=f'Ego-MLP + PREDICTED intent (classifier acc {acc:.0%})')
    y2 = np.mean([np.nanmean(PER[('L2', s, 0.0)]['ade']) for s in seeds])
    ax.axhline(y2, color=BLUE, ls=':', lw=1.2)
    ax.text(1, y2 - 0.13, 'Ego-MLP + cmd (no intent)', color=BLUE, ha='left', fontsize=12)
    a2 = [f'{RES}/w1_dump_A2_val_f{f}.pkl' for f in F]
    if all(os.path.exists(p) for p in a2) and os.path.exists(f'{RES}/w1_a2_flip_ade.pkl'):
        v = pickle.load(open(f'{RES}/w1_a2_flip_ade.pkl', 'rb'))
        ax.plot([100 * f for f in F], [v[f] for f in F], marker='o', ms=8, lw=2, color=YELLOW, label='VLA A2')
    ax.axhline(np.nanmean(PER['CV']['ade']), color=GREY, ls=':', lw=1.5)
    ax.text(40, np.nanmean(PER['CV']['ade']) + 0.03, 'CV', color=INK2, ha='right')
    ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xlabel('meta-action label error rate at test (%)')
    ax.set_ylabel('ADE@6s (m), official val'); ax.legend()
    ax.set_title('How much each model leans on the oracle meta-action')
    save(fig, 'f7_flip_rate')


if __name__ == '__main__':
    names = sys.argv[1:] or ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
    for n in names:
        try:
            globals()[n]()
        except FileNotFoundError as e:
            print(f'skip {n}: missing {e.filename}')
