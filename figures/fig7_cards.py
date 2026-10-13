"""fig7: one qualitative card per arm on ONE val sample chosen by rule (not by hand):
  among val samples in C2 MOVING (v0_can > 0.2 m/s) with n_fut = 12, the one whose M1-v2a s42
  L2@3s (holdout tau, per-sample dump) is closest to the median of that set; ties -> smaller
  sample token. Arms (seed 42 runs, existing dumps, holdout tau): no-cam VLA (A3), B1, M1 10 Hz,
  M1-v2a, M1-v2a + flow expert (mean of 6), CoC v3. Shown: CAM_FRONT_LEFT / FRONT / FRONT_RIGHT at
  t0; per arm a BEV (LiDAR frame, x right, y forward; GT future 6 s, prediction, annotated agents
  at t0) and the generated words / CoC text. No metric is recomputed beyond reading the dumps."""
import sys, json, pickle, textwrap
import numpy as np
from PIL import Image
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/ar1')
import evalw1 as E, q2_traj as Q
from gate_a import load, preds
from meta_actions import LON, LAT
import coc_template as CT
from _common import plt, save

NUS = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
RES = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'
LIDAR_FWD = 0.985793          # LiDAR ahead of the rear axle (ego_pose origin)


def main():
    C = Q.ctx(); rva = C['rva']; W, V0, T = E.data(); tokv = [r['sample_token'] for r in rva]
    v0 = np.array([V0[t]['v0_can'] for t in tokv]); n12 = np.array([r['n_fut'] == 12 for r in rva])
    l2 = Q.run('M1v2a')['l2'][:, 5]
    cand = np.where((v0 > 0.2) & n12)[0]
    med = np.median(l2[cand])
    i = min(cand, key=lambda k: (abs(l2[k] - med), tokv[k])); st = tokv[i]; rec = rva[i]
    print(f'rule pick: {st} ({rec["scene_name"]}), M1-v2a s42 L2@3s {l2[i]:.3f}, median {med:.3f} over {len(cand)}')
    G = np.asarray(T[st]['P'])[:12]
    arms = []
    for lab, tag in (('no-cam VLA (A3)', 'A3'), ('B1 (3 cams)', 'B1'), ('M1, 10 Hz words', 'M1'),
                     ('M1-v2a, 2 Hz words', 'M1v2a'), ('CoC v3', 'COC3')):
        d = load(tag, 'val'); P = preds(d, [rec], 'hybrid', Q.run(tag)['tau'])[0]; x = d['data'][st]
        if 'coc_gen' in x and x['coc_gen']:
            txt = x['coc_gen']
        elif 'meta_gen' in x:
            w = x['meta_gen']; txt = 'words: ' + ' '.join(LON[c] for c in w[:6]) + ' | ' + ' '.join(LAT[c] for c in w[6:])
        else:
            txt = '(trajectory tokens only)'
        arms.append((lab, P, txt))
    X = pickle.load(open(f'{RES}/ar1_expert_M1v2a.pkl', 'rb'))['val']
    k = X['tokens'].index(st); arms.insert(4, ('M1-v2a + flow expert (mean of 6)', X['traj'][k].mean(0),
                                               'same words as M1-v2a; trajectory from the expert'))
    CT._init_globals(); Gl = CT._G; r = Gl['W'][st]; e = CT.ego_pose(r)
    mn = Gl['loc'][r['scene_name']]; M = CT.Map(mn)
    objs = CT.objects(r, Gl['inst'], Gl['S'], Gl['A'], Gl['by_s'], M, set())
    J = lambda n: json.load(open(f'{NUS}/v1.0-trainval/{n}.json'))
    sd = {(d['sample_token'], d['filename'].split('/')[1]): d['filename'] for d in J('sample_data') if d['is_key_frame']}
    fig = plt.figure(figsize=(16, 15))
    for j, ch in enumerate(('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT')):
        ax = fig.add_axes([0.01 + j * 0.33, 0.775, 0.32, 0.175]); ax.imshow(Image.open(f'{NUS}/{sd[(st, ch)]}'))
        ax.set_title(f'{ch} at t0', fontsize=11); ax.axis('off')
    for a, (lab, P, txt) in enumerate(arms):
        row, col = divmod(a, 3)
        ax = fig.add_axes([0.04 + col * 0.33, 0.47 - row * 0.40, 0.26, 0.25])
        for o in objs:
            if np.hypot(o['x'], o['y']) < 45:
                lx, ly = -o['y'], o['x'] - LIDAR_FWD
                ax.add_patch(plt.Rectangle((lx - 0.9, ly - 2.2), 1.8, 4.4, color='#ffb000' if o['kind'] not in CT.VRU else '#e8336d', alpha=0.6))
        ax.plot(G[:, 0], G[:, 1], 'o-', color='k', ms=3, label='GT 6 s')
        ax.plot(P[:, 0], P[:, 1], 's-', color='#c44e52', ms=3, label='prediction')
        ax.add_patch(plt.Rectangle((-0.925, -2.0), 1.85, 4.08, color='#2ecc71'))
        e3 = np.linalg.norm(P[5] - G[5])
        ax.set_xlim(-20, 20); ax.set_ylim(-8, 45); ax.set_aspect('equal'); ax.set_title(f'{lab}\nL2@3s {e3:.2f} m', fontsize=11)
        ax.legend(fontsize=9, loc='lower right'); ax.tick_params(labelsize=9)
        fig.text(0.02 + col * 0.33, 0.445 - row * 0.40, '\n'.join(textwrap.wrap(txt, 62)[:6]), fontsize=9, va='top', family='monospace')
    fig.suptitle(f'Rule-chosen example: {r["scene_name"]} {st} (C2 MOVING, median M1-v2a L2@3s); BEV in the LiDAR frame, '
                 'x right, y forward (m);\nagent boxes axis-aligned approximations: vehicles orange, VRU pink', fontsize=12, y=0.995)
    save(fig, 'fig7_cards')


if __name__ == '__main__':
    main()
