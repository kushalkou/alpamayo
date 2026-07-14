"""visualize.py — Alpamayo VLA demo visualizer.

Per keyframe, renders:
  • Bird's-eye view (ego at origin, facing UP, 6 s horizon): GROUND TRUTH (green),
    MODEL PREDICTION (red), CONSTANT-VELOCITY BASELINE (grey dashed).
  • The 6-camera grid for that keyframe.
Animates across all keyframes of a scene -> GIF.

Ships 3 scenes chosen by GT behaviour: straight / turning / braking-or-accelerating.

Usage:
  python visualize.py --checkpoint /path/to/ckpt.pt --out_dir ../viz [--max_frames 40]

Checkpoint is a CLI arg so the W2 (fixed-ego) model can be swapped in when ready.
Decode / rollout / frame conventions match inference.py (V1-fixed v0 = future_speeds[0]).
"""
import os, sys, argparse, pickle, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from tokenizer import TrajectoryTokenizer
from dataset import compute_ego_state, quat_to_yaw, NUSCENES_ROOT
from vision_live import preprocess_image, encode_normalized_images, CAMERAS
from ar_eval import ar_decode, unicycle_rollout, encode_live_one

DEVICE = 'cuda:0'
DT = 0.5
N_STEPS = 12
TRAJ = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/trajectories_full.pkl'
COSMOS = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'

GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]


def ego_frame(points_global, cx, cy, yaw0):
    """Rotate/translate global points into ego frame (ego at origin, heading +x)."""
    d = np.asarray(points_global, float) - np.array([cx, cy])
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    R = np.array([[c, -s], [s, c]])
    return (R @ d.T).T


def rollout_ego(accels, curvs, v0):
    """Unicycle rollout in ego frame (yaw0=0 => forward +x)."""
    p, _ = unicycle_rollout(np.asarray(accels), np.asarray(curvs), v0, 0.0)
    return p


def cv_ego(v0, n=N_STEPS):
    """Constant velocity in ego frame: straight along +x at v0."""
    return np.array([[v0 * DT * (t + 1), 0.0] for t in range(n)])


def draw_bev(ax, gt, pred, cv):
    """gt/pred/cv are [n,2] ego-frame (x=forward, y=left). Plot forward=UP."""
    def xy(p):   # forward up, left to the left
        return -p[:, 1], p[:, 0]
    for arr, color, style, lab in [(cv, '0.5', '--', 'const-vel'),
                                   (gt, '#2ca02c', '-', 'ground truth'),
                                   (pred, '#d62728', '-', 'prediction')]:
        x, y = xy(arr)
        x = np.concatenate([[0], x]); y = np.concatenate([[0], y])
        ax.plot(x, y, style, color=color, lw=2.2, label=lab,
                marker='o', ms=3, zorder=3)
    ax.scatter([0], [0], c='k', s=60, marker='^', zorder=4, label='ego')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('left  ←  lateral (m)  →  right'); ax.set_ylabel('forward (m)')
    ax.set_title('BEV — 6 s horizon'); ax.legend(loc='upper right', fontsize=8)
    R = 40
    ax.set_xlim(-R/2, R/2); ax.set_ylim(-5, R)


def load_cam(path, size=(320, 180)):
    try:
        return np.asarray(Image.open(path).convert('RGB').resize(size))
    except Exception:
        return np.zeros((size[1], size[0], 3), np.uint8)


def render_frame(model, visual, tok, traj, title):
    cam_paths = traj['cam_paths']
    imgs = torch.stack([preprocess_image(cam_paths[c], augment=False) for c in CAMERAS])
    vt = encode_live_one(visual, imgs, DEVICE)
    ego = compute_ego_state(traj)
    _, accels, curvs = ar_decode(model, vt, ego, tok, DEVICE)

    v0 = float(traj['future_speeds'][0])
    cx, cy = traj['current_pose']['translation'][0], traj['current_pose']['translation'][1]
    yaw0 = quat_to_yaw(traj['current_pose']['rotation'])
    gt = ego_frame(np.array(traj['future_positions'])[:N_STEPS], cx, cy, yaw0)
    pred = rollout_ego(accels, curvs, v0)
    cv = cv_ego(v0)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.4, 1, 1])
    axb = fig.add_subplot(gs[0, :]); draw_bev(axb, gt, pred, cv)
    ade6 = float(np.linalg.norm(pred - gt, axis=1).mean())
    fig.suptitle(f"{title}   |   v0={v0:.1f} m/s   ADE@6s={ade6:.2f} m", fontsize=13)
    for r in range(2):
        for cc in range(3):
            ax = fig.add_subplot(gs[r + 1, cc])
            ax.imshow(load_cam(cam_paths[GRID[r][cc]]))
            ax.set_title(GRID[r][cc], fontsize=8); ax.axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def classify(traj):
    """Return (max|curv|, max|accel|) over the 6 s GT window."""
    cur = np.abs(np.array(traj.get('future_curvatures', [0])))
    acc = np.abs(np.array(traj.get('future_accelerations', [0])))
    return float(cur.max()), float(acc.max())


def pick_scenes(nusc, test_trajs):
    """Pick one straight, one turning, one braking/accelerating exemplar traj; return
    (label -> scene_token)."""
    st2scene = {}
    for scene in nusc.scene:
        tk = scene['first_sample_token']
        while tk:
            st2scene[tk] = scene['token']; tk = nusc.get('sample', tk)['next']
    scored = []
    for t in test_trajs:
        c, a = classify(t)
        scored.append((t, c, a, st2scene.get(t['sample_token'])))
    scored = [s for s in scored if s[3] is not None]
    turning = max(scored, key=lambda s: s[1])
    braking = max(scored, key=lambda s: s[2])
    straight = min(scored, key=lambda s: s[1] + s[2])
    return {'straight': straight[3], 'turning': turning[3], 'braking_accel': braking[3]}


def scene_trajs_ordered(nusc, scene_token, by_sample):
    scene = nusc.get('scene', scene_token)
    out = []
    tk = scene['first_sample_token']
    while tk:
        if tk in by_sample:
            out.append(by_sample[tk])
        tk = nusc.get('sample', tk)['next']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out_dir', default='/home/dgx1user/Alpamayo-Kushal/Alpamayo/viz')
    ap.add_argument('--max_frames', type=int, default=40)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from nuscenes.nuscenes import NuScenes
    print('[viz] loading nuScenes...')
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSCENES_ROOT, verbose=False)
    allt = pickle.load(open(TRAJ, 'rb'))
    # resolve cam_paths for every traj + index by sample_token
    st2scene = {}
    for scene in nusc.scene:
        tk = scene['first_sample_token']
        while tk:
            st2scene[tk] = scene['token']; tk = nusc.get('sample', tk)['next']
    by_sample = {}
    from dataset import build_scene_split
    _, _, test = build_scene_split(allt, NUSCENES_ROOT)   # resolves cam_paths on split
    for t in test:
        by_sample[t['sample_token']] = t

    model = load_model(cosmos_path=COSMOS, device=DEVICE)
    ck = torch.load(a.checkpoint, map_location='cpu')
    model.load_state_dict(ck['model_state'], strict=False)
    model.zero_ego = False
    model.cosmos.model.language_model.gradient_checkpointing_disable()
    model.eval()
    visual = model.cosmos.model.visual
    tok = TrajectoryTokenizer()
    print(f"[viz] checkpoint {a.checkpoint} (epoch={ck.get('epoch')}, "
          f"val_ade6={ck.get('val_ade6')}, val_loss={ck.get('val_loss')})")

    scenes = pick_scenes(nusc, test)
    print(f"[viz] scenes: {scenes}")
    for label, sc in scenes.items():
        trajs = scene_trajs_ordered(nusc, sc, by_sample)[:a.max_frames]
        if not trajs:
            print(f"[viz] {label}: no trajs, skip"); continue
        frames = []
        for i, tr in enumerate(trajs):
            frames.append(render_frame(model, visual, tok, tr, f"{label}  frame {i+1}/{len(trajs)}"))
            if i % 5 == 0: print(f"  {label}: {i+1}/{len(trajs)}", flush=True)
        gif = os.path.join(a.out_dir, f"demo_{label}.gif")
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=400, loop=0)
        print(f"[viz] saved {gif} ({len(frames)} frames)")
    print("VIZ_DONE")


if __name__ == '__main__':
    main()
