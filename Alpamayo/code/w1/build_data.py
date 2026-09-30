"""w1/build_data.py -- WEEK1 items 2/3/4: official-split dataset + literature-style GT.

For every nuScenes v1.0-trainval sample with >= 6 future keyframes (3 s, the planning
protocol horizon) this builds one record containing:

  MODEL INPUT/TARGET fields, identical in construction to extract_trajectories.py
  (CAM_FRONT ego poses): sample_token, current_pose, past_poses (<= 4), cam_paths,
  future_positions / future_yaws / future_speeds / future_accelerations /
  future_curvatures for the n_fut <= 12 available future keyframes, n_fut.

  LITERATURE GT fields, copied from VAD tools/data_converter/vad_nuscenes_converter.py:
    gt_lidar6  [6,2]  LIDAR_TOP sensor position at future steps 1..6, expressed in the
                      LIDAR frame at t0 (x right, y forward), cumulative (i.e. BEFORE
                      VAD's per-step offset conversion)
    command    0 right / 1 left / 2 straight:
                 if ego_fut_trajs[-1][0] >= 2: Turn Right
                 elif ego_fut_trajs[-1][0] <= -2: Turn Left
                 else: Go Straight
               (ego_fut_trajs[-1] = lidar-frame position at future step 6 = 3 s)
    lidar_ep0, lidar_cs: ego pose of the LIDAR_TOP sample_data at t0 and the LIDAR_TOP
                      calibrated_sensor record (for converting predictions)
    agents     boxes for collision: every annotation at t0 whose category is vehicle.*
               or human.pedestrian.* and has >= 1 lidar/radar point, followed along its
               'next' chain for future steps 1..6 (VAD convention: only agents present
               at t0; a missing future annotation is masked). Each entry is
               (x, y, yaw, length, width) in the t0 LIDAR frame, yaw from +x CCW.
  scene_name, split in {train, holdout, val} (official nuScenes splits; 50 holdout
  scenes drawn from official train with RandomState(42) over sorted scene names).

Output: Alpamayo/data/w1_data.pkl  {'records': [...], 'holdout_scenes': [...]}
"""
import sys, pickle, time
import numpy as np
from pyquaternion import Quaternion
CODE = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code'
sys.path.insert(0, CODE)
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from extract_trajectories import get_ego_pose, compute_curvature, quaternion_to_yaw
from vision_live import CAMERAS

ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_data.pkl'
DT, N_FUT, N_PAST, N_EVAL = 0.5, 12, 4, 6
N_HOLDOUT = 50


def kin(cur, fut):
    """extract_trajectory's speed/accel/curvature over [current]+future (any length)."""
    allp = [cur] + fut
    pos = np.array([p['translation'][:2] for p in allp])
    yaw = np.array([quaternion_to_yaw(p['rotation']) for p in allp])
    sp = np.zeros(len(allp))
    for i in range(1, len(allp)):
        sp[i] = np.linalg.norm(pos[i] - pos[i - 1]) / DT
    sp[0] = sp[1]
    ac = np.zeros(len(allp))
    for i in range(1, len(allp)):
        ac[i] = (sp[i] - sp[i - 1]) / DT
    ac[0] = ac[1]
    cu = compute_curvature(pos, yaw)
    return pos[1:], yaw[1:], sp[1:], ac[1:], cu[1:]


def to_lidar(pts_global, ep, cs):
    """VAD: global -> ego at lcf -> lidar at lcf."""
    x = pts_global - np.array(ep['translation'])
    x = np.dot(Quaternion(ep['rotation']).inverse.rotation_matrix, x.T).T
    x = x - np.array(cs['translation'])
    return np.dot(Quaternion(cs['rotation']).inverse.rotation_matrix, x.T).T


def lidar_global_pos(nusc, s):
    sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    return (Quaternion(ep['rotation']).rotation_matrix @ np.array(cs['translation'])
            + np.array(ep['translation']))


def box_in_lidar(nusc, ann_token, ep, cs):
    a = nusc.get('sample_annotation', ann_token)
    b = Box(a['translation'], a['size'], Quaternion(a['rotation']))
    b.translate(-np.array(ep['translation'])); b.rotate(Quaternion(ep['rotation']).inverse)
    b.translate(-np.array(cs['translation'])); b.rotate(Quaternion(cs['rotation']).inverse)
    w, l, _ = b.wlh
    yaw = b.orientation.yaw_pitch_roll[0]
    return [float(b.center[0]), float(b.center[1]), float(yaw), float(l), float(w)]


def main():
    t0 = time.time()
    nusc = NuScenes(version='v1.0-trainval', dataroot=ROOT, verbose=False)
    sp = create_splits_scenes()
    tr_names = sorted(sp['train']); va_names = set(sp['val'])
    rs = np.random.RandomState(42)
    hold = set(rs.choice(tr_names, N_HOLDOUT, replace=False).tolist())
    recs = []
    for scene in nusc.scene:
        name = scene['name']
        split = 'val' if name in va_names else ('holdout' if name in hold else 'train')
        toks = []; t = scene['first_sample_token']
        while t:
            toks.append(t); t = nusc.get('sample', t)['next']
        for j, st in enumerate(toks):
            n_fut = min(N_FUT, len(toks) - 1 - j)
            if n_fut < N_EVAL:
                continue
            s = nusc.get('sample', st)
            cur = get_ego_pose(nusc, st)
            fut = [get_ego_pose(nusc, toks[j + k]) for k in range(1, n_fut + 1)]
            past = [get_ego_pose(nusc, toks[k]) for k in range(max(0, j - N_PAST), j)]
            fp, fy, fs, fa, fc = kin(cur, fut)
            # literature GT (VAD converter)
            sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
            ep = nusc.get('ego_pose', sd['ego_pose_token'])
            cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
            g = np.stack([lidar_global_pos(nusc, nusc.get('sample', toks[j + k]))
                          for k in range(0, N_EVAL + 1)])
            gl = to_lidar(g, ep, cs)                     # [7,3], row 0 = t0 (~0)
            x6 = gl[-1][0]
            command = 0 if x6 >= 2 else (1 if x6 <= -2 else 2)
            # agents for collision
            agents = []
            for at in s['anns']:
                a = nusc.get('sample_annotation', at)
                cat = a['category_name']
                if not (cat.startswith('vehicle.') or cat.startswith('human.pedestrian.')):
                    continue
                if a['num_lidar_pts'] + a['num_radar_pts'] <= 0:
                    continue
                traj = []; nxt = a['next']
                for k in range(1, N_EVAL + 1):
                    if nxt == '':
                        traj.append(None); continue
                    traj.append(box_in_lidar(nusc, nxt, ep, cs))
                    nxt = nusc.get('sample_annotation', nxt)['next']
                agents.append({'cls': 'vehicle' if cat.startswith('vehicle.') else 'human',
                               'fut': traj})
            recs.append({
                'sample_token': st, 'scene_name': name, 'split': split, 'idx_in_scene': j,
                'current_pose': cur, 'past_poses': past, 'n_fut': n_fut,
                'future_positions': fp, 'future_yaws': fy, 'future_speeds': fs,
                'future_accelerations': fa, 'future_curvatures': fc,
                'cam_paths': {c: nusc.get_sample_data_path(s['data'][c]) for c in CAMERAS},
                'gt_lidar6': gl[1:, :2].astype(np.float64), 'command': command,
                'lidar_ep0': {'translation': ep['translation'], 'rotation': ep['rotation']},
                'lidar_cs': {'translation': cs['translation'], 'rotation': cs['rotation']},
                'lidar_ts': sd['timestamp'], 'cam_ts': cur['timestamp'],
                'agents': agents,
            })
    pickle.dump({'records': recs, 'holdout_scenes': sorted(hold)}, open(OUT, 'wb'))
    print(f'[w1] {len(recs)} records -> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
