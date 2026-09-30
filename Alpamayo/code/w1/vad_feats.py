"""w1/vad_feats.py -- WEEK1 planner item 2: VAD's ego status (gt_ego_lcf_feat), computed
EXACTLY as tools/data_converter/vad_nuscenes_converter.py does (hustvl/VAD main), for
every w1 record. Faithful reproduction, including:
  - quart_to_rpy (L32-37) unpacks (x,y,z,w) while nuScenes stores (w,x,y,z), so
    ego_yaw ~ 0 for planar motion -> ego_w ~ 0 and (vx,vy) ~ (0, v);
  - ego_v / ego_w from the previous LIDAR keyframe, else FORWARD from the next (L473-480);
  - can_bus accel (-> ax, ay) from `pose`, the first CAN pose message AFTER t0, via
    the L180-190 loop-variable bug; zeros(18) for scenes without CAN (L175-176);
  - v0 and steering from locate_message (L39-43, NEAREST message, possibly after t0);
    Kappa = 2*steering/2.588, sign flipped in Singapore;
  - except-path (scene without CAN): v0 = |ego_his_trajs[-1] + ego_fut_trajs[0]|
    (L503-506; the history filler L415-417 itself uses next-sample offsets when the
    previous sample is missing), Kappa = 0.
Output feature (9): [vx, vy, ax, ay, w, length, width, v0, Kappa] -> data/w1_vadfeat.pkl
"""
import sys, math, pickle
import numpy as np
from pyquaternion import Quaternion
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code/w1')

ROOT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
OUT = '/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_vadfeat.pkl'
ego_width, ego_length = 1.85, 4.084


def quart_to_rpy(qua):                         # VAD L32-37, verbatim
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw


def locate_message(utimes, utime):             # VAD L39-43, verbatim
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


def get_global_sensor_pose(rec, nusc):
    sd = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    from nuscenes.utils.geometry_utils import transform_matrix
    return transform_matrix(ep['translation'], Quaternion(ep['rotation'])) @ \
        transform_matrix(cs['translation'], Quaternion(cs['rotation']))


def can_bus_accel(pose_list, t):
    """_get_can_bus_info L178-190: returns can_bus[7:9] = pose[accel][:2] where `pose` is
    the loop variable after the break (first message with utime > t)."""
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > t:
            break
        last_pose = pose
    return list(pose['accel'][:2])


def feats(nusc, can, s, loc_cache):
    sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
    pose_record = nusc.get('ego_pose', sd['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    prev = nusc.get('ego_pose', nusc.get('sample_data', nusc.get('sample', s['prev'])['data']['LIDAR_TOP'])['ego_pose_token']) if s['prev'] else None
    nxt = nusc.get('ego_pose', nusc.get('sample_data', nusc.get('sample', s['next'])['data']['LIDAR_TOP'])['ego_pose_token']) if s['next'] else None
    scene = nusc.get('scene', s['scene_token'])
    try:
        pose_list = can.get_messages(scene['name'], 'pose')
        ax, ay = can_bus_accel(pose_list, s['timestamp'])
    except Exception:
        ax, ay = 0.0, 0.0                       # np.zeros(18)
    _, _, ego_yaw = quart_to_rpy(pose_record['rotation'])
    ego_pos = np.array(pose_record['translation'])
    if prev is not None:
        _, _, ego_yaw_prev = quart_to_rpy(prev['rotation'])
        ego_w = (ego_yaw - ego_yaw_prev) / 0.5
        ego_v = np.linalg.norm(ego_pos[:2] - np.array(prev['translation'])[:2]) / 0.5
    else:
        _, _, ego_yaw_next = quart_to_rpy(nxt['rotation'])
        ego_w = (ego_yaw_next - ego_yaw) / 0.5
        ego_v = np.linalg.norm(np.array(nxt['translation'])[:2] - ego_pos[:2]) / 0.5
    ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)
    try:
        pose_msgs = can.get_messages(scene['name'], 'pose')
        steer_msgs = can.get_messages(scene['name'], 'steeranglefeedback')
        pu = [m['utime'] for m in pose_msgs]; su = [m['utime'] for m in steer_msgs]
        v0 = pose_msgs[locate_message(pu, s['timestamp'])]['vel'][0]
        steering = steer_msgs[locate_message(su, s['timestamp'])]['value']
        loc = nusc.get('log', scene['log_token'])['location']
        if loc.startswith('singapore'):
            steering *= -1
        Kappa = 2 * steering / 2.588
    except Exception:
        # except-path: ego_his_trajs[-1] + ego_fut_trajs[0] in the lcf lidar frame
        def lcf(p):
            x = p - np.array(pose_record['translation'])
            x = Quaternion(pose_record['rotation']).inverse.rotation_matrix @ x
            x = x - np.array(cs_record['translation'])
            return Quaternion(cs_record['rotation']).inverse.rotation_matrix @ x
        g0 = get_global_sensor_pose(s, nusc)[:3, 3]
        gn = get_global_sensor_pose(nusc.get('sample', s['next']), nusc)[:3, 3] if s['next'] else g0
        if s['prev']:
            gp = get_global_sensor_pose(nusc.get('sample', s['prev']), nusc)[:3, 3]
        else:
            gp = g0 - (gn - g0)                 # L415-417 filler: uses the next offset
        his_last = lcf(g0) - lcf(gp); fut0 = lcf(gn) - lcf(g0)
        d = his_last + fut0
        v0 = float(np.sqrt(d[0] ** 2 + d[1] ** 2)); Kappa = 0
    return [ego_vx, ego_vy, ax, ay, ego_w, ego_length, ego_width, v0, Kappa]


def main():
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    W = pickle.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/w1_data.pkl', 'rb'))
    nusc = NuScenes(version='v1.0-trainval', dataroot=ROOT, verbose=False)
    can = NuScenesCanBus(dataroot=ROOT)
    cache = {}

    class Cached:
        def get_messages(self, sc, kind):
            if (sc, kind) not in cache:
                try: cache[(sc, kind)] = can.get_messages(sc, kind)
                except Exception as e: cache[(sc, kind)] = e
            v = cache[(sc, kind)]
            if isinstance(v, Exception): raise v
            return [dict(m) for m in v]

    C = Cached()
    out = {}
    for r in W['records']:
        out[r['sample_token']] = np.array(feats(nusc, C, nusc.get('sample', r['sample_token']), None),
                                          dtype=np.float64)
    pickle.dump(out, open(OUT, 'wb'))
    F = np.stack(list(out.values()))
    print(f'[vadfeat] {len(out)} records -> {OUT}')
    for j, n in enumerate(['vx', 'vy', 'ax', 'ay', 'w', 'length', 'width', 'v0', 'Kappa']):
        print(f'  {n:6} mean {F[:, j].mean():+.4f}  sd {F[:, j].std():.4f}  |max| {np.abs(F[:, j]).max():.4f}')


if __name__ == '__main__':
    main()
