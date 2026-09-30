"""w1/plan_metrics.py -- WEEK1 item 3: standard nuScenes open-loop planning metrics.

SOURCES (ported, not rewritten):
  VAD  projects/mmdet3d_plugin/VAD/planner/metric_stp3.py  ("same as stp3")
       PlanningMetric: X/Y bounds [-50,50,0.5] (200x200 BEV), ego W=1.85 m, H(length)
       =4.084 m, ego box shifted +0.5 m forward; _get_poly_region_in_image,
       evaluate_single_coll and evaluate_coll are copied VERBATIM below (only the
       class wrapper is simplified; torch kept).
  VAD  projects/mmdet3d_plugin/VAD/VAD.py  compute_planner_metric_stp3:
       for i in range(3): cur_time=(i+1)*2; L2 = compute_L2(pred[:cur_time],
       gt[:cur_time]) (mean over the prefix); box collision = obj_box_coll.mean() over
       the prefix; occupancy = logical_or(vehicle, pedestrian).        -> "TemAvg"
  UniAD projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py
       same W/H/+0.5 ego box and the same GT-collision masking; L2 is per timestep
       (compute_L2 returns [B, n_future]) and results are read at the step itself;
       occupancy = vehicles only.                                     -> "NoAvg"
  UniAD's rasterisation uses its own occupancy-GT orientation. Here BOTH conventions
  use VAD's verbatim engine; NoAvg differs only in category set (vehicles) and in
  reading step k instead of the prefix mean.

Occupancy input (VAD builds it via its mmdet3d pipeline, not runnable here): agents
present at t0 (vehicle.* / human.pedestrian.*, >= 1 lidar/radar point), boxed at their
annotated future pose at each of the 6 future keyframes, in the t0 LIDAR frame
(w1/build_data.py). The reported collision is VAD/UniAD's obj_box_col: ego-box overlap
with occupancy, counted only at timesteps where the GT ego box does NOT itself collide.
Consequence: the GT trajectory scores exactly 0 by construction; the raw (unmasked)
GT box-collision rate is reported separately as the annotation-noise check.

Frames: all trajectories are LIDAR_TOP positions in the t0 LIDAR frame (x right,
y forward), as in VAD's gt_ego_fut_trajs. Model/CV predictions (CAM-ego-origin
positions + yaw) are converted by pred_to_lidar().
"""
import copy, math
import numpy as np, torch, cv2
from skimage.draw import polygon
from pyquaternion import Quaternion

ego_width, ego_length = 1.85, 4.084
DT, N = 0.5, 12


class PlanningMetric():
    def __init__(self):
        self.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]
        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()
        self.W = ego_width
        self.H = ego_length

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        return dx, bx, nx

    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                     dtype=torch.long)
        return bev_resolution, bev_start_position, bev_dimension

    # ---- VERBATIM (VAD metric_stp3.py) ----
    def _get_poly_region_in_image(self, param):
        lidar2cv_rot = np.array([[1, 0], [0, -1]])
        x_a, y_a, yaw_a, agent_length, agent_width = param
        trans_a = np.array([[x_a, y_a]]).T
        rot_mat_a = np.array([[np.cos(yaw_a), -np.sin(yaw_a)],
                              [np.sin(yaw_a), np.cos(yaw_a)]])
        agent_corner = np.array([
            [agent_length/2, -agent_length/2, -agent_length/2, agent_length/2],
            [agent_width/2, agent_width/2, -agent_width/2, -agent_width/2]])  # (2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a  # (2,4)
        # convert to cv frame
        agent_corner_cv2 = (np.matmul(lidar2cv_rot, agent_corner_lidar)
            - self.bev_start_position[:2, None] + self.bev_resolution[:2, None] / 2.0).T / self.bev_resolution[:2]  # (4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)
        return agent_corner_cv2

    def evaluate_single_coll(self, traj, segmentation, input_gt):
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        trajs_ = copy.deepcopy(trajs)
        trajs_[:, :, [0, 1]] = trajs_[:, :, [1, 0]]  # can also change original tensor
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc  # (n_future, 32, 2)

        r = (self.bev_dimension[0] - trajs_[:, :, 0]).astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:, :, 1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())
        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        B, n_future, _ = trajs.shape
        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)
        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i], input_gt=True)
            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            xi = ((-self.bx[0]/2 - yy) / self.dx[0]).long()
            yi = ((-self.bx[1]/2 + xx) / self.dx[1]).long()
            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))
            ti = torch.arange(n_future)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], xi[m1], yi[m1]].long()
            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i], input_gt=False).to(ti.device)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()
        return obj_coll_sum, obj_box_coll_sum
    # ---- end verbatim ----

    def occupancy(self, rec, classes):
        """[6,200,200] occupancy from rec['agents'] (VAD get_birds_eye_view_label logic)."""
        seg = np.zeros((6, self.bev_dimension[0], self.bev_dimension[1]))
        for a in rec['agents']:
            if a['cls'] not in classes:
                continue
            for t, b in enumerate(a['fut']):
                if b is None:
                    continue
                x, y, yaw, l, w = b
                cv2.fillPoly(seg[t], [self._get_poly_region_in_image([x, y, yaw, l, w])], 1.0)
        return torch.from_numpy(seg)


PM = PlanningMetric()


def sample_metrics(rec, pred6, occ_cache=None):
    """Per-sample per-timestep arrays for one prediction [6,2] (t0 LIDAR frame).
    Returns l2[6], box_col_veh[6], box_col_vp[6] (masked as in the source code)."""
    gt = torch.tensor(rec['gt_lidar6'], dtype=torch.float32)
    pr = torch.tensor(np.asarray(pred6), dtype=torch.float32)
    l2 = torch.sqrt(((pr - gt) ** 2).sum(-1)).numpy()
    occ = occ_cache if occ_cache is not None else occupancies(rec)
    out = {'l2': l2}
    for k, o in occ.items():
        _, box = PM.evaluate_coll(pr[None], gt[None], o[None])
        out[f'col_{k}'] = box.numpy()
    return out


def occupancies(rec):
    return {'veh': PM.occupancy(rec, ('vehicle',)),
            'vp': PM.occupancy(rec, ('vehicle', 'human'))}


def gt_raw_collision(rec, occ):
    """UNMASKED GT ego-box collision per timestep (annotation-noise check)."""
    gt = torch.tensor(rec['gt_lidar6'], dtype=torch.float32)
    return {k: PM.evaluate_single_coll(gt, o, True).numpy() for k, o in occ.items()}


def aggregate(per):
    """per: dict of arrays [n,6] -> literature numbers.
    NoAvg (UniAD): L2 at step k, collision (vehicles) at step k.
    TemAvg (ST-P3/VAD): L2 mean over steps < k, collision (veh+ped) mean over steps < k.
    k = 2, 4, 6 -> 1 s, 2 s, 3 s. Collision in %."""
    r = {}
    for s, k in ((1, 2), (2, 4), (3, 6)):
        r[f'L2_NoAvg_{s}s'] = float(per['l2'][:, k - 1].mean())
        r[f'L2_TemAvg_{s}s'] = float(per['l2'][:, :k].mean())
        r[f'Col_NoAvg_{s}s'] = 100 * float(per['col_veh'][:, k - 1].mean())
        r[f'Col_TemAvg_{s}s'] = 100 * float(per['col_vp'][:, :k].mean())
    return r


# ---------------- prediction -> t0 LIDAR frame ----------------

def rollout_xy_yaw(acc, cur, v0, yaw0):
    """shrink_lib.rollout, also returning the heading after each step."""
    x = y = 0.0; yaw = yaw0; v = v0
    P = np.empty((N, 2)); Y = np.empty(N)
    for j in range(N):
        v = max(0.0, v + acc[j] * DT)
        yaw = yaw + v * cur[j] * DT
        x += v * math.cos(yaw) * DT; y += v * math.sin(yaw) * DT
        P[j] = (x, y); Y[j] = yaw
    return P, Y


def pred_to_lidar(rec, xy_rel, yaw, origin='cam'):
    """xy_rel [n,2]: ego-origin displacement from the t0 origin, GLOBAL axes;
    yaw [n]: global heading. -> LIDAR_TOP position in the t0 LIDAR frame [n,2].
    origin 'cam': t0 origin = CAM_FRONT ego pose (the model's frame);
           'lidar': t0 origin = LIDAR_TOP ego pose (the literature GT's frame)."""
    ep, cs = rec['lidar_ep0'], rec['lidar_cs']
    o = (np.array(rec['current_pose']['translation'][:3]) if origin == 'cam'
         else np.array(ep['translation']))
    csr = Quaternion(cs['rotation']); cst = np.array(cs['translation'])
    out = []
    for (dx, dy), h in zip(xy_rel, yaw):
        q = Quaternion(axis=[0, 0, 1], angle=h)
        g = np.array([o[0] + dx, o[1] + dy, o[2]]) + q.rotation_matrix @ cst
        x = g - np.array(ep['translation'])
        x = Quaternion(ep['rotation']).inverse.rotation_matrix @ x
        x = csr.inverse.rotation_matrix @ (x - cst)
        out.append(x[:2])
    return np.array(out)
