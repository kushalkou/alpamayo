"""W4 — scenario stratification (straight vs turning) by GT curvature.

Splits the test set by max |GT curvature| over the 6 s window and reports ADE@1/2/3/6s
per subset for: full live-vision, ego-only, zero-both, and the constant-velocity baseline.

Consistency note: the existing full/ego-only checkpoints were TRAINED on the OLD
(pre-W1) ego encoding, and V3a evaluated them that way. To stay matched, this script
feeds the model the OLD ego (inlined `old_ego`) while seeding the rollout with the
V1-fixed v0=future_speeds[0]. zero-both ignores inputs; CV uses no model. (The clean
fixed-ego stratification will be re-run on the W2 checkpoints.)

Launch: torchrun --nproc_per_node=8 eval_stratified.py
"""
import os, sys, json, pickle
import numpy as np, torch
import torch.distributed as dist
sys.path.insert(0,'/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset, pose_to_xyyaw
from tokenizer import TrajectoryTokenizer
from ar_eval import ar_decode, unicycle_rollout, encode_live_one, HORIZONS, N_STEPS
import inference as INF

CK='/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
MODELS=[
    ('full',      f'{CK}/_livevision_run_jul9/alpamayo_best_e1_val2.0806.pt', False, False),
    ('ego_only',  f'{CK}/_egoonly_run_jul9/alpamayo_best_e1_val2.0035.pt',    True,  False),
    ('zero_both', f'{CK}/_zeroboth_run_jul12/alpamayo_best_e1_val2.0392.pt',  True,  True),
]
CURV_THRESH=0.05   # rad/m; turning if max|future_curvature| exceeds this

def old_ego(traj):
    """Pre-W1 compute_ego_state (backward diff), to match how these ckpts were trained."""
    dt=0.5
    raw=list(traj.get('past_poses',[])); raw.append(traj['current_pose'])
    poses=[pose_to_xyyaw(p) for p in raw]
    while len(poses)<4: poses=[poses[0]]+poses
    poses=poses[-4:]; states=[]; speeds=[]
    for i in range(4):
        x,y,yaw=poses[i]
        if i==0: sp=yr=ac=0.0
        else:
            xp,yp,yawp=poses[i-1]; sp=float(np.hypot(x-xp,y-yp))/dt
            yr=float(np.arctan2(np.sin(yaw-yawp),np.cos(yaw-yawp)))/dt
            ac=(sp-speeds[-1])/dt if speeds else 0.0
        speeds.append(sp); states.append([sp,yaw,yr,ac])
    return torch.tensor(states,dtype=torch.float32)

def gt_local(traj):
    gt=np.array(traj['future_positions'])[:N_STEPS]
    c=np.array(traj['current_pose']['translation'][:2]); return gt-c

def ades(pred, gt):
    out={}
    for step,lab in HORIZONS.items():
        e=np.linalg.norm(pred[:step]-gt[:step],axis=1); out[lab]=float(e.mean())
    return out

def main():
    dist.init_process_group(backend='nccl')
    lr=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device=f'cuda:{lr}'; ws=dist.get_world_size(); m0=(lr==0)
    with open(INF.TRAJECTORIES_PATH,'rb') as f: allt=pickle.load(f)
    _,_,test=build_scene_split(allt, INF.NUSCENES_ROOT)
    ds=NuScenesVLADataset(test, split='test', augment=False)
    tok=TrajectoryTokenizer()
    model=load_model(device=device); model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual=model.cosmos.model.visual
    my=list(range(lr,len(test),ws))

    # curvature label per sample
    def maxcurv(t):
        c=np.abs(np.array(t.get('future_curvatures',[0]))); return float(c.max()) if len(c) else 0.0

    # CV baseline (no model)
    per={'CV':{}}
    for nm,_,_,_ in MODELS: per[nm]={}
    for i in my:
        t=test[i]; v0=float(t['future_speeds'][0]); g=gt_local(t)
        # CV in GLOBAL-axes ego-origin frame: straight along current heading
        yaw0=float(old_ego(t)[3,1])  # == current global yaw
        cvp=np.array([[v0*0.5*(k+1)*np.cos(yaw0), v0*0.5*(k+1)*np.sin(yaw0)] for k in range(N_STEPS)])
        per['CV'][i]=ades(cvp,g)

    # models
    for nm,path,zv,ze in MODELS:
        ck=torch.load(path,map_location='cpu'); model.load_state_dict(ck['model_state'],strict=False)
        model.zero_ego=ze
        for c,i in enumerate(my):
            t=test[i]; item=ds[i]
            vt=encode_live_one(visual,item['images'],device)
            if zv: vt=torch.zeros_like(vt)
            ego=old_ego(t)
            _,acc,cur=ar_decode(model,vt,ego,tok,device)
            v0=float(t['future_speeds'][0]); yaw0=float(ego[3,1])
            pred,_=unicycle_rollout(acc,cur,v0,yaw0)
            per[nm][i]=ades(pred,gt_local(t))
            if m0 and nm=='full' and c%80==0: print(f"  full {c}/{len(my)}",flush=True)

    # File-based gather (NOT all_gather_object): the ~45-min imbalanced decode makes ranks
    # drift far past the 10-min NCCL collective timeout. Each rank writes its shard; rank 0
    # polls for all shards and aggregates. No collective over the long loop.
    import time as _time
    SHARD_DIR='/tmp/claude-1000/w4_shards'; os.makedirs(SHARD_DIR, exist_ok=True)
    payload={'per':per,'curv':{i:maxcurv(test[i]) for i in my}}
    tmp=os.path.join(SHARD_DIR,f'shard_{lr}.pkl.tmp'); fin=os.path.join(SHARD_DIR,f'shard_{lr}.pkl')
    with open(tmp,'wb') as f: pickle.dump(payload,f)
    os.replace(tmp,fin)
    if not m0:
        dist.destroy_process_group(); return
    # rank 0 waits for all shards
    while sum(os.path.exists(os.path.join(SHARD_DIR,f'shard_{r}.pkl')) for r in range(ws)) < ws:
        _time.sleep(5)
    gathered=[pickle.load(open(os.path.join(SHARD_DIR,f'shard_{r}.pkl'),'rb')) for r in range(ws)]
    if m0:
        PER={k:{} for k in per}; CURV={}
        for g in gathered:
            for k in g['per']:
                PER[k].update(g['per'][k])
            CURV.update(g['curv'])
        idx=list(CURV.keys())
        straight=[i for i in idx if CURV[i]<=CURV_THRESH]
        turning =[i for i in idx if CURV[i]> CURV_THRESH]
        print(f"\n=== W4 STRATIFICATION (threshold max|curv|>{CURV_THRESH} rad/m) ===")
        print(f"straight n={len(straight)}   turning n={len(turning)}   total={len(idx)}")
        def report(subset,name):
            print(f"\n-- {name} (n={len(subset)}) --")
            print(f"{'model':12} {'ADE1s':>7} {'ADE2s':>7} {'ADE3s':>7} {'ADE6s':>7}")
            for k in ['CV']+[m[0] for m in MODELS]:
                row=[np.mean([PER[k][i][l] for i in subset]) for l in ['1s','2s','3s','6s']]
                print(f"{k:12} "+" ".join(f"{v:7.3f}" for v in row))
        report(idx,'ALL'); report(straight,'STRAIGHT'); report(turning,'TURNING')
        res={'threshold':CURV_THRESH,'n_straight':len(straight),'n_turning':len(turning),
             'PER':{k:{str(i):PER[k][i] for i in idx} for k in PER},
             'curv':{str(i):CURV[i] for i in idx}}
        json.dump(res, open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/res_w4_stratified.json','w'))
        print("\nW4_DONE")
    dist.destroy_process_group()

if __name__=='__main__': main()
