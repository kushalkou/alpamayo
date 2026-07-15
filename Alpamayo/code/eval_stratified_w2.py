"""X2 — straight/turning stratification on the W2 FIXED-EGO models.

Same split as W4 (max|GT curv|>0.05 rad/m) but on the W2 full + ego-only checkpoints,
evaluated with the CORRECT ego (current compute_ego_state, matching W2 training) and the
V1-fixed rollout. Reports ADE@1/2/3/6s per subset for full, ego-only, CV.

Launch: torchrun --nproc_per_node=8 --master_port=29541 eval_stratified_w2.py
"""
import os, sys, json, pickle, time
import numpy as np, torch
import torch.distributed as dist
sys.path.insert(0,'/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset, compute_ego_state
from tokenizer import TrajectoryTokenizer
from ar_eval import ar_decode, unicycle_rollout, encode_live_one, HORIZONS, N_STEPS
import inference as INF

CK='/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
MODELS=[
    ('full',     f'{CK}/_w2_full_fixed/alpamayo_best.pt',    False, False),
    ('ego_only', f'{CK}/_w2_egoonly_fixed/alpamayo_best.pt', True,  False),
]
CURV=0.05
SHARD='/tmp/claude-1000/x2_shards'

def gt_local(t):
    return np.array(t['future_positions'])[:N_STEPS]-np.array(t['current_pose']['translation'][:2])
def ades(pred,gt):
    return {lab:float(np.linalg.norm(pred[:s]-gt[:s],axis=1).mean()) for s,lab in HORIZONS.items()}
def maxcurv(t):
    c=np.abs(np.array(t.get('future_curvatures',[0]))); return float(c.max()) if len(c) else 0.0

def main():
    dist.init_process_group(backend='nccl')
    lr=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device=f'cuda:{lr}'; ws=dist.get_world_size(); m0=(lr==0)
    os.makedirs(SHARD,exist_ok=True)
    with open(INF.TRAJECTORIES_PATH,'rb') as f: allt=pickle.load(f)
    _,_,test=build_scene_split(allt, INF.NUSCENES_ROOT)
    ds=NuScenesVLADataset(test, split='test', augment=False)
    tok=TrajectoryTokenizer()
    model=load_model(device=device); model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual=model.cosmos.model.visual
    my=list(range(lr,len(test),ws))

    per={'CV':{}}
    for nm,_,_,_ in MODELS: per[nm]={}
    for i in my:
        t=test[i]; ego=compute_ego_state(t); v0=float(t['future_speeds'][0]); yaw0=float(ego[3,1])
        cvp=np.array([[v0*0.5*(k+1)*np.cos(yaw0), v0*0.5*(k+1)*np.sin(yaw0)] for k in range(N_STEPS)])
        per['CV'][i]=ades(cvp,gt_local(t))
    for nm,path,zv,ze in MODELS:
        ck=torch.load(path,map_location='cpu'); model.load_state_dict(ck['model_state'],strict=False); model.zero_ego=ze
        for c,i in enumerate(my):
            t=test[i]; item=ds[i]; vt=encode_live_one(visual,item['images'],device)
            if zv: vt=torch.zeros_like(vt)
            ego=item['ego_state']; _,acc,cur=ar_decode(model,vt,ego,tok,device)
            v0=float(t['future_speeds'][0]); yaw0=float(ego[3,1])
            pred,_=unicycle_rollout(acc,cur,v0,yaw0); per[nm][i]=ades(pred,gt_local(t))
            if m0 and nm=='full' and c%80==0: print(f"  full {c}/{len(my)}",flush=True)

    payload={'per':per,'curv':{i:maxcurv(test[i]) for i in my}}
    with open(os.path.join(SHARD,f's_{lr}.pkl.tmp'),'wb') as f: pickle.dump(payload,f)
    os.replace(os.path.join(SHARD,f's_{lr}.pkl.tmp'),os.path.join(SHARD,f's_{lr}.pkl'))
    if not m0: dist.destroy_process_group(); return
    while sum(os.path.exists(os.path.join(SHARD,f's_{r}.pkl')) for r in range(ws))<ws: time.sleep(5)
    gathered=[pickle.load(open(os.path.join(SHARD,f's_{r}.pkl'),'rb')) for r in range(ws)]
    PER={k:{} for k in per}; CURVd={}
    for g in gathered:
        for k in g['per']: PER[k].update(g['per'][k])
        CURVd.update(g['curv'])
    idx=list(CURVd.keys()); straight=[i for i in idx if CURVd[i]<=CURV]; turning=[i for i in idx if CURVd[i]>CURV]
    print(f"\n=== X2 W2 FIXED-EGO STRATIFICATION (max|curv|>{CURV}) ===")
    print(f"straight n={len(straight)}  turning n={len(turning)}  total={len(idx)}")
    def rep(sub,name):
        print(f"\n-- {name} (n={len(sub)}) --")
        print(f"{'model':10} {'ADE1s':>7} {'ADE2s':>7} {'ADE3s':>7} {'ADE6s':>7}")
        for k in ['CV']+[m[0] for m in MODELS]:
            row=[np.mean([PER[k][i][l] for i in sub]) for l in ['1s','2s','3s','6s']]
            print(f"{k:10} "+" ".join(f"{v:7.3f}" for v in row))
    rep(idx,'ALL'); rep(straight,'STRAIGHT'); rep(turning,'TURNING')
    json.dump({'threshold':CURV,'n_straight':len(straight),'n_turning':len(turning),
               'PER':{k:{str(i):PER[k][i] for i in idx} for k in PER}},
              open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/res_x2_w2_stratified.json','w'))
    print("X2_DONE")
    dist.destroy_process_group()

if __name__=='__main__': main()
