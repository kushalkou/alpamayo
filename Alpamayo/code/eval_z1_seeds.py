"""Z1 — seed robustness. Evaluate full-vision and ego-only across 3 seeds (42=Y1, 123, 2024),
report OVERALL and TURNING (max|GT curv|>0.05) ADE@6s per model per seed. Sharded 8-GPU.

Launch: torchrun --nproc_per_node=8 --master_port=29567 eval_z1_seeds.py
"""
import os, sys, json, pickle, time
import numpy as np, torch
import torch.distributed as dist
sys.path.insert(0,'/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset, compute_ego_state
from tokenizer import TrajectoryTokenizer
from ar_eval import ar_decode, unicycle_rollout, encode_live_one, N_STEPS
import inference as INF

CK='/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
MODELS=[
    ('full_s42',   f'{CK}/_y1_full_turnw/alpamayo_best.pt',    False),
    ('ego_s42',    f'{CK}/_y1_egoonly_turnw/alpamayo_best.pt', True),
    ('full_s123',  f'{CK}/_z1_full_s123/alpamayo_best.pt',     False),
    ('ego_s123',   f'{CK}/_z1_ego_s123/alpamayo_best.pt',      True),
    ('full_s2024', f'{CK}/_z1_full_s2024/alpamayo_best.pt',    False),
    ('ego_s2024',  f'{CK}/_z1_ego_s2024/alpamayo_best.pt',     True),
]
CURV=0.05; SHARD='/tmp/claude-1000/z1_shards'

def gtl(t): return np.array(t['future_positions'])[:N_STEPS]-np.array(t['current_pose']['translation'][:2])
def maxcurv(t):
    c=np.abs(np.array(t.get('future_curvatures',[0]))); return float(c.max()) if len(c) else 0.0

def main():
    dist.init_process_group(backend='nccl')
    lr=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device=f'cuda:{lr}'; ws=dist.get_world_size(); m0=(lr==0); os.makedirs(SHARD,exist_ok=True)
    with open(INF.TRAJECTORIES_PATH,'rb') as f: allt=pickle.load(f)
    _,_,test=build_scene_split(allt, INF.NUSCENES_ROOT)
    ds=NuScenesVLADataset(test, split='test', augment=False); tok=TrajectoryTokenizer()
    model=load_model(device=device); model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual=model.cosmos.model.visual; my=list(range(lr,len(test),ws)); torch.set_grad_enabled(False)
    per={nm:{} for nm,_,_ in MODELS}
    for nm,path,zv in MODELS:
        if not os.path.exists(path):
            if m0: print(f"[z1] MISSING {path}",flush=True)
            continue
        ck=torch.load(path,map_location='cpu'); model.load_state_dict(ck['model_state'],strict=False); model.zero_ego=False
        for c,i in enumerate(my):
            t=test[i]; item=ds[i]; vt=encode_live_one(visual,item['images'],device)
            if zv: vt=torch.zeros_like(vt)
            ego=item['ego_state']; _,acc,cur=ar_decode(model,vt,ego,tok,device)
            v0=float(t['future_speeds'][0]); yaw0=float(ego[3,1])
            pred,_=unicycle_rollout(acc,cur,v0,yaw0); e=np.linalg.norm(pred-gtl(t),axis=1).mean()
            per[nm][i]=float(e)
            if m0 and nm=='full_s123' and c%80==0: print(f"  {nm} {c}/{len(my)}",flush=True)
    with open(os.path.join(SHARD,f's_{lr}.pkl.tmp'),'wb') as f: pickle.dump({'per':per,'curv':{i:maxcurv(test[i]) for i in my}},f)
    os.replace(os.path.join(SHARD,f's_{lr}.pkl.tmp'),os.path.join(SHARD,f's_{lr}.pkl'))
    if not m0: dist.destroy_process_group(); return
    while sum(os.path.exists(os.path.join(SHARD,f's_{r}.pkl')) for r in range(ws))<ws: time.sleep(5)
    g=[pickle.load(open(os.path.join(SHARD,f's_{r}.pkl'),'rb')) for r in range(ws)]
    PER={nm:{} for nm,_,_ in MODELS}; CURVd={}
    for gg in g:
        for nm in PER: PER[nm].update(gg['per'].get(nm,{}))
        CURVd.update(gg['curv'])
    idx=[i for i in CURVd if i in PER['full_s42']]
    turn=[i for i in idx if CURVd[i]>CURV]
    def ade(nm,sub):
        vals=[PER[nm][i] for i in sub if i in PER[nm]]; return float(np.mean(vals)) if vals else float('nan')
    print(f"\n=== Z1 SEED ROBUSTNESS (turn-weighted; ADE@6s mean) ===  n_all={len(idx)} n_turn={len(turn)}")
    print(f"{'seed':8} {'full OVERALL':>13} {'ego OVERALL':>12} {'full TURN':>10} {'ego TURN':>9} {'full-ego OVR':>13}")
    rows={'full':{'overall':[],'turn':[]}, 'ego':{'overall':[],'turn':[]}}
    for s in ['42','123','2024']:
        fo=ade(f'full_s{s}',idx); eo=ade(f'ego_s{s}',idx); ft=ade(f'full_s{s}',turn); et=ade(f'ego_s{s}',turn)
        print(f"{s:8} {fo:13.3f} {eo:12.3f} {ft:10.3f} {et:9.3f} {fo-eo:+13.3f}")
        rows['full']['overall'].append(fo); rows['ego']['overall'].append(eo)
        rows['full']['turn'].append(ft); rows['ego']['turn'].append(et)
    import numpy as _np
    def ms(x): x=[v for v in x if v==v]; return (_np.mean(x), _np.std(x), min(x), max(x)) if x else (float('nan'),)*4
    for m in ['full','ego']:
        mo=ms(rows[m]['overall']); mt=ms(rows[m]['turn'])
        print(f"[{m:4}] OVERALL mean={mo[0]:.3f} sd={mo[1]:.3f} [{mo[2]:.3f},{mo[3]:.3f}]  "
              f"TURN mean={mt[0]:.3f} sd={mt[1]:.3f} [{mt[2]:.3f},{mt[3]:.3f}]")
    deltas=[rows['full']['overall'][k]-rows['ego']['overall'][k] for k in range(3)]
    wins=sum(1 for d in deltas if d<0)   # full beats ego (lower ADE)
    print(f"[VERDICT] full-vision beats ego-only OVERALL in {wins}/3 seeds (deltas full-ego = "
          f"{[round(d,3) for d in deltas]}; negative = vision wins)")
    json.dump({'PER':{nm:{str(i):PER[nm][i] for i in PER[nm]} for nm in PER},
               'curv':{str(i):CURVd[i] for i in idx}},
              open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/res_z1_seeds.json','w'))
    print("Z1_EVAL_DONE"); dist.destroy_process_group()

if __name__=='__main__': main()
