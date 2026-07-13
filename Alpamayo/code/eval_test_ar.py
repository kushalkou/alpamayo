"""Sharded (8-GPU) full-test autoregressive eval. Same decode/rollout/frame as
inference.py, but sharded across DDP ranks for speed. Reports ADE/FDE per horizon +
AR token-acc / seq-acc. Use for P3/P4 checkpoints.

Launch: torchrun --nproc_per_node=8 eval_test_ar.py --checkpoint CK [--zero_vision] [--zero_ego] --out res_x.json
"""
import os, sys, json, pickle, argparse
import numpy as np, torch
import torch.distributed as dist
sys.path.insert(0,'/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset
from tokenizer import TrajectoryTokenizer
from ar_eval import ar_decode, unicycle_rollout, encode_live_one, HORIZONS, N_STEPS

TRAJ='/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/trajectories_full.pkl'
ROOT='/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'
OUT_DIR='/home/dgx1user/Alpamayo-Kushal/Alpamayo/results'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--zero_vision',action='store_true')
    ap.add_argument('--zero_ego',action='store_true')
    ap.add_argument('--out',default='res_test_ar.json')
    a=ap.parse_args()

    dist.init_process_group(backend='nccl')
    lr=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device=f'cuda:{lr}'; ws=dist.get_world_size(); main0=(lr==0)

    model=load_model(device=device)
    ck=torch.load(a.checkpoint,map_location='cpu')
    model.load_state_dict(ck['model_state'],strict=False)
    model.zero_ego=a.zero_ego
    model.cosmos.model.language_model.gradient_checkpointing_disable()
    model.eval()
    visual=model.cosmos.model.visual; tok=TrajectoryTokenizer()

    with open(TRAJ,'rb') as f: allt=pickle.load(f)
    _,_,test=build_scene_split(allt, ROOT)
    ds=NuScenesVLADataset(test, split='test', augment=False)
    n=len(test)
    if main0: print(f"[eval] test n={n} ws={ws} ckpt={a.checkpoint} zv={a.zero_vision} ze={a.zero_ego}",flush=True)

    my=list(range(lr,n,ws))
    ade={h:[] for h in HORIZONS.values()}; fde={h:[] for h in HORIZONS.values()}
    tc=tt=sc=0
    for c,i in enumerate(my):
        item=ds[i]; traj=test[i]
        vt=encode_live_one(visual,item['images'],device)
        if a.zero_vision: vt=torch.zeros_like(vt)
        toks,accels,curvs=ar_decode(model,vt,item['ego_state'],tok,device)
        ego=item['ego_state']; v0=float(ego[3,0]); yaw0=float(ego[3,1])
        pred,_=unicycle_rollout(accels,curvs,v0,yaw0)
        gt=np.array(traj['future_positions'])[:N_STEPS]
        cx,cy=traj['current_pose']['translation'][0],traj['current_pose']['translation'][1]
        gtl=gt-np.array([cx,cy])
        for step,lab in HORIZONS.items():
            if step<=len(pred) and step<=len(gtl):
                e=np.linalg.norm(pred[:step]-gtl[:step],axis=1)
                ade[lab].append(float(e.mean())); fde[lab].append(float(e[-1]))
        gtok=item['traj_tokens'].tolist()
        for g,p in zip(gtok,toks): tc+=int(g==p); tt+=1
        if gtok==list(toks): sc+=1
        if main0 and c%50==0: print(f"  rank0 {c}/{len(my)}",flush=True)

    payload={'ade':ade,'fde':fde,'tc':tc,'tt':tt,'sc':sc,'nlocal':len(my)}
    gathered=[None]*ws
    dist.all_gather_object(gathered,payload)
    if main0:
        A={h:[] for h in HORIZONS.values()}; F={h:[] for h in HORIZONS.values()}
        TC=TT=SC=NL=0
        for g in gathered:
            for h in A: A[h].extend(g['ade'][h]); F[h].extend(g['fde'][h])
            TC+=g['tc']; TT+=g['tt']; SC+=g['sc']; NL+=g['nlocal']
        res={'ade':{},'fde':{},'n':len(A['6s']),
             'token_accuracy':TC/max(TT,1),'sequence_accuracy':SC/max(NL,1),
             'checkpoint':a.checkpoint,'zero_vision':a.zero_vision,'zero_ego':a.zero_ego}
        print("\n=== TEST AR EVAL ===")
        for h in ['1s','2s','3s','6s']:
            res['ade'][h]={'mean':float(np.mean(A[h])),'median':float(np.median(A[h]))}
            res['fde'][h]={'mean':float(np.mean(F[h])),'median':float(np.median(F[h]))}
            print(f"  ADE {h}: mean={res['ade'][h]['mean']:.4f} median={res['ade'][h]['median']:.4f} | "
                  f"FDE mean={res['fde'][h]['mean']:.4f}")
        print(f"  tok_acc={100*res['token_accuracy']:.2f}%  seq_acc={100*res['sequence_accuracy']:.2f}%  n={res['n']}")
        os.makedirs(OUT_DIR,exist_ok=True)
        with open(os.path.join(OUT_DIR,a.out),'w') as f: json.dump(res,f,indent=2)
        print(f"[eval] saved -> {a.out}\nEVAL_DONE")
    dist.destroy_process_group()

if __name__=='__main__': main()
