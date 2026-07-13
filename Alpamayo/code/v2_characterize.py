"""V2: (a) zero-both emits a constant 24-token sequence -> count unique across N samples.
(b) [proven by code inspection] rollout seeds v0/yaw0 from GT, independent of zeroing.
(c) naive constant-velocity + constant-turn-rate baselines on the full test set,
    with the V1-fixed frame/seeding. No model needed for (c)."""
import sys, os, pickle, argparse
import numpy as np, torch
sys.path.insert(0,'/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model, TRAJ_LEN, TEXT_DIM
from dataset import build_scene_split, NuScenesVLADataset
from tokenizer import TrajectoryTokenizer
from ar_eval import ar_decode, unicycle_rollout, encode_live_one, HORIZONS, N_STEPS
import inference as INF

DEVICE='cuda:0'
ZB='/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints/_zeroboth_run_jul12/alpamayo_best_e1_val2.0392.pt'

def cv_baselines(test):
    """Naive baselines, no model. Fixed frame: origin, gt_local = future - translation."""
    res={}
    for name in ['CV_straight','CTR_const_yawrate']:
        ade={h:[] for h in HORIZONS.values()}; fde={h:[] for h in HORIZONS.values()}
        from dataset import compute_ego_state
        for traj in test:
            e=compute_ego_state(traj)
            v0=float(traj['future_speeds'][0]); yaw0=float(e[3,1]); yr=float(e[3,2])
            x=y=0.0; yaw=yaw0; P=[]
            for t in range(N_STEPS):
                if name=='CTR_const_yawrate': yaw=yaw+yr*0.5
                x+=v0*np.cos(yaw)*0.5; y+=v0*np.sin(yaw)*0.5; P.append([x,y])
            P=np.array(P)
            gt=np.array(traj['future_positions'])[:N_STEPS]-np.array(traj['current_pose']['translation'][:2])
            for step,lab in HORIZONS.items():
                er=np.linalg.norm(P[:step]-gt[:step],axis=1); ade[lab].append(er.mean()); fde[lab].append(er[-1])
        res[name]={'ade':{h:{'mean':float(np.mean(ade[h])),'median':float(np.median(ade[h]))} for h in ade},
                   'fde':{h:{'mean':float(np.mean(fde[h])),'median':float(np.median(fde[h]))} for h in fde}}
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n_seq',type=int,default=400); a=ap.parse_args()
    with open(INF.TRAJECTORIES_PATH,'rb') as f: allt=pickle.load(f)
    _,_,test=build_scene_split(allt, INF.NUSCENES_ROOT)
    tok=TrajectoryTokenizer()

    # (a) zero-both constant-sequence check
    model=load_model(device=DEVICE)
    ck=torch.load(ZB,map_location='cpu'); model.load_state_dict(ck['model_state'],strict=False)
    model.zero_ego=True; model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual=model.cosmos.model.visual
    ds=NuScenesVLADataset(test[:a.n_seq], split='test', augment=False)
    seqs=set(); first=None
    for i in range(a.n_seq):
        item=ds[i]
        vt=encode_live_one(visual,item['images'],DEVICE); vt=torch.zeros_like(vt)  # zero_vision
        toks,acc,cur=ar_decode(model,vt,item['ego_state'],tok,DEVICE)
        seqs.add(tuple(toks))
        if first is None: first=(toks,acc,cur)
        if i%100==0: print(f"  decoded {i}/{a.n_seq}, unique so far={len(seqs)}",flush=True)
    print(f"\n[V2a] zero-both: {len(seqs)} UNIQUE 24-token sequence(s) across {a.n_seq} test samples")
    print(f"[V2a] the sequence = {list(first[0])}")
    print(f"[V2a] decoded accels = {[round(float(x),3) for x in first[1]]}")
    print(f"[V2a] decoded curvs  = {[round(float(x),4) for x in first[2]]}")

    # (c) naive baselines on full test
    print(f"\n[V2c] computing naive baselines on full test n={len(test)} ...",flush=True)
    res=cv_baselines(test)
    for name,r in res.items():
        print(f"\n[V2c] {name}:")
        for h in ['1s','2s','3s','6s']:
            print(f"   ADE {h}: mean={r['ade'][h]['mean']:.4f} median={r['ade'][h]['median']:.4f} | FDE mean={r['fde'][h]['mean']:.4f}")
    import json
    json.dump(res, open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/res_v2_cv_baselines.json','w'), indent=2)
    print("V2_DONE")

if __name__=='__main__': main()
