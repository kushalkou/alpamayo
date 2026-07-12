"""P2 sanity: AR val-ADE on the full-live checkpoint over (a) the seed-fixed 400 subset
and (b) the full val set (3572). They should be close, validating the selection metric.

Launch: torchrun --nproc_per_node=8 sanity_valade.py
"""
import os, sys, pickle
import numpy as np, torch
import torch.distributed as dist
sys.path.insert(0, '/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from dataset import build_scene_split, NuScenesVLADataset
from tokenizer import TrajectoryTokenizer
from ar_eval import compute_val_ade, fixed_val_indices

CKPT='/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints/_livevision_run_jul9/alpamayo_best_e1_val2.0806.pt'
TRAJ='/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/trajectories_full.pkl'
ROOT='/home/dgx1user/Alpamayo-Kushal/Alpamayo/nuscenes'

def main():
    dist.init_process_group(backend='nccl')
    lr = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr)
    device=f'cuda:{lr}'; ws=dist.get_world_size()
    main0 = (lr==0)

    model=load_model(device=device)
    ck=torch.load(CKPT,map_location='cpu')
    model.load_state_dict(ck['model_state'],strict=False)
    model.zero_ego=False
    model.cosmos.model.language_model.gradient_checkpointing_disable()
    visual=model.cosmos.model.visual
    tok=TrajectoryTokenizer()

    with open(TRAJ,'rb') as f: allt=pickle.load(f)
    _,val_trajs,_=build_scene_split(allt, ROOT)
    val_ds=NuScenesVLADataset(val_trajs, split='val', augment=False)
    n=len(val_ds)
    if main0: print(f"[sanity] val n={n}, world_size={ws}", flush=True)

    sub_idx=fixed_val_indices(n, k=400, seed=1234)
    s=compute_val_ade(model,val_ds,val_trajs,sub_idx,device,visual,tokenizer=tok,
                      world_size=ws,rank=lr,zero_vision=False,zero_ego=False)
    if main0:
        a=s['ade']
        print(f"\n[400-SUBSET] n={s['n']}")
        for h in ['1s','2s','3s','6s']:
            print(f"  ADE {h}: mean={a[h]['mean']:.4f} median={a[h]['median']:.4f}")

    full_idx=np.arange(n)
    s2=compute_val_ade(model,val_ds,val_trajs,full_idx,device,visual,tokenizer=tok,
                       world_size=ws,rank=lr,zero_vision=False,zero_ego=False)
    if main0:
        a=s2['ade']
        print(f"\n[FULL-VAL] n={s2['n']}")
        for h in ['1s','2s','3s','6s']:
            print(f"  ADE {h}: mean={a[h]['mean']:.4f} median={a[h]['median']:.4f}")
        d6=s['ade']['6s']['median']; f6=s2['ade']['6s']['median']
        print(f"\n[COMPARE] median ADE@6s  subset={d6:.4f}  full={f6:.4f}  "
              f"|Δ|={abs(d6-f6):.4f}  ({100*abs(d6-f6)/f6:.1f}%)")
        print("SANITY_DONE")
    dist.destroy_process_group()

if __name__=='__main__': main()
