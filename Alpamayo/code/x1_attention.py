"""X1 — attention rollout: do the trajectory-prediction positions attend to VISION?

Layout of the LM input (length 1563):
  [ vision: 0..1535 | ego: 1536..1539 | fed traj tokens: 1540..1562 ]
The 24 trajectory-PREDICTION query positions are hidden[1539..1562] (1539 predicts token 0
from context; 1540+j predicts token j+1). For each such query we measure how much of its
(causal, row-normalized) attention mass lands on the vision / ego / prior-traj key groups.

If vision mass ≪ its count-uniform share, the trajectory tokens are effectively ignoring
vision at the attention level → the bottleneck is routing/adapter-capacity, not "vision useless".

Usage: python x1_attention.py --checkpoint <w2 full> --n 16
"""
import sys, argparse, pickle
import numpy as np, torch
sys.path.insert(0,'/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model, TRAJ_LEN
from dataset import build_scene_split, NuScenesVLADataset
from vision_live import encode_normalized_images
import inference as INF

DEVICE='cuda:0'
V0,V1=0,1536        # vision keys
E0,E1=1536,1540     # ego keys
CTX=1540
Q0=CTX-1            # first traj-prediction query = 1539
QN=CTX+TRAJ_LEN-1   # last+1 = 1563

@torch.no_grad()
def encode_live(visual, images):
    flat=images.to(DEVICE); pooled=encode_normalized_images(visual, flat)
    return pooled.reshape(1,6*256,-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints/_w2_full_fixed/alpamayo_best.pt')
    ap.add_argument('--n',type=int,default=16)
    a=ap.parse_args()

    with open(INF.TRAJECTORIES_PATH,'rb') as f: allt=pickle.load(f)
    _,val,_=build_scene_split(allt, INF.NUSCENES_ROOT)
    ds=NuScenesVLADataset(val, split='val', augment=False)
    model=load_model(device=DEVICE); ck=torch.load(a.checkpoint,map_location='cpu')
    model.load_state_dict(ck['model_state'],strict=False); model.zero_ego=False
    model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual=model.cosmos.model.visual; lm=model.cosmos.model.language_model
    # SDPA ignores output_attentions; force eager so real weights are returned. The
    # per-layer null-hook frees each [1,H,S,S] tensor immediately (peak ~17GB).
    lm.config._attn_implementation='eager'
    try: model.cosmos.config._attn_implementation='eager'
    except Exception: pass
    print(f"[x1] ckpt {a.checkpoint} (val_ade6={ck.get('val_ade6')})",flush=True)

    layers=lm.layers
    L=len(layers); vis_acc=np.zeros(L); ego_acc=np.zeros(L); traj_acc=np.zeros(L)
    cur={}   # layer_idx -> (vis,ego,traj) for current sample, filled by hooks

    def make_hook(li):
        def hook(module, inp, out):
            # out = (attn_output, attn_weights, [past_kv]); reduce weights then drop them
            if not isinstance(out, tuple) or len(out) < 2 or out[1] is None:
                return out
            aw = out[1]                                          # [B,H,S,S]
            # slice the 24 query rows FIRST (2MB), then cast — never float the full [H,S,S]
            q = aw[0][:, Q0:QN, :].float().mean(0)               # [24,S] mean over heads
            vmass = q[:, V0:V1].sum(-1); emass = q[:, E0:E1].sum(-1)
            tmass = (1.0 - vmass - emass).clamp(min=0)
            cur[li] = (float(vmass.mean()), float(emass.mean()), float(tmass.mean()))
            return (out[0], None) + tuple(out[2:])               # free the big tensor
        return hook

    handles=[layers[li].self_attn.register_forward_hook(make_hook(li)) for li in range(L)]
    nseen=0
    torch.set_grad_enabled(False)   # critical: no autograd graph retention across samples
    for i in range(a.n):
        item=ds[i]
        vt=encode_live(visual,item['images']).to(DEVICE,torch.float16)
        ego=item['ego_state'].to(DEVICE,torch.float32).unsqueeze(0)
        ctx=model._build_context(vt,ego)                         # [1,1540,3584]
        tt=item['traj_tokens'].to(DEVICE).unsqueeze(0)           # [1,24]
        gt=model.traj_embed(tt).to(ctx.dtype)
        lm_in=torch.cat([ctx, gt[:,:-1,:]],dim=1)                # [1,1563,3584]
        cur.clear()
        lm(inputs_embeds=lm_in, use_cache=False, output_attentions=True)
        for li in range(L):
            if li in cur:
                v,e,t=cur[li]; vis_acc[li]+=v; ego_acc[li]+=e; traj_acc[li]+=t
        nseen+=1
        torch.cuda.empty_cache()
        if i%4==0: print(f"  {i+1}/{a.n}",flush=True)
    for h in handles: h.remove()

    vis=vis_acc/nseen; ego=ego_acc/nseen; traj=traj_acc/nseen
    # count-uniform baseline (avg over the 24 queries): vision share if attention were uniform
    qs=np.arange(Q0,QN)
    base_v=np.mean(1536/(qs+1)); base_e=np.mean(4/(qs+1)); base_t=np.mean((qs-1539)/(qs+1))
    print("\n===== X1 ATTENTION MASS from the 24 traj-prediction positions =====")
    print(f"{'layer':6} {'vision':>8} {'ego':>8} {'prior-traj':>11}")
    for li in range(L):
        print(f"{li:6d} {vis[li]:8.4f} {ego[li]:8.4f} {traj[li]:11.4f}")
    print(f"{'POOLED':6} {vis.mean():8.4f} {ego.mean():8.4f} {traj.mean():11.4f}")
    print(f"\ncount-uniform baseline: vision={base_v:.4f} ego={base_e:.4f} prior-traj={base_t:.4f}")
    print(f"vision attention / uniform-share = {vis.mean()/base_v:.3f}  "
          f"(>1 = over-attended, <1 = under-attended)")
    print(f"ego attention / uniform-share    = {ego.mean()/base_e:.3f}")
    print("X1_DONE")

if __name__=='__main__': main()
