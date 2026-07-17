"""Y2 — turning-scene demo: full-vision vs ego-only, side by side.

Two BEV panels (LEFT = full live-vision prediction, RIGHT = ego-only prediction), each
overlaying GROUND TRUTH (green), the model PREDICTION (red), and the CONSTANT-VELOCITY
baseline (grey dashed); the 6-camera grid below. Animates the turning scene -> GIF, so the
turn where vision helps is visible side by side.

Usage: python visualize_y2.py [--full CK] [--ego CK] [--out_dir DIR] [--max_frames N]
"""
import os, sys, argparse, pickle, math
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,'/home/dgx1user/Alpamayo-Kushal/Alpamayo/code')
from model import load_model
from tokenizer import TrajectoryTokenizer
from dataset import compute_ego_state, quat_to_yaw, NUSCENES_ROOT, build_scene_split
from vision_live import preprocess_image, CAMERAS
from ar_eval import ar_decode, unicycle_rollout, encode_live_one

DEVICE='cuda:0'; DT=0.5; N_STEPS=12
TRAJ='/home/dgx1user/Alpamayo-Kushal/Alpamayo/data/trajectories_full.pkl'
COSMOS='/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/cosmos_reason'
CKD='/home/dgx1user/Alpamayo-Kushal/Alpamayo/models/checkpoints'
GRID=[['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT'],
      ['CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']]

def ego_frame(pts, cx, cy, yaw0):
    d=np.asarray(pts,float)-np.array([cx,cy]); c,s=math.cos(-yaw0),math.sin(-yaw0)
    return (np.array([[c,-s],[s,c]])@d.T).T
def cv_ego(v0): return np.array([[v0*DT*(t+1),0.0] for t in range(N_STEPS)])
def load_cam(p,size=(300,168)):
    try: return np.asarray(Image.open(p).convert('RGB').resize(size))
    except Exception: return np.zeros((size[1],size[0],3),np.uint8)

def draw_bev(ax,gt,pred,cv,title,pred_label):
    def xy(p): return -p[:,1], p[:,0]
    for arr,color,style,lab in [(cv,'0.5','--','const-vel'),(gt,'#2ca02c','-','ground truth'),
                                (pred,'#d62728','-',pred_label)]:
        x,y=xy(arr); x=np.concatenate([[0],x]); y=np.concatenate([[0],y])
        ax.plot(x,y,style,color=color,lw=2.4,label=lab,marker='o',ms=3,zorder=3)
    ax.scatter([0],[0],c='k',s=70,marker='^',zorder=4)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.set_title(title,fontsize=11)
    ax.set_xlabel('lateral (m)'); ax.set_ylabel('forward (m)'); ax.legend(loc='upper right',fontsize=8)
    ax.set_xlim(-20,20); ax.set_ylim(-5,40)

@torch.no_grad()
def decode_all(model, visual, tok, trajs, zero_vision):
    """Return list of (accels,curvs) per frame for the loaded checkpoint."""
    out=[]
    for tr in trajs:
        imgs=torch.stack([preprocess_image(tr['cam_paths'][c],augment=False) for c in CAMERAS])
        vt=encode_live_one(visual,imgs,DEVICE)
        if zero_vision: vt=torch.zeros_like(vt)
        _,acc,cur=ar_decode(model,vt,compute_ego_state(tr),tok,DEVICE)
        out.append((acc,cur))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--full',default=f'{CKD}/_y1_full_turnw/alpamayo_best.pt')
    ap.add_argument('--ego', default=f'{CKD}/_y1_egoonly_turnw/alpamayo_best.pt')
    ap.add_argument('--out_dir',default='/home/dgx1user/Alpamayo-Kushal/Alpamayo/viz')
    ap.add_argument('--max_frames',type=int,default=40)
    a=ap.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    torch.set_grad_enabled(False)

    from nuscenes.nuscenes import NuScenes
    print('[y2] loading nuScenes...'); nusc=NuScenes(version='v1.0-trainval',dataroot=NUSCENES_ROOT,verbose=False)
    allt=pickle.load(open(TRAJ,'rb'))
    _,_,test=build_scene_split(allt,NUSCENES_ROOT)
    by_sample={t['sample_token']:t for t in test}
    st2sc={}
    for sc in nusc.scene:
        tk=sc['first_sample_token']
        while tk: st2sc[tk]=sc['token']; tk=nusc.get('sample',tk)['next']
    # Pick the scene where full-vision most beats ego-only on turning frames, using the
    # per-sample ADEs from the Y1 stratified eval (so the demo shows a scene where vision
    # genuinely helps, not just a high-curvature one).
    import json as _json
    scene=None
    try:
        per=_json.load(open('/home/dgx1user/Alpamayo-Kushal/Alpamayo/results/res_y1_turnw_stratified.json'))['PER']
        agg={}   # scene -> [sum(ego-full delta @6s), n_turn, sum(full_ade@6s)]
        for idx_s in per['full']:
            i=int(idx_s); t=test[i]
            mc=float(np.abs(np.array(t.get('future_curvatures',[0]))).max())
            if mc<=0.05: continue
            d=per['ego_only'][idx_s]['6s']-per['full'][idx_s]['6s']   # >0 => vision helps
            sc=st2sc.get(t['sample_token'])
            if sc is None: continue
            av=agg.setdefault(sc,[0.0,0,0.0]); av[0]+=d; av[1]+=1; av[2]+=per['full'][idx_s]['6s']
        # scenes with >=4 turning frames where full-vision itself is GOOD (mean ADE<5m),
        # then pick the largest vision advantage among them (genuine "vision helps", not
        # an ego-only blow-up).
        cand={s:v[0]/v[1] for s,v in agg.items() if v[1]>=4 and (v[2]/v[1])<5.0 and v[0]/v[1]>0}
        if not cand:   # relax if none qualify
            cand={s:v[0]/v[1] for s,v in agg.items() if v[1]>=4 and v[0]/v[1]>0}
        scene=max(cand,key=cand.get)
        fa=agg[scene][2]/agg[scene][1]
        print(f"[y2] selected turning scene {scene}: mean vision-advantage@6s={cand[scene]:+.2f} m, "
              f"full mean ADE@6s={fa:.2f} m over {agg[scene][1]} turning frames",flush=True)
    except Exception as e:
        print(f"[y2] scene-selection fallback ({e})",flush=True)
        best=max(test,key=lambda t:float(np.abs(np.array(t.get('future_curvatures',[0]))).max()))
        scene=st2sc[best['sample_token']]
    trajs=[]; tk=nusc.get('scene',scene)['first_sample_token']
    while tk:
        if tk in by_sample: trajs.append(by_sample[tk])
        tk=nusc.get('sample',tk)['next']
    trajs=trajs[:a.max_frames]
    print(f"[y2] turning scene {scene}: {len(trajs)} frames",flush=True)

    model=load_model(cosmos_path=COSMOS,device=DEVICE)
    model.cosmos.model.language_model.gradient_checkpointing_disable(); model.eval()
    visual=model.cosmos.model.visual; tok=TrajectoryTokenizer()

    # full-vision predictions
    ckf=torch.load(a.full,map_location='cpu'); model.load_state_dict(ckf['model_state'],strict=False); model.zero_ego=False
    print(f"[y2] full ckpt val_ade6={ckf.get('val_ade6')}",flush=True)
    full_preds=decode_all(model,visual,tok,trajs,zero_vision=False)
    # ego-only predictions
    cke=torch.load(a.ego,map_location='cpu'); model.load_state_dict(cke['model_state'],strict=False); model.zero_ego=False
    print(f"[y2] ego ckpt val_ade6={cke.get('val_ade6')}",flush=True)
    ego_preds=decode_all(model,visual,tok,trajs,zero_vision=True)

    frames=[]
    for i,tr in enumerate(trajs):
        v0=float(tr['future_speeds'][0]); cx,cy=tr['current_pose']['translation'][:2]
        yaw0=quat_to_yaw(tr['current_pose']['rotation'])
        gt=ego_frame(np.array(tr['future_positions'])[:N_STEPS],cx,cy,yaw0); cv=cv_ego(v0)
        pf,_=unicycle_rollout(full_preds[i][0],full_preds[i][1],v0,0.0)
        pe,_=unicycle_rollout(ego_preds[i][0],ego_preds[i][1],v0,0.0)
        adef=float(np.linalg.norm(pf-gt,axis=1).mean()); adee=float(np.linalg.norm(pe-gt,axis=1).mean())
        fig=plt.figure(figsize=(12,9))
        # two BEV panels on top
        axl=fig.add_axes([0.06,0.56,0.40,0.38]); axr=fig.add_axes([0.55,0.56,0.40,0.38])
        draw_bev(axl,gt,pf,cv,f"FULL live-vision   ADE@6s={adef:.2f} m",'full pred')
        draw_bev(axr,gt,pe,cv,f"EGO-ONLY   ADE@6s={adee:.2f} m",'ego pred')
        fig.suptitle(f"Turning scene — frame {i+1}/{len(trajs)}   |   v0={v0:.1f} m/s   "
                     f"(vision {'helps' if adef<adee else 'ties/loses'}: {adee-adef:+.2f} m)",fontsize=13)
        # 6-camera grid below
        for r in range(2):
            for cc in range(3):
                ax=fig.add_axes([0.05+cc*0.315, 0.26-r*0.225, 0.29, 0.205])
                ax.imshow(load_cam(tr['cam_paths'][GRID[r][cc]])); ax.set_title(GRID[r][cc],fontsize=7); ax.axis('off')
        fig.canvas.draw(); buf=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy(); plt.close(fig)
        frames.append(buf)
        if i%5==0: print(f"  frame {i+1}/{len(trajs)}",flush=True)
    gif=os.path.join(a.out_dir,'demo_turning_full_vs_ego.gif')
    ims=[Image.fromarray(f) for f in frames]
    ims[0].save(gif,save_all=True,append_images=ims[1:],duration=450,loop=0)
    print(f"[y2] saved {gif} ({len(frames)} frames)\nY2_DONE")

if __name__=='__main__': main()
