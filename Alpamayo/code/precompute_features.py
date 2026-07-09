"""
precompute_features.py
Precomputes CLIP ViT perception tokens + ego tokens for all trajectories.
Saves to disk so training doesn't recompute them every batch.
"""

import torch
import numpy as np
import pickle
from tqdm import tqdm
import sys
sys.path.append('/home/drive1/Alpamayo/data')

from camera_loader import CameraLoader
from vision_encoder import VisionEncoder
from ego_encoder import EgoEncoder

TRAJ_PATH    = '/home/drive1/Alpamayo/data/trajectories_full.pkl'
FEATURES_PATH = '/home/drive1/Alpamayo/data/perception_features.pkl'
DEVICE = torch.device('cuda:0')

if __name__ == '__main__':
    # Load trajectories
    with open(TRAJ_PATH, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trajectories")

    # Initialize models
    cam_loader    = CameraLoader()
    vision_enc    = VisionEncoder()
    ego_enc       = EgoEncoder()

    features = []
    failed   = 0

    for i, traj in enumerate(tqdm(trajectories, desc='Extracting features')):
        try:
            sample_token = traj['sample_token']

            # Load 6 camera images
            images = cam_loader.get_sample_cameras(sample_token)

            # Get perception tokens [294, 768]
            with torch.no_grad():
                perception_tokens = vision_enc.get_perception_tokens(images)  # [294, 768]

            # Get 2s ego history
            history = cam_loader.get_2s_history(sample_token)
            ego_tokens, _ = ego_enc.encode(history)  # [4, 768]

            # Concatenate [298, 768]
            combined = torch.cat([
                perception_tokens,
                ego_tokens
            ], dim=0).cpu().to(torch.float16)  # save as float16 to save space

            features.append(combined)

        except Exception as e:
            print(f"Failed on trajectory {i}: {e}")
            # Use zeros as fallback
            features.append(torch.zeros(298, 768, dtype=torch.float16))
            failed += 1

        # Save checkpoint every 1000
        if (i + 1) % 1000 == 0:
            print(f"  Checkpoint at {i+1} | Failed so far: {failed}")
            with open(FEATURES_PATH, 'wb') as f:
                pickle.dump(features, f)

    # Final save
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(features, f)

    print(f"\nDone! {len(features)} feature vectors saved")
    print(f"Failed: {failed}")
    print(f"Shape per sample: {features[0].shape}")
    print(f"Saved to {FEATURES_PATH}")