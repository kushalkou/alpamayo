"""
camera_loader.py
Loads all 6 synchronized camera views for a given nuScenes sample.
"""

import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes

# The 6 cameras in order: front group, then back group
CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT', 
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]

class CameraLoader:
    def __init__(self, dataroot='/home/drive1/Alpamayo/nuscenes_full', version='v1.0-trainval'):
        print("Loading NuScenes...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.dataroot = dataroot
        print(f"Ready. {len(self.nusc.scene)} scenes, {len(self.nusc.sample)} samples.")

    def get_sample_cameras(self, sample_token):
        """
        Given a sample token, returns all 6 camera images as PIL Images.
        Returns a dict: { camera_name -> PIL.Image }
        """
        sample = self.nusc.get('sample', sample_token)
        images = {}

        for cam in CAMERAS:
            # Get the camera sample data token
            cam_token = sample['data'][cam]
            cam_data = self.nusc.get('sample_data', cam_token)

            # Build full path and load image
            img_path = f"{self.dataroot}/{cam_data['filename']}"
            images[cam] = Image.open(img_path).convert('RGB')

        return images

    def get_scene_samples(self, scene_index=0):
        """
        Returns all sample tokens for a given scene index.
        """
        scene = self.nusc.scene[scene_index]
        samples = []
        
        # Walk the linked list of samples in this scene
        sample_token = scene['first_sample_token']
        while sample_token:
            samples.append(sample_token)
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample['next']  # empty string at end of scene
        
        return samples


    def get_ego_state(self, sample_token):
        """
        Returns ego-state for a given sample:
        speed, yaw_rate, acceleration (from CAN bus via ego_pose)
        """
        sample = self.nusc.get('sample', sample_token)
        
        # Get ego pose at this timestamp
        # We use CAM_FRONT as the reference sensor for ego pose
        cam_token = sample['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_token)
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

        # Extract translation and rotation
        translation = ego_pose['translation']   # [x, y, z]
        rotation = ego_pose['rotation']          # quaternion [w, x, y, z]

        return {
            'translation': np.array(translation),
            'rotation': np.array(rotation),
            'timestamp': cam_data['timestamp']
        }

    def get_2s_history(self, sample_token):
        """
        Returns the 2-second history window of ego poses leading up to this sample.
        nuScenes samples are at 2Hz, so 2 seconds = 4 samples back.
        """
        history = []
        token = sample_token

        for _ in range(4):  # 4 steps back = 2 seconds at 2Hz
            ego = self.get_ego_state(token)
            history.append(ego)
            sample = self.nusc.get('sample', token)
            if sample['prev'] == '':
                break
            token = sample['prev']

        history.reverse()  # chronological order
        return history


if __name__ == '__main__':
    loader = CameraLoader()

    scene_samples = loader.get_scene_samples(scene_index=0)
    print(f"Scene 0 has {len(scene_samples)} samples")

    # Test cameras
    first_sample_token = scene_samples[0]
    images = loader.get_sample_cameras(first_sample_token)
    print(f"\nLoaded {len(images)} camera views:")
    for cam_name, img in images.items():
        print(f"  {cam_name}: {img.size} pixels")

    # Test ego state
    ego = loader.get_ego_state(first_sample_token)
    print(f"\nEgo state at first sample:")
    print(f"  Translation (x,y,z): {ego['translation']}")
    print(f"  Rotation (quaternion): {ego['rotation']}")

    # Test 2s history
    sample_token_mid = scene_samples[5]  # pick one with history
    history = loader.get_2s_history(sample_token_mid)
    print(f"\n2-second history window ({len(history)} poses):")
    for i, h in enumerate(history):
        print(f"  t-{len(history)-1-i}: x={h['translation'][0]:.2f}, y={h['translation'][1]:.2f}")