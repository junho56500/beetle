import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

class NuScenesBEVDataset(Dataset):
    def __init__(self, nusc, layer_names=['lane', 'road_segment']):
        self.nusc = nusc
        self.samples = nusc.sample
        self.cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                          'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        imgs = []
        mats = [] # To store Intrinsics and Extrinsics

        for cam_name in self.cam_names:
            cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
            
            # 1. Load Image
            img = Image.open(f"/content/data/sets/nuscenes/{cam_data['filename']}")
            imgs.append(self.transform(img))

            # 2. Get Geometry (Intrinsics & Extrinsics)
            sensor_pos = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
            # Intrinsic Matrix (K)
            intrinsic = torch.tensor(sensor_pos['camera_intrinsic'])
            
            # Extrinsic Matrix (Camera to Ego Car)
            rotation = torch.tensor(sensor_pos['rotation']) # Quaternion
            translation = torch.tensor(sensor_pos['translation'])
            
            # For simplicity, we wrap these into a dict
            mats.append({'K': intrinsic, 'T': translation, 'R': rotation})

        return torch.stack(imgs), mats

# --- Training Initialization ---
nusc = NuScenes(version='v1.0-mini', dataroot='/content/data/sets/nuscenes', verbose=False)
dataset = NuScenesBEVDataset(nusc)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)

# Example Loop

images, mats = dataset[0]
img_to_show = images[0].permute(1, 2, 0).numpy()
plt.imshow(img_to_show)

for batch in loader:
    images = torch.stack([item[0] for item in batch]) # [B, 6, 3, 224, 224]
    print(f"Loaded batch of images: {images.shape}")
    break