import os
import glob
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HPointLocHDF5Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        
        hdf5_files = sorted(glob.glob(os.path.join(root, '**', '*.hdf5'), recursive=True))
        print(f"Found {len(hdf5_files)} hdf5 files")
        
        self.samples = []
        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                for key in ['rgb', 'rgb_base']:
                    n = f[key].shape[0]
                    for i in range(n):
                        self.samples.append((hdf5_path, key, i))
        
        print(f"Total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hdf5_path, key, frame_idx = self.samples[idx]
        with h5py.File(hdf5_path, 'r') as f:
            img = f[key][frame_idx]
        
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img


class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', out_channels=256):
        super().__init__()
        self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino.eval()
        embed_dim = {'dinov2_vits14': 384, 'dinov2_vitb14': 768}[model_name]
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h_patches = H // 14
        w_patches = W // 14
        with torch.no_grad():
            patch_tokens = self.dino.forward_features(x)['x_norm_patchtokens']
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, -1, h_patches, w_patches)
        return self.proj(feat_map)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT = '/media/hdd/ihsuan/ClipVPR/HPointLoc/HPointLoc_all'
    
    transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = HPointLocHDF5Dataset(DATA_ROOT, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=False)
    
    model = DINOv2FeatureExtractor('dinov2_vitb14', out_channels=256).to(device)
    model.eval()
    
    all_features = []
    with torch.no_grad():
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            feat = model(imgs)  # [B, 256, 20, 20]
            all_features.append(feat.cpu())
            if i % 50 == 0:
                print(f"[{i}/{len(dataloader)}] batch shape: {feat.shape}")
    
    all_features = torch.cat(all_features, dim=0)
    print(f"Final shape: {all_features.shape}")
    torch.save(all_features, 'hpointloc_all_features.pt')
    print("Saved to hpointloc_all_features.pt")


if __name__ == '__main__':
    main()