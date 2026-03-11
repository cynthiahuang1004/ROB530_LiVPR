import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

HARDCODE_PATH = '/media/hdd/ihsuan/ClipVPR/Nordland/data'
FILTERED_LIST = '/media/hdd/ihsuan/ClipVPR/Nordland/dataset_imageNames/nordland_imageNames.txt'

class NordlandDataset(Dataset):
    def __init__(self,
                 data_folder=HARDCODE_PATH,
                 query_season='winter',
                 reference_season='summer',
                 transform=None,
                 split='train',
                 img_per_place=2,
                 use_filtered=True):

        self.data_folder = data_folder
        self.query_season = query_season
        self.reference_season = reference_season
        self.transform = transform
        self.split = split
        self.img_per_place = img_per_place

        # 讀取過濾後的 frame index list
        if use_filtered and os.path.exists(FILTERED_LIST):
            with open(FILTERED_LIST, 'r') as f:
                # 檔案內容是 image name，例如 images-00285
                names = [line.strip() for line in f.readlines()]
            self.frame_ids = sorted([n.replace('.png', '') for n in names])
        else:
            # 直接掃 summer 資料夾
            all_files = sorted(os.listdir(os.path.join(data_folder, reference_season)))
            self.frame_ids = [f.replace('.png', '') for f in all_files if f.endswith('.png')]

        self.num_frames = len(self.frame_ids)

        if self.split == 'val':
            # val: references = summer, queries = winter
            # 每個 frame 是一個獨立的 place
            self.references = np.arange(self.num_frames)       # index 0..N-1 對應 reference
            self.queries = np.arange(self.num_frames)          # index 0..N-1 對應 query
            # positives[i] = [i]，因為同index就是同地點
            self.positives = [np.array([i]) for i in range(self.num_frames)]
            self.num_references = self.num_frames

    def __len__(self):
        if self.split == 'train':
            return self.num_frames  # 每個frame當一個place
        elif self.split == 'val':
            # reference + query 都放進來，總長度 = 2 * num_frames
            return 2 * self.num_frames

    def __getitem__(self, idx):
        if self.split == 'train':
            # 每個 place 回傳 img_per_place 張（從不同季節各取一張）
            frame_id = self.frame_ids[idx]
            seasons = [self.reference_season, self.query_season]
            imgs = []
            for season in seasons:
                path = os.path.join(self.data_folder, season, f'{frame_id}.png')
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))
            place = torch.cat(imgs, dim=0)  # [2, C, H, W]
            label = torch.tensor([idx, idx])  # 同一place
            return place, place, label

        elif self.split == 'val':
            # 前 num_frames 個 idx 是 reference（summer）
            # 後 num_frames 個 idx 是 query（winter）
            if idx < self.num_frames:
                season = self.reference_season
                frame_id = self.frame_ids[idx]
            else:
                season = self.query_season
                frame_id = self.frame_ids[idx - self.num_frames]

            path = os.path.join(self.data_folder, season, f'{frame_id}.png')
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img, -1