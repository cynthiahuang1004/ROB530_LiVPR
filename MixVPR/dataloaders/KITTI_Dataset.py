import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

HARDCODE_PATH = 'C:/Users/User/Desktop/ROB530/clip-slcd/MixVPR/dataloaders/KITTI_dataset'


class KITTIVPRDataset(Dataset):
    def __init__(self, data_folder=HARDCODE_PATH, sequence='00', transform=None, threshold=5.0, imgs_per_place=4):
        """
        Args:
            data_folder: KITTI Odometry dataset 的根目錄
            sequence: 序列名稱 (例如 '00', '05')
            threshold: 判定為「正樣本」的距離門檻 (公尺)
            imgs_per_place: 訓練時每個地點採樣的影像張數
        """
        self.data_folder = data_folder
        self.sequence = sequence
        self.transform = transform
        self.imgs_per_place = imgs_per_place
        self.threshold = threshold

        # 1. 設定路徑
        self.img_dir = os.path.join(data_folder, sequence, 'image_2')
        self.pose_file = os.path.join(data_folder, 'poses', f'{sequence}.txt')

        # 2. 載入影像檔案清單
        self.filenames = sorted([os.path.join(self.img_dir, f) for f in os.listdir(self.img_dir) if f.endswith('.png')])
        
        # 3. 載入 Poses (計算距離用)
        # KITTI pose 格式是 3x4 矩陣的展開 (12 numbers)
        self.poses = np.loadtxt(self.pose_file)
        self.locations = self.poses[:, [3, 7, 11]]  # 提取 x, y, z 座標

        # 4. 預先計算正樣本索引 (類似你的 getPositives)
        self.all_positives = self._build_positive_map()

    def _build_positive_map(self):
        """根據歐幾里得距離找出每個 index 的鄰居"""
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(radius=self.threshold)
        knn.fit(self.locations)
        distances, indices = knn.radius_neighbors(self.locations)
        return indices

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 模擬 HPointLoc 的訓練邏輯：一個地點回傳多張影像
        # 我們把當前影像作為 Anchor，並從它的鄰居中隨機挑選正樣本
        
        pos_indices = self.all_positives[idx]
        
        # 如果鄰居不夠，就重複採樣；如果夠，就隨機挑
        if len(pos_indices) < self.imgs_per_place:
            selected_indices = np.random.choice(pos_indices, self.imgs_per_place, replace=True)
        else:
            selected_indices = np.random.choice(pos_indices, self.imgs_per_place, replace=False)

        place_list = []
        for s_idx in selected_indices:
            img = Image.open(self.filenames[s_idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            place_list.append(img.unsqueeze(0))

        # 拼接成 (imgs_per_place, C, H, W)
        place_tensor = torch.cat(place_list, dim=0)
        
        # label 用當前的 index 代表這個地點
        label = torch.tensor([idx]).repeat(self.imgs_per_place)

        return place_tensor, place_tensor, label # 這裡回傳兩次以符合你目前的介面