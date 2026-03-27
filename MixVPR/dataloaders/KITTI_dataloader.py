import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
from prettytable import PrettyTable
from MixVPR.dataloaders.KITTI_Dataset import KITTIVPRDataset

class KITTIValidationModule(pl.LightningDataModule):
    def __init__(self,
                 data_folder='C:/Users/User/Desktop/ROB530/clip-slcd/MixVPR/dataloaders/KITTI_dataset',
                 batch_size=32,
                 image_size=(480, 640),
                 num_workers=4,
                 # 這裡列出所有你想測試的序列 (00-10 是有 GT 的)
                 val_seqs=['00'], 
                 mean_std={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.val_seqs = val_seqs

        self.val_set_names = [f"KITTI_{s}" for s in val_seqs]
        
        # 驗證不需要 RandAugment，保持圖像真實性
        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
        ])

    def setup(self, stage=None):
        # 僅在 validate 或 test 階段初始化
        if stage in ['validate', 'test', None]:
            self.val_datasets = []
            for seq in self.val_seqs:
                # 注意：imgs_per_place 設為 1，因為驗證只需單圖特徵
                self.val_datasets.append(
                    KITTIVPRDataset(
                        data_folder=self.hparams.data_folder,
                        sequence=seq,
                        transform=self.valid_transform,
                        imgs_per_place=1,
                        threshold=5.0 # 5米內的視為正樣本
                    )
                )

    def val_dataloader(self):
        # 回傳一個 DataLoader 列表
        return [DataLoader(
            ds, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=False, 
            pin_memory=True
        ) for ds in self.val_datasets]

    def test_dataloader(self):
        return self.val_dataloader() # test 邏輯通常與 val 一致

    def print_stats(self):
        table = PrettyTable(['Sequence', 'Total Images', 'GPS Threshold'])
        for ds in self.val_datasets:
            table.add_row([ds.sequence, len(ds), f"{ds.threshold}m"])
        print("\n" + table.get_string(title="KITTI Zero-shot Evaluation Sets") + "\n")