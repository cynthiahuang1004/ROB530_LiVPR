import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T
from MixVPR.dataloaders.NordlandDataset import NordlandDataset

class NordlandDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 img_per_place=2,
                 shuffle_all=True,
                 image_size=(322, 322),
                 num_workers=4,
                 val_set_names=['nordland'],
                 query_season='winter',
                 reference_season='summer'):

        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.query_season = query_season
        self.reference_season = reference_season

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.train_dataset = NordlandDataset(
            query_season=self.query_season,
            reference_season=self.reference_season,
            transform=self.transform,
            split='train',
            img_per_place=self.img_per_place
        )
        self.val_datasets = []
        for name in self.val_set_names:
            val_ds = NordlandDataset(
                query_season=self.query_season,
                reference_season=self.reference_season,
                transform=self.transform,
                split='val'
            )
            self.val_datasets.append(val_ds)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle_all,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return [DataLoader(ds,
                           batch_size=self.batch_size,
                           shuffle=False,
                           num_workers=self.num_workers,
                           pin_memory=True)
                for ds in self.val_datasets]