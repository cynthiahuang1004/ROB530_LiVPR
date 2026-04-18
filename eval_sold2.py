# eval_sold2.py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import argparse
import sys
import os
import pandas as pd
import glob
from PIL import Image
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, '.')
from main_GSV_sold2 import VPRModel
import MixVPR.utils as utils
from MixVPR.dataloaders.NordlandDataset import NordlandDataset
from MixVPR.dataloaders import PittsburgDataset

@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    model.eval()
    all_descriptors = []
    for batch in tqdm(dataloader):
        imgs = batch[0].to(device)
        desc = model(imgs)
        all_descriptors.append(desc.cpu())
    return torch.cat(all_descriptors, dim=0)

def get_transform(image_size=(322, 322)):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(ckpt_path, device='cuda'):
    model = VPRModel(
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4],
        agg_arch='MixVPR',
        agg_config={
            'in_channels': 256 + 256 + 1024,  # SOLD2: DinoV2 + SOLD2 + ResNet50
            'in_h': 23,
            'in_w': 23,
            'out_channels': 1024,
            'mix_depth': 4,
            'mlp_ratio': 1,
            'out_rows': 4,
        },
        faiss_gpu=False
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model

def eval_nordland(model, batch_size=64, device='cuda'):
    print("\n===== NordLand =====")
    transform = get_transform()
    val_dataset = NordlandDataset(
        query_season='winter', reference_season='summer',
        transform=transform, split='val'
    )
    loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    feats = evaluate(model, loader, device)
    r_list = feats[:val_dataset.num_references]
    q_list = feats[val_dataset.num_references:]
    pitts_dict, _ = utils.get_validation_recalls(
        r_list=r_list, q_list=q_list,
        k_values=[1, 5, 10], gt=val_dataset.positives,
        print_results=True, dataset_name='NordLand', faiss_gpu=False
    )
    return pitts_dict

def eval_pitts250k(model, batch_size=64, device='cuda'):
    print("\n===== Pitts250k-test =====")
    transform = get_transform()
    dataset = PittsburgDataset.get_250k_test_set(input_transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    feats = evaluate(model, loader, device)
    r_list = feats[:dataset.dbStruct.numDb]
    q_list = feats[dataset.dbStruct.numDb:]
    pitts_dict, _ = utils.get_validation_recalls(
        r_list=r_list, q_list=q_list,
        k_values=[1, 5, 10], gt=dataset.getPositives(),
        print_results=True, dataset_name='Pitts250k-test', faiss_gpu=False
    )
    return pitts_dict

class SPEDDataset(Dataset):
    def __init__(self, dataset_path, split='ref', transform=None):
        self.transform = transform
        self.image_paths = sorted(
            glob.glob(os.path.join(dataset_path, split, '**', '*.jpg'), recursive=True) +
            glob.glob(os.path.join(dataset_path, split, '**', '*.png'), recursive=True)
        )
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, idx

def eval_sped(model, batch_size=64, device='cuda'):
    print("\n===== SPED =====")
    transform = get_transform()
    sped_path = '/media/hdd/ihsuan/ClipVPR/SPED/SPEDTEST/SPEDTEST'
    ref_dataset = SPEDDataset(sped_path, split='ref', transform=transform)
    query_dataset = SPEDDataset(sped_path, split='query', transform=transform)
    print(f"References: {len(ref_dataset)}, Queries: {len(query_dataset)}")
    r_list = evaluate(model, DataLoader(ref_dataset, batch_size=batch_size, num_workers=8), device)
    q_list = evaluate(model, DataLoader(query_dataset, batch_size=batch_size, num_workers=8), device)
    gt = np.load(os.path.join(sped_path, 'ground_truth_new.npy'), allow_pickle=True)
    pitts_dict, _ = utils.get_validation_recalls(
        r_list=r_list, q_list=q_list,
        k_values=[1, 5, 10], gt=gt,
        print_results=True, dataset_name='SPED', faiss_gpu=False
    )
    return pitts_dict

class MSLSDataset(Dataset):
    def __init__(self, dataset_path, cities, split='database', transform=None):
        self.transform = transform
        self.image_paths = []
        for city in cities:
            csv_path = os.path.join(dataset_path, 'train_val', city, split, 'subtask_index.csv')
            img_dir = os.path.join(dataset_path, 'train_val', city, split, 'images')
            df = pd.read_csv(csv_path)
            for key in df[df['all'] == True]['key'].tolist():
                self.image_paths.append(os.path.join(img_dir, key + '.jpg'))
        print(f"MSLS {split}: {len(self.image_paths)} images")
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, idx

def eval_msls(model, batch_size=64, device='cuda'):
    print("\n===== MSLS Val =====")
    transform = get_transform()
    msls_path = '/media/hdd/ihsuan/ClipVPR/MSLS'
    val_cities = ['cph', 'sf']
    ref_dataset = MSLSDataset(msls_path, val_cities, split='database', transform=transform)
    query_dataset = MSLSDataset(msls_path, val_cities, split='query', transform=transform)
    r_list = evaluate(model, DataLoader(ref_dataset, batch_size=batch_size, num_workers=8), device)
    q_list = evaluate(model, DataLoader(query_dataset, batch_size=batch_size, num_workers=8), device)

    def get_utms(split):
        utms = []
        for city in val_cities:
            idx_df = pd.read_csv(os.path.join(msls_path, 'train_val', city, split, 'subtask_index.csv'))
            raw_df = pd.read_csv(os.path.join(msls_path, 'train_val', city, split, 'postprocessed.csv'))
            for key in idx_df[idx_df['all'] == True]['key'].tolist():
                row = raw_df[raw_df['key'] == key].iloc[0]
                utms.append([row['easting'], row['northing']])
        return np.array(utms)

    ref_utms = get_utms('database')
    query_utms = get_utms('query')
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(ref_utms)
    positives = knn.radius_neighbors(query_utms, radius=25, return_distance=False)
    pitts_dict, _ = utils.get_validation_recalls(
        r_list=r_list, q_list=q_list,
        k_values=[1, 5, 10], gt=positives,
        print_results=True, dataset_name='MSLS Val', faiss_gpu=False
    )
    return pitts_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--datasets', nargs='+', default=['nordland'],
                        choices=['nordland', 'pitts250k', 'sped', 'msls'])
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.ckpt_dir is not None:
        ckpt_list = sorted(glob.glob(os.path.join(args.ckpt_dir, '*.ckpt')))
        print(f"Found {len(ckpt_list)} checkpoints in {args.ckpt_dir}")
    elif args.ckpt_path is not None:
        ckpt_list = [args.ckpt_path]
    else:
        raise ValueError("Provide --ckpt_path or --ckpt_dir")

    save_dir = './LOGS/eval_results_sold2'
    os.makedirs(save_dir, exist_ok=True)
    summary_file = os.path.join(save_dir, 'summary.csv')
    summary_rows = []

    for ckpt_path in ckpt_list:
        ckpt_name = os.path.basename(ckpt_path).replace('.ckpt', '')
        print(f"\n{'='*60}\nEvaluating: {ckpt_name}\n{'='*60}")

        model = load_model(ckpt_path, device)
        row = {'checkpoint': ckpt_name}

        eval_fns = {
            'nordland': eval_nordland,
            'pitts250k': eval_pitts250k,
            'sped': eval_sped,
            'msls': eval_msls,
        }

        for ds in args.datasets:
            try:
                pitts_dict = eval_fns[ds](model, batch_size=args.batch_size, device=device)
                row[f'{ds}_R1']  = round(pitts_dict[1], 4)
                row[f'{ds}_R5']  = round(pitts_dict[5], 4)
                row[f'{ds}_R10'] = round(pitts_dict[10], 4)
                with open(os.path.join(save_dir, f'{ckpt_name}_{ds}.txt'), 'w') as f:
                    f.write(f"Checkpoint: {ckpt_path}\n{'='*50}\n{ds.upper()}\n")
                    f.write(f"R@1:  {pitts_dict[1]:.4f}\nR@5:  {pitts_dict[5]:.4f}\nR@10: {pitts_dict[10]:.4f}\n")
            except Exception as e:
                print(f"Error on {ds}: {e}")
                row[f'{ds}_R1'] = row[f'{ds}_R5'] = row[f'{ds}_R10'] = -1

        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
        print(f"Summary updated: {summary_file}")
        del model
        torch.cuda.empty_cache()

    print(f"\nAll done. Summary: {summary_file}")


'''
python eval_sold2.py \
  --ckpt_dir /media/hdd/ihsuan/ClipVPR/livpr/LOGS/resnet50/lightning_logs/Final_ResNet_DepthAnythingV2_SOLD2/checkpoints \
  --datasets nordland pitts250k sped msls \
  --batch_size 64
'''