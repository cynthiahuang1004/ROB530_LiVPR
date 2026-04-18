'''
python viz_activation_map.py --dataset sped
python viz_activation_map.py --single_query path/to/query.jpg
python viz_activation_map.py --dataset nordland --max_samples 20 --gpu 0
'''

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pandas as pd
from sklearn.neighbors import BallTree
from main_GSV import VPRModel


CKPT_PATH = (
    "/media/hdd/ihsuan/ClipVPR/livpr/LOGS/resnet50/lightning_logs/Final_ResNet_DepthAnythingV2_SOLD2/checkpoints/resnet50_epoch(05)_step(2820)_R1[0.9303]_R5[0.9828].ckpt"
)

'''
nordland: /media/hdd/ihsuan/ClipVPR/livpr/LOGS/resnet50/lightning_logs/Final_ResNet_DepthAnythingV2_SOLD2/checkpoints/resnet50_epoch(22)_step(10810)_R1[0.9335]_R5[0.9857].ckpt
pitts: /media/hdd/ihsuan/ClipVPR/livpr/LOGS/resnet50/lightning_logs/Final_ResNet_DepthAnythingV2_SOLD2/checkpoints/resnet50_epoch(17)_step(8460)_R1[0.9347]_R5[0.9842].ckpt
msls: /media/hdd/ihsuan/ClipVPR/livpr/LOGS/resnet50/lightning_logs/Final_ResNet_DepthAnythingV2_SOLD2/checkpoints/resnet50_epoch(20)_step(9870)_R1[0.9332]_R5[0.9851].ckpt
sped: /media/hdd/ihsuan/ClipVPR/livpr/LOGS/resnet50/lightning_logs/Final_ResNet_DepthAnythingV2_SOLD2/checkpoints/resnet50_epoch(05)_step(2820)_R1[0.9303]_R5[0.9828].ckpt
'''

transform = T.Compose([
    T.Resize((322, 322)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0).cuda()
    except Exception as e:
        print(f"Warning: Cannot open image, skipping: {path} ({e})")
        return None

def _glob_images(folder):
    folder = Path(folder)
    imgs = []
    for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg'):
        imgs.extend(folder.glob(ext))
    return sorted(imgs)

def get_query_paths(dataset_name):
    if dataset_name == 'sped':
        q_dir  = Path('/media/hdd/ihsuan/ClipVPR/SPED/SPEDTEST/SPEDTEST/query')
        q_imgs = _glob_images(q_dir)
        print(f"[SPED] {len(q_imgs)} queries")
        return [str(q) for q in q_imgs]

    elif dataset_name == 'pitts':
        import scipy.io as sio
        root      = Path('/media/hdd/ihsuan/ClipVPR/Pittsburgh250k')
        gt_dir    = root / 'groundtruth'
        qfnames   = sio.loadmat(str(gt_dir / 'qfnames.mat'))['qfnames'].flatten()
        query_ids = sio.loadmat(str(gt_dir / 'pittsburgh_queryID_1000.mat'))['query_id'].flatten()
        with open(gt_dir / 'gt_export.txt') as f:
            lines = [l.strip() for l in f.readlines()]
        paths = []
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            if int(parts[1]) == 0:
                continue
            real_query_idx = query_ids[int(parts[0])]
            q_path = root / 'queries_real' / (qfnames[real_query_idx][0] + '.jpg')
            paths.append(str(q_path))
        print(f"[Pittsburgh] {len(paths)} queries")
        return paths

    elif dataset_name == 'nordland':
        q_dir        = Path('/media/hdd/ihsuan/ClipVPR/Nordland/data/summer')
        filtered_txt = Path('/media/hdd/ihsuan/ClipVPR/Nordland/dataset_imageNames/nordland_imageNames.txt')
        if filtered_txt.exists():
            with open(filtered_txt) as f:
                valid_names = set(line.strip() for line in f)
            print(f"[Nordland] filtered list: {len(valid_names)} frames available")
        else:
            valid_names = None
            print("[Nordland] filtered list does not exist, using all frames (will skip very dark ones)")
        paths, skipped = [], 0
        for q in _glob_images(q_dir):
            if valid_names is not None:
                if q.name not in valid_names:
                    skipped += 1
                    continue
            else:
                if np.array(Image.open(q).convert('RGB')).mean() < 5:
                    skipped += 1
                    continue
            paths.append(str(q))
        print(f"[Nordland] {len(paths)} queries, skipped: {skipped}")
        return paths

    elif dataset_name == 'msls':
        train_val = Path('/media/hdd/ihsuan/ClipVPR/MSLS/train_val')
        ALL_CITIES = [
            'amman', 'amsterdam', 'austin', 'bangkok', 'berlin', 'boston',
            'budapest', 'cph', 'goa', 'helsinki', 'london', 'manila',
            'melbourne', 'moscow', 'nairobi', 'ottawa', 'paris', 'phoenix',
            'saopaulo', 'sf', 'tokyo', 'toronto', 'trondheim', 'zurich'
        ]
        THRESHOLD_M = 25.0
        paths = []
        for city_name in ALL_CITIES:
            base    = train_val / city_name
            pq_path = base / 'query'    / 'postprocessed.csv'
            pr_path = base / 'database' / 'postprocessed.csv'
            if not pq_path.exists() or not pr_path.exists():
                continue
            pq = pd.read_csv(str(pq_path))
            pr = pd.read_csv(str(pr_path))
            q_img_dir = base / 'query'    / 'images'
            r_img_dir = base / 'database' / 'images'
            pq['img_path'] = pq['key'].apply(lambda k: str(q_img_dir / f'{k}.jpg'))
            pr['img_path'] = pr['key'].apply(lambda k: str(r_img_dir / f'{k}.jpg'))
            pq = pq[pq['img_path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
            pr = pr[pr['img_path'].apply(lambda p: Path(p).exists())].reset_index(drop=True)
            if len(pq) == 0 or len(pr) == 0:
                continue
            for vd in pq['view_direction'].unique():
                pq_vd = pq[pq['view_direction'] == vd].reset_index(drop=True)
                pr_vd = pr[pr['view_direction'] == vd].reset_index(drop=True)
                if len(pq_vd) == 0 or len(pr_vd) == 0:
                    continue
                tree = BallTree(pr_vd[['easting', 'northing']].values, metric='euclidean')
                distances, _ = tree.query(pq_vd[['easting', 'northing']].values, k=1)
                for i, dist in enumerate(distances[:, 0]):
                    if dist <= THRESHOLD_M:
                        paths.append(pq_vd['img_path'].iloc[i])
            print(f'[MSLS/{city_name}] {len(paths)} queries so far')
        print(f'\n[MSLS] 全部城市合計: {len(paths)} queries')
        return paths

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class _StopForward(Exception):
    """在 aggregator 之前中斷 forward，避免 channel shape mismatch"""
    pass

def visualize_three_branches(model, image_path, save_path='branches.png'):
    img_tensor = load_image(image_path)
    if img_tensor is None:
        return

    branch_feats = {}

    def make_hook(name):
        def hook(module, input, output):
            branch_feats[name] = output.detach()
        return hook

    def stop_hook(module, input):
        raise _StopForward()

    hooks = [
        model.backbone.register_forward_hook(make_hook('ResNet50')),
        model.proj.register_forward_hook(make_hook('DinoV2')),
        model.reducers[-1].register_forward_hook(make_hook('Depth')),
        model.aggregator.register_forward_pre_hook(stop_hook),
    ]

    try:
        with torch.no_grad():
            model(img_tensor)
    except _StopForward:
        pass  
    finally:
        for h in hooks:
            h.remove()

    if len(branch_feats) < 3:
        print(f"Warning: Only extracted {len(branch_feats)} branches, skipping: {image_path}")
        return

    orig_img = Image.open(image_path).convert('RGB').resize((322, 322))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(orig_img); axes[0, 0].set_title('Original'); axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    for col, name in enumerate(['ResNet50', 'DinoV2', 'Depth'], start=1):
        feat   = branch_feats[name]               # [1, C, H, W]
        act    = feat[0].mean(dim=0).cpu().numpy() # [H, W]
        act    = (act - act.min()) / (act.max() - act.min() + 1e-8)
        act_up = F.interpolate(
            torch.tensor(act).unsqueeze(0).unsqueeze(0).float(),
            size=(322, 322), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        axes[0, col].imshow(act_up, cmap='jet')
        axes[0, col].set_title(f'{name} activation'); axes[0, col].axis('off')
        axes[1, col].imshow(orig_img)
        axes[1, col].imshow(act_up, cmap='jet', alpha=0.5)
        axes[1, col].set_title(f'{name} overlay');    axes[1, col].axis('off')

    plt.suptitle(Path(image_path).name, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

def batch_visualize(model, dataset_name, output_dir, max_samples):
    query_paths = get_query_paths(dataset_name)[:max_samples]
    out_dir     = Path(output_dir) / dataset_name / 'branches'
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, q_path in enumerate(query_paths):
        stem = Path(q_path).stem
        print(f"[{i+1}/{len(query_paths)}] {stem}")
        visualize_three_branches(
            model, q_path,
            save_path=str(out_dir / f'{stem}_branches.png')
        )

    print(f"\n=== Completed [{dataset_name}] ===")
    print(f"Results saved to: {out_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str, default='sped',
                        choices=['sped', 'pitts', 'nordland', 'msls'])
    parser.add_argument('--output_dir',   type=str, default='./viz_results')
    parser.add_argument('--max_samples',  type=int, default=20)
    parser.add_argument('--single_query', type=str, default=None,
                        help='Only run a single query, no need to specify --dataset')
    parser.add_argument('--gpu',          type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    print(f"Loading checkpoint：{CKPT_PATH}")
    model = VPRModel.load_from_checkpoint(CKPT_PATH, strict=False)
    model.eval().cuda()
    print("Model loaded successfully!\n")

    if args.single_query:
        os.makedirs(args.output_dir, exist_ok=True)
        visualize_three_branches(
            model, args.single_query,
            save_path=f'{args.output_dir}/single_branches.png'
        )
        print(f"Results saved to {args.output_dir}/single_branches.png")
    else:
        batch_visualize(model, args.dataset, args.output_dir, args.max_samples)