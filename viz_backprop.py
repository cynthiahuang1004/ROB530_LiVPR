'''
python viz_backprop.py --dataset sped
python viz_backprop.py --single_query path/to/query.jpg
python viz_backprop.py --dataset sped --max_samples 20 --gpu 1
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
    "/media/hdd/ihsuan/ClipVPR/livpr/LOGS/resnet50/lightning_logs/"
    "version_4/checkpoints/"
    "resnet50_epoch(06)_step(3290)_R1[0.9246]_R5[0.9815].ckpt"
)

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
        print(f"[Warning: cannot open image, skipping]: {path} ({e})")
        return None

def _glob_images(folder):
    folder = Path(folder)
    imgs = []
    for ext in ('*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg'):
        imgs.extend(folder.glob(ext))
    return sorted(imgs)

def get_dataset_pairs(dataset_name):
    """回傳 list of (query_path_str, ref_path_str)"""

    if dataset_name == 'sped':
        q_dir   = Path('/media/hdd/ihsuan/ClipVPR/SPED/SPEDTEST/SPEDTEST/query')
        r_dir   = Path('/media/hdd/ihsuan/ClipVPR/SPED/SPEDTEST/SPEDTEST/ref')
        gt_path = Path('/media/hdd/ihsuan/ClipVPR/SPED/SPEDTEST/SPEDTEST/ground_truth_new.npy')
        q_imgs  = _glob_images(q_dir)
        r_imgs  = _glob_images(r_dir)

        if gt_path.exists():
            gt = np.load(str(gt_path), allow_pickle=True)
            print(f"[SPED] GT loaded, shape={gt.shape}")
            pairs = []
            for i, q in enumerate(q_imgs):
                ref_idx = int(gt[i][0]) if hasattr(gt[i], '__len__') else int(gt[i])
                ref_idx = min(ref_idx, len(r_imgs) - 1)
                pairs.append((str(q), str(r_imgs[ref_idx])))
        else:
            print("[SPED] GT not found, using same-index pairing")
            pairs = [(str(q), str(r_imgs[i % len(r_imgs)])) for i, q in enumerate(q_imgs)]

        print(f"[SPED] {len(pairs)} pairs")
        return pairs

    elif dataset_name == 'pitts':
        import scipy.io as sio
        root   = Path('/media/hdd/ihsuan/ClipVPR/Pittsburgh250k')
        gt_dir = root / 'groundtruth'

        qfnames   = sio.loadmat(str(gt_dir / 'qfnames.mat'))['qfnames'].flatten()
        dbfnames  = sio.loadmat(str(gt_dir / 'dbfnames.mat'))['dbfnames'].flatten()
        query_ids = sio.loadmat(str(gt_dir / 'pittsburgh_queryID_1000.mat'))['query_id'].flatten()

        with open(gt_dir / 'gt_export.txt') as f:
            lines = [l.strip() for l in f.readlines()]

        pairs, skipped = [], 0
        for i in range(0, len(lines), 2):
            parts           = lines[i].split()
            query_local_idx = int(parts[0])
            n_refs          = int(parts[1])
            if n_refs == 0:
                skipped += 1
                continue
            ref_indices    = list(map(int, lines[i + 1].split()))
            real_query_idx = query_ids[query_local_idx]
            q_name         = qfnames[real_query_idx][0]
            folder, fname  = dbfnames[ref_indices[0]][0].split('/')
            q_path = root / 'queries_real' / (q_name + '.jpg')
            r_path = root / folder          / (fname  + '.jpg')
            pairs.append((str(q_path), str(r_path)))

        print(f"[Pittsburgh] {len(pairs)} pairs, skipped GT: {skipped}")
        return pairs

    elif dataset_name == 'nordland':
        q_dir        = Path('/media/hdd/ihsuan/ClipVPR/Nordland/data/summer')
        r_dir        = Path('/media/hdd/ihsuan/ClipVPR/Nordland/data/winter')
        filtered_txt = Path('/media/hdd/ihsuan/ClipVPR/Nordland/dataset_imageNames/nordland_imageNames.txt')

        if filtered_txt.exists():
            with open(filtered_txt) as f:
                valid_names = set(line.strip() for line in f)
            print(f"[Nordland] filtered list: {len(valid_names)} frames available")
        else:
            valid_names = None
            print("[Nordland] filtered list does not exist, using brightness filtering")

        r_map   = {p.name: p for p in _glob_images(r_dir)}
        pairs, skipped = [], 0
        for q in _glob_images(q_dir):
            if q.name not in r_map:
                continue
            if valid_names is not None:
                if q.name not in valid_names:
                    skipped += 1
                    continue
            else:
                if np.array(Image.open(q).convert('RGB')).mean() < 5:
                    skipped += 1
                    continue
            pairs.append((str(q), str(r_map[q.name])))

        print(f"[Nordland] {len(pairs)} valid pairs, skipped: {skipped}")
        return pairs

    elif dataset_name == 'msls':
        train_val = Path('/media/hdd/ihsuan/ClipVPR/MSLS/train_val')
        ALL_CITIES = [
            'amman', 'amsterdam', 'austin', 'bangkok', 'berlin', 'boston',
            'budapest', 'cph', 'goa', 'helsinki', 'london', 'manila',
            'melbourne', 'moscow', 'nairobi', 'ottawa', 'paris', 'phoenix',
            'saopaulo', 'sf', 'tokyo', 'toronto', 'trondheim', 'zurich'
        ]
        THRESHOLD_M = 25.0
        all_pairs = []

        for city_name in ALL_CITIES:
            base    = train_val / city_name
            pq_path = base / 'query'    / 'postprocessed.csv'
            pr_path = base / 'database' / 'postprocessed.csv'
            if not pq_path.exists() or not pr_path.exists():
                print(f'[MSLS/{city_name}] 缺 postprocessed.csv，跳過')
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

            city_pairs = []
            for vd in pq['view_direction'].unique():
                pq_vd = pq[pq['view_direction'] == vd].reset_index(drop=True)
                pr_vd = pr[pr['view_direction'] == vd].reset_index(drop=True)
                if len(pq_vd) == 0 or len(pr_vd) == 0:
                    continue
                tree = BallTree(pr_vd[['easting', 'northing']].values, metric='euclidean')
                distances, indices = tree.query(pq_vd[['easting', 'northing']].values, k=1)
                for i, (dist, ref_idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
                    if dist <= THRESHOLD_M:
                        city_pairs.append((pq_vd['img_path'].iloc[i], pr_vd['img_path'].iloc[ref_idx]))

            matched_rate = len(city_pairs) / len(pq) * 100 if len(pq) > 0 else 0
            print(f'[MSLS/{city_name}] query={len(pq)}  matched={len(city_pairs)} ({matched_rate:.1f}%)')
            all_pairs.extend(city_pairs)

        print(f'\n[MSLS] Total pairs across all cities: {len(all_pairs)}')
        return all_pairs

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def visualize_gradcam_all_branches(model, image_path, save_path='gradcam_all.png'):
    img_tensor = load_image(image_path)
    if img_tensor is None:
        return None

    saved_feat, saved_grad = {}, {}

    def make_fwd_hook(name):
        def hook(module, input, output):
            saved_feat[name] = output
        return hook

    def make_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            saved_grad[name] = grad_output[0].detach()
        return hook

    hooks = [
        model.proj.register_forward_hook(make_fwd_hook('DinoV2')),
        model.proj.register_full_backward_hook(make_bwd_hook('DinoV2')),
        model.reducers[-1].register_forward_hook(make_fwd_hook('Depth')),
        model.reducers[-1].register_full_backward_hook(make_bwd_hook('Depth')),
        model.backbone.register_forward_hook(make_fwd_hook('ResNet50')),
        model.backbone.register_full_backward_hook(make_bwd_hook('ResNet50')),
    ]

    model.zero_grad()
    output = model(img_tensor)
    output.norm().backward()
    for h in hooks:
        h.remove()

    cams = {}
    for name in ['DinoV2', 'Depth', 'ResNet50']:
        feat    = saved_feat[name].detach()
        grad    = saved_grad[name]
        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * feat).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=(322, 322), mode='bilinear', align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams[name] = cam

    orig_img = Image.open(image_path).convert('RGB').resize((322, 322))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for row, name in enumerate(['DinoV2', 'Depth', 'ResNet50']):
        cam = cams[name]
        axes[row, 0].imshow(orig_img);                          axes[row, 0].set_title(f'{name} | Original')
        axes[row, 1].imshow(cam, cmap='jet');                   axes[row, 1].set_title(f'{name} | Grad-CAM')
        axes[row, 2].imshow(orig_img);
        axes[row, 2].imshow(cam, cmap='jet', alpha=0.5);        axes[row, 2].set_title(f'{name} | Overlay')
        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle(Path(image_path).name, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    return cams

def batch_visualize(model, dataset_name, output_dir, max_samples):
    pairs   = get_dataset_pairs(dataset_name)[:max_samples]
    out_dir = Path(output_dir) / dataset_name / 'gradcam_all'
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (q_path, _) in enumerate(pairs):
        stem = Path(q_path).stem
        print(f"[{i+1}/{len(pairs)}] {stem}")
        visualize_gradcam_all_branches(
            model, q_path,
            save_path=str(out_dir / f'{stem}_gradcam_all.png')
        )

    print(f"\n=== Completed [{dataset_name}] ===")
    print(f"Result saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      type=str, default='sped',
                        choices=['sped', 'pitts', 'nordland', 'msls'])
    parser.add_argument('--output_dir',   type=str, default='./viz_results')
    parser.add_argument('--max_samples',  type=int, default=20)
    parser.add_argument('--single_query', type=str, default=None,
                        help='Only run single query, don\'t specify --dataset')
    parser.add_argument('--gpu',          type=int, default=0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    print(f"Loading checkpoint：{CKPT_PATH}")
    model = VPRModel.load_from_checkpoint(CKPT_PATH)
    model.eval().cuda()
    print("Model loaded!\n")

    if args.single_query:
        os.makedirs(args.output_dir, exist_ok=True)
        visualize_gradcam_all_branches(
            model, args.single_query,
            save_path=f'{args.output_dir}/single_gradcam_all.png'
        )
        print(f"Result saved to {args.output_dir}/single_gradcam_all.png")
    else:
        batch_visualize(model, args.dataset, args.output_dir, args.max_samples)