import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
import MixVPR.utils as utils
import torchvision.transforms as T
import clip
import faiss
from matplotlib import pyplot as plt
from MixVPR.dataloaders.KITTI_dataloader import KITTIValidationModule

# from MixVPR.dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from MixVPR.dataloaders.HPointLocDataloader import HPointLocDataModule
from MixVPR.models import helper
from MixVPR.dataloaders import HPointLocDataset, HPointLocDataloader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForDepthEstimation # Load depth anything model

import os 
tf_vpr = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False,
                vlm_model_id = 'dinov2_vitb14',
                depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf",
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin    

        self.tf_vpr = T.Compose([
            T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
            #add a transform convert_img_type to float from uint8
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        # res = faiss.StandardGpuResources()
        # flat_config = faiss.GpuIndexFlatConfig()
        # flat_config.useFloat16 = True
        # flat_config.device = 0
        self.embed_size = 4096
        self.l2_search = faiss.IndexFlatL2(self.embed_size)
        self.faiss_index = faiss.IndexIDMap(self.l2_search)

        # resnet50 backbone
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        
        # DinoV2 extractor
        out_channels = 256
        self.dino = torch.hub.load('facebookresearch/dinov2', vlm_model_id)
        self.dino.eval()
        embed_dim = {'dinov2_vits14': 384, 'dinov2_vitb14': 768}[vlm_model_id]
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id)
        self.depth_encoder = self.depth_model.backbone

        # Reduce feature map from 384 channels to 256 channels
        self.reducers = nn.ModuleList([
            nn.Conv2d(384, 256, kernel_size=1) for _ in range(4)
        ])

        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    # the forward pass of the lightning model
    def forward(self, x, is_visualizing=False):
        grad_context = torch.enable_grad() if is_visualizing else torch.no_grad()

        # DinoV2 feature map
        if x.dim() == 5:
            B, S, C, H, W = x.shape
            x = x.view(B * S, C, H, W) 
        else:
            B, C, H, W = x.shape
            S = 1 # 紀錄序列長度為 1

        # --- 新增：強制調整大小以符合 DINOv2 (14的倍數) ---
        h_new = (H // 14) * 14
        w_new = (W // 14) * 14
        if H != h_new or W != w_new:
            import torch.nn.functional as F
            x = F.interpolate(x, size=(h_new, w_new), mode='bilinear', align_corners=False)
        
        h_patches = h_new // 14
        w_patches = w_new // 14

        with grad_context:
            patch_tokens = self.dino.forward_features(x)['x_norm_patchtokens']

        # with torch.no_grad():
        #     patch_tokens = self.dino.forward_features(x)['x_norm_patchtokens']

        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, -1, h_patches, w_patches)
        feat_map = self.proj(feat_map)
    

        # Depth feature map
        # with torch.no_grad():
        #     encoder_output = self.depth_encoder(x)

        with grad_context:
            encoder_output = self.depth_encoder(x)

        processed_maps = []
        for i, f in enumerate(encoder_output.feature_maps):
            # Reshape from [B, N, C] to [B, C, H, W]
            # N-1 to remove CLS token
            b, n, c = f.shape

            # in case input images isn't square
            # h = w = int((n - 1)**0.5)
            # assert h * w == n - 1
            h = h_patches
            w = w_patches
            assert h * w == n - 1
            
            # Remove CLS token (index 0), then permute to [B, C, N-1]
            feat_2d = f[:, 1:, :].transpose(1, 2).contiguous().reshape(b, c, h, w)
            
            # Reduce channels 384 -> 256
            feat_reduced = self.reducers[i](feat_2d)
            processed_maps.append(feat_reduced)

        # Concatenate 4 stages along channel dimension -> [B, 1024, H, W]
        depth_feats = torch.cat(processed_maps, dim=1)
        
        # res_feats = self.backbone(x) # 1024 channels inside res_feats
        with grad_context:
            res_feats = self.backbone(x)

        depth_feats = torch.nn.functional.interpolate(depth_feats, size=(23, 23), mode='bilinear', align_corners=False)
        feat_map = torch.nn.functional.interpolate(feat_map, size=(23, 23), mode='bilinear', align_corners=False)
        res_feats = torch.nn.functional.interpolate(res_feats, size=(23, 23), mode='bilinear', align_corners=False)
        
        # cat dinov2 feature and depth feature here
        x = torch.cat([feat_map, depth_feats, res_feats], dim=1)
        x = self.aggregator(x)
        return x
    
#     # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, llm_places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        print("End of training !")
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        print("Now is validating 2")
        places, llm_places, labels = batch
        # calculate descriptors
        descriptors = self(places)
        # 將 labels 轉為 1D tensor: (batch_size,)
        val_loss = self.loss_function(descriptors, labels.view(-1))
        print("         ")
        print("This is val loss !!", val_loss)
        print("         ")

        self.last_batch = batch # keep the last batch for visualization

        return descriptors.detach().cpu()
    
    def validation_epoch_end(self, val_step_outputs):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            print("Now is validating 1")
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.getPositives() #length of positives is num_queries
                r_list = feats[ : num_references]
                q_list = feats[num_references : ]
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.pIdx
                r_list = feats[ : num_references]
                q_list = feats[num_references : ]
            elif 'hloc' in val_set_name:
                #blah blah blah
                num_references = val_dataset.references.shape[0]
                reference_indices = val_dataset.references 
                query_indices = val_dataset.queries
                num_queries = len(query_indices)
                positives = val_dataset.positives #should be np array of arrays 
                # [[...] * num_q]
                offset = 0
                new_positives = []
                for l in positives:
                    new_positives.append(np.array(range(len(l))) + offset)
                    offset += len(l)
                r_list = feats[reference_indices]
                q_list = feats[query_indices]
                # new_positives = np.concatenate(new_positives)
            elif 'KITTI' in val_set_name:
                # KITTI 通常是 Self-Retrieval (自己找自己) 或序列對齊
                # 這裡假設所有影像既是 Reference 也是 Query
                num_total = len(val_dataset)
                all_indices = np.arange(num_total)
                
                # 定義 Reference 和 Query (通常 KITTI 驗證會用同一組特徵進行 All-to-All 檢索)
                reference_indices = all_indices
                query_indices = all_indices
                num_references = len(reference_indices)
                num_queries = len(query_indices)

                # 取得 Ground Truth (positives)
                # 假設 KITTIVPRDataset 有實作 get_positives() 或是 positives 屬性
                # 這通常是根據距離門檻 (如 5m) 預先算好的鄰居索引列表
                if hasattr(val_dataset, 'positives'):
                    new_positives = val_dataset.all_positives
                else:
                    # 如果 Dataset 沒寫，這裡會根據你提供的 threshold 現場算 (較慢)
                    # 建議確保 Dataset 類別裡有 self.positives
                    new_positives = val_dataset.all_positives

                r_list = feats[reference_indices]
                q_list = feats[query_indices]

            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            if 'KITTI' in val_set_name:
                exclusion_radius = 25  # 排除前後 2.5 秒內的影像
                filtered_positives = []
                for i, pos_list in enumerate(new_positives):
                    # 只保留索引差距大於 25 的正樣本
                    mask = np.abs(pos_list - i) > exclusion_radius
                    filtered_pos = pos_list[mask]
                    filtered_positives.append(filtered_pos)
                
                # 替換掉原本的 gt
                new_positives = filtered_positives
            
            pitts_dict, predictions = utils.get_validation_recalls(r_list=r_list, 
                                                q_list=q_list,
                                                k_values=[1, 5, 10, 15, 20, 50, 100],
                                                gt=new_positives,
                                                print_results=True,
                                                dataset_name=val_set_name,
                                                faiss_gpu=self.faiss_gpu
                                                )
            
            print("This is num_queries :", num_queries)
            retrieved_images = []
            num_viz_rows = min(num_queries, 10) # 準備前 10 組資料
            exclusion_radius = 25 # KITTI 的排除半徑

            for idx in range(num_viz_rows):
                q_dataset_idx = query_indices[idx]
                
                # 第一張永遠是 Query
                # 使用安全索引 [0] 解決 ValueError: too many values to unpack
                query_img = val_dataset[q_dataset_idx][0] 
                mini = [query_img] 
                
                # 尋找前 5 個「非鄰居」的 Rank 結果
                count = 0
                for rel_idx in predictions[idx]:
                    if np.abs(rel_idx - idx) > exclusion_radius:
                        ref_dataset_idx = int(reference_indices[rel_idx])
                        mini.append(val_dataset[ref_dataset_idx][0])
                        count += 1
                    if count >= 5: # 找夠 5 張就停
                        break
                
                retrieved_images.append(mini)

            # --- 開始繪圖 ---
            num_viz = min(num_queries, 3)
            top_k_viz = 2 # 畫兩張 Rank
            fig, ax = plt.subplots(num_viz, 1 + top_k_viz, figsize=(15, 3 * num_viz))
            if num_viz == 1: ax = np.expand_dims(ax, axis=0) 

            for r in range(num_viz):
                for c in range(1 + top_k_viz):
                    # 安全檢查：如果這組 Query 找出來的 valid 圖片不夠多
                    if c >= len(retrieved_images[r]):
                        ax[r, c].text(0.5, 0.5, "No Loop Found", ha='center', va='center')
                        ax[r, c].axis('off')
                        continue
                        
                    img_data = retrieved_images[r][c]
                    
                    # 處理 Tensor 與格式轉換 (維持你原本的邏輯)
                    if torch.is_tensor(img_data):
                        if img_data.dim() == 4: img_data = img_data.squeeze(0)
                        img_data = img_data.permute(1, 2, 0).cpu().numpy()
                        img_data = (img_data * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
                        img_data = np.clip(img_data, 0, 1)

                    ax[r, c].imshow(np.ascontiguousarray(img_data))
                    ax[r, c].axis('off')

                    # 設定標題
                    if c == 0:
                        ax[r, c].set_title(f"Query ID:{query_indices[r]}", fontsize=8)
                    else:
                        # 重新計算正確的 ID 用於標題
                        valid_ranks = [j for j in predictions[r] if np.abs(j - r) > exclusion_radius]
                        rel_idx = valid_ranks[c-1]
                        abs_id = int(reference_indices[rel_idx])
                        ax[r, c].set_title(f"Rank {c} ID:{abs_id} (Loop)", fontsize=8)

            # --- 存檔 ---
            save_dir = './LOGS/retrieved_images'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/epoch_{self.current_epoch}_{val_set_name}.png', bbox_inches='tight', dpi=100)
            plt.close(fig)

            print(f"\n" + "="*50)
            print(f"Epoch {self.current_epoch} - {val_set_name}")
            print("-" * 50)
            
            num_viz_print = min(num_queries, 3)
            for r in range(num_viz_print):
                q_idx = query_indices[r]
                
                # 關鍵：找出這組 Query 的「非鄰居」預測列表
                valid_preds = [j for j in predictions[r] if np.abs(j - r) > exclusion_radius]
                
                # 取得 Rank 1 & 2 (如果有的話)
                top1_abs = int(reference_indices[valid_preds[0]]) if len(valid_preds) > 0 else "N/A"
                top2_abs = int(reference_indices[valid_preds[1]]) if len(valid_preds) > 1 else "N/A"
                
                print(f"Query {r} (ID: {q_idx}) -> Rank 1: {top1_abs} (Loop), Rank 2: {top2_abs}")
            print("="*50 + "\n")
            import json

            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                return obj

            predictions_abs = [[int(reference_indices[rel_idx]) for rel_idx in row] for row in predictions]

            loop_data = {
                "val_set_name": val_set_name,
                "epoch": self.current_epoch,
                "query_indices": query_indices.tolist() if hasattr(query_indices, 'tolist') else list(query_indices),
                "top_k_predictions": predictions_abs
            }

            json_save_path = f'{save_dir}/indices_epoch_{self.current_epoch}_{val_set_name}.json'
            with open(json_save_path, 'w') as f:
                json.dump(loop_data, f, indent=4, default=convert_to_serializable)

            print(f">>> 已將 Index 對應關係存至: {json_save_path}")

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        if hasattr(self, 'last_batch'):
            print(f"Generating contribution visualization for epoch {self.current_epoch}...")
            save_path = f'./LOGS/retrieved_images/interpret_epoch_{self.current_epoch}.png'
            
            # 注意：你的 visualize_contribution 定義在類別內，呼叫時要用 self
            try:
                self.visualize_contribution(self.last_batch, save_name=save_path)
                print(f">>> 視覺化圖表已存至: {save_path}")
            except Exception as e:
                print(f"視覺化失敗: {e}")
    
    def visualize_contribution(self, batch, save_name="interpret.png"):
        self.eval()
        
        places, _, labels = batch
        img = places[0:1].clone().detach().to(self.device)
        img.requires_grad = True 
        
        with torch.enable_grad():
            descriptors = self(img, is_visualizing=True) 
            
            target = labels[0:1].view(-1).to(self.device)
            loss = self.loss_fn(descriptors, target)
            
            self.zero_grad()
            loss.backward()
            
        if img.grad is not None:
            grad_data = img.grad.data.cpu().numpy()[0]
            learned_weights = grad_data.mean(axis=0) 
        else:
            print("Error: Gradient is None. Make sure forward pass is differentiable.")
            learned_weights = np.zeros((img.shape[2], img.shape[3]))

        self.zero_grad()
        
        self._plot_results(img, learned_weights, save_name)

    def _plot_results(self, img, weights, save_name):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2
        import torch

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        rgb = img[0].detach().permute(1, 2, 0).cpu().numpy()
        rgb = (rgb * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        rgb = np.clip(rgb, 0, 1)
        axes[0].imshow(rgb)
        axes[0].set_title("Original RGB")

        with torch.no_grad():
            depth_feats = self.depth_encoder(img.detach()).feature_maps[-1]
            if depth_feats.shape[1] == 530:
                d_tokens = depth_feats[0, 1:, :].mean(dim=-1)
            else:
                d_tokens = depth_feats[0, :, :].mean(dim=-1)
            d_map = d_tokens.reshape(23, 23).cpu().numpy()
            d_map = cv2.resize(d_map, (rgb.shape[1], rgb.shape[0]))
        axes[1].imshow(d_map, cmap='magma')
        axes[1].set_title("Geometric (Depth) Feature")

        with torch.no_grad():
            dino_layers = self.dino.get_intermediate_layers(img.detach())
            dino_out = dino_layers[-1]
            if dino_out.shape[1] == 530:
                s_tokens = dino_out[0, 1:, :].mean(dim=-1)
            else:
                s_tokens = dino_out[0, :, :].mean(dim=-1)
            semantic_map = s_tokens.reshape(23, 23).cpu().numpy()
            semantic_map = cv2.resize(semantic_map, (rgb.shape[1], rgb.shape[0]))
        axes[2].imshow(semantic_map, cmap='viridis')
        axes[2].set_title("Semantic (DINO) Feature")

        v_limit = np.percentile(np.abs(weights), 99) if np.any(weights) else 1.0
        axes[3].imshow(weights, cmap='RdBu_r', vmin=-v_limit, vmax=v_limit)
        axes[3].set_title("Learned Weights (Gradients)")
        
        for ax in axes: ax.axis('off')
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
        plt.close(fig)
            
            
if __name__ == '__main__':
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
        
    # datamodule = GSVCitiesDataModule(
    #     batch_size=32,
    #     img_per_place=2,
    #     min_img_per_place=2,
    #     shuffle_all=False, # shuffle all images or keep shuffling in-city only
    #     random_sample_from_each_place=True,
    #     image_size=(320, 320),
    #     num_workers=28,
    #     show_data_stats=True,
    #     val_set_names=['pitts30k_val'], # pitts30k_val, pitts30k_test, msls_val
    # )

    
    
    # examples of backbones
    # resnet18, resnet50, resnet101, resnet152,
    # resnext50_32x4d, resnext50_32x4d_swsl , resnext101_32x4d_swsl, resnext101_32x8d_swsl
    # efficientnet_b0, efficientnet_b1, efficientnet_b2
    # swinv2_base_window12to16_192to256_22kft1k
    model = VPRModel(
        #---- Encoder
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4], # 4 crops the last resnet layer, 3 crops the 3rd, ...etc
        
        #---- Aggregator
        # agg_arch='CosPlace',
        # agg_config={'in_dim': 2048,
        #             'out_dim': 2048},
        # agg_arch='GeM',
        # agg_config={'p': 3},
        
        # agg_arch='ConvAP',
        # agg_config={'in_channels': 2048,
        #             'out_channels': 2048},

        agg_arch='MixVPR',
        agg_config={'in_channels' : 256+1024+1024, #change this to 1024 if no clip, but 2048 with clip 
                'in_h' : 23,
                'in_w' : 23,
                'out_channels' : 1024,
                'mix_depth' : 4,
                'mlp_ratio' : 1,
                'out_rows' : 4}, # the output dim will be (out_rows * out_channels)
        
        #---- Train hyperparameters
        lr=0.05, # 0.0002 for adam, 0.05 or sgd (needs to change according to batch size)
        optimizer='sgd', # sgd, adamw
        weight_decay=0.001, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        warmpup_steps=650,
        milestones=[5, 10, 15, 25, 45],
        lr_mult=0.3,

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    #load model weights frm weight_path 
    # print("test-1")
    # model.load_from_checkpoint('')# path to resnet checkpoint
    print("test0")
    val_set = 'hloc'
    datamodule = HPointLocDataModule(
        batch_size=32,
        img_per_place=2,
        min_img_per_place=2,
        shuffle_all=True, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(322, 322),
        num_workers=2,
        show_data_stats=True,
        val_set_names=[val_set], # pitts30k_val, pitts30k_test, msls_val
    )
    
    # val_set = 'kitti'
    # val_sequences = ['00'] 

    # datamodule = KITTIValidationModule(
    #     data_folder='C:/Users/User/Desktop/ROB530/clip-slcd/MixVPR/dataloaders/KITTI_dataset',
    #     batch_size=32,            # 根據你的顯存大小調整，驗證通常可以設大一點
    #     image_size=(480, 640),    # KITTI 原始比例接近 3:10，這裡建議維持 480x640 或 320x640
    #     num_workers=4,            # 建議設為 CPU 核心數的一半
    #     val_seqs=val_sequences,   # 傳入你想驗證的序列清單
    #     mean_std={
    #         'mean': [0.485, 0.456, 0.406], 
    #         'std': [0.229, 0.224, 0.225]
    #     } # 使用 ImageNet 標準歸一化
    # )
    
    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val

    print("test1")
    checkpoint_cb = ModelCheckpoint(
        monitor=f'{val_set}/R1',
        filename=f'{model.encoder_arch}' +
        '_epoch({epoch:02d})_step({step:04d})_R1[{hloc/R1:.4f}]_R5[{hloc/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max',)
    
    print("test2")

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='cuda', gpus=1,
        default_root_dir=f'./LOGS/{model.encoder_arch}', # Tensorflow can be used to viz 

        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=16, # we use half precision to reduce  memory usage
        max_epochs=80,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.validate(model=model, datamodule=datamodule)
