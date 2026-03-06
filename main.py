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
        self.validation_step_outputs = []
        self.l2_search = faiss.IndexFlatL2(self.embed_size)
        self.faiss_index = faiss.IndexIDMap(self.l2_search)

        # resnet50 backbone
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        
        # DinoV2 extractor
        out_channels = 256
        self.dino = torch.hub.load(
            '/home/shared/.cache/torch/hub/facebookresearch_dinov2_main',
            vlm_model_id,
            source='local'
        )        
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
    def forward(self, x):

        # DinoV2 feature map
        B, C, H, W = x.shape
        h_patches = H // 14
        w_patches = W // 14
        with torch.no_grad():
            patch_tokens = self.dino.forward_features(x)['x_norm_patchtokens']
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, -1, h_patches, w_patches)
        feat_map = self.proj(feat_map)
    

        # Depth feature map
        with torch.no_grad():
            encoder_output = self.depth_encoder(x)

        processed_maps = []
        for i, f in enumerate(encoder_output.feature_maps):
            # Reshape from [B, N, C] to [B, C, H, W]
            # N-1 to remove CLS token
            b, n, c = f.shape

            # in case input images isn't square
            h = w = int((n - 1)**0.5)
            assert h * w == n - 1
            
            # Remove CLS token (index 0), then permute to [B, C, N-1]
            feat_2d = f[:, 1:, :].transpose(1, 2).contiguous().reshape(b, c, h, w)
            
            # Reduce channels 384 -> 256
            feat_reduced = self.reducers[i](feat_2d)
            processed_maps.append(feat_reduced)

        # Concatenate 4 stages along channel dimension -> [B, 1024, H, W]
        depth_feats = torch.cat(processed_maps, dim=1)
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
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
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
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        print("End of training !")
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        #print("Now is validating 2")
        places, llm_places, _ = batch
        descriptors = self(places)
        output = descriptors.detach().cpu()
        self.validation_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.validation_step_outputs
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

            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            
            pitts_dict, predictions = utils.get_validation_recalls(r_list=r_list, 
                                                q_list=q_list,
                                                k_values=[1, 5, 10, 15, 20, 50, 100],
                                                gt=new_positives,
                                                print_results=True,
                                                dataset_name=val_set_name,
                                                faiss_gpu=self.faiss_gpu
                                                )
            
            retrieved_images = []
            for idx, i in enumerate(query_indices[:5]):  # idx是predictions的index，i是dataset的index
                mini = [val_dataset[i]] 
                for j in predictions[idx][:5]:  # 用 idx 而不是 i
                    mini.append(val_dataset[reference_indices[j]])  # j是reference的相對index，要轉回絕對index
                retrieved_images.append(mini)

            # retrieved_images = []
            # for i in query_indices[:5]:
            #     mini = [val_dataset[i]] 
            #     for j in predictions[i][:5]:
            #         mini.append(val_dataset[j])
            #     retrieved_images.append(mini)
            
            #create a subplot of 5 rows and 6 columns 
            fig, ax = plt.subplots(5, 6, figsize=(20, 20))
            #plot the images in retrived_images in each subplot 
            retrieved_images = []
            for idx, i in enumerate(query_indices[:5]):
                item = val_dataset[i]
                print(f"type: {type(item)}, ", end="")
                if isinstance(item, (list, tuple)):
                    print(f"len: {len(item)}, types: {[type(x) for x in item]}")
                else:
                    print(f"shape: {item.shape if hasattr(item, 'shape') else 'no shape'}")
                break  # 只印一次就好
                
            save_dir = f'./LOGS/retrieved_images'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/epoch_{self.current_epoch}_{val_set_name}.png',
                        bbox_inches='tight', dpi=100)
            plt.close(fig)
            
            # for i in range(5):
            #     for j in range(6):
            #         ax[i, j].imshow(retrieved_images[i][j])
            #         ax[i, j].axis('off')
            # del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')
        self.validation_step_outputs = []
            
            
if __name__ == '__main__':
    pl.seed_everything(seed=190223, workers=True)
        
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
        agg_config={'in_channels' : 256 + 1024 + 1024, #256 (DinoV2) + 1024 (Depth) + 1024 (ResNet50) = 2304 
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
    # model.load_from_checkpoint('')# path to resnet checkpoint
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
    
    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = ModelCheckpoint(
        monitor=f'{val_set}/R1',
        filename=f'{model.encoder_arch}' +
        '_epoch({epoch:02d})_step({step:04d})_R1[{hloc/R1:.4f}]_R5[{hloc/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max',)

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='cuda', devices=1,
        default_root_dir=f'./LOGS/{model.encoder_arch}', # Tensorflow can be used to viz 

        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=16, # we use half precision to reduce  memory usage
        max_epochs=150,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.validate(model=model, datamodule=datamodule)
