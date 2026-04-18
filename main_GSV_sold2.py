from MixVPR.dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.optim import lr_scheduler, optimizer
import MixVPR.utils as utils
import torchvision.transforms as T
import faiss
from matplotlib import pyplot as plt 

# from MixVPR.dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from MixVPR.dataloaders.HPointLocDataloader import HPointLocDataModule
from MixVPR.models import helper
from MixVPR.dataloaders import HPointLocDataset, HPointLocDataloader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
                lr=0.0002, 
                optimizer='adamw',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=200,
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
        self.train_losses = []
        self.val_r1_scores = []

        self.tf_vpr = T.Compose([
            T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
            #add a transform convert_img_type to float from uint8
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] 
        self.faiss_gpu = faiss_gpu
        # res = faiss.StandardGpuResources()
        # flat_config = faiss.GpuIndexFlatConfig()
        # flat_config.useFloat16 = True
        # flat_config.device = 0
        #self.embed_size = 4096
        self.embed_size = agg_config.get('out_rows', 4) * agg_config.get('out_channels', 1024)
        self.validation_step_outputs = []
        self.l2_search = faiss.IndexFlatL2(self.embed_size)
        self.faiss_index = faiss.IndexIDMap(self.l2_search)

        # resnet50 backbone
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        

        # resnet50 backbone
        # self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        
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

        # SOLD2 backbone
        from kornia.feature.sold2 import SOLD2
        sold2 = SOLD2(pretrained=True)
        self.sold2_backbone = sold2.model.backbone_net.net
        for param in self.sold2_backbone.parameters():
            param.requires_grad = False
        self.sold2_backbone.eval()

        # SOLD2 adapter (since the cat feature map size shrinks too much for SOLD2 encoder outputs)
        self.sold2_adapter = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )


        # Reduce feature map from 384 channels to 256 channels
        self.reducers = nn.ModuleList([
            nn.Conv2d(384, 256, kernel_size=1) for _ in range(4)
        ])

        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
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
            # encoder_output = self.depth_encoder(x)
            x_gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            sold2_output_ = self.sold2_backbone(x_gray)

        # sold2_output (N, 128, 23, 23)
        sold2_output = self.sold2_adapter(sold2_output_)

        """
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
        """

        # Concatenate 4 stages along channel dimension -> [B, 1024, H, W]
        # depth_feats = torch.cat(processed_maps, dim=1)
        res_feats = self.backbone(x) # 1024 channels inside res_feats
        line_feats = sold2_output

        line_feats = torch.nn.functional.interpolate(line_feats, size=(23, 23), mode='bilinear', align_corners=False)
        # depth_feats = torch.nn.functional.interpolate(depth_feats, size=(23, 23), mode='bilinear', align_corners=False)
        feat_map = torch.nn.functional.interpolate(feat_map, size=(23, 23), mode='bilinear', align_corners=False)
        res_feats = torch.nn.functional.interpolate(res_feats, size=(23, 23), mode='bilinear', align_corners=False)
        res_feats = torch.nn.functional.interpolate(res_feats, size=(23, 23), mode='bilinear', align_corners=False)
        
        x = torch.cat([feat_map, line_feats, res_feats], dim=1)
        x = self.aggregator(x)
        return x
    
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
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)
        
    def loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                loss, batch_acc = loss

        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
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
        avg_loss = sum(self.batch_acc) / len(self.batch_acc) if self.batch_acc else 0
        # 從 trainer log 取 loss
        loss_val = self.trainer.callback_metrics.get('loss', None)
        if loss_val is not None:
            self.train_losses.append(loss_val.item())
        self.batch_acc = []
        print("End of training !")

    def on_train_batch_start(self, batch, batch_idx):
        self.dino.eval()
        self.depth_encoder.eval()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, labels = batch
        descriptors = self(places)
        output = descriptors.detach().cpu()
        self.validation_step_outputs.append(output)
        return output
    
    def _plot_curves(self):
        save_dir = './LOGS/curves'
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curve
        if self.train_losses:
            axes[0].plot(self.train_losses, 'b-o', markersize=3)
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True)
        
        # R@1 curve
        if self.val_r1_scores:
            axes[1].plot(self.val_r1_scores, 'r-o', markersize=3)
            axes[1].set_title('Validation R@1')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('R@1')
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/curves_epoch_{self.current_epoch}.png', dpi=100)
        plt.close(fig)

    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.validation_step_outputs
        dm = self.trainer.datamodule

        if len(dm.val_datasets)==1: 
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)

            print("Now is validating 1")
            
            if 'pitts' in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.getPositives() 
                new_positives = positives 
                r_list = feats[ : num_references]
                q_list = feats[num_references : ]
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.pIdx
                r_list = feats[ : num_references]
                q_list = feats[num_references : ]
            elif 'hloc' in val_set_name:
                num_references = val_dataset.references.shape[0]
                reference_indices = val_dataset.references 
                query_indices = val_dataset.queries 
                positives = val_dataset.positives 
                offset = 0
                new_positives = []
                for l in positives:
                    new_positives.append(np.array(range(len(l))) + offset)
                    offset += len(l)
                r_list = feats[reference_indices]
                q_list = feats[query_indices]

                # print("len(reference_indices):", len(reference_indices))
                # print("len(query_indices):", len(query_indices))
                # print("positives[0]:", positives[0])
                # print("new_positives[0]:", new_positives[0])
                # print("positives[1]:", positives[1])
                # print("new_positives[1]:", new_positives[1])
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

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')
        self.validation_step_outputs = []
        r1 = pitts_dict.get(1, 0)
        self.val_r1_scores.append(r1)

        self._plot_curves()
            
    

if __name__ == '__main__':
    pl.seed_everything(seed=190223, workers=True)
        
    datamodule = GSVCitiesDataModule(
        batch_size=60,#32
        img_per_place=4,#2
        min_img_per_place=4,#2
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(322, 322),
        num_workers=28,
        show_data_stats=True,
        val_set_names=['pitts30k_val'], # pitts30k_val, pitts30k_test, msls_val
    )
   
    
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
        # agg_config={'in_channels' : 256+1024+1024, #change this to 1024 if no clip, but 2048 with clip 
        agg_config={'in_channels' : 256 + 256 + 1024, #256 (DinoV2) + 1024 (Depth) + 1024 (ResNet50) = 2304 
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
    # val_set = 'hloc'
    # datamodule = HPointLocDataModule(
    #     batch_size=32,
    #     img_per_place=2,
    #     min_img_per_place=2,
    #     shuffle_all=True, # shuffle all images or keep shuffling in-city only
    #     random_sample_from_each_place=True,
    #     image_size=(322, 322),
    #     num_workers=2,
    #     show_data_stats=True,
    #     val_set_names=[val_set], # pitts30k_val, pitts30k_test, msls_val
    # )
    
    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = ModelCheckpoint(
    monitor='pitts30k_val/R1',
    filename=f'{model.encoder_arch}' +
    '_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]',
    auto_insert_metric_name=False,
    save_weights_only=True,
    save_top_k=-1,
    mode='max',
    )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='cuda', devices=1,
        default_root_dir=f'./LOGS/{model.encoder_arch}', # Tensorflow can be used to viz 

        num_sanity_val_steps=0, # runs a validation step before stating training
        precision=16, # we use half precision to reduce  memory usage
        max_epochs=30,
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
        # fast_dev_run=True # uncomment or dev mode (only runs a one iteration train and validation, no checkpointing).
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
    # trainer.validate(model=model, datamodule=datamodule)