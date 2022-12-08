import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dataset import LungDatasets
from pathlib import Path

from utils import thin_plate_dense

def default_unet_features():
    nb_features = [[32, 48, 48, 64], [64, 48, 48, 48, 48, 32, 64]]  #  encoder,decoder
    return nb_features


inshape = (224//2,224//2,224//2)
H = 192
W = 192
D = 208
grid_sp = 2
unet_half_res=True
nb_unet_features=None
nb_unet_levels=None
unet_feat_mult=1
nb_unet_conv_per_level=1
int_steps=7
int_downsize=2
bidir=False
use_probs=False
src_feats=1
trg_feats=1
unet_half_res=False
unet_half_res=True

class VoxelMorphPP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        

        self.UNet = Unet(ConvBlock,inshape,
                        infeats=2,
                        nb_features=nb_unet_features,
                        nb_levels=nb_unet_levels,
                        feat_mult=unet_feat_mult,
                        nb_conv_per_level=nb_unet_conv_per_level,
                        half_res=unet_half_res)
        
        self.heatmap_head = nn.Sequential(nn.ConvTranspose3d(64,16,7),nn.InstanceNorm3d(16),nn.ReLU(),\
                        nn.Conv3d(16,32,3,padding=1),nn.InstanceNorm3d(32),\
                        nn.ReLU(),nn.Conv3d(32,32,3,padding=1),nn.Upsample(size=(11,11,11),mode='trilinear'),\
                        nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(64),nn.ReLU(),                
                        nn.Conv3d(32,32,3,padding=1),nn.InstanceNorm3d(32),nn.ReLU(),\
                        nn.Conv3d(32,16,3,padding=1),nn.InstanceNorm3d(16),nn.ReLU(),nn.Conv3d(16,1,3,padding=1))
        
    
    def forward(self, input, sample_xyz):
        output = self.UNet(input)[:,:,4:-4,4:-4,2:-2]
        sampled = F.grid_sample(output, sample_xyz.view(1,-1,1,1,3), mode='bilinear', align_corners=False )
        disp_pred = self.heatmap_head(sampled.permute(2,1,0,3,4))

        return output, disp_pred
    
    def training_step(self, batch, batch_idx):
        
        s = batch
        mind_fix  = s['scan_e'].view(1,1,H,W,D)*s['mask_e'].view(1,1,H,W,D)/500 # -> torch.Size([1, 1, 192, 192, 208])
        mind_mov = s['scan_i'].view(1,1,H,W,D)*s['mask_i'].view(1,1,H,W,D)/500 # -> torch.Size([1, 1, 192, 192, 208])
        
        mind_fix = F.avg_pool3d(mind_fix, grid_sp, stride=grid_sp) # -> torch.Size([1, 1, 96, 96, 104])
        mind_mov = F.avg_pool3d(mind_mov, grid_sp, stride=grid_sp) # -> torch.Size([1, 1, 96, 96, 104])
        
        keypts_fix = s['kps_e']
        keypts_mov = s['kps_i']
        disp_gt = keypts_mov-keypts_fix
        input = F.pad(torch.cat((mind_fix,mind_mov),1),(4,4,8,8,8,8)) # -> torch.Size([1, 2, 112, 112, 112])

        # sample 512 random keypoints which to use for loss computation
        idx = torch.randperm(keypts_fix.shape[0])[:512]
        sample_xyz = keypts_fix[idx]
        
        output, disp_pred = self(input, sample_xyz)
        

        #discretised heatmap grid (you may have to adapt the capture range from .3)
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0)*.3,(1,1,11,11,11), align_corners=False)

        pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)
        loss = nn.MSELoss()(torch.unsqueeze(pred_xyz, 0), disp_gt[idx])
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True,
                 batch_size=1)
        return loss

    def validation_step(self, batch, dataset_idx):
        s = batch
        mind_fix  = s['scan_e'].view(1,1,H,W,D)*s['mask_e'].view(1,1,H,W,D)/500 # -> torch.Size([1, 1, 192, 192, 208])
        mind_mov = s['scan_i'].view(1,1,H,W,D)*s['mask_i'].view(1,1,H,W,D)/500 # -> torch.Size([1, 1, 192, 192, 208])
        
        mind_fix = F.avg_pool3d(mind_fix, grid_sp, stride=grid_sp) # -> torch.Size([1, 1, 96, 96, 104])
        mind_mov = F.avg_pool3d(mind_mov, grid_sp, stride=grid_sp) # -> torch.Size([1, 1, 96, 96, 104])
        
        keypts_fix = s['kps_e']
        keypts_mov = s['kps_i']
        disp_gt = keypts_mov-keypts_fix
        input = F.pad(torch.cat((mind_fix,mind_mov),1),(4,4,8,8,8,8)) # -> torch.Size([1, 2, 112, 112, 112])

        # sample 512 random keypoints which to use for loss computation
        idx = torch.randperm(keypts_fix.shape[0])[:512]
        sample_xyz = keypts_fix[idx]
        
        output, disp_pred = self(input, sample_xyz)
        

        #discretised heatmap grid (you may have to adapt the capture range from .3)
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0)*.3,(1,1,11,11,11), align_corners=False)

        pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)
        loss = nn.MSELoss()(torch.unsqueeze(pred_xyz, 0), disp_gt[idx])
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("val_loss", loss, batch_size=1)
        # return loss
    
    def predict_step(self, batch, dataset_idx):
        #discretised heatmap grid (you may have to adapt the capture range from .3)
        mesh = F.affine_grid(torch.eye(3,4).unsqueeze(0)*.3,(1,1,11,11,11), align_corners=False)

        patch = F.affine_grid(0.05*torch.eye(3,4).unsqueeze(0),(1,1,3,3,3), align_corners=True)

        s = batch
        mind_fix  = s['scan_e'].view(1,1,H,W,D)*s['mask_e'].view(1,1,H,W,D)/500 # -> torch.Size([1, 1, 192, 192, 208])
        mind_mov = s['scan_i'].view(1,1,H,W,D)*s['mask_i'].view(1,1,H,W,D)/500 # -> torch.Size([1, 1, 192, 192, 208])
        
        mind_fix = F.avg_pool3d(mind_fix, grid_sp, stride=grid_sp) # -> torch.Size([1, 1, 96, 96, 104])
        mind_mov = F.avg_pool3d(mind_mov, grid_sp, stride=grid_sp) # -> torch.Size([1, 1, 96, 96, 104])
        
        keypts_fix = s['kps_e']
        keypts_mov = s['kps_i'] # can be None or whatever

        tre_net = torch.zeros(3,30)
        disp_copd = torch.zeros(10,3,192,192,208)
        
        # gpu_usage()
        
        input = F.pad(torch.cat((mind_fix,mind_mov),1),(4,4,8,8,8,8))
        output = self.UNet(input)[:,:,4:-4,4:-4,2:-2]

        sample_xyz = keypts_fix
        sampled = F.grid_sample(output, sample_xyz.view(1,-1,1,1,3), mode='bilinear')
        #sampled = F.grid_sample(output,sample_xyz.view(1,-1,1,1,3)+patch.view(1,1,-1,1,3),mode='bilinear')
        disp_pred = self.heatmap_head(sampled.permute(2,1,0,3,4))
#            disp_pred = heatmap(sampled.permute(2,1,0,3,4).view(keypts_fix.shape[0],-1,3,3,3))

        pred_xyz = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)

        # torch.Size([2013, 3]) pred_xyz
        # torch.Size([1, 2013, 3]) kp_fixed
        dense_flow_ = thin_plate_dense(keypts_fix, pred_xyz.unsqueeze(0), (H, W, D), 4, 0.1)
        # dense_flow_ = thin_plate_dense(keypts_fix.unsqueeze(0), pred_xyz.unsqueeze(0), (H, W, D), 4, 0.1)
        dense_flow = dense_flow_.flip(4).permute(0,4,1,2,3)*torch.tensor([H-1,W-1,D-1]).view(1,3,1,1,1)/2

        disp_hr = dense_flow

        disp_lr = F.interpolate(disp_hr,size=(H//grid_sp,W//grid_sp,D//grid_sp),mode='trilinear',align_corners=False)
        
        net = nn.Sequential(nn.Conv3d(3,1,(H//grid_sp,W//grid_sp,D//grid_sp),bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data/grid_sp
        # net
        optimizer = torch.optim.Adam(net.parameters(), lr=1)
        grid0 = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H//grid_sp,W//grid_sp,D//grid_sp),align_corners=False)
        #run Adam optimisation with diffusion regularisation and B-spline smoothing
        lambda_weight = .65# with tps: .5, without:0.7
        
        with torch.set_grad_enabled(True):
            for iter in range(50):#80
                optimizer.zero_grad()
                disp_sample = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(net[0].weight,3,stride=1,padding=1),3,stride=1,padding=1),3,stride=1,padding=1).permute(0,2,3,4,1)
                reg_loss = lambda_weight*((disp_sample[0,:,1:,:]-disp_sample[0,:,:-1,:])**2).mean()+\
                lambda_weight*((disp_sample[0,1:,:,:]-disp_sample[0,:-1,:,:])**2).mean()+\
                lambda_weight*((disp_sample[0,:,:,1:]-disp_sample[0,:,:,:-1])**2).mean()
                scale = torch.tensor([(H//grid_sp-1)/2,(W//grid_sp-1)/2,(D//grid_sp-1)/2]).unsqueeze(0)
                grid_disp = grid0.view(-1,3).float()+((disp_sample.view(-1,3))/scale).flip(1).float()
                patch_mov_sampled = F.grid_sample(mind_mov.float(),grid_disp.view(1,H//grid_sp,W//grid_sp,D//grid_sp,3),align_corners=False,mode='bilinear')#,padding_mode='border')
                sampled_cost = (patch_mov_sampled-mind_fix).pow(2).mean(1)*12
                loss = sampled_cost.mean()
                (loss+reg_loss).backward()
                optimizer.step()
        
        fitted_grid = disp_sample.permute(0,4,1,2,3).detach()
        disp_hr = F.interpolate(fitted_grid*grid_sp,size=(H,W,D),mode='trilinear',align_corners=False)

        disp_smooth = F.avg_pool3d(F.avg_pool3d(F.avg_pool3d(disp_hr,3,padding=1,stride=1),3,padding=1,stride=1),3,padding=1,stride=1)


        disp_hr = torch.flip(disp_smooth/torch.tensor([H-1,W-1,D-1]).view(1,3,1,1,1)*2,[1])
        disp_copd = disp_hr

        pred_xyz = F.grid_sample(disp_hr.float(),keypts_fix.view(1,-1,1,1,3),mode='bilinear').squeeze().t()

        disp_gt = keypts_mov-keypts_fix

        moving_image = s['scan_i']
        warped = F.grid_sample(moving_image,
                               disp_copd.permute(0,2,3,4,1)+F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D), align_corners=False))

        return warped
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)
    

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.ReLU()#nn.LeakyReLU(0.2)
        self.main2 = Conv(out_channels, out_channels, 1, stride,0)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.activation2 = nn.ReLU()#nn.LeakyReLU(0.2)


    def forward(self, x):
        out = self.activation(self.norm(self.main(x)))
        out = self.activation2(self.norm2(self.main2(out)))
        return out

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 ConvBlock,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x

        