# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool import bev_pool
from mmdet3d.ops.voxel_pooling import voxel_pooling
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
from projects.mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from mmdet.models.backbones.resnet import BasicBlock
from projects.mmdet3d_plugin.utils.semkitti import semantic_kitti_class_frequencies, kitti_class_names, CE_ssc_loss
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import pdb
from .ViewTransformerLSSBEVDepth import *
from .semkitti_depthnet import SemKITTIDepthNet
from .attention import attention, CA3D, CrossAttentionFusion, CrossLayer,AttentionLayer,ClipFusion
from .aggregator import EmbedAggregator,project_occ_mask_to_voxel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, save_checkpoint,
                         wrap_fp16_model)
from collections import OrderedDict
from projects.mmdet3d_plugin.utils.tools import tools
from .gmflow.gmflow import GMFlow
from projects.mmdet3d_plugin.occupancy.modules.nat_attention import NATCrossLayer,Attention_Cross
from PIL import Image
norm_cfg = dict(type='GN', num_groups=2, requires_grad=True)
class stereofeature_net(nn.Module):
    def __init__(self, in_channels, mid_channels,
                 depth_channels, cam_channels):
        
        super(stereofeature_net, self).__init__()
        
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
      
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(),
        )
        self.bn = nn.Identity()
        self.depth_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        
        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
    def forward(self, x, mlp_input):  
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1])) 
        x = self.reduce_conv(x) 
        depth_se = self.depth_mlp(mlp_input)[..., None, None]   
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return depth
def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=pad, bias=False),
                         build_norm_layer(norm_cfg, out_channels)[1] )
class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))
        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6
def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                        num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
def build_concat_volume( refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def warp( x, calib, down, maxdepth ):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, D, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    # B,C,D,H,W to B,H,W,C,D
    x = x.transpose(1, 3).transpose(2, 4)
    B, H, W, C, D = x.size()
    x = x.view(B, -1, C, D)
    # mesh grid
    xx = (calib / ( down * 4.))[:, None] / torch.arange(1, 1 +  maxdepth //  down,
                                                            device='cuda').float()[None, :]
    new_D =  maxdepth //  down
    xx = xx.view(B, 1, new_D).repeat(1, C, 1)
    xx = xx.view(B, C, new_D, 1)
    yy = torch.arange(0, C, device='cuda').view(-1, 1).repeat(1, new_D).float()
    yy = yy.view(1, C, new_D, 1).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), -1).float()
    vgrid = Variable(grid)
    # scale grid to [-1,1]
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(D - 1, 1) - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(C - 1, 1) - 1.0
    if float(torch.__version__[:3])>1.2:
        output = nn.functional.grid_sample(x, vgrid, align_corners=True).contiguous()
    else:
        output = nn.functional.grid_sample(x, vgrid).contiguous()
    output = output.view(B, H, W, C, new_D).transpose(1, 3).transpose(2, 4)
    return output.contiguous()

class GwcNet_volume_encoder(nn.Module):
    def __init__(self, maxdisp, out_c ,numC_input):
        super(GwcNet_volume_encoder, self).__init__()
        self.maxdisp = maxdisp
        self.num_groups = 32

        self.feature_withcam = stereofeature_net( in_channels=numC_input, mid_channels=numC_input,
                 depth_channels=64, cam_channels= 30)

        self.dres0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif3_1 = nn.Sequential(convbn_3d(32, out_c, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                       )

        self.classif3_2 = nn.Sequential( 
                                      nn.Conv3d(out_c, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, features_left, features_right, mlp_input_left, mlp_input_right, calib):
        B = features_left.shape[0]
        features=torch.cat([ features_left, features_right ],0)
        mlp_input=torch.cat([ mlp_input_left, mlp_input_right ],0)
        fea = self.feature_withcam(features,  mlp_input)
        refimg_fea, targetimg_fea = fea[:B ], fea[B: ]

 
        gwc_volume = build_gwc_volume(refimg_fea, targetimg_fea, self.maxdisp, self.num_groups)   
        volume = warp(gwc_volume , calib, down=1, maxdepth=gwc_volume.shape[2] ) 
        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        out3 = self.dres4(out2)
        cost3_1 = self.classif3_1(out3)    
        cost3 = self.classif3_2(cost3_1)     
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
 
        return {"multi_channel":cost3_1, "single_channel":pred3}


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        return  self.alpha * self.fn(x, *args, **kwargs) + x

class volume_interaction(nn.Module):  
    def __init__(self,  out_channels=1):
        super(volume_interaction, self).__init__()
        self.redir1 = nn.Conv3d(2, 32, kernel_size=3, stride=1, padding=1)
        self.dres1 = hourglass(32)
        self.redir2 = nn.Conv3d(32 , out_channels, kernel_size=3, stride=1, padding=1)

        self.lss2stereo =   attention(in_dim=1) 
        self.stereo2lss =   attention(in_dim=1)  

        self.CA3D = Residual( CA3D(32) )

    def forward(self, stereo_volume, lss_volume):   

        stereo_volume=stereo_volume.unsqueeze(1)
        lss_volume=lss_volume.unsqueeze(1)


        lss_volume_from_stereoguidance = self.lss2stereo( q = stereo_volume ,   kv = lss_volume     )
        stereo_volume_from_lssguidance = self.stereo2lss( q = lss_volume ,      kv = stereo_volume  ) 



        all_volume=torch.cat( (lss_volume_from_stereoguidance, stereo_volume_from_lssguidance ), dim=1)
        data1 = F.relu(self.redir1(all_volume))
        data2 =  self.dres1(data1)
        data2 = self.CA3D(data2)

        data3 = F.relu(self.redir2(data2))  

        data3 = data3.squeeze(1)
        data3 = F.softmax(data3, dim=1)
        return data3


class Bottle_embedding(nn.Module):

    def __init__(self, in_planes, planes, downsample=None, norm_cfg=dict(type='BN2d', requires_grad=True),):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes,1)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1 )
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.conv3 = nn.Conv2d(planes, in_planes,1 )
        self.bn3 = build_norm_layer(norm_cfg, in_planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
  

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class TempBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(TempBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Squeeze: Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Fully Connected layers with ReLU and Sigmoid
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, in_channels // reduction)
        self.fc3 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        batch_size, channels, _, _ = x.size()
        squeeze_out = self.global_avg_pool(x)  # (B, C, 1, 1)
        squeeze_out = squeeze_out.view(batch_size, channels)  # (B, C)
        
        # Excitation: Fully Connected Layers
        excitation_out = F.relu(self.fc1(squeeze_out))  # (B, C//reduction)
        excitation_out = F.relu(self.fc2(excitation_out))  # (B, C)
        excitation_out = self.fc3(excitation_out)  # (B, C)
        excitation_out = self.sigmoid(excitation_out)  # (B, C)
        
        # Reshape excitation_out and apply it to input features
        excitation_out = excitation_out.view(batch_size, channels, 1, 1)  # (B, C, 1, 1)
        return x * excitation_out  # Apply channel-wise attention



@NECKS.register_module()
class ViewTransformerLiftSplatShootVoxelGmflow(ViewTransformerLSSBEVDepth):
    def __init__(
            self, 
            loss_depth_weight,
            semkitti=False,
            imgseg=True,
            imgseg_class=21,
            lift_with_imgseg=False,
            point_cloud_range=None,
            loss_seg_weight=1.0,
            loss_depth_type='bce', 
            point_xyz_channel=0,
            point_xyz_mode='cat',
            **kwargs,
        ):
        
        super(ViewTransformerLiftSplatShootVoxelGmflow, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)

        self.flow_net = GMFlow(feature_channels=128,
                   num_scales=2,
                   upsample_factor=4,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   )
        flow_checkpoint_path = './ckpt/gmflow_with_refine_kitti-8d3b9786.pth'
        if flow_checkpoint_path is not None:
            checkpoint = torch.load(flow_checkpoint_path, map_location='cpu')
            weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
            self.flow_net.load_state_dict(weights, strict=False)
        self.occ_att = NATCrossLayer(dim=128,num_heads=8, kernel_size=7)
        self.aggregator = EmbedAggregator(channels=128)
        self.stereo_volume_net = GwcNet_volume_encoder( maxdisp=self.D, out_c=32, numC_input=self.numC_input )
        self.volume_interaction = volume_interaction()
        self.loss_depth_type = loss_depth_type
        self.cam_depth_range = self.grid_config['dbound']
        self.constant_std = 0.5
        self.point_cloud_range = point_cloud_range
        self.reduce_feat = nn.Sequential(
            nn.Conv2d(256,
                      128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
        )

        ''' Extra input for Splating: except for the image features, the lifted points should also contain their positional information '''
        self.point_xyz_mode = point_xyz_mode
        self.point_xyz_channel = point_xyz_channel
        
        assert self.point_xyz_mode in ['cat', 'add']
        if self.point_xyz_mode == 'add':
            self.point_xyz_channel = self.numC_Trans
        
        if self.point_xyz_channel > 0:
            assert self.point_cloud_range is not None
            self.point_cloud_range = torch.tensor(self.point_cloud_range)
            
            mid_channel = self.point_xyz_channel // 2
            self.point_xyz_encoder = nn.Sequential(
                nn.Linear(in_features=3, out_features=mid_channel),
                nn.BatchNorm1d(mid_channel),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=mid_channel, out_features=self.point_xyz_channel),
            )
        
        self.semkitti = semkitti
        

        ''' Auxiliary task: image-view segmentation '''
        self.imgseg = imgseg
        if self.imgseg:
            self.imgseg_class = imgseg_class
            self.loss_seg_weight = loss_seg_weight
            self.lift_with_imgseg = lift_with_imgseg
            
            # build a small segmentation head
            in_channels = self.numC_input
            self.img_seg_head = nn.Sequential(
                BasicBlock(in_channels, in_channels),
                BasicBlock(in_channels, in_channels),
                nn.Conv2d(in_channels, self.imgseg_class, kernel_size=1, padding=0),
            )
        
        self.forward_dic = {}


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape  
        gt_depths = gt_depths.view(B * N,
                                   H // self.downsample, self.downsample,
                                   W // self.downsample, self.downsample, 1)   
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()  
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample) 
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)  
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values   
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)  
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]  
        gt_depths_vals = gt_depths.clone()
        
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))  
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]  
        
        return gt_depths_vals, gt_depths.float()
    
    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):  
   
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)  
 
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)  
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(depth_preds, depth_labels, reduction='none').sum() / max(1.0, fg_mask.sum())
        
        return depth_loss

    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.cam_depth_range, constant_std=self.constant_std)
        
        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))        
        
        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]
        
        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)
        
        return depth_loss
    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)
        
        elif self.loss_depth_type == 'kld':
            depth_loss = self.get_klv_depth_loss(depth_labels, depth_preds)
        
        else:
            pdb.set_trace()
        
        return self.loss_depth_weight * depth_loss

    @force_fp32()
    def get_seg_loss(self, seg_labels):
        criterion = nn.CrossEntropyLoss(
            reduction="mean",
        )
        seg_preds = self.forward_dic['imgseg_logits']
        if seg_preds.shape[-2:] != seg_labels.shape[-2:]:
            seg_preds = F.interpolate(seg_preds, size=seg_labels.shape[1:])
        
        loss_seg = criterion(seg_preds, seg_labels.squeeze().long())
        
        return self.loss_seg_weight * loss_seg
    

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_xyz = geom_feats.clone()
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        
        if self.point_xyz_channel > 0:
            geom_xyz = geom_xyz.view(Nprime, 3)
            geom_xyz = geom_xyz[kept]
            
            pc_range = self.point_cloud_range.type_as(geom_xyz) # normalize points to [-1, 1]
            geom_xyz = (geom_xyz - pc_range[:3]) / (pc_range[3:] - pc_range[:3])
            geom_xyz = (geom_xyz - 0.5) * 2
            geom_xyz_feats = self.point_xyz_encoder(geom_xyz)
            
            if self.point_xyz_mode == 'cat':
                # concatenate image features & geometric features
                x = torch.cat((x, geom_xyz_feats), dim=1)
            
            elif self.point_xyz_mode == 'add':
                x += geom_xyz_feats
                
            else:
                raise NotImplementedError
        
        # [b, c, z, x, y] == [b, c, x, y, z]
        final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = final.permute(0, 1, 3, 4, 2)

        return final

    def forward(self, input,is_train=True,img_metas=None):
        
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

 
        feature_left, _, _, _, _, _, _, mlp_input_left=input[:8]
        feature_left = feature_left.squeeze(1)
        feature_right, _, _, _, _, _, _, mlp_input_right=input[8:16]
        feature_right = feature_right.squeeze(1)  
        calib = input[16]
        img_flow = input[-2]
        flow_feat = input[-1]
        img_l = input[-3][0].squeeze(1)
        stereo_volume = self.stereo_volume_net(feature_left,feature_right,mlp_input_left, mlp_input_right, calib)
        stereo_volume = stereo_volume["single_channel"]  
        B, N, C, H, W = x.shape
        img_feat,depth_prob,img_feat_agg,occ_mask = self.FlowGuidedTemporalAlignement(img_flow,img_l,flow_feat,x,mlp_input,intrins,img_metas)
        depth_prob = self.volume_interaction(stereo_volume, depth_prob)
        # # Lift
        volume = depth_prob.unsqueeze(1) * torch.cat([img_feat,img_feat_agg,occ_mask],dim=1).unsqueeze(2)  
        volume = volume.view(B, N, -1, self.D, H, W)   
        volume = volume.permute(0, 1, 3, 4, 5, 2)   
        # # Splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)  
        bev_feat = self.voxel_pooling(geom, volume)  

        bev_feat_t = bev_feat[:,:C,]
        bev_feat_agg = bev_feat[:,C:2*C,]
        occ_voxel = bev_feat[:,-1:,]



        bev_fuse = OcclusionGuidedVoxelRefinement(bev_feat_agg,bev_feat_t,occ_voxel)

        return bev_fuse, depth_prob

    def FlowGuidedTemporalAlignement(self, temp_imgs, cur_img, temp_feats, curr_feat, mlp_input, intrins,img_metas):
        """
        Flow-Guided Temporal Alignment
        output: img_feat: B*128*H*W
                depth_prob: B*112*H*W

        """

        temp_feats = torch.cat(temp_feats,dim=1)
        B, N, C, H, W = temp_feats.shape
        self.flow_net.eval()
        flows_f = []
        flows_b = []
        for i in range(N):
            with torch.no_grad():
                flow_res = self.flow_net(temp_imgs[i], cur_img,
                                    attn_splits_list=[2,8],
                                    corr_radius_list=[-1,4],
                                    prop_radius_list=[-1,1],
                                    )
                flows = flow_res['flow_preds'][-1]
                flows_f.append(flows[:B])
                flows_b.append(flows[B:])
        flows_b_tensor = torch.stack(flows_b,dim=1)
        flows_f_tensor = torch.stack(flows_f,dim=1)
        B, _, C, H, W = curr_feat.shape
        agg_x = []
        ref_x = []
        for i in range(B):
            ref_x_single = tools.torch_warp(temp_feats[i], flows_b_tensor[i])
            agg_x_single = self.aggregator(curr_feat[i], ref_x_single)
            ref_x.append(ref_x_single)
            agg_x.append(agg_x_single)
        ref_x = torch.stack(ref_x,dim=0)
        agg_x_g = torch.cat(agg_x,dim=0) # B C H W

        occ_feat = torch.zeros_like(curr_feat.view(B, C, H, W))
        vox_mask = torch.zeros([B, 1, H, W ]).cuda()

        _,_,flow_H,_ = flows_f[0].shape
        for idx in range(N):
            flows_f[idx] =  F.interpolate(flows_f[idx], size=(H, W), mode='bilinear', align_corners=False)/ (flow_H / H) 
            flows_b[idx] =  F.interpolate(flows_b[idx], size=(H, W), mode='bilinear', align_corners=False)/ (flow_H / H) 
            occ_mask = compute_occlusion_mask(flows_f[idx], flows_b[idx])
            occ_feat = (1-occ_mask) * ref_x[:,idx,] + occ_feat 
            vox_mask = torch.max(occ_mask,vox_mask)
        curr_feat = curr_feat.view(B, C, H, W)
        x = self.occ_att( (curr_feat).permute(0,2,3,1),(occ_feat).permute(0,2,3,1)).permute(0,3,1,2)+curr_feat
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]
        depth_prob = self.get_depth_dist(depth_digit)


        agg_x_g = self.occ_att((agg_x_g).permute(0,2,3,1),occ_feat.permute(0,2,3,1)).permute(0,3,1,2)+agg_x_g
        agg_x = self.depth_net(agg_x_g, mlp_input)
        img_feat_agg = agg_x[:, self.D:self.D + self.numC_Trans, ...]

        return img_feat,depth_prob,img_feat_agg,vox_mask


def OcclusionGuidedVoxelRefinement(agg_voxel, voxel, occ_voxel):

    weight = torch.zeros_like(voxel)
    fused_voxel = torch.zeros_like(voxel)
    occ_voxel = occ_voxel==0
    # 扩展遮挡掩码，使其与体素特征一致
    occ_voxel_expanded = occ_voxel.float().repeat(1, agg_voxel.shape[1], 1, 1, 1)  # (B, C, X, Y, Z)

    # 对于非遮挡区域
    fused_voxel = agg_voxel * (1-occ_voxel_expanded) + fused_voxel
    weight = weight + (1-occ_voxel_expanded)

    # 对于遮挡区域
    fused_voxel = fused_voxel + voxel 
    fused_weights = weight + torch.ones_like(voxel) 

    fused_weights = torch.clamp(fused_weights, min=1e-6)  
    fused_voxel = fused_voxel / fused_weights
    
    return fused_voxel

def depth_warp(depth, flow):

    B, D, H, W = depth.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing='ij'
    )
    coords = torch.stack((x, y), dim=0).float()  # (2, H, W)
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)
    target_coords = coords + flow
    target_coords_norm = torch.stack([
        2.0 * target_coords[:, 0] / (W - 1) - 1.0,
        2.0 * target_coords[:, 1] / (H - 1) - 1.0
    ], dim=1)  # (B, 2, H, W)
    target_coords_norm = target_coords_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)
    warped_depth = F.grid_sample(
        depth, target_coords_norm, mode='bilinear', padding_mode='zeros', align_corners=False
    )
    return warped_depth

def compute_occlusion_mask(fwd_flow, bwd_flow, alpha=0.01, beta=0.5):

    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = depth_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = depth_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return torch.max(fwd_occ,bwd_occ).unsqueeze(1)

