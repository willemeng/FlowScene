
import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import BaseModule
import torch.nn.functional as F



class EmbedAggregator(BaseModule):
    """Embedding convs to aggregate multi feature maps.

    This module is proposed in "Flow-Guided Feature Aggregation for Video
    Object Detection". `FGFA <https://arxiv.org/abs/1703.10025>`_.

    Args:
        num_convs (int): Number of embedding convs.
        channels (int): Channels of embedding convs. Defaults to 256.
        kernel_size (int): Kernel size of embedding convs, Defaults to 3.
        norm_cfg (dict): Configuration of normlization method after each
            conv. Defaults to None.
        act_cfg (dict): Configuration of activation method after each
            conv. Defaults to dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 num_convs=1,
                 channels=256,
                 kernel_size=3,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(EmbedAggregator, self).__init__(init_cfg)
        assert num_convs > 0, 'The number of convs must be bigger than 1.'
        self.embed_convs = nn.ModuleList()
        for i in range(num_convs):
            if i == num_convs - 1:
                new_norm_cfg = None
                new_act_cfg = None
            else:
                new_norm_cfg = norm_cfg
                new_act_cfg = act_cfg
            self.embed_convs.append(
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    norm_cfg=new_norm_cfg,
                    act_cfg=new_act_cfg))

    def forward(self, x, ref_x):
        """Aggregate reference feature maps `ref_x`.

        The aggregation mainly contains two steps:
        1. Computing the cos similarity between `x` and `ref_x`.
        2. Use the normlized (i.e. softmax) cos similarity to weightedly sum
        `ref_x`.

        Args:
            x (Tensor): of shape [1, C, H, W]
            ref_x (Tensor): of shape [N, C, H, W]. N is the number of reference
                feature maps.

        Returns:
            Tensor: The aggregated feature map with shape [1, C, H, W].
        """
        assert len(x.shape) == 4 and len(x) == 1, \
            "Only support 'batch_size == 1' for x"
        x_embed = x
        for embed_conv in self.embed_convs:
            x_embed = embed_conv(x_embed)
        # x_embed = x_embed / x_embed.norm(p=2, dim=1, keepdim=True)

        ref_x_embed = ref_x
        for embed_conv in self.embed_convs:
            ref_x_embed = embed_conv(ref_x_embed)
        # ref_x_embed = ref_x_embed / ref_x_embed.norm(p=2, dim=1, keepdim=True)

        cos_sim = F.cosine_similarity(ref_x_embed, x_embed, dim=1, eps=1e-8)
        cos_sim = cos_sim.unsqueeze(1)  # Add channel dimension for broadcasting


        ada_weights = cos_sim.softmax(dim=0)

        agg_x = torch.sum(ref_x * ada_weights, dim=0, keepdim=True)

        return agg_x


def create_voxel_grid(voxel_size, grid_range):
    x = torch.arange(grid_range[0], grid_range[1], step=voxel_size[0])
    y = torch.arange(grid_range[2], grid_range[3], step=voxel_size[1])
    z = torch.arange(grid_range[4], grid_range[5], step=voxel_size[2])
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    voxel_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (X, Y, Z, 3)
    return voxel_grid

def voxel_to_image_projection(voxel_grid, intrinsics, B):
    # 扩展到批次维度
    voxel_grid = voxel_grid.reshape(-1, 3).T  # (3, N)
    voxel_grid_h = torch.cat([voxel_grid, torch.ones(1, voxel_grid.shape[1]).cuda()], dim=0)  # Homogeneous (4, N)
    intrinsics = intrinsics[:,:,:3,:3]
    # 投影到图像空间
    pixel_coords = intrinsics[:,0,].bmm(voxel_grid_h[:3, :].unsqueeze(0).expand(B, -1, -1))  # (B, 3, N)
    pixel_coords = pixel_coords[:, :2, :] / pixel_coords[:, 2:3, :]  # Normalize by depth (B, 2, N)

    # 获取像素坐标
    pixel_x = pixel_coords[:, 0, :].long()
    pixel_y = pixel_coords[:, 1, :].long()

    return pixel_x, pixel_y


def project_occ_mask_to_voxel(occ_mask, depth_bins, intrinsics, voxel_size=[128,128,16], grid_range=[0, 51.2, -25.6, 25.6, -2, 4.4]):
    """
    将遮挡掩码从图像空间投影到体素空间。
    Args:
        occ_mask: 遮挡掩码 (B, H, W)。
        depth_bins: 深度分布，用于对体素和图像进行深度一致性约束 (B, D, H, W)。
        intrinsics: 相机内参矩阵 (B, 3, 3)。
        voxel_size: 体素分辨率 [x_res, y_res, z_res]。
        grid_range: 体素网格范围 [x_min, x_max, y_min, y_max, z_min, z_max]。
    Returns:
        occ_voxel: 投影到体素网格的遮挡掩码 (B, X, Y, Z)。
    """
    B,_, H, W = occ_mask.shape
    D = depth_bins.shape[1]

    # 创建体素网格
    voxel_grid = create_voxel_grid(voxel_size, grid_range).to(occ_mask.device)  # (X, Y, Z, 3)
    voxel_shape = voxel_grid.shape[:3]  # (X, Y, Z)

    # 投影到图像空间
    pixel_x, pixel_y = voxel_to_image_projection(voxel_grid, intrinsics, B)  # (B, N)

    # 限制坐标范围
    pixel_x = torch.clamp(pixel_x, 0, W - 1)
    pixel_y = torch.clamp(pixel_y, 0, H - 1)
    occ_mask = 1 - occ_mask
    # 获取每个体素对应的遮挡值
    occ_values = occ_mask[torch.arange(B).unsqueeze(1),:, pixel_y, pixel_x]  # (B, N)

    # 将遮挡值 reshape 为体素形状
    occ_voxel = occ_values.view(B,1, *voxel_shape)  # (B,1, X, Y, Z)

    return occ_voxel
