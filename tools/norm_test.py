import torch
import numpy as np


def occ_norm(img, img_norm_cfg=None):

    from mmcv.image.photometric import imnormalize
    
    if img_norm_cfg is None:
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        to_rgb = True
    else:
        mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(img_norm_cfg['std'], dtype=np.float32)
        to_rgb = img_norm_cfg['to_rgb']
    
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    
    return img

def pred_norm(flow_img):

    # flow_img,_,_ = self.img_transform(flow_img, post_rot, post_tran, resize=resize, 
    #         resize_dims=resize_dims, crop=crop,flip=flip, rotate=rotate)
    # flow_img = flow_img.resize((832,256))
    # if self.colorjitter and self.is_train:
    #     flow_img = self.pipeline_colorjitter(flow_img)
    # flow_img = torch.tensor(np.array(flow_img)).float().permute(2, 0, 1).contiguous()
    flow_img = torch.from_numpy(np.array(flow_img).copy()).permute(2, 0, 1)
    flow_img = flow_img/255.
    # flow_img = self.normalize_img(flow_img, img_norm_cfg=self.img_norm_cfg)
    return flow_img

def flow_norm(image):
    mean = [104.920005, 110.1753, 114.785955]
    stddev = 1 / 0.0039216
    image = (image - mean) / stddev
    image = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)
    # img = np.transpose(img, [2, 0, 1])
    return image


def convert_pred_to_occ(img_tensor, to_rgb=True):
    """Inplace normalize an image tensor with mean and std.

    Args:
        img_tensor (Tensor): Image tensor to be normalized, shape [B, C, H, W].
        mean (Tensor): The mean to be used for normalization, shape [C].
        std (Tensor): The std to be used for normalization, shape [C].
        to_rgb (bool): Whether to convert to RGB (from BGR).

    Returns:
        Tensor: The normalized image tensor.
    """
    assert img_tensor.dtype != torch.uint8, "Input tensor should not be of dtype uint8."

    # Ensure mean and std are on the same device as img_tensor
    img_tensor = img_tensor*255.
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
    mean = mean.to(img_tensor.device).view(-1, 1, 1)  # Shape: [1, C, 1, 1]
    stdinv = 1 / std.to(img_tensor.device).view(-1, 1, 1)  # Shape: [1, C, 1, 1]

    # Step 1: Convert BGR to RGB if needed (default is BGR in PyTorch)
    if to_rgb:
        img_tensor = img_tensor[ [2, 1, 0], :, :]  # Convert from BGR to RGB by channel swapping
    # Step 2: Inplace normalization (subtract mean and multiply by inverse std)
    img_tensor.sub_(mean)  # Multiply by inverse std (inplace)
    img_tensor.mul_(stdinv)  # Subtract mean (inplace)

    
    return img_tensor


def convert_pred_to_flow(normalized_img3,to_rgb=True):
    """
    将 norm3 归一化的 tensor 转换为 norm2 归一化的 tensor使用 inplace 操作。
    
    Args:
        normalized_img3 (Tensor): norm3 归一化后的图像 (C, H, W)，像素范围在 [0, 1]
    
    Returns:
        Tensor: norm2 归一化后的图像 (C, H, W)
    """
    # norm2 的均值和标准差
    mean2 = torch.tensor([104.920005, 110.1753, 114.785955], dtype=torch.float32).to(normalized_img3.device).view( -1, 1, 1)
    
    std2 = 1 / 0.0039216  # 标准化因子，相当于除以255

    # 1. 恢复像素值范围到 [0, 255]
    normalized_img3.mul_(255.0)  # 使用 inplace 操作恢复到 [0, 255]
    
    # 2. 对恢复后的图像应用 norm2 归一化
    normalized_img3.sub_(mean2)  # 使用 inplace 操作进行减法 (img - mean2)

    normalized_img3.div_(std2)   # 使用 inplace 操作进行乘法 (img * std2)

    return normalized_img3
def convert_occ_to_flow(normalized_img1):
    """
    将 norm1 归一化的 tensor 转换为 norm2 归一化的 tensor。
    
    Args:
        normalized_img1 (Tensor): norm1 归一化后的图像 (C, H, W)
    
    Returns:
        Tensor: norm2 归一化后的图像 (C, H, W)
    """
    # norm1 的均值和标准差
    mean1 = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).to(normalized_img1.device).view(3, 1, 1)
    std1 = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).to(normalized_img1.device).view(3, 1, 1)
    
    # norm2 的均值和标准差
    mean2 = torch.tensor([104.920005, 110.1753, 114.785955], dtype=torch.float32).to(normalized_img1.device).view(3, 1, 1)
    std2 = 1 / 0.0039216  # 标准化因子，相当于除以255

    # 1. 恢复 norm1 归一化图像到原始像素范围
    
     # 1. 恢复 norm1 归一化图像到原始像素范围
    normalized_img1.mul_(std1)  # inplace 操作: 恢复为原始像素值，乘以 std1
    normalized_img1.add_(mean1) # inplace 操作: 恢复为原始像素值，添加 mean1
    normalized_img1 = normalized_img1[ [2, 1, 0], :, :]  # Convert from BGR to RGB by channel swapping
    # (image - mean) / stddev
    # 2. 对恢复后的图像应用 norm2 归一化

    normalized_img1.sub_(mean2) # inplace 操作: 减去 mean2
    normalized_img1.div_(std2)  # inplace 操作: 乘以 std2

    return normalized_img1

import mmcv
from PIL import Image
flow_img = mmcv.imread('/data/B221000559-XYJ/project/WM-Project/semantickitti/sequences/00/image_2/000000.png'   , 'unchanged')
flow_img = Image.fromarray(flow_img)
flow_img = np.array(flow_img)
img_occ = occ_norm(flow_img.copy())
img_pred = pred_norm(flow_img.copy())
img_flow = flow_norm(flow_img.copy())

occ_to_flow = convert_occ_to_flow(img_occ.clone())
pred_to_flow = convert_pred_to_flow(img_pred.clone())
pred_to_occ = convert_pred_to_occ(img_pred.clone())

error_flow = torch.norm(img_flow - occ_to_flow)
error_occ = torch.norm(img_occ - pred_to_occ)
print(f"Reconstruction Error (norm1)")