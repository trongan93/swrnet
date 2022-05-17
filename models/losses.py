from typing import Optional

import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn
from math import exp

import warnings
import torch.nn.functional as F
from torch.autograd import Variable

def dice_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, smooth=1.) -> float:
    """
    Dice loss masking invalids (it masks the 0 value in the target tensor)

    Args:
        logits: (B, C, H, W) tensor with logits (no softmax)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        smooth: Value to avoid div by zero

    Returns:
        averaged loss (float)

    """
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    pred = torch.softmax(logits, dim=1)
    valid = (target != 0) # (B, H, W) tensor
    target_without_invalids = (target - 1) * valid  # Set invalids to land

    target_one_hot_without_invalid = torch.nn.functional.one_hot(target_without_invalids,
                                                                 num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    axis_red = (2, 3) # H, W reduction

    pred_valid = pred * valid.unsqueeze(1).float()  # # Set invalids to 0 (all values in prob tensor are 0

    intersection = (pred_valid * target_one_hot_without_invalid).sum(dim=axis_red) # (B, C) tensor

    union = pred_valid.sum(dim=axis_red) + target_one_hot_without_invalid.sum(dim=axis_red)  # (B, C) tensor

    dice_score = ((2. * intersection + smooth) /
                 (union + smooth))

    loss = (1 - dice_score)  # (B, C) tensor

    return torch.mean(loss)

def cross_entropy_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, weight:Optional[torch.Tensor]=None) -> float:
    """
    F.cross_entropy loss masking invalids (it masks the 0 value in the target tensor)

    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:
        averaged loss

    """
#     print('trongan93-test losses: Cross entropy loss function for mask')
#     print(logits.dim())
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    valid = (target != 0)
    target_without_invalids = (target - 1) * valid

    # BCE Loss (ignoring invalid values)
    bce = F.cross_entropy(logits, target_without_invalids,
                          weight=weight, reduction='none')  # (B, 1, H, W)

    bce *= valid  # mask out invalid pixels

    return torch.sum(bce / (torch.sum(valid) + 1e-6))


def calc_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted BCE and Dice loss masking invalids:
     bce_loss * bce_weight + dice_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """

    bce = cross_entropy_loss_mask_invalid(logits, target, weight=weight)

    # Dice Loss
    # Perform spatial softmax over NxCxHxW
    dice = dice_loss_mask_invalid(logits, target) # (B, C)

    # Weighted sum
    return bce * bce_weight + dice * (1 - bce_weight)

def calc_loss_mask_invalid_original_unet(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted BCE and Dice loss masking invalids:
     bce_loss * bce_weight + dice_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """

    bce = cross_entropy_loss_mask_invalid(logits, target, weight=weight)
    return bce


def binary_cross_entropy_loss_mask_invalid(logits: torch.Tensor, target: torch.Tensor,
                                           pos_weight:Optional=None) -> float:
    """

    Args:
        logits: (B, H, W) tensor with logits (no signmoid!)
        target: (B, H, W) Tensor with values encoded as {0: invalid, 1: neg_xxx, 2:pos_xxx}
        pos_weight: weight of positive class

    Returns:

    """
    assert logits.dim() == 3, f"Unexpected shape of logits. Logits: {logits.shape} target: {target.shape}"
    assert target.dim() == 3, f"Unexpected shape of target. Logits: {logits.shape} target: {target.shape}"

    valid = (target != 0)
    target_without_invalids = (target - 1) * valid

    pixelwise_bce = F.binary_cross_entropy_with_logits(logits, target_without_invalids.float(), reduction='none',
                                                       pos_weight=pos_weight)

    pixelwise_bce *= valid  # mask out invalid pixels

    return torch.sum(pixelwise_bce / (torch.sum(valid) + 1e-6))

def calc_loss_multioutput_logistic_mask_invalid(logits: torch.Tensor, target: torch.Tensor,
                                                pos_weight_problem:Optional[List[float]]=None,
                                                weight_problem:Optional[List[float]]=None) -> float:

    assert logits.dim() == 4, "Unexpected shape of logits"
    assert target.dim() == 4, "Unexpected shape of target"

    if weight_problem is None:
        weight_problem = [ 1/logits.shape[1] for _ in range(logits.shape[1])]

    total_loss = 0
    for i in range(logits.shape[1]):
        pos_weight = torch.tensor(pos_weight_problem[i], device=logits.device) if pos_weight_problem is not None else None
        curr_loss = binary_cross_entropy_loss_mask_invalid(logits[:, i], target[:, i], pos_weight=pos_weight)
        total_loss += curr_loss*weight_problem[i]

    return total_loss


# Optimize by Trong-An Bui (trongan93@gmail.com) 
# - Change the Cross Entropy Loss to Focal Loss function

def focal_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, weight:Optional[torch.Tensor]=None, gamma=2, alpha=.25) -> float:
    """
    F.cross_entropy loss masking invalids (it masks the 0 value in the target tensor)

    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:
        averaged loss

    """
#     print('trongan93-test losses: Cross entropy loss function for mask')
#     print(logits.dim())
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    valid = (target != 0)
    target_without_invalids = (target - 1) * valid

    ce_loss = F.cross_entropy(logits, target_without_invalids,reduction='none',weight=weight)
    ce_loss *= valid # mask out invalid pixels
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha*(1 - pt) ** gamma * ce_loss)
    
    return torch.sum(focal_loss / (torch.sum(valid) + 1e-6))

def calc_loss_mask_invalid_2(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted Focal loss and Dice loss masking invalids:
     focal_loss * focal_weight + dice_loss * (1-focal_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """

    # bce = focal_loss_mask_invalid(logits, target, weight=weight)
    bce = focal_loss_mask_invalid(logits, target, weight=weight, gamma=0, alpha=1)

    # Dice Loss
    # Perform spatial softmax over NxCxHxW
    dice = dice_loss_mask_invalid(logits, target) # (B, C)

    # Weighted sum
    return bce * bce_weight + dice * (1 - bce_weight)

# Optimize 2 by Trong-An Bui (trongan93@gmail.com) 
# - Change the Cross Entropy Loss to Focal Loss function
# - Different loss functions for RGB and Nir
def iou_loss_mask_invalid(logits: torch.Tensor, target:torch.Tensor, smooth=1.) -> float:
    """
    IoU loss masking invalids (it masks the 0 value in the target tensor)

    Args:
        logits: (B, C, H, W) tensor with logits (no softmax)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        smooth: Value to avoid div by zero

    Returns:
        averaged loss (float)

    """
    assert logits.dim() == 4, f"Expected 4D tensor logits"
    assert target.dim() == 3, f"Expected 3D tensor target"

    pred = torch.softmax(logits, dim=1)
    valid = (target != 0) # (B, H, W) tensor
    target_without_invalids = (target - 1) * valid  # Set invalids to land

    target_one_hot_without_invalid = torch.nn.functional.one_hot(target_without_invalids,
                                                                 num_classes=pred.shape[1]).permute(0, 3, 1, 2)
    axis_red = (2, 3) # H, W reduction

    pred_valid = pred * valid.unsqueeze(1).float()  # # Set invalids to 0 (all values in prob tensor are 0

    intersection = (pred_valid * target_one_hot_without_invalid).sum(dim=axis_red) # (B, C) tensor
    total = (pred_valid + target_one_hot_without_invalid).sum(dim=axis_red) # (B, C) tensor

    union = total - intersection
    iou_score = (intersection + smooth)/(union + smooth)
    loss = (1 - iou_score)  # (B, C) tensor
    return torch.mean(loss)

def calc_loss_mask_invalid_3(logits: torch.Tensor, target:torch.Tensor,
                           bce_weight:float=0.5, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted Focal loss and IoU loss masking invalids:
     focal_loss * bce_weight + iou_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """
    # print(f"Shape of logits: {logits.shape}")
    logits_rgb = logits[:,0:2,:,:]
    # print(f"Shape of logits_rgb: {logits_rgb.shape}")
    
    fc = focal_loss_mask_invalid(logits, target, weight=weight, gamma=5, alpha=0.001)

    # Dice Loss
    # Perform spatial softmax over NxCxHxW
    iou_loss = iou_loss_mask_invalid(logits, target) # (B, C)

    # Weighted sum
    return fc * bce_weight + iou_loss * (1 - bce_weight)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    # 所有元素的和等於1
    return gauss/gauss.sum()

def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # 使用高斯函數生成一維張量
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 該一維張量與其轉置交叉相乘得到二維張量(這保持了高斯特性) # 增加兩個額外的維度，將其轉換爲四維
    window =  Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    # .expand將引數中的列表合併到原列表的末尾
    # 返回一个内存连续的有相同数据的tensor，如果原tensor内存连续，则返回原tensor
    # Variable就是变量的意思。实质上也就是可以变化的量，区别于int变量，它是一种可以变化的变量，这正好就符合了反向传播，参数更新的属性。
    return window

def ssim(logits: torch.Tensor, target:torch.Tensor, window_size=13, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    
    logits = logits.type(torch.float).cpu()
    target = target.type(torch.float).cpu()
    
    if val_range is None:
        if torch.max(logits) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(logits) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    _, channel, height, width = logits.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel)  # .to(logits.device)
    
    # target = torch.Tensor(target)
#     int size = 0
#     Tensor target = torch::tensor(size, dtype(int))
    target = torch.Tensor(target).unsqueeze(0)
    target = target.permute(1,0,2,3)
    mu1 = F.conv2d(logits, window, padding=padd, groups=channel)
    mu2 = F.conv2d(target, window, padding=padd, groups=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(logits * logits, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv2d(logits * target, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    if full:
        return ret, cs
    
    return ret 

def calc_loss_mask_invalid_4(logits: torch.Tensor, target:torch.Tensor, weight:Optional[torch.Tensor]=None) -> float:
    """
    Weighted Focal loss and IoU loss masking invalids:
     focal_loss * bce_weight + iou_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function
    Returns:
    """
    # print(f"Shape of logits: {logits.shape}")
    logits_rgb = logits[:,0:2,:,:]
    # print(f"Shape of logits_rgb: {logits_rgb.shape}")
    
    fc = focal_loss_mask_invalid(logits, target, weight=weight, gamma=0, alpha=0.0)

    # Dice Loss
    # Perform spatial softmax over NxCxHxW
    iou_loss = iou_loss_mask_invalid(logits, target) # (B, C)
    
    ssim_loss = ssim(logits, target)
    ss_loss = 1-ssim_loss
    
    loss = ss_loss/0.73694 + iou_loss/0.60932 + fc/0.33947
    # Weighted sum
    return loss*0.1 ,ss_loss
