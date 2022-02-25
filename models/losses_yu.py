from typing import Optional

import torch
import torch.nn as nn
from typing import Optional, List
import numpy as np
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

    bce *= valid  # mask out invalid pixels > 屏蔽無效像素

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
    Weighted BCE and Dice loss masking invalids:
     bce_loss * bce_weight + dice_loss * (1-bce_weight)
    Args:
        logits: (B, C, H, W) tensor with logits (no softmax!)
        target: (B, H, W) tensor. int values in {0,...,C} it considers 0 to be the invalid value
        bce_weight: weight of the bce part
        weight: (C, ) tensor. weight per class value to cross_entropy function

    Returns:

    """

    bce = focal_loss_mask_invalid(logits, target, weight=weight)
    #ce改focal

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
                                                                 num_classes=pred.shape[1]).permute(0, 3, 1, 2)  #將tensor的维度换位
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
    
    fc = focal_loss_mask_invalid(logits, target, weight=weight, gamma=0, alpha=0.0)

    # Dice Loss
    # Perform spatial softmax over NxCxHxW
    iou_loss = iou_loss_mask_invalid(logits, target) # (B, C)

    # Weighted sum
    return fc * bce_weight + iou_loss * (1 - bce_weight)


###
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

def ssim1(logits: torch.Tensor, target:torch.Tensor, window_size=11, window=None, size_average=True, full=False, val_range=None):
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

def calc_loss_mask_invalid_4(logits: torch.Tensor, target:torch.Tensor, bce_weight:float=1/3, weight:Optional[torch.Tensor]=None) -> float:
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
    
    ssim_loss = ssim1(logits, target)

    # Weighted sum
    return (1-ssim_loss)* bce_weight + fc * bce_weight + iou_loss * bce_weight


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    
    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
        
    pow1 = mcs ** weights
    pow2 = mssim ** weights
    
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


#####
def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g .unsqueeze(0).unsqueeze(0)  # 增加兩個維度
    # squeeze() 能夠去除維度、unsqueeze() 則能增加維度


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
# input.cpu()
    input=input.cpu()
    input=torch.Tensor(input) # permute(0,3,1,2)
    
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    # shape除了第一個和最後一個 (C,H)
    
#     if len(input.shape) == 3:
#         conv = F.conv2d 
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):  # 0>C, 1>H 
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        if s >= win.shape[-1]:  # 最後一項 w
            win = torch.FloatTensor().to(win)
            out = out.view(-1,2)
            # -1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成4列
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
            
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out

def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
        
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
#     if win is not None:  # set win_size
#         win_size = win.shape[-1]
#     if win is None:
#         win = _fspecial_gauss_1d(win_size, win_sigma)
#         win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))


    win = win.to(X.device, dtype=X.dtype)
    # tensor.to() 这个函数功能是产生一个新的tensor，并不会改变原数据
#     Y = torch.squeeze(Y)
    
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim2(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
#     if not X.shape == Y.shape:
#         raise ValueError("Input images should have the same dimensions.")

    
    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
   #     Y = Y.squeeze(dim=d)
    #Y = torch.squeeze(Y)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

#     if not X.type() == Y.type():
#         raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = torch.FloatTensor(weights, device=X.device, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)
    
def calc_loss_mask_invalid_5(logits: torch.Tensor, target:torch.Tensor, weight:Optional[torch.Tensor]=None) -> float:
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
    
    ssim_loss = ssim2(logits, target)

    # Weighted sum
    return fc + iou_loss + ssim_loss