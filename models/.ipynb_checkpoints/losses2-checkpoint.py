# USING KORNIA
import torch
import kornia as K
import torchvision
from matplotlib import pyplot as plt
from models import losses as l1
from typing import Optional

def dice_loss_edge_water(logits: torch.Tensor, target:torch.Tensor, smooth=1.) -> float:
    """
    Dice loss masking invalids (it masks the 0 value in the target tensor)

    Args:
        logits: (B, 1, H, W) tensor 
        target: (B, 1, H, W) tensor.
        smooth: Value to avoid div by zero

    Returns:
        averaged loss (float)

    """
    logits = logits.squeeze()
    target = target.squeeze()
    assert logits.dim() == 3, f"Expected 1D tensor logits"
    assert target.dim() == 3, f"Expected 1D tensor target"
    axis_red = (2, 3)
        
    intersection = (logits * target).sum(dim=axis_red) # (B, C) tensor
    
    logit_target = logits * target
    union = logit_target.sum(dim=axis_red) + target_one_hot_without_invalid.sum(dim=axis_red)  # (B, C) tensor

    dice_score = ((2. * intersection + smooth) /
                 (union + smooth))

    loss = (1 - dice_score)  # (B, C) tensor

    return torch.mean(loss)
    
def optimize_loss(logits: torch.Tensor, target: torch.Tensor, weight:Optional[torch.Tensor]=None) -> float:
    """

    Args:
        logits: (B, H, W) tensor with logits (no signmoid!)
        target: (B, H, W) Tensor with values encoded as {0: invalid, 1: neg_xxx, 2:pos_xxx}
        pos_weight: weight of positive class

    Returns:
        loss value
    """
#     water_predict_logits = logits[:,2,:,:]
#     water_predict_logits = torch.unsqueeze(water_predict_logits,1)
    
#     # water_predict_imgs_show = K.tensor_to_image(water_predict_logits[0])
#     # print(water_predict_imgs_show.shape)
#     # plt.imshow(water_predict_imgs_show)
#     # plt.show()
#     # edge_water_target_show = K.tensor_to_image(edge_water_target[0])
#     # print(edge_water_target_show)
#     # plt.imshow(edge_water_target_show,cmap='Greys')
#     # plt.show()
    
#     # 0: invalied, 1: land, 2: water, 3: cloud
#     target = torch.unsqueeze(target,1)
#     water_target = target*(target == 2)
#     print('water_target shape: ', water_target.shape)
#     print(torch.any(water_target, 0))
#     if (torch.any(water_target, 0)):
#         magnitude_water_target, edge_water_target = K.filters.canny(water_target)
#         magnitude_water_predict_logits, edge_water_predict_logits = K.filters.canny(water_predict_logits)
#         loss_edge = dice_loss_edge_water(edge_water_predict_logits, edge_water_target)
#         loss = l1.calc_loss_mask_invalid_3(logits, target, weight) + loss_edge
#     else:
#         loss = l1.calc_loss_mask_invalid_3(logits, target, weight)
    
#     return loss
    # print('optimize loss function')
    valid = (target != 0)
    target_without_invalids = (target - 1) * valid
    
    water_region = (target_without_invalids == 1)
    water_target = target * water_region
    water_target = torch.unsqueeze(water_target,1).float()
    magnitude_water_target, edge_water_target = K.filters.canny(water_target)
    
    water_predict_logits = logits[:,1,:,:]
    water_predict = torch.unsqueeze(water_predict_logits,1).float()
    print('water predict size: ', water_predict.shape)
    magnitude_water_predict_logits, edge_water_predict_logits = K.filters.canny(water_predict)
    
#     Continue working here!
    loss_edge = dice_loss_edge_water(edge_water_predict_logits, edge_water_target)
    
    loss = l1.calc_loss_mask_invalid_3(logits, target_without_invalids, weight) + loss_edge
    return loss