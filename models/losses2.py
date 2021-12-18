# USING KORNIA
import torch
import kornia as K
import torchvision
from matplotlib import pyplot as plt
# from models import losses as l1

def dice_loss_edge_nir(logits: torch.Tensor, target:torch.Tensor, smooth=1.) -> float:
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
    
def optimize_loss(logits: torch.Tensor, target: torch.Tensor) -> float:
    """

    Args:
        logits: (B, H, W) tensor with logits (no signmoid!)
        target: (B, H, W) Tensor with values encoded as {0: invalid, 1: neg_xxx, 2:pos_xxx}
        pos_weight: weight of positive class

    Returns:
        loss value
    """
    nir_logits = logits[:,3,:,:]
    nir_logits = torch.unsqueeze(nir_logits,1)
#     nir_imgs_show = K.tensor_to_image(nir_logits[0])
#     print(nir_imgs_show.shape)
#     plt.imshow(nir_imgs_show)
#     plt.show()
    # 0: invalied, 1: land, 2: water, 3: cloud
    water_target = target[:,2,:,:]
    water_target = torch.unsqueeze(water_target,1)
    
    magnitude_nir_logits, edge_nir_logits = K.filters.canny(nir_logits)
    magnitude_water_target, edge_water_target = K.filters.canny(water_target)
#     edge_nir_logits_show = K.tensor_to_image(edge_nir_logits[0])
#     print(edge_nir_logits_show)
#     plt.imshow(edge_nir_logits_show)
#     plt.show()
    
    loss = dice_loss_edge_nir(edge_nir_logits, edge_water_target)
    
    return loss