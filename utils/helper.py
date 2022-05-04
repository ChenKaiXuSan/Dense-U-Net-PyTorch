import numpy as np

import torch 
import torch.nn.functional as F

from utils.utils import tensor2var

def compute_iou(model, loader, threshold=0.3):
    '''
    computes accuracy on the dataset wrapped in a loader

    Args:
        model (nn): test model
        loader (function): test data loader
        threshold (float, optional): threshold for the output. Defaults to 0.3.
    '''

    valloss = 0

    with torch.no_grad():

        for i_step, (data, target) in enumerate(loader):

            # configure input 
            data = tensor2var(data)    
            target = tensor2var(target)

            outputs = model(data)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

    return valloss / i_step

def dice_coef_metric(inputs, target):
    '''
    calc the IoU, for the segmentation quality metric

    Args:
        inputs (tensor): inputs image 
        target (tensor): target image

    Returns:
        _type_: the IoU
    '''    

    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union


# segmentation loss function 
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

# segmentation loss
def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred).clone()

    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss
