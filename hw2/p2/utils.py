#utils
import math
import scipy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from PIL import Image

def calcu_iou(b_pred, b_labels):
    b = b_pred.size(0)
    b_pred = b_pred.max(1)[1]

    pred, labels = b_pred.detach().cpu().numpy(), b_labels.detach().cpu().numpy()

    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred==i)
        tp_fn = np.sum(labels==i)
        tp = np.sum((pred==i)*(labels==i))
        iou = tp/(tp_fp+tp_fn-tp)
        mean_iou += 0 if math.isnan(iou) else iou/6

    return mean_iou 

def mask2label(filepath):
    mask = Image.open(filepath).convert('RGB')
    mask = np.array(mask).astype(np.uint8)
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    masks = np.empty((512,512))
    masks[ mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[ mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[ mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[ mask == 2] = 3  # (Green: 010) Forest land 
    masks[ mask == 1] = 4  # (Blue: 001) Water 
    masks[ mask == 7] = 5  # (White: 111) Barren land 
    masks[ mask == 0] = 6  # (Black: 000) Unknown
    masks[ mask == 4] = 6  # (Red: 100) Unknown
    return masks

def mean(ls):
    return sum(ls)/len(ls)

def accuracy(outputs, labels):
    outputs = outputs.permute(0,2,3,1)
    outputs, labels = outputs.detach().cpu(), labels.detach().cpu()
    _, preds = outputs.max(-1)
    preds = preds.view(-1)
    labels = labels.view(-1)
    return float(preds.eq(labels).sum()) / preds.size(0)

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

    
