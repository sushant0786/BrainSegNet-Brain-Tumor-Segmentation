import torch
from torch import nn




POS_WEIGHT    = 2.0           
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pos_w = torch.tensor([POS_WEIGHT], device=DEVICE)
bce_logits = nn.BCEWithLogitsLoss(pos_weight=pos_w)

def dice_coef(probs, targets, eps=1e-7):
    inter = (probs * targets).sum(dim=(2,3))
    union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice  = (2*inter + eps) / (union + eps)
    return dice.mean()

def dice_loss(probs, targets): return 1 - dice_coef(probs, targets)

def criterion(logits, targets):
    bce  = bce_logits(logits, targets)
    probs = torch.sigmoid(logits)
    return bce + dice_loss(probs, targets)
