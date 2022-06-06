import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, x, y):
        if x.shape[0] != y.shape[0]:
            print("Predict size != target size")
            return
            
        #x = x.contiguous().view(x.shape[0], -1)
        #y = y.contiguous().view(y.shape[0], -1)

        #dice = (torch.sum(torch.mul(x, y))*2.0 + 1.0) / (torch.sum(x + y) + 1.0)
        incr = (2.0 * (x*y).sum() + 0.99) 
        union = (x.sum() + y.sum() + 0.99)
        dice = incr/union
        return 1 - dice

def DiceCoeff(x, y):
    #x = x.flatten()
    #y = y.flatten()

    if x.sum() == 0 and y.sum() == 0:
        return 1

    dice = float(2 * (y * x).sum()) / float(y.sum() + x.sum())
    return dice
