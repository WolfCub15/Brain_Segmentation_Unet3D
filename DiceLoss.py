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
            
        x = x.contiguous().view(x.shape[0], -1)
        y = y.contiguous().view(y.shape[0], -1)

        dice = (torch.sum(torch.mul(x, y))*2 + 1) / (torch.sum(x.pow(2) + y.pow(2)) + 1)

        return 1 - dice

def DiceCoeff(x, y, c = 0.5):
    x = x.flatten()
    y = y.flatten()
    xx = np.power(x,2)
    yy = np.power(y,2)
    dice = float(2 * (y * x).sum() + 1) / float(yy.sum() + xx.sum() + 1)
    return dice
