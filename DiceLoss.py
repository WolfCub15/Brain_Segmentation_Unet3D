import torch
import torch.nn as nn
import numpy as np

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth = 1, p = 2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0]
        
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        dice = (torch.sum(torch.mul(predict, target))*2 + self.smooth) / (torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth)

        return 1 - dice

def DiceCoeff(x, y, c = 0.5):
    x = x.flatten()
    y = y.flatten()
    x[x > c] = np.float32(1)
    x[x < c] = np.float32(0)
    dice = float(2 * (y * x).sum()) / float(y.sum() + x.sum())
    return dice
