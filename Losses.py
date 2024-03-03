import torch.nn as nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight.float()  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.3, beta=0.7, smooth=1e-10):
        super(TverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2)

        tp = torch.sum(inputs * targets, dim=(0, 2, 3))
        fn = torch.sum(targets * (1 - inputs), dim=(0, 2, 3))
        fp = torch.sum((1 - targets) * inputs, dim=(0, 2, 3))

        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky = torch.sum(ti)

        return 1 - tversky
