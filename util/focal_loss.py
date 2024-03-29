import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=4):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)
    
