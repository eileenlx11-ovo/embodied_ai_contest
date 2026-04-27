import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(nn.Module):
    def __init__(self, num_classes=1000, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets):
        pred = F.softmax(logits, dim=-1)
        pred = torch.clamp(pred, min=1e-8)

        one_hot = F.one_hot(targets, self.num_classes).float()

        ce = -(one_hot * torch.log(pred)).sum(dim=-1).mean()
        rce = -(pred * torch.log(one_hot.clamp(min=1e-8))).sum(dim=-1).mean()

        return self.alpha * ce + self.beta * rce
