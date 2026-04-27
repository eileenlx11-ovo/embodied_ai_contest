import torch
import torch.nn as nn
import torch.nn.functional as F


class GCELoss(nn.Module):
    def __init__(self, num_classes=1000, q=0.7):
        super().__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits, targets):
        pred = F.softmax(logits, dim=-1)
        pred = torch.clamp(pred, min=1e-8)

        one_hot = F.one_hot(targets, self.num_classes).float()
        pred_y = (pred * one_hot).sum(dim=-1)

        loss = (1.0 - pred_y ** self.q) / self.q
        return loss.mean()
