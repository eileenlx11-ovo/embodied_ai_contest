import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes=1000, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = F.one_hot(targets, self.num_classes).float()
            smooth_targets = smooth_targets * (1.0 - self.smoothing) + self.smoothing / self.num_classes
        return -(smooth_targets * log_probs).sum(dim=-1).mean()
