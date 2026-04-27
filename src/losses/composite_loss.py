import torch.nn as nn


class CompositeLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(self, logits, targets, **kwargs):
        total = 0.0
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(logits, targets, **kwargs)
        return total
