import inspect
import torch.nn as nn


class CompositeLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)
        self.forward_params = [set(inspect.signature(loss_fn.forward).parameters) for loss_fn in self.losses]

    def forward(self, logits, targets, **kwargs):
        total = 0.0
        for loss_fn, w, params in zip(self.losses, self.weights, self.forward_params):
            loss_kwargs = {key: value for key, value in kwargs.items() if key in params}
            total = total + w * loss_fn(logits, targets, **loss_kwargs)
        return total
