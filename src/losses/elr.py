import torch
import torch.nn as nn
import torch.nn.functional as F


class ELRLoss(nn.Module):
    def __init__(self, num_samples, num_classes=1000, beta=0.9, lam=3.0):
        super().__init__()
        self.beta = beta
        self.lam = lam
        self.num_classes = num_classes
        self.register_buffer(
            "target_estimates",
            torch.zeros(num_samples, num_classes),
        )

    def forward(self, logits, targets, indices):
        pred = F.softmax(logits, dim=-1)

        with torch.no_grad():
            self.target_estimates[indices] = (
                self.beta * self.target_estimates[indices]
                + (1.0 - self.beta) * pred
            )

        ce = F.cross_entropy(logits, targets)

        t = self.target_estimates[indices].detach()
        elr_reg = torch.log(1.0 - (pred * t).sum(dim=-1) + 1e-8).mean()

        return ce + self.lam * elr_reg
