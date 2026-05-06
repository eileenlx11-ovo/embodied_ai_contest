import torch
import torch.nn as nn
import torch.nn.functional as F


class ELRLoss(nn.Module):
    def __init__(self, num_samples, num_classes=1000, beta=0.9, lam=3.0, cpu_buffer=True):
        super().__init__()
        self.beta = beta
        self.lam = lam
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.cpu_buffer = cpu_buffer
        self.register_buffer(
            "target_estimates",
            torch.zeros(num_samples, num_classes),
            persistent=False,
        )

    def forward(self, logits, targets, indices):
        assert indices.min().item() >= 0 and indices.max().item() < self.num_samples, "ELR indices must be global sample IDs"
        pred = F.softmax(logits, dim=-1)

        if self.cpu_buffer:
            buffer_indices = indices.detach().cpu()
            pred_for_update = pred.detach().cpu()
            with torch.no_grad():
                self.target_estimates[buffer_indices] = (
                    self.beta * self.target_estimates[buffer_indices]
                    + (1.0 - self.beta) * pred_for_update
                )
            t = self.target_estimates[buffer_indices].to(logits.device, non_blocking=True)
        else:
            with torch.no_grad():
                self.target_estimates[indices] = (
                    self.beta * self.target_estimates[indices]
                    + (1.0 - self.beta) * pred
                )
            t = self.target_estimates[indices]

        ce = F.cross_entropy(logits, targets)

        t = t.detach()
        elr_reg = torch.log(1.0 - (pred * t).sum(dim=-1) + 1e-8).mean()

        return ce + self.lam * elr_reg
