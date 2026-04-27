import torch
import torch.nn as nn


class PeerLoss(nn.Module):
    def __init__(self, num_classes=1000, lam=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lam = lam

    def forward(self, logits, targets):
        main_loss = self.ce(logits, targets)

        peer_indices = torch.randperm(targets.size(0), device=targets.device)
        peer_loss = self.ce(logits, targets[peer_indices])

        return main_loss - self.lam * peer_loss
