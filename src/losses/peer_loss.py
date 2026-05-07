import torch
import torch.nn as nn


class PeerLoss(nn.Module):
    """Simplified Peer Loss (Liu & Guo, 2020; arxiv:1908.07229).

    The original formulation uses two independently sampled pairs —
    `CE(f(x_i), y_i) - lam * CE(f(x_{i1}), y_{i2})`, with `x_{i1}` and
    `y_{i2}` drawn from separate samples. This implementation uses the
    same batch of logits paired with permuted targets, which is cheaper
    (no second forward pass) but not identical to the paper. Use it as a
    noise-robust regularizer, not as a strict reproduction.
    """

    def __init__(self, num_classes=1000, lam=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lam = lam

    def forward(self, logits, targets):
        main_loss = self.ce(logits, targets)

        peer_indices = torch.randperm(targets.size(0), device=targets.device)
        peer_loss = self.ce(logits, targets[peer_indices])

        return main_loss - self.lam * peer_loss
