import torch.nn as nn
from src.losses.sce import SCELoss
from src.losses.gce import GCELoss
from src.losses.peer_loss import PeerLoss
from src.losses.elr import ELRLoss
from src.losses.label_smoothing import LabelSmoothingLoss
from src.losses.composite_loss import CompositeLoss


def build_loss(cfg, num_samples=None):
    tc = cfg["training"]
    loss_name = tc.get("loss", "ce")
    num_classes = cfg["model"]["num_classes"]

    if loss_name == "ce":
        smoothing = tc.get("label_smoothing", 0.0)
        return nn.CrossEntropyLoss(label_smoothing=smoothing)

    if loss_name == "label_smoothing":
        smoothing = tc.get("label_smoothing", 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)

    if loss_name == "sce":
        lp = tc.get("loss_params", {})
        return SCELoss(
            num_classes=num_classes,
            alpha=lp.get("alpha", 0.1),
            beta=lp.get("beta", 1.0),
        )

    if loss_name == "gce":
        lp = tc.get("loss_params", {})
        return GCELoss(num_classes=num_classes, q=lp.get("q", 0.7))

    if loss_name == "peer_loss":
        lp = tc.get("loss_params", {})
        return PeerLoss(num_classes=num_classes, lam=lp.get("lam", 0.5))

    if loss_name == "elr":
        assert num_samples is not None, "ELR requires num_samples"
        lp = tc.get("loss_params", {})
        return ELRLoss(
            num_samples=num_samples,
            num_classes=num_classes,
            beta=lp.get("beta", 0.9),
            lam=lp.get("lam", 3.0),
        )

    raise ValueError(f"Unknown loss: {loss_name}. Available: ce, label_smoothing, sce, gce, peer_loss, elr")
