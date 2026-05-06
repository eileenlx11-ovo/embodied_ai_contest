import torch.nn as nn
from src.losses.sce import SCELoss
from src.losses.gce import GCELoss
from src.losses.peer_loss import PeerLoss
from src.losses.elr import ELRLoss
from src.losses.label_smoothing import LabelSmoothingLoss
from src.losses.composite_loss import CompositeLoss


def _build_single_loss(loss_name, params, num_classes, num_samples):
    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.0))

    if loss_name == "label_smoothing":
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=params.get("smoothing", 0.1))

    if loss_name == "sce":
        return SCELoss(
            num_classes=num_classes,
            alpha=params.get("alpha", 0.1),
            beta=params.get("beta", 1.0),
        )

    if loss_name == "gce":
        return GCELoss(num_classes=num_classes, q=params.get("q", 0.7))

    if loss_name == "peer_loss":
        return PeerLoss(num_classes=num_classes, lam=params.get("lam", 0.5))

    if loss_name == "elr":
        assert num_samples is not None, "ELR requires num_samples"
        return ELRLoss(
            num_samples=num_samples,
            num_classes=num_classes,
            beta=params.get("beta", 0.9),
            lam=params.get("lam", 3.0),
            cpu_buffer=params.get("cpu_buffer", True),
        )

    raise ValueError(f"Unknown loss: {loss_name}. Available: ce, label_smoothing, sce, gce, peer_loss, elr, composite")


def build_loss(cfg, num_samples=None):
    tc = cfg["training"]
    loss_name = tc.get("loss", "ce")
    num_classes = cfg["model"]["num_classes"]
    lp = tc.get("loss_params", {})

    if loss_name == "ce":
        return _build_single_loss("ce", {"label_smoothing": tc.get("label_smoothing", 0.0)}, num_classes, num_samples)

    if loss_name == "composite":
        specs = lp.get("losses", [])
        assert specs, "CompositeLoss requires loss_params.losses"
        losses = [
            _build_single_loss(spec["name"], spec.get("params", {}), num_classes, num_samples)
            for spec in specs
        ]
        weights = [spec.get("weight", 1.0) for spec in specs]
        return CompositeLoss(losses, weights)

    return _build_single_loss(loss_name, lp, num_classes, num_samples)
