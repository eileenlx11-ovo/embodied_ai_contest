import torch.nn as nn


def build_loss(cfg):
    smoothing = cfg["training"].get("label_smoothing", 0.0)
    return nn.CrossEntropyLoss(label_smoothing=smoothing)
