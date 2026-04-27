import torch.nn as nn
from torchvision.models import resnet50


def build_resnet50(num_classes: int = 1000) -> nn.Module:
    model = resnet50(weights=None)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
