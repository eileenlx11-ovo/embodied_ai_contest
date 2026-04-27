from src.models.resnet import build_resnet50


def build_model(cfg):
    name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]

    if name == "resnet50":
        return build_resnet50(num_classes)
    else:
        raise NotImplementedError(f"Model '{name}' not implemented yet. Available: resnet50")
