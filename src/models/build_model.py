from src.models.resnet import build_resnet50
from src.models.starnet import build_starnet_s1, build_starnet_s2
from src.models.convnext_v2 import build_convnext_v2_t


def build_model(cfg):
    name = cfg["model"]["name"]
    num_classes = cfg["model"]["num_classes"]
    drop_path = cfg["model"].get("drop_path_rate", 0.0)

    if name == "resnet50":
        return build_resnet50(num_classes)
    if name == "starnet_s1":
        return build_starnet_s1(num_classes, drop_path)
    if name == "starnet_s2":
        return build_starnet_s2(num_classes, drop_path)
    if name == "convnext_v2_t":
        return build_convnext_v2_t(num_classes, drop_path)

    raise NotImplementedError(
        f"Model '{name}' not implemented. "
        f"Available: resnet50, starnet_s1, starnet_s2, convnext_v2_t"
    )
