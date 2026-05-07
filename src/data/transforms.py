import torch
import numpy as np
from torchvision import transforms


def get_train_transforms(cfg):
    size = cfg["data"]["input_size"]
    aug_cfg = cfg.get("augmentation", {})

    if cfg["data"]["dataset"] == "cifar10":
        return transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    t = [
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]

    if aug_cfg.get("randaug", False):
        n = aug_cfg.get("randaug_n", 2)
        m = aug_cfg.get("randaug_m", 9)
        t.append(transforms.RandAugment(num_ops=n, magnitude=m))

    if aug_cfg.get("color_jitter", False):
        t.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))

    t.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if aug_cfg.get("random_erasing", False):
        t.append(transforms.RandomErasing(p=0.25))

    return transforms.Compose(t)


def get_val_transforms(cfg):
    size = cfg["data"]["input_size"]
    if cfg["data"]["dataset"] == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.Resize(int(size / 0.875)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class Mixup:
    """Placeholder — defined but not yet wired into BaseTrainer/NoisyTrainer.

    Returns soft targets, so a downstream trainer needs to use a soft-target
    loss (e.g. `F.cross_entropy` with class-probability targets). Integrate
    inside the trainer's train_one_epoch before `self.criterion(...)`.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, images, targets, num_classes):
        if self.alpha <= 0:
            return images, targets
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        mixed_images = lam * images + (1 - lam) * images[idx]
        targets_a = torch.nn.functional.one_hot(targets, num_classes).float()
        targets_b = torch.nn.functional.one_hot(targets[idx], num_classes).float()
        mixed_targets = lam * targets_a + (1 - lam) * targets_b
        return mixed_images, mixed_targets


class CutMix:
    """Placeholder — defined but not yet wired into BaseTrainer/NoisyTrainer.

    Same integration caveat as `Mixup`: produces soft targets and needs
    trainer-side hookup before it takes effect.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, images, targets, num_classes):
        if self.alpha <= 0:
            return images, targets
        lam = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        B, C, H, W = images.shape

        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = np.random.randint(H)
        cx = np.random.randint(W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)

        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

        targets_a = torch.nn.functional.one_hot(targets, num_classes).float()
        targets_b = torch.nn.functional.one_hot(targets[idx], num_classes).float()
        mixed_targets = lam * targets_a + (1 - lam) * targets_b
        return images, mixed_targets
