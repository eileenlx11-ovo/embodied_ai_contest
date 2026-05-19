import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from src.data.transforms import get_train_transforms, get_val_transforms
import numpy as np


class IndexedDataset(Dataset):
    """Wraps a dataset so __getitem__ returns (data, target, local_idx).

    `local_idx` is an index into this wrapper (0..len(self)-1), not the
    underlying dataset's global sample id. This keeps index-aware losses
    (e.g. ELR) correctly sized when training on a Subset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, int(idx)


def _build_datasets(cfg):
    dataset_name = cfg["data"]["dataset"]
    root = cfg["data"]["root"]
    subset_ratio = cfg["data"].get("subset", 1.0)

    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    if dataset_name == "cifar10":
        train_dataset = CIFAR10(root=root, train=True, download=True, transform=train_transform)
        val_dataset = CIFAR10(root=root, train=False, download=True, transform=val_transform)
    elif dataset_name == "imagenet":
        train_dataset = ImageFolder(root=f"{root}/train", transform=train_transform)
        val_dataset = ImageFolder(root=f"{root}/val", transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if subset_ratio < 1.0:
        n_train = int(len(train_dataset) * subset_ratio)
        indices = np.random.permutation(len(train_dataset))[:n_train]
        train_dataset = Subset(train_dataset, indices)

    return train_dataset, val_dataset


def _build_loaders(cfg, train_dataset, val_dataset):
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    pin_memory = cfg["data"]["pin_memory"]

    use_cuda = cfg.get("device", "auto") != "cpu" and torch.cuda.is_available()
    if not use_cuda:
        pin_memory = False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def get_dataloaders(cfg):
    train_dataset, val_dataset = _build_datasets(cfg)
    return _build_loaders(cfg, train_dataset, val_dataset)


def get_indexed_dataloaders(cfg):
    """Same as get_dataloaders but train set returns (image, target, index)."""
    train_dataset, val_dataset = _build_datasets(cfg)
    return _build_loaders(cfg, IndexedDataset(train_dataset), val_dataset)
