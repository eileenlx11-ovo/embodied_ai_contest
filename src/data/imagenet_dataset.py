import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
from torchvision.datasets import CIFAR10, ImageFolder
from src.data.transforms import get_train_transforms, get_val_transforms
import numpy as np


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target, idx


def get_dataloaders(cfg):
    dataset_name = cfg["data"]["dataset"]
    root = cfg["data"]["root"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    pin_memory = cfg["data"]["pin_memory"]
    subset_ratio = cfg["data"].get("subset", 1.0)

    use_cuda = cfg.get("device", "auto") != "cpu" and torch.cuda.is_available()
    if not use_cuda:
        pin_memory = False

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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def get_indexed_dataloaders(cfg):
    """Same as get_dataloaders but train set returns (image, target, index)."""
    dataset_name = cfg["data"]["dataset"]
    root = cfg["data"]["root"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    pin_memory = cfg["data"]["pin_memory"]
    subset_ratio = cfg["data"].get("subset", 1.0)

    use_cuda = cfg.get("device", "auto") != "cpu" and torch.cuda.is_available()
    if not use_cuda:
        pin_memory = False

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

    indexed_train = IndexedDataset(train_dataset)

    train_loader = DataLoader(
        indexed_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader
