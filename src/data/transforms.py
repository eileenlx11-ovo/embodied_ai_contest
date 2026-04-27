from torchvision import transforms


def get_train_transforms(cfg):
    size = cfg["data"]["input_size"]
    if cfg["data"]["dataset"] == "cifar10":
        return transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


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
