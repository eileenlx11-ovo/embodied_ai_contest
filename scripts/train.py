import sys
import os
import argparse
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seed import set_seed
from src.data.imagenet_dataset import get_dataloaders
from src.models.build_model import build_model
from src.losses.base_loss import build_loss
from src.trainers.base_trainer import BaseTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="auto=自动检测, cuda=强制GPU, cpu=强制CPU")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.subset is not None:
        cfg["data"]["subset"] = args.subset
    cfg["device"] = args.device

    set_seed(cfg.get("seed", 42))

    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg)
    criterion = build_loss(cfg)

    trainer = BaseTrainer(model, train_loader, val_loader, criterion, cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()
