import sys
import os
import argparse
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.seed import set_seed
from src.data.imagenet_dataset import get_dataloaders, get_indexed_dataloaders
from src.models.build_model import build_model
from src.losses.base_loss import build_loss
from src.trainers.base_trainer import BaseTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="覆盖 cfg.training.warmup_epochs，方便小 epoch 烟囱测试")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="覆盖 cfg.logging.eval_interval")
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
    if args.warmup_epochs is not None:
        cfg["training"]["warmup_epochs"] = args.warmup_epochs
    if args.eval_interval is not None:
        cfg["logging"]["eval_interval"] = args.eval_interval
    cfg["device"] = args.device

    set_seed(cfg.get("seed", 42))

    loss_name = cfg["training"].get("loss", "ce")
    if loss_name in {"elr", "composite"}:
        train_loader, val_loader = get_indexed_dataloaders(cfg)
    else:
        train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg)
    num_samples = getattr(train_loader.dataset, "num_global_samples", len(train_loader.dataset))
    criterion = build_loss(cfg, num_samples=num_samples)

    trainer = BaseTrainer(model, train_loader, val_loader, criterion, cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()
