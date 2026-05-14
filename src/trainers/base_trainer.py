import csv
import json
import math
import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from src.utils.metrics import accuracy
from src.utils.ema import ModelEMA

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


def resolve_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg.get("device", "auto"))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

        tc = cfg["training"]
        self.epochs = tc["epochs"]
        self.accum_steps = tc.get("accumulation_steps", 1)
        self.grad_clip = tc.get("grad_clip", 0.0)
        self.log_interval = cfg["logging"]["log_interval"]
        self.eval_interval = cfg["logging"]["eval_interval"]
        self.save_interval = cfg["logging"]["save_interval"]
        self.ckpt_dir = cfg["logging"]["checkpoint_dir"]
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        if tc.get("channels_last", False) and self.device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        self.model = model.to(self.device)

        # torch.compile
        if tc.get("compile", False):
            self.model = torch.compile(self.model)

        # optimizer
        if tc["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=tc["base_lr"],
                momentum=tc.get("momentum", 0.9), weight_decay=tc["weight_decay"],
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=tc["base_lr"], weight_decay=tc["weight_decay"],
            )

        # scheduler
        warmup_epochs = tc.get("warmup_epochs", 5)
        total_epochs = tc["epochs"]
        if total_epochs <= warmup_epochs:
            warmup_epochs = max(0, total_epochs - 1)
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_epochs - warmup_epochs))
            self.scheduler = SequentialLR(
                self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_epochs))

        self.use_amp = tc.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        # EMA
        ema_decay = tc.get("ema_decay", 0.0)
        self.ema = ModelEMA(self.model, decay=ema_decay) if ema_decay > 0 else None

        self.best_acc = 0.0
        self.global_step = 0
        self.epoch_history = []
        self.metrics_path = None
        self.run_dir = None
        run_name = cfg.get("logging", {}).get("run_name")
        if run_name:
            log_dir = cfg.get("logging", {}).get("log_dir", "./logs")
            self.run_dir = os.path.join(log_dir, run_name)
            os.makedirs(self.run_dir, exist_ok=True)
            self.metrics_path = os.path.join(self.run_dir, "metrics.csv")
            with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "epoch", "train_loss", "train_acc", "val_top1", "val_top5",
                    "best_val_top1", "lr", "epoch_time_sec", "loss", "loss_params",
                ])
                writer.writeheader()

        # WandB (optional)
        self.wandb_run = None
        wb_cfg = cfg.get("logging", {}).get("wandb", {})
        if wb_cfg.get("enabled", False):
            if not _WANDB_AVAILABLE:
                print("WARN: wandb enabled in config but package not installed; skipping.")
            else:
                self.wandb_run = wandb.init(
                    project=wb_cfg.get("project", "embodied-ai"),
                    name=wb_cfg.get("run_name"),
                    config=cfg,
                    mode=wb_cfg.get("mode", "online"),
                    resume="allow",
                )

    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                images, targets, indices = batch
            else:
                images, targets = batch
                indices = None

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if indices is not None:
                indices = indices.to(self.device, non_blocking=True)
            if self.cfg["training"].get("channels_last", False):
                images = images.to(memory_format=torch.channels_last)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                if indices is not None:
                    loss = self.criterion(outputs, targets, indices=indices) / self.accum_steps
                else:
                    loss = self.criterion(outputs, targets) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    self.ema.update(self.model)

            total_loss += loss.item() * self.accum_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            self.global_step += 1

            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f} Acc: {acc:.2f}%")
                if self.wandb_run:
                    self.wandb_run.log({
                        "train/loss_running": avg_loss,
                        "train/acc_running": acc,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    }, step=self.global_step)

        self.scheduler.step()
        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def validate(self):
        model = self.ema.ema if self.ema else self.model
        model.eval()
        top1_sum, top5_sum, total = 0.0, 0.0, 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                outputs = model(images)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = targets.size(0)
            top1_sum += acc1 * batch_size
            top5_sum += acc5 * batch_size
            total += batch_size

        return top1_sum / total, top5_sum / total

    def _write_epoch_metrics(self, metrics):
        if not self.metrics_path:
            return
        tc = self.cfg["training"]
        row = {
            "epoch": metrics["epoch"],
            "train_loss": metrics["train/loss_epoch"],
            "train_acc": metrics["train/acc_epoch"],
            "val_top1": metrics.get("val/top1", ""),
            "val_top5": metrics.get("val/top5", ""),
            "best_val_top1": metrics.get("val/best_top1", self.best_acc),
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch_time_sec": metrics["train/epoch_time_sec"],
            "loss": tc.get("loss", "ce"),
            "loss_params": json.dumps(tc.get("loss_params", {}), sort_keys=True),
        }
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def _write_summary(self):
        if not self.run_dir or not self.epoch_history:
            return
        train_losses = [m["train/loss_epoch"] for m in self.epoch_history]
        values = []
        for metrics in self.epoch_history:
            for value in metrics.values():
                if isinstance(value, (int, float)):
                    values.append(float(value))
        summary = {
            "run_name": self.cfg.get("logging", {}).get("run_name"),
            "loss": self.cfg["training"].get("loss", "ce"),
            "loss_params": self.cfg["training"].get("loss_params", {}),
            "epochs": self.epochs,
            "best_val_top1": self.best_acc,
            "initial_train_loss": train_losses[0],
            "final_train_loss": train_losses[-1],
            "has_nan_or_inf": any(not math.isfinite(v) for v in values),
            "train_loss_decreased": train_losses[-1] < train_losses[0],
            "final_metrics": self.epoch_history[-1],
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def save_checkpoint(self, epoch, acc1, filename="latest.pth"):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_acc": self.best_acc,
            "cfg": self.cfg,
        }
        if self.ema:
            state["ema"] = self.ema.state_dict()
        torch.save(state, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if self.ema and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        return ckpt.get("epoch", 0)

    def fit(self):
        print(f"Training on {self.device} for {self.epochs} epochs")
        print(f"Train samples: {len(self.train_loader.dataset)}, "
              f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_one_epoch(epoch)
            elapsed = time.time() - t0

            log = f"Epoch {epoch}/{self.epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Time: {elapsed:.1f}s"

            epoch_metrics = {
                "epoch": epoch,
                "train/loss_epoch": train_loss,
                "train/acc_epoch": train_acc,
                "train/epoch_time_sec": elapsed,
            }

            if epoch % self.eval_interval == 0:
                val_top1, val_top5 = self.validate()
                log += f" | Val Top-1: {val_top1:.2f}% Top-5: {val_top5:.2f}%"
                epoch_metrics["val/top1"] = val_top1
                epoch_metrics["val/top5"] = val_top5
                if val_top1 > self.best_acc:
                    self.best_acc = val_top1
                    self.save_checkpoint(epoch, val_top1, "best.pth")
                    log += " *best*"
                epoch_metrics["val/best_top1"] = self.best_acc

            print(log)
            self.epoch_history.append(epoch_metrics.copy())
            self._write_epoch_metrics(epoch_metrics)
            if self.wandb_run:
                self.wandb_run.log(epoch_metrics, step=self.global_step)

            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, self.best_acc, "latest.pth")

        self.save_checkpoint(self.epochs, self.best_acc, "latest.pth")
        self._write_summary()
        print(f"Training complete. Best Val Top-1: {self.best_acc:.2f}%")
        if self.wandb_run:
            self.wandb_run.summary["best_top1"] = self.best_acc
            self.wandb_run.finish()
