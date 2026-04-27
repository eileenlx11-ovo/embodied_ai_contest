import os
import time
import torch
import torch.nn as nn
import numpy as np
from src.trainers.base_trainer import BaseTrainer
from src.utils.metrics import accuracy


class NoisyTrainer(BaseTrainer):
    """Trainer with per-sample loss tracking and index-aware loss support."""

    def __init__(self, model, train_loader, val_loader, criterion, cfg):
        super().__init__(model, train_loader, val_loader, criterion, cfg)
        self.num_samples = len(train_loader.dataset)
        self.sample_losses = torch.zeros(self.num_samples)
        self.sample_counts = torch.zeros(self.num_samples)
        self.needs_indices = hasattr(criterion, "forward") and "indices" in str(
            criterion.forward.__code__.co_varnames
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
            if self.cfg["training"].get("channels_last", False):
                images = images.to(memory_format=torch.channels_last)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                if self.needs_indices and indices is not None:
                    loss_unreduced = self.criterion(outputs, targets, indices.to(self.device))
                else:
                    loss_unreduced = self.criterion(outputs, targets)
                loss = loss_unreduced / self.accum_steps

            self.scaler.scale(loss).backward()

            if indices is not None:
                with torch.no_grad():
                    per_sample = nn.functional.cross_entropy(
                        outputs, targets, reduction="none"
                    )
                    idx_cpu = indices.cpu()
                    self.sample_losses[idx_cpu] += per_sample.cpu()
                    self.sample_counts[idx_cpu] += 1

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

            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f} Acc: {acc:.2f}%")

        self.scheduler.step()
        return total_loss / len(self.train_loader), 100.0 * correct / total

    def get_avg_sample_losses(self):
        mask = self.sample_counts > 0
        avg = torch.zeros_like(self.sample_losses)
        avg[mask] = self.sample_losses[mask] / self.sample_counts[mask]
        return avg

    def get_clean_noisy_split(self, clean_ratio=0.8):
        avg_losses = self.get_avg_sample_losses()
        valid = self.sample_counts > 0
        valid_indices = torch.where(valid)[0]
        valid_losses = avg_losses[valid_indices]

        k = int(len(valid_indices) * clean_ratio)
        _, sorted_idx = valid_losses.sort()
        clean = valid_indices[sorted_idx[:k]].numpy()
        noisy = valid_indices[sorted_idx[k:]].numpy()
        return clean, noisy

    def reset_loss_tracking(self):
        self.sample_losses.zero_()
        self.sample_counts.zero_()
