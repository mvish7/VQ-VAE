"""Training utilities for Trajectory VQ-VAE."""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


class Trainer:
    """Handles training loop, validation, logging, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        config: dict,
        device: torch.device | str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.epochs = config.get("epochs", 100)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.log_interval = config.get("log_interval", 50)
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=config.get("log_dir", "runs"))
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        self.global_step = 0

    def train(self, resume_from: str | None = None) -> None:
        """Run full training loop."""
        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(
            f"Starting training | epochs={self.epochs} | "
            f"train_batches={len(self.train_loader)} | "
            f"val_batches={len(self.val_loader)} | device={self.device}"
        )

        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_recon={val_metrics['reconstruction_loss']:.4f} | "
                f"perplexity={val_metrics['perplexity']:.1f} | lr={lr:.2e}"
            )

            # Tensorboard epoch-level
            for key, val in train_metrics.items():
                self.writer.add_scalar(f"epoch/train_{key}", val, epoch)
            for key, val in val_metrics.items():
                self.writer.add_scalar(f"epoch/val_{key}", val, epoch)
            self.writer.add_scalar("epoch/lr", lr, epoch)

            # Checkpointing
            self._save_checkpoint(epoch, val_metrics["reconstruction_loss"])

        self.writer.close()
        logger.info("Training complete.")

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        accum = {}
        num_batches = 0

        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            output = self.model(batch)
            loss = output["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            # Accumulate metrics
            metrics = {
                "loss": output["loss"].item(),
                "reconstruction_loss": output["reconstruction_loss"].item(),
                "dynamics_loss": output["dynamics_loss"].item(),
                "commitment_loss": output["commitment_loss"].item(),
                "perplexity": output["perplexity"],
            }
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            num_batches += 1

            # Step-level tensorboard
            self.writer.add_scalar("step/loss", metrics["loss"], self.global_step)
            self.writer.add_scalar("step/reconstruction_loss", metrics["reconstruction_loss"], self.global_step)
            self.writer.add_scalar("step/commitment_loss", metrics["commitment_loss"], self.global_step)
            self.writer.add_scalar("step/perplexity", metrics["perplexity"], self.global_step)
            self.global_step += 1

            # Periodic console log
            if (i + 1) % self.log_interval == 0:
                logger.info(
                    f"  [Epoch {epoch+1} | Step {i+1}/{len(self.train_loader)}] "
                    f"loss={metrics['loss']:.4f} | "
                    f"recon={metrics['reconstruction_loss']:.4f} | "
                    f"dyn={metrics['dynamics_loss']:.4f} | "
                    f"commit={metrics['commitment_loss']:.4f} | "
                    f"perp={metrics['perplexity']:.1f}"
                )

        return {k: v / num_batches for k, v in accum.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation pass."""
        self.model.eval()
        accum = {}
        num_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            output = self.model(batch)

            metrics = {
                "loss": output["loss"].item(),
                "reconstruction_loss": output["reconstruction_loss"].item(),
                "dynamics_loss": output["dynamics_loss"].item(),
                "commitment_loss": output["commitment_loss"].item(),
                "perplexity": output["perplexity"],
            }
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in accum.items()}

    def _save_checkpoint(self, epoch: int, val_recon_loss: float) -> None:
        """Save checkpoint and track best model."""
        state = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
        }

        torch.save(state, self.checkpoint_dir / "last.pt")

        if val_recon_loss < self.best_val_loss:
            self.best_val_loss = val_recon_loss
            torch.save(state, self.checkpoint_dir / "best.pt")
            logger.info(f"  ✓ New best model saved (val_recon={val_recon_loss:.4f})")

    def _load_checkpoint(self, path: str) -> None:
        """Resume training from checkpoint."""
        logger.info(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.start_epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        logger.info(f"Resumed at epoch {self.start_epoch}")
