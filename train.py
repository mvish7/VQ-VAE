"""Training entrypoint for Trajectory VQ-VAE."""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.dataloader import TrajDataset
from model import TrajectoryVQVAE
from trainer import Trainer

# ── Config ──────────────────────────────────────────────────────────
config = {
    # Data
    "dataset_path": "/media/vishal/datasets/ar1_vae_dataset1/",
    "batch_size": 64,
    "num_workers": 4,
    # Model
    "in_channels": 9,
    "hidden_dim": 256,
    "num_embeddings": 1024,
    "embedding_dim": 256,
    "commitment_cost": 0.25,
    "dynamics_weight": 1.0,
    "num_groups": 32,
    # Training
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "grad_clip": 1.0,
    "log_interval": 500,
    # Paths
    "checkpoint_dir": "checkpoints",
    "log_dir": "runs",
}

# ── Data ────────────────────────────────────────────────────────────
train_set = TrajDataset(config["dataset_path"], split="train")
val_set = TrajDataset(config["dataset_path"], split="test")

train_loader = DataLoader(
    train_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=True,
)
val_loader = DataLoader(
    val_set,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True,
)

# ── Model ───────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TrajectoryVQVAE(
    in_channels=config["in_channels"],
    hidden_dim=config["hidden_dim"],
    num_embeddings=config["num_embeddings"],
    embedding_dim=config["embedding_dim"],
    commitment_cost=config["commitment_cost"],
    dynamics_weight=config["dynamics_weight"],
    num_groups=config["num_groups"],
)

# ── Optimizer & Scheduler ───────────────────────────────────────────
optimizer = AdamW(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"],
)
scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

# ── Train ───────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    device=device,
)

trainer.train()
