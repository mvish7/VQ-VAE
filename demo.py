"""Demo script for Trajectory VQ-VAE (epoch 60 checkpoint).

Demonstrates the full encode → quantize → decode pipeline on real test
samples and prints per-sample reconstruction metrics.

Usage:
    python demo.py
    python demo.py --checkpoint checkpoints/batch512_aug02_epoch60_perp820/best.pt
    python demo.py --num_samples 10 --dataset_path /path/to/data
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from dataset.dataloader import TrajDataset
from model import TrajectoryVQVAE


CHECKPOINT_DEFAULT = "checkpoints/batch512_aug02_epoch60_perp820/best.pt"
DATASET_DEFAULT = "/media/vishal/datasets/ar1_vae_dataset/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trajectory VQ-VAE Demo")
    parser.add_argument(
        "--checkpoint", type=str, default=CHECKPOINT_DEFAULT, help="Path to checkpoint"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=DATASET_DEFAULT, help="Path to dataset"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to demo"
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> TrajectoryVQVAE:
    """Load trained VQ-VAE from checkpoint."""
    model = TrajectoryVQVAE()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Strip _orig_mod. prefix from torch.compile
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return model


@torch.no_grad()
def demo_encode_decode(
    model: TrajectoryVQVAE, samples: torch.Tensor, device: str
) -> None:
    """Run encode → decode and print per-sample metrics."""
    samples = samples.to(device)
    batch_size = samples.shape[0]

    # --- Step 1: Encode to discrete codes ---
    indices, z_q = model.encode(samples)
    logger.info(
        f"Encoded {batch_size} trajectories → codebook indices shape: {indices.shape}"
    )

    # --- Step 2: Decode from quantized latents ---
    recon_from_latents = model.decode(z_q)

    # --- Step 3: Decode from indices only (VLA integration path) ---
    recon_from_indices = model.decode_from_indices(indices)

    # --- Metrics ---
    logger.info("")
    logger.info("=" * 65)
    logger.info("  Per-Sample Reconstruction Results")
    logger.info("=" * 65)
    logger.info(f"  {'Sample':<8} {'MSE':>12} {'Vel MSE':>12} {'Codes':>25}")
    logger.info("-" * 65)

    for i in range(batch_size):
        orig = samples[i]                 # (5, 64)
        recon = recon_from_indices[i]     # (5, 64)

        mse = F.mse_loss(recon, orig).item()

        # Velocity error
        pred_vel = recon[:, 1:] - recon[:, :-1]
        tgt_vel = orig[:, 1:] - orig[:, :-1]
        vel_mse = F.mse_loss(pred_vel, tgt_vel).item()

        codes = indices[i].cpu().tolist()
        logger.info(f"  {i:<8} {mse:>12.6f} {vel_mse:>12.6f} {str(codes):>25}")

    # Overall batch stats
    batch_mse = F.mse_loss(recon_from_indices, samples).item()
    latent_mse = F.mse_loss(recon_from_latents, samples).item()
    cross_mse = F.mse_loss(recon_from_indices, recon_from_latents).item()

    logger.info("-" * 65)
    logger.info(f"  Batch MSE (from indices):   {batch_mse:.6f}")
    logger.info(f"  Batch MSE (from latents):   {latent_mse:.6f}")
    logger.info(f"  Index vs Latent decode gap:  {cross_mse:.2e}")
    logger.info("=" * 65)


@torch.no_grad()
def demo_decode_from_random_codes(model: TrajectoryVQVAE, device: str) -> None:
    """Demonstrate decoding from arbitrary codebook indices."""
    num_embeddings = model.quantizer.num_embeddings

    # Generate random valid codebook indices (B=3, T=8)
    random_indices = torch.randint(0, num_embeddings, (3, 8), device=device)
    decoded = model.decode_from_indices(random_indices)

    logger.info("")
    logger.info("── Random Code Decoding ────────────────────────────────")
    logger.info(f"  Random indices shape: {random_indices.shape}")
    logger.info(f"  Decoded trajectory shape: {decoded.shape}")
    logger.info(f"  Output range: [{decoded.min():.4f}, {decoded.max():.4f}]")

    for i in range(random_indices.shape[0]):
        codes = random_indices[i].cpu().tolist()
        logger.info(f"  Sample {i} codes: {codes}")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load a few test samples
    logger.info(f"Loading test data from: {args.dataset_path}")
    test_set = TrajDataset(args.dataset_path, split="test", portion=1.0, augment=False)
    logger.info(f"Test set size: {len(test_set)} samples")

    num_samples = min(args.num_samples, len(test_set))
    samples = torch.stack([test_set[i] for i in range(num_samples)])
    # logger.info(f"Shape of samples: {samples.shape}")
    logger.info(f"Selected {num_samples} samples, shape: {samples.shape}")

    # Demo 1: encode → decode pipeline
    t1 = time.time()
    demo_encode_decode(model, samples, device)
    t2 = time.time()
    logger.info(f"Encode → decode pipeline took: {t2 - t1:.4f} seconds")

    # Demo 2: decode from random codebook indices
    t3 = time.time()
    demo_decode_from_random_codes(model, device)
    t4 = time.time()
    logger.info(f"Decode from random codes took: {t4 - t3:.4f} seconds")

    logger.info("\nDemo complete.")


if __name__ == "__main__":
    main()