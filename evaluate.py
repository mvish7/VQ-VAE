"""Evaluation script for Trajectory VQ-VAE.

Computes reconstruction quality, dynamics preservation, and codebook health
metrics on the test set from a trained checkpoint.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --dataset_path /path/to/data
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

from dataset.dataloader import TrajDataset
from model import TrajectoryVQVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Trajectory VQ-VAE")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/media/vishal/datasets/ar1_vae_dataset/",
        help="Path to dataset",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for results JSON (defaults to checkpoint directory)",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: TrajectoryVQVAE, dataloader: DataLoader, device: str) -> dict:
    """Run evaluation over the full dataset.

    Returns:
        Dictionary of aggregated metrics.
    """
    model.eval()

    total_mse = 0.0
    total_vel_mse = 0.0
    total_acc_mse = 0.0
    total_perplexity = 0.0
    num_batches = 0
    all_indices = []

    for batch in dataloader:
        batch = batch.to(device)
        output = model(batch)
        reconstruction = output["reconstruction"]

        # Reconstruction MSE
        total_mse += F.mse_loss(reconstruction, batch).item()

        # Dynamics: velocity MSE
        pred_vel = reconstruction[:, :, 1:] - reconstruction[:, :, :-1]
        target_vel = batch[:, :, 1:] - batch[:, :, :-1]
        total_vel_mse += F.mse_loss(pred_vel, target_vel).item()

        # Dynamics: acceleration MSE
        pred_acc = pred_vel[:, :, 1:] - pred_vel[:, :, :-1]
        target_acc = target_vel[:, :, 1:] - target_vel[:, :, :-1]
        total_acc_mse += F.mse_loss(pred_acc, target_acc).item()

        # Codebook
        total_perplexity += output["perplexity"]
        all_indices.append(output["indices"].cpu())

        num_batches += 1

    # Aggregate
    all_indices = torch.cat(all_indices, dim=0).flatten()  # (N*8,)
    num_embeddings = model.quantizer.num_embeddings

    # Code usage histogram
    code_counts = torch.bincount(all_indices, minlength=num_embeddings)
    active_codes = (code_counts > 0).sum().item()
    code_freq = code_counts.float() / code_counts.sum()

    # Global perplexity from full index distribution (more accurate than batch avg)
    global_perplexity = torch.exp(
        -torch.sum(code_freq * torch.log(code_freq + 1e-10))
    ).item()

    metrics = {
        "reconstruction_mse": total_mse / num_batches,
        "velocity_mse": total_vel_mse / num_batches,
        "acceleration_mse": total_acc_mse / num_batches,
        "avg_batch_perplexity": total_perplexity / num_batches,
        "global_perplexity": global_perplexity,
        "active_codes": active_codes,
        "total_codes": num_embeddings,
        "codebook_utilization_pct": round(active_codes / num_embeddings * 100, 1),
        "num_samples": len(all_indices),
        "code_frequency_top10": {
            str(idx.item()): count.item()
            for idx, count in zip(
                *torch.topk(code_counts, k=min(10, num_embeddings))
            )
        },
        "code_frequency_bottom10": {
            str(idx.item()): count.item()
            for idx, count in zip(
                *torch.topk(code_counts, k=min(10, num_embeddings), largest=False)
            )
        },
    }

    return metrics


def print_report(metrics: dict) -> None:
    """Print a formatted evaluation report."""
    logger.info("=" * 60)
    logger.info("  TRAJECTORY VQ-VAE EVALUATION REPORT")
    logger.info("=" * 60)

    logger.info("\n── Reconstruction ──────────────────────────────────────")
    logger.info(f"  MSE:                {metrics['reconstruction_mse']:.6f}")

    logger.info("\n── Dynamics ────────────────────────────────────────────")
    logger.info(f"  Velocity MSE:       {metrics['velocity_mse']:.6f}")
    logger.info(f"  Acceleration MSE:   {metrics['acceleration_mse']:.6f}")

    logger.info("\n── Codebook Health ─────────────────────────────────────")
    logger.info(f"  Batch Perplexity:   {metrics['avg_batch_perplexity']:.1f}")
    logger.info(f"  Global Perplexity:  {metrics['global_perplexity']:.1f}")
    logger.info(
        f"  Active Codes:       {metrics['active_codes']} / {metrics['total_codes']} "
        f"({metrics['codebook_utilization_pct']}%)"
    )

    logger.info("\n  Most-used codes:")
    for code_id, count in metrics["code_frequency_top10"].items():
        logger.info(f"    Code {code_id:>4s}: {count}")
    logger.info("  Least-used codes:")
    for code_id, count in metrics["code_frequency_bottom10"].items():
        logger.info(f"    Code {code_id:>4s}: {count}")

    logger.info("=" * 60)


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)

    # Load model
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model = TrajectoryVQVAE()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info("Model loaded successfully")

    # Load test dataset
    logger.info(f"Loading test dataset from: {args.dataset_path}")
    test_set = TrajDataset(args.dataset_path, split="test", portion=1.0, augment=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Test set: {len(test_set)} samples, {len(test_loader)} batches")

    # Evaluate
    metrics = evaluate(model, test_loader, device)

    # Report
    print_report(metrics)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_path = output_dir / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
