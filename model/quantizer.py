"""EMA-based Vector Quantizer with codebook restart."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """Vector Quantizer with Exponential Moving Average updates.
    
    Features:
    - EMA codebook updates (no gradient descent on embeddings)
    - Dead code restart mechanism
    - Perplexity tracking for codebook utilization
    - Straight-Through Estimator for gradient flow
    """

    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_code_threshold: int = 2,
    ):
        """Initialize VectorQuantizerEMA.
        
        Args:
            num_embeddings: Size of codebook (K).
            embedding_dim: Dimension of each embedding (D).
            commitment_cost: Weight for commitment loss (beta).
            decay: EMA decay rate.
            epsilon: Small constant for numerical stability.
            dead_code_threshold: Re-init codes unused for this many batches.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.dead_code_threshold = dead_code_threshold

        # Codebook embeddings (not updated via gradients)
        self.register_buffer("embeddings", torch.randn(num_embeddings, embedding_dim))
        self.embeddings.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA cluster size and sum
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding_sum", self.embeddings.clone())

        # Track batches since each code was last used
        self.register_buffer("usage_count", torch.zeros(num_embeddings, dtype=torch.long))

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Quantize continuous features to discrete codebook entries.
        
        Args:
            z: Encoder output of shape (B, D, T).
            
        Returns:
            quantized: Quantized tensor (B, D, T) with gradients via STE.
            indices: Codebook indices (B, T).
            loss: Commitment loss term.
            perplexity: Codebook utilization metric.
        """
        # (B, D, T) -> (B, T, D) -> (B*T, D)
        z = z.permute(0, 2, 1).contiguous()
        batch_size, seq_len, _ = z.shape
        flat_z = z.view(-1, self.embedding_dim)

        # Compute distances to all codebook entries
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(flat_z, self.embeddings.t())
        )

        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).float()

        # Quantize
        quantized_flat = torch.matmul(encodings, self.embeddings)
        quantized = quantized_flat.view(batch_size, seq_len, self.embedding_dim)

        # EMA updates (training only)
        if self.training:
            self._update_ema(flat_z, encodings)
            self._restart_dead_codes(flat_z)

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        loss = self.commitment_cost * e_latent_loss

        # Straight-Through Estimator: copy gradients from decoder to encoder
        quantized = z + (quantized - z).detach()

        # Perplexity (codebook utilization)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()

        # (B, T, D) -> (B, D, T)
        quantized = quantized.permute(0, 2, 1).contiguous()
        indices = indices.view(batch_size, seq_len)

        return quantized, indices, loss, perplexity

    def _update_ema(self, flat_z: torch.Tensor, encodings: torch.Tensor) -> None:
        """Update codebook via EMA."""
        # Count codes used in this batch
        cluster_size = torch.sum(encodings, dim=0)
        embedding_sum = torch.matmul(encodings.t(), flat_z)

        # EMA update
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.ema_embedding_sum.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)

        # Laplace smoothing
        n = torch.sum(self.ema_cluster_size)
        cluster_size_smoothed = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )

        # Update embeddings
        self.embeddings.data.copy_(self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1))

        # Update usage tracking
        used_codes = cluster_size > 0
        self.usage_count[used_codes] = 0
        self.usage_count[~used_codes] += 1

    def _restart_dead_codes(self, flat_z: torch.Tensor) -> None:
        """Re-initialize codes that haven't been used recently."""
        dead_mask = self.usage_count >= self.dead_code_threshold
        num_dead = dead_mask.sum().item()

        if num_dead > 0 and flat_z.shape[0] >= num_dead:
            # Sample random encoder outputs to replace dead codes
            random_indices = torch.randperm(flat_z.shape[0], device=flat_z.device)[:num_dead]
            new_embeddings = flat_z[random_indices].detach()

            dead_indices = torch.where(dead_mask)[0]
            self.embeddings.data[dead_indices] = new_embeddings
            self.ema_embedding_sum.data[dead_indices] = new_embeddings
            self.ema_cluster_size.data[dead_indices] = 1.0
            self.usage_count[dead_indices] = 0
