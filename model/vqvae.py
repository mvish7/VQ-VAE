"""Trajectory VQ-VAE main model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Encoder
from model.decoder import Decoder
from model.quantizer import VectorQuantizerEMA


class TrajectoryVQVAE(nn.Module):
    """Trajectory VQ-VAE for learning discrete latent representations.
    
    Compresses 6.4s vehicle trajectories (T=64) to discrete codes (T=8)
    and reconstructs them with full temporal resolution.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 256,
        num_embeddings: int = 768,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        dynamics_weight: float = 1.0,
        unit_circle_weight: float = 0.05,
        num_groups: int = 32,
    ):
        """Initialize TrajectoryVQVAE.
        
        Args:
            in_channels: Number of trajectory channels [x,y,z,sin_yaw,cos_yaw].
            hidden_dim: Hidden dimension for encoder/decoder.
            num_embeddings: Codebook size (K).
            embedding_dim: Dimension of codebook entries (D).
            commitment_cost: Weight for commitment loss (beta).
            dynamics_weight: Weight for dynamics/smoothing loss (lambda).
            unit_circle_weight: Weight for sin²+cos²=1 regularization.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        self.dynamics_weight = dynamics_weight
        self.unit_circle_weight = unit_circle_weight

        self.encoder = Encoder(in_channels, hidden_dim, num_groups)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost
        )
        self.decoder = Decoder(in_channels, hidden_dim, num_groups)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | float]:
        """Forward pass with loss computation.
        
        Args:
            x: Input trajectory of shape (B, 5, 64).
            
        Returns:
            Dictionary containing:
                - loss: Total loss
                - reconstruction_loss: SmoothL1 reconstruction loss
                - dynamics_loss: Velocity + acceleration MSE
                - commitment_loss: VQ commitment loss
                - unit_circle_loss: sin²+cos² = 1 regularization
                - perplexity: Codebook utilization metric
                - reconstruction: Reconstructed trajectory (B, 5, 64)
                - indices: Codebook indices (B, 8)
        """
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, indices, commitment_loss, entropy_loss, perplexity = self.quantizer(z)

        # Decode
        reconstruction = self.decoder(z_q)

        # Calculate losses (both train and eval for monitoring)
        reconstruction_loss = F.smooth_l1_loss(reconstruction, x)
        dynamics_loss = self._compute_dynamics_loss(reconstruction, x)
        unit_circle_loss = self._compute_unit_circle_loss(reconstruction)

        # Total loss: L = L_rec + λ * L_dyn + β * L_commit + γ * L_uc + L_entropy
        total_loss = (
            reconstruction_loss
            + self.dynamics_weight * dynamics_loss
            + commitment_loss
            + self.unit_circle_weight * unit_circle_loss
            + entropy_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "dynamics_loss": dynamics_loss,
            "commitment_loss": commitment_loss,
            "unit_circle_loss": unit_circle_loss,
            "entropy_loss": entropy_loss,
            "perplexity": perplexity,
            "reconstruction": reconstruction,
            "indices": indices,
        }

    def _compute_dynamics_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity and acceleration MSE loss.
        
        Ensures reconstructed trajectories are physically smooth.
        
        Args:
            pred: Predicted trajectory (B, C, T).
            target: Ground truth trajectory (B, C, T).
            
        Returns:
            Combined velocity and acceleration MSE loss.
        """
        # Velocity: v_t = x_t - x_{t-1}
        pred_vel = pred[:, :, 1:] - pred[:, :, :-1]
        target_vel = target[:, :, 1:] - target[:, :, :-1]
        velocity_loss = F.mse_loss(pred_vel, target_vel)

        # Acceleration: a_t = v_t - v_{t-1}
        pred_acc = pred_vel[:, :, 1:] - pred_vel[:, :, :-1]
        target_acc = target_vel[:, :, 1:] - target_vel[:, :, :-1]
        acceleration_loss = F.mse_loss(pred_acc, target_acc)

        return velocity_loss + acceleration_loss

    def _compute_unit_circle_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Penalize sin²(yaw) + cos²(yaw) deviating from 1.

        Args:
            pred: Predicted trajectory (B, C, T) with channels 3=sin, 4=cos.

        Returns:
            MSE between sin²+cos² and 1.
        """
        sin_pred = pred[:, 3, :]
        cos_pred = pred[:, 4, :]
        return F.mse_loss(sin_pred**2 + cos_pred**2, torch.ones_like(sin_pred))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory to discrete codes (for inference).
        
        Args:
            x: Input trajectory (B, 5, 64).
            
        Returns:
            indices: Codebook indices (B, 8).
            z_q: Quantized latents (B, D, 8).
        """
        z = self.encoder(x)
        z_q, indices, _, _, _ = self.quantizer(z)
        return indices, z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to trajectory (for inference).
        
        Args:
            z_q: Quantized latents (B, D, 8).
            
        Returns:
            Reconstructed trajectory (B, 5, 64).
        """
        return self.decoder(z_q)

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices (for VLA integration).
        
        Args:
            indices: Codebook indices (B, 8).
            
        Returns:
            Reconstructed trajectory (B, 5, 64).
        """
        # Look up embeddings
        z_q = F.embedding(indices, self.quantizer.embeddings)
        z_q = z_q.permute(0, 2, 1).contiguous()  # (B, T, D) -> (B, D, T)
        return self.decoder(z_q)
