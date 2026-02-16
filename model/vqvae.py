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
        in_channels: int = 9,
        hidden_dim: int = 256,
        num_embeddings: int = 1024,
        embedding_dim: int = 256,
        commitment_cost: float = 0.25,
        dynamics_weight: float = 1.0,
        num_groups: int = 32,
    ):
        """Initialize TrajectoryVQVAE.
        
        Args:
            in_channels: Number of trajectory channels [x,y,z,r1_x,r1_y,r1_z,r2_x,r2_y,r2_z].
            hidden_dim: Hidden dimension for encoder/decoder.
            num_embeddings: Codebook size (K).
            embedding_dim: Dimension of codebook entries (D).
            commitment_cost: Weight for commitment loss (beta).
            dynamics_weight: Weight for dynamics/smoothing loss (lambda).
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        self.dynamics_weight = dynamics_weight

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
            x: Input trajectory of shape (B, 9, 64).
            
        Returns:
            Dictionary containing:
                - loss: Total loss
                - reconstruction_loss: SmoothL1 reconstruction loss
                - dynamics_loss: Velocity + acceleration MSE
                - commitment_loss: VQ commitment loss
                - perplexity: Codebook utilization metric
                - reconstruction: Reconstructed trajectory (B, 9, 64)
                - indices: Codebook indices (B, 8)
        """
        # Encode
        z = self.encoder(x)

        # Quantize
        z_q, indices, commitment_loss, perplexity = self.quantizer(z)

        # Decode
        reconstruction = self.decoder(z_q)

        # Calculate losses (both train and eval for monitoring)
        reconstruction_loss = F.smooth_l1_loss(reconstruction, x)
        dynamics_loss = self._compute_dynamics_loss(reconstruction, x)

        # Total loss: L = L_rec + λ * L_dyn + β * L_commit
        total_loss = (
            reconstruction_loss
            + self.dynamics_weight * dynamics_loss
            + commitment_loss
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "dynamics_loss": dynamics_loss,
            "commitment_loss": commitment_loss,
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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode trajectory to discrete codes (for inference).
        
        Args:
            x: Input trajectory (B, 9, 64).
            
        Returns:
            indices: Codebook indices (B, 8).
            z_q: Quantized latents (B, D, 8).
        """
        z = self.encoder(x)
        z_q, indices, _, _ = self.quantizer(z)
        return indices, z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to trajectory (for inference).
        
        Args:
            z_q: Quantized latents (B, D, 8).
            
        Returns:
            Reconstructed trajectory (B, 9, 64).
        """
        return self.decoder(z_q)

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices (for VLA integration).
        
        Args:
            indices: Codebook indices (B, 8).
            
        Returns:
            Reconstructed trajectory (B, 9, 64).
        """
        # Look up embeddings
        z_q = F.embedding(indices, self.quantizer.embeddings)
        z_q = z_q.permute(0, 2, 1).contiguous()  # (B, T, D) -> (B, D, T)
        return self.decoder(z_q)
