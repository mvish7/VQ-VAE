"""1D ResNet Encoder for trajectory compression."""

import torch
import torch.nn as nn

from model.blocks import ResNetBlock1D


class Encoder(nn.Module):
    """1D ResNet encoder that compresses temporal dynamics.
    
    Compresses T=64 -> T=8 via 3 downsampling stages.
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 256,
        num_groups: int = 32,
    ):
        """Initialize Encoder.
        
        Args:
            in_channels: Number of input channels (trajectory features).
            hidden_dim: Hidden dimension throughout the encoder.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)

        # 3 downsampling stages: 64 -> 32 -> 16 -> 8
        self.stages = nn.ModuleList()
        for _ in range(3):
            stage = nn.Sequential(
                ResNetBlock1D(hidden_dim, num_groups),
                ResNetBlock1D(hidden_dim, num_groups),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            )
            self.stages.append(stage)

        # Pre-quantization normalization (critical for codebook utilization)
        self.output_norm = nn.GroupNorm(num_groups, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode trajectory to latent representation.
        
        Args:
            x: Input trajectory of shape (B, 9, 64).
            
        Returns:
            Latent representation of shape (B, hidden_dim, 8).
        """
        x = self.input_proj(x)

        for stage in self.stages:
            x = stage(x)

        x = self.output_norm(x)
        return x
