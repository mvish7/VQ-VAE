"""Upsampling Decoder for trajectory reconstruction."""

import torch
import torch.nn as nn

from model.blocks import ResNetBlock1D


class Decoder(nn.Module):
    """Upsampling decoder that reconstructs trajectories from latent codes.
    
    Expands T=8 -> T=64 via 3 upsampling stages.
    Uses linear interpolation + Conv to avoid checkerboard artifacts.
    """

    def __init__(
        self,
        out_channels: int = 9,
        hidden_dim: int = 256,
        num_groups: int = 32,
    ):
        """Initialize Decoder.
        
        Args:
            out_channels: Number of output channels (trajectory features).
            hidden_dim: Hidden dimension throughout the decoder.
            num_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection from quantized latents
        self.input_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # 3 upsampling stages: 8 -> 16 -> 32 -> 64
        self.stages = nn.ModuleList()
        for _ in range(3):
            stage = nn.Sequential(
                ResNetBlock1D(hidden_dim, num_groups),
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            )
            self.stages.append(stage)

        # Output head: GN -> ReLU -> Conv
        self.output_head = nn.Sequential(
            nn.GroupNorm(num_groups, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents to trajectory.
        
        Args:
            z_q: Quantized latents of shape (B, hidden_dim, 8).
            
        Returns:
            Reconstructed trajectory of shape (B, out_channels, 64).
        """
        x = self.input_proj(z_q)

        for stage in self.stages:
            x = stage(x)

        x = self.output_head(x)
        return x
