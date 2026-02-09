"""ResNet building blocks for 1D temporal convolutions."""

import torch
import torch.nn as nn


class ResNetBlock1D(nn.Module):
    """Pre-activation ResNet block for 1D temporal data.
    
    Structure: ReLU -> Conv1d(3x3) -> GN -> ReLU -> Conv1d(3x3) -> GN + Residual
    """

    def __init__(self, channels: int, num_groups: int = 32):
        """Initialize ResNetBlock1D.
        
        Args:
            channels: Number of input and output channels.
            num_groups: Number of groups for GroupNorm. Defaults to 32.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-activation and residual connection.
        
        Args:
            x: Input tensor of shape (B, C, T).
            
        Returns:
            Output tensor of shape (B, C, T).
        """
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual
