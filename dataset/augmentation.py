"""On-the-fly trajectory augmentation for VQ-VAE training."""

import math
import torch


class TrajectoryAugmentor:
    """Stochastic augmentation for (9, 64) trajectory tensors.

    Channel layout:
        0-2 : x, y, z  positions
        3-5 : rotation col-1 (r1_x, r1_y, r1_z)
        6-8 : rotation col-2 (r2_x, r2_y, r2_z)

    Each augmentation is applied independently with probability `aug_prob`,
    so not every sample in a batch gets the same transforms.
    """

    def __init__(self, aug_prob: float = 0.5, max_rot_deg: float = 5.0, noise_std: float = 1e-3):
        self.aug_prob = aug_prob
        self.max_rot_deg = max_rot_deg
        self.noise_std = noise_std

    def __call__(self, feature: torch.Tensor) -> torch.Tensor:
        """Apply augmentations stochastically."""
        if torch.rand(1).item() < self.aug_prob:
            feature = self._mirror_y(feature)
        if torch.rand(1).item() < self.aug_prob:
            feature = self._random_yaw_rotation(feature, self.max_rot_deg)
        if torch.rand(1).item() < self.aug_prob:
            feature = self._add_gaussian_noise(feature, self.noise_std)
        return feature

    @staticmethod
    def _mirror_y(feature: torch.Tensor) -> torch.Tensor:
        """Lateral mirror around the Y-axis (negate y, flip rot cols)."""
        feature = feature.clone()
        feature[1] = -feature[1]       # y position
        feature[3] = -feature[3]       # r1_x
        feature[5] = -feature[5]       # r1_z
        feature[6] = -feature[6]       # r2_x
        feature[8] = -feature[8]       # r2_z
        return feature

    @staticmethod
    def _random_yaw_rotation(feature: torch.Tensor, max_deg: float = 5.0) -> torch.Tensor:
        """Apply a small random yaw rotation (around Z-axis) to the trajectory."""
        angle = torch.empty(1).uniform_(-max_deg, max_deg).item()
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=feature.dtype)

        feature = feature.clone()
        feature[:2] = rot @ feature[:2]      # xyz positions (x, y)
        feature[3:5] = rot @ feature[3:5]    # rotation col-1 (r1_x, r1_y)
        feature[6:8] = rot @ feature[6:8]    # rotation col-2 (r2_x, r2_y)
        return feature

    @staticmethod
    def _add_gaussian_noise(feature: torch.Tensor, std: float = 1e-3) -> torch.Tensor:
        """Add tiny Gaussian noise."""
        return feature + torch.randn_like(feature) * std
