import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

from dataset.augmentation import TrajectoryAugmentor


class TrajDataset(Dataset):
    """Trajectory dataset with optional on-the-fly augmentation."""

    def __init__(
        self,
        root_path: str = "/dataset",
        split: str = "train",
        portion: float = 0.1,
        augment: bool = False,
        aug_prob: float = 0.5,
        max_rot_deg: float = 5.0,
        noise_std: float = 1e-3,
    ):
        self.traj_data = load_from_disk(root_path)[split]
        self.traj_data = self.traj_data.select(range(int(len(self.traj_data) * portion)))
        self.augmentor = TrajectoryAugmentor(aug_prob=aug_prob, max_rot_deg=max_rot_deg, noise_std=noise_std) if augment else None

    def __len__(self):
        return len(self.traj_data)

    @staticmethod
    def convert_rot_mat_to_6d(rot_mat):
        """Converts 3x3 rotation matrix to 6d continuous rotation (Gram-Schmidt)."""
        rot_mat_cols = torch.tensor(rot_mat)[:, :, :2]
        return rot_mat_cols.permute(0, 2, 1).reshape(-1, 6)

    def __getitem__(self, item: int):
        ego_xyz = self.traj_data[item]["ego_future_xyz"][0][0]
        ego_rot = self.traj_data[item]["ego_future_rot"][0][0]
        # converting 3x3 rotation matrix to 6d rotation (gram schmidt)
        ego_rot = self.convert_rot_mat_to_6d(ego_rot)
        # (9, 64) for each sample
        feature = torch.cat((torch.tensor(ego_xyz), ego_rot), dim=1).T

        if self.augmentor is not None:
            feature = self.augmentor(feature)

        return feature

