import torch
from torch.utils.data import Dataset
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
    def extract_yaw_sincos(rot_mat):
        """Extract yaw from 3x3 rotation matrices as [sin(yaw), cos(yaw)].

        Args:
            rot_mat: Rotation matrices of shape (T, 3, 3).

        Returns:
            Tensor of shape (T, 2) containing [sin(yaw), cos(yaw)].
        """
        rot = torch.tensor(rot_mat, dtype=torch.float32)
        yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
        return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=1)

    def __getitem__(self, item: int):
        ego_xyz = self.traj_data[item]["ego_future_xyz"][0][0]
        ego_rot = self.traj_data[item]["ego_future_rot"][0][0]
        # Extract yaw as sin/cos from rotation matrix
        ego_yaw_sincos = self.extract_yaw_sincos(ego_rot)
        # (5, 64) for each sample: [x, y, z, sin_yaw, cos_yaw]
        feature = torch.cat((torch.tensor(ego_xyz, dtype=torch.float32), ego_yaw_sincos), dim=1).T

        if self.augmentor is not None:
            feature = self.augmentor(feature)

        return feature

