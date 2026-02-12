import torch
import numpy as np
from sympy.printing.pytorch import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk

class TrajDataset:
    # datasets.config.IN_MEMORY_MAX_SIZE = 16000
    def __init__(self, root_path: str = "/dataset", split: str = "train"):
        self.traj_data = load_from_disk(root_path)[split]

    def __len__(self):
        return len(self.traj_data)

    @staticmethod
    def convert_rot_mat_to_6d(rot_mat):
        """
        converts 3x3 rotation matrix to 6d continuous rotations using gram schmidt process
        """
        rot_mat_cols = torch.tensor(rot_mat)[:, :, :2]
        return rot_mat_cols.permute(0, 2, 1).reshape(-1, 6)

    def __getitem__(self, item: int):
        ego_xyz, ego_rot = self.traj_data[item]["ego_future_xyz"][0][0], self.traj_data[item]["ego_future_rot"][0][0]
        # converting 3x3 rotation matrix to 6d rotation (gram schmidt)
        ego_rot = self.convert_rot_mat_to_6d(ego_rot)
        # 64 x 9 for each sample
        feature = torch.cat((ego_xyz, ego_rot), dim=1)
        return feature
