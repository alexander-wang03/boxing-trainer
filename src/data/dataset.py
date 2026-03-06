"""
PyTorch Dataset and DataLoader classes for punch and defense classification.

Usage:
    from src.data.dataset import get_punch_loaders, get_defense_loaders
    train_loader, val_loader, test_loader = get_punch_loaders(batch_size=32)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


class BoxingDataset(Dataset):
    """
    Generic dataset for loading preprocessed keypoint sequences.

    Expects .npz file with 'X' (sequences) and 'y' (labels).
    """

    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_loaders(dataset_prefix: str,
                splits_dir: Path = config.SPLITS_DIR,
                batch_size: int = config.BATCH_SIZE,
                num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for a given dataset prefix.

    Args:
        dataset_prefix: 'punch' or 'defense'.
        splits_dir: Directory containing the .npz split files.
        batch_size: Batch size.
        num_workers: Number of dataloader workers.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = BoxingDataset(splits_dir / f"{dataset_prefix}_train.npz")
    val_ds = BoxingDataset(splits_dir / f"{dataset_prefix}_val.npz")
    test_ds = BoxingDataset(splits_dir / f"{dataset_prefix}_test.npz")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Loaded {dataset_prefix} data: "
          f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    return train_loader, val_loader, test_loader


def get_punch_loaders(**kwargs) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get DataLoaders for punch classification."""
    return get_loaders("punch", **kwargs)


def get_defense_loaders(**kwargs) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Get DataLoaders for defense classification."""
    return get_loaders("defense", **kwargs)
