"""Training utilities for nanoRecSys."""

import numpy as np
import torch
import pandas as pd
from ..config import settings


def get_vocab_sizes():
    """
    Load user and item maps to determine vocabulary sizes.
    """
    try:
        user_map = np.load(settings.processed_data_dir / "user_map.npy")
        item_map = np.load(settings.processed_data_dir / "item_map.npy")
        return len(user_map), len(item_map)
    except Exception:
        # Fallback if processing wasn't standard or maps missing
        train_users = np.load(settings.processed_data_dir / "train_users.npy")
        train_items = np.load(settings.processed_data_dir / "train_items.npy")
        return train_users.max() + 1, train_items.max() + 1


def compute_item_probabilities(
    n_items, return_log_probs=False, device=None, smooth=True
):
    """
    Compute item probabilities from training data.

    Args:
        n_items: Total number of items
        return_log_probs: If True, return log probabilities
        device: torch device
        smooth: If True, apply Laplace smoothing. If False, use raw frequencies.

    Returns:
        torch.Tensor
    """
    print("Loading training data to compute item probabilities...")
    train_df = pd.read_parquet(settings.processed_data_dir / "train.parquet")

    # Calculate item counts
    item_counts = (
        train_df["item_idx"].value_counts().reindex(range(n_items), fill_value=0)
    )
    counts_np = item_counts.to_numpy(dtype=np.float32)

    total_count = len(train_df)

    if smooth:
        # Laplace smoothing
        probs = (counts_np + 1.0) / (total_count + n_items)
    else:
        # Raw frequencies
        if total_count > 0:
            probs = counts_np / total_count
        else:
            probs = np.zeros(n_items, dtype=np.float32)

    if return_log_probs:
        epsilon = 1e-10
        probs = np.log(probs + epsilon)

    probs_tensor = torch.from_numpy(probs).float()
    if device:
        probs_tensor = probs_tensor.to(device)
    return probs_tensor


def collate_fn_numpy_to_tensor(batch):
    """
    Collate function for BatchSampler that converts a tuple of numpy arrays
    into a tuple of torch tensors.
    """
    return tuple(torch.from_numpy(arr) for arr in batch)
