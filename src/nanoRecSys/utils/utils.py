# Copyright (c) 2026 Zhe Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training utilities for nanoRecSys."""

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR

from nanoRecSys.config import settings
from nanoRecSys.utils.logging_config import get_logger


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


def get_linear_warmup_scheduler(optimizer, warmup_steps):
    """Create a linear warmup scheduler."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def load_all_positives(
    threshold: float | None = None, splits: list[str] | None = None
) -> dict:
    """
    Load all positive interactions for each user across specified splits.
    Returns:
        dict: user_idx -> set of positive item_idxs
    """
    logger = get_logger()
    threshold = threshold if threshold is not None else 0
    splits = splits or ["train", "val", "test"]

    logger.info(
        f"Loading interactions from {splits} to build positive sets (threshold={threshold})..."
    )

    dfs = []

    for split in splits:
        path = settings.processed_data_dir / f"{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df = df[df["rating"] >= threshold]
            dfs.append(df[["user_idx", "item_idx"]])

    if not dfs:
        raise FileNotFoundError(
            "No processed data files found (train/val/test.parquet)"
        )

    full_df = pd.concat(dfs, ignore_index=True)

    # Group by user
    logger.info(f"Grouping {len(full_df)} positive interactions by user...")
    user_positives = full_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    return user_positives


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
    logger = get_logger()
    logger.info("Loading training data to compute item probabilities...")
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


def compute_seq_item_probabilities(
    n_items, return_log_probs=False, device=None, smooth=True
):
    """
    Compute item probabilities from sequential training data (seq_train_sequences.npy).

    The sequences file contains 1-based item indices with 0 padding.
    This function counts actual item occurrences (shifted back to 0-based)
    to estimate popularity for sequential tasks.

    Args:
        n_items: Total number of items
        return_log_probs: If True, return log probabilities
        device: torch device
        smooth: If True, apply Laplace smoothing.

    Returns:
        torch.Tensor
    """
    logger = get_logger()
    logger.info("Loading sequential training data to compute item probabilities...")
    seq_path = settings.processed_data_dir / "seq_train_sequences.npy"

    if not seq_path.exists():
        logger.warning(
            f"Sequence file not found at {seq_path}. Falling back to compute_item_probabilities."
        )
        return compute_item_probabilities(n_items, return_log_probs, device, smooth)

    # Load sequences
    seqs = np.load(seq_path)
    flat = seqs.ravel()

    # Filter padding (0)
    nonpad = flat[flat != 0]

    # Shift to 0-based index
    items = (nonpad - 1).astype(np.int64)

    # Count
    counts = np.bincount(items, minlength=n_items).astype(np.float32)

    total_count = counts.sum()

    if smooth:
        # Laplace smoothing
        probs = (counts + 1.0) / (total_count + n_items)
    else:
        if total_count > 0:
            probs = counts / total_count
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
    Collate function for converting numpy batches to tensor batches.
    Handles standard list-of-samples, pre-batched tuples, and single-sample edge cases.
    """
    # 1. Standard DataLoader: List of tuples -> Tuple of stacked arrays
    if isinstance(batch, list):
        batch = tuple(np.stack(items) for items in zip(*batch))

    # 2. Single sample edge case (e.g. DDP yielding single item):
    # (Array, Scalar) -> (Array[1, ...], Array[1])
    elif isinstance(batch, tuple) and any(np.isscalar(x) for x in batch):
        batch = tuple(
            x[None, ...] if isinstance(x, np.ndarray) else np.array([x]) for x in batch
        )

    # 3. Ensure tuple (handle unexpected single array input)
    if isinstance(batch, np.ndarray):
        batch = (batch,)

    # 4. Convert all arrays to tensors
    return tuple(torch.from_numpy(arr) for arr in batch)


class OnDemandEmbeddings:
    """Memory-mapped embeddings for on-demand loading without loading full array into memory."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._memmap: np.ndarray | None = None

    def _ensure_loaded(self) -> None:
        """Lazily load memory-mapped array on first access."""
        if self._memmap is None:
            self._memmap = np.load(self.filepath, mmap_mode="r")

    def __getitem__(self, indices):
        """Get embeddings for given indices, returning as torch tensor."""
        self._ensure_loaded()
        # Convert torch tensor indices to numpy if needed
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        assert self._memmap is not None
        return torch.from_numpy(np.array(self._memmap[indices])).float()

    @property
    def shape(self):
        """Return shape of embeddings."""
        self._ensure_loaded()
        assert self._memmap is not None
        return self._memmap.shape

    @property
    def dtype(self):
        """Return dtype of embeddings."""
        self._ensure_loaded()
        assert self._memmap is not None
        return self._memmap.dtype


def collate_fn_with_embeddings(
    batch, user_embeddings: Union[OnDemandEmbeddings, torch.Tensor]
):
    """Collate function that loads embeddings on-demand for the batch.

    Args:
        batch: List of tuples (user_idx, item_idx, label, weight) or (user_idx, item_idx, label)
        user_embeddings: OnDemandEmbeddings instance for lazy loading

    Returns:
        Tuple of (user_embeddings_tensor, item_indices_tensor, labels_tensor, [weights_tensor])
    """
    has_weights = len(batch) == 4

    # Unpack batch
    if has_weights:
        users, items, labels, weights = batch
        weights = np.array(weights)
    else:
        users, items, labels = batch
        weights = None

    # Convert to numpy arrays
    user_idx = np.array(users)
    item_idx = np.array(items)
    label_data = np.array(labels)

    # Load user embeddings on-demand for this batch
    user_emb = user_embeddings[user_idx]

    # Build result tuple (optionally include weights)
    result = [
        user_emb,
        torch.from_numpy(item_idx).long(),
        torch.from_numpy(label_data).long(),
    ]

    if weights is not None:
        result.append(torch.from_numpy(weights).float())

    return tuple(result)
