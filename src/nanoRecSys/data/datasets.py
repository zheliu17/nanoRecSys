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

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nanoRecSys.utils.logging_config import get_logger


def load_item_metadata(item_map_path, movies_path, cache_dir=None):
    """
    Load item metadata: Genres (Multi-Hot) and Years (Indices).

    Year binning strategy:
    Unknown, Pre-1970, 70s, 80s, 90s, 2000+

    Returns:
        genre_matrix: (num_items, num_genres) float tensor
        year_indices: (num_items,) long tensor
        num_genres: int
        num_years: int (Total bins)
    """
    import os

    if cache_dir:
        g_path = os.path.join(cache_dir, "genre_matrix_binned.npy")
        y_path = os.path.join(cache_dir, "year_indices_binned.npy")
        if os.path.exists(g_path) and os.path.exists(y_path):
            logger = get_logger()
            logger.info("Loading cached metadata (binned)...")
            g_mat = torch.from_numpy(np.load(g_path)).float()
            y_ind = torch.from_numpy(np.load(y_path)).long()
            # Inferred dimensions
            # Year bin size is fixed at 6
            return g_mat, y_ind, g_mat.shape[1], 6

    item_map = np.load(item_map_path)
    movies_df = pd.read_csv(movies_path)

    movie_to_genres = dict(zip(movies_df["movieId"], movies_df["genres"]))
    movie_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    # --- GENRES ---
    all_genres = set()
    for g_str in movies_df["genres"].dropna():
        for g in g_str.split("|"):
            all_genres.add(g)
    sorted_genres = sorted(list(all_genres))
    sorted_genres = ["Unknown"] + sorted_genres
    genre2id = {g: i for i, g in enumerate(sorted_genres)}
    num_genres = len(genre2id)

    # --- YEARS (Binned) ---
    year_buckets = ["Unknown", "Pre-1970", "70s", "80s", "90s", "2000+"]
    bucket2id = {b: i for i, b in enumerate(year_buckets)}
    num_years = len(year_buckets)

    def get_year_bucket_idx(title):
        match = re.search(r"\((\d{4})\)", str(title))
        if match:
            year = int(match.group(1))
            if year < 1970:
                bucket = "Pre-1970"
            elif 1970 <= year < 1980:
                bucket = "70s"
            elif 1980 <= year < 1990:
                bucket = "80s"
            elif 1990 <= year < 2000:
                bucket = "90s"
            else:  # 2000+
                bucket = "2000+"
        else:
            bucket = "Unknown"
        return bucket2id[bucket]

    num_items = len(item_map)
    genre_matrix = torch.zeros((num_items, num_genres), dtype=torch.float)
    year_indices = torch.zeros((num_items,), dtype=torch.long)

    logger = get_logger()
    logger.info("Building Genre (Multi-Hot) + Year (Binned) Metadata...")
    for idx, raw_id in enumerate(item_map):
        # Genres
        g_str = movie_to_genres.get(raw_id, "")
        has_genre = False
        if not (pd.isna(g_str) or g_str == "(no genres listed)" or not g_str):
            for g in g_str.split("|"):
                if g in genre2id:
                    genre_matrix[idx, genre2id[g]] = 1.0
                    has_genre = True

        if not has_genre:
            genre_matrix[idx, genre2id["Unknown"]] = 1.0

        # Year
        title = movie_to_title.get(raw_id, "")
        year_indices[idx] = get_year_bucket_idx(title)

    if cache_dir:
        g_path = os.path.join(cache_dir, "genre_matrix_binned.npy")
        y_path = os.path.join(cache_dir, "year_indices_binned.npy")
        np.save(g_path, genre_matrix.numpy())
        np.save(y_path, year_indices.numpy())

    return genre_matrix, year_indices, num_genres, num_years


class InteractionsDataset(Dataset):
    """
    Unified dataset for training/validation interactions from parquet file.
    Can filter by rating thresholds.
    Yields (user_idx, item_idx, rating).
    """

    def __init__(
        self,
        interactions_path: str,
        positive_threshold: float | None = None,
        negative_threshold: float | None = None,
    ):
        self.df = pd.read_parquet(interactions_path)

        if positive_threshold is not None and negative_threshold is not None:
            mask = (self.df["rating"] >= positive_threshold) | (
                self.df["rating"] <= negative_threshold
            )
            self.df = self.df[mask]
        elif positive_threshold is not None:
            self.df = self.df[self.df["rating"] >= positive_threshold]

        self.users = self.df["user_idx"].values.astype(np.int64)
        self.items = self.df["item_idx"].values.astype(np.int64)
        self.ratings = self.df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

    def __getitems__(self, idxs):
        return self.users[idxs], self.items[idxs], self.ratings[idxs]


class UniqueUserDataset(Dataset):
    def __init__(self, user_histories):
        self.users = list(user_histories.keys())
        self.histories = user_histories

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        items = self.histories[user]

        item = np.random.choice(items)

        # Return a dummy rating (5.0) to match InteractionsDataset signature
        return user, item, 5.0

    def __getitems__(self, idxs):
        users_batch = []
        items_batch = []
        ratings_batch = []
        for idx in idxs:
            user = self.users[idx]
            items = self.histories[user]
            item = np.random.choice(items)
            users_batch.append(user)
            items_batch.append(item)
            ratings_batch.append(5.0)

        return (
            np.array(users_batch, dtype=np.int64),
            np.array(items_batch, dtype=np.int64),
            np.array(ratings_batch, dtype=np.float32),
        )


def _load_interactions_data(path, pos_threshold=None, neg_threshold=None):
    """Returns (pos_data, neg_data) numpy arrays (N, 2) or None."""
    if not os.path.exists(path):
        return None, None
    df = pd.read_parquet(path)
    pos_data = None
    neg_data = None
    if pos_threshold is not None:
        pos_data = df[df["rating"] >= pos_threshold][["user_idx", "item_idx"]].values
    if neg_threshold is not None:
        neg_data = df[df["rating"] <= neg_threshold][["user_idx", "item_idx"]].values
    return pos_data, neg_data


def _load_hard_negatives_data(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return df[["user_idx", "neg_item_idx"]].values


def _load_random_negatives_data(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_parquet(path)

    # Identify negative columns dynamically
    neg_cols = [c for c in df.columns if c.startswith("neg_item_idx_")]
    if not neg_cols:
        return None

    u = df["user_idx"].values

    user_list = []
    item_list = []

    for col in neg_cols:
        user_list.append(u)
        item_list.append(df[col].values)

    rand_users = np.concatenate(user_list, axis=0)
    rand_items = np.concatenate(item_list, axis=0)

    return np.stack([rand_users, rand_items], axis=1)


def _make_block(data, label, weight, return_weight):
    if data is None or len(data) == 0:
        return None
    n = len(data)
    block = [
        data,
        np.full((n, 1), label, dtype=np.float32),
    ]
    if return_weight:
        block.append(np.full((n, 1), weight, dtype=np.float32))
    return np.hstack(block)


class RankerTrainDataset(Dataset):
    """
    Unified dataset for Ranker training.
    Positives (W=1), Explicit Negs (W=explicit_neg_weight), Hard Negs (W=1), Random Negs (W=1).
    """

    def __init__(
        self,
        interactions_path: str,
        hard_neg_path: str,
        random_neg_path: str,
        pos_threshold: float | None,
        neg_threshold: float | None,
        explicit_neg_weight: float,
        random_neg_ratio: float,
    ):
        assert 0.0 <= random_neg_ratio <= 1.0, "random_neg_ratio must be in [0.0, 1.0]"
        logger = get_logger()
        logger.info("Building Ranker Training Dataset...")
        data_blocks = []

        # 1. Interactions
        pos_data, exp_neg_data = _load_interactions_data(
            interactions_path, pos_threshold, neg_threshold
        )

        b_pos = _make_block(pos_data, label=1.0, weight=1.0, return_weight=True)
        if b_pos is not None:
            data_blocks.append(b_pos)
            if pos_data is not None:
                logger.info(f"  - Loaded {pos_data.shape[0]} positives")

        if explicit_neg_weight > 0.0 and exp_neg_data is not None:
            b_exp = _make_block(
                exp_neg_data, label=0.0, weight=explicit_neg_weight, return_weight=True
            )
            data_blocks.append(b_exp)
            logger.info(
                f"  - Loaded {exp_neg_data.shape[0]} explicit negatives (weight={explicit_neg_weight})"
            )
        else:
            logger.info("  - Skipping explicit negatives")

        # 2. Hard Negatives
        hard_data = _load_hard_negatives_data(hard_neg_path)
        b_hard = _make_block(hard_data, label=0.0, weight=1.0, return_weight=True)
        if b_hard is not None:
            data_blocks.append(b_hard)
            if hard_data is not None:
                logger.info(f"  - Loaded {hard_data.shape[0]} hard negatives")

        # 3. Random Negatives
        assert 0.0 <= random_neg_ratio <= 1.0, "random_neg_ratio must be in [0.0, 1.0]"
        if random_neg_ratio > 0:
            rand_data = _load_random_negatives_data(random_neg_path)
            if rand_data is not None:
                if random_neg_ratio < 1.0:
                    n_keep = int(len(rand_data) * random_neg_ratio)
                    logger.info(
                        f"  - Subsampling random negatives: {random_neg_ratio:.2f} ({n_keep}/{len(rand_data)})"
                    )
                    # Random sampling without replacement
                    indices = np.random.choice(len(rand_data), n_keep, replace=False)
                    rand_data = rand_data[indices]

                b_rand = _make_block(
                    rand_data, label=0.0, weight=1.0, return_weight=True
                )
                if b_rand is not None:
                    data_blocks.append(b_rand)
                    logger.info(f"  - Loaded {rand_data.shape[0]} random negatives")
        else:
            logger.info("  - Skipping random negatives (ratio=0)")

        if not data_blocks:
            raise ValueError("No training data found!")

        # Keep columns separate and typed to avoid casting in __getitem__
        full_data = np.vstack(data_blocks)
        self.users = full_data[:, 0].astype(np.int64)
        self.items = full_data[:, 1].astype(np.int64)
        self.labels = full_data[:, 2].astype(np.float32)
        self.weights = full_data[:, 3].astype(np.float32)

        logger.info(f"Total Ranker Training Samples: {len(self.users)}")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # Return numpy scalars, let default_collate convert to tensor (faster)
        return self.users[idx], self.items[idx], self.labels[idx], self.weights[idx]

    def __getitems__(self, idxs):
        return self.users[idxs], self.items[idxs], self.labels[idxs], self.weights[idxs]


class RankerEvalDataset(Dataset):
    """
    Dataset for Ranker Validation with specific negative sources.
    mode: 'explicit', 'hard', 'random'
    """

    def __init__(
        self,
        interactions_path: str,
        negatives_path: str | None,
        mode: str,
        pos_threshold: float,
        neg_threshold: float | None = None,
    ):
        logger = get_logger()
        data_blocks = []

        # 1. Positives (Always loaded)
        pos_data, _ = _load_interactions_data(
            interactions_path, pos_threshold=pos_threshold
        )
        b_pos = _make_block(pos_data, label=1.0, weight=1.0, return_weight=False)
        if b_pos is not None:
            data_blocks.append(b_pos)

        # 2. Negatives
        neg_data = None
        if mode == "explicit":
            if neg_threshold is None:
                raise ValueError("neg_threshold required for explicit mode")
            _, neg_data = _load_interactions_data(
                interactions_path, neg_threshold=neg_threshold
            )
        elif mode == "hard":
            neg_data = _load_hard_negatives_data(negatives_path)
        elif mode == "random":
            neg_data = _load_random_negatives_data(negatives_path)
        else:
            logger.warning(f"Unknown mode {mode}")

        b_neg = _make_block(neg_data, label=0.0, weight=1.0, return_weight=False)
        if b_neg is not None:
            data_blocks.append(b_neg)
        elif (
            mode != "explicit" and negatives_path
        ):  # Warn if path provided but data empty/missing
            logger.warning(f"No negatives loaded from {negatives_path} for mode {mode}")

        if data_blocks:
            full_data = np.vstack(data_blocks)
            # Shuffle in-place so batches have mixed labels (avoids AUROC warning)
            np.random.shuffle(full_data)
        else:
            full_data = np.empty((0, 3))

        self.users = full_data[:, 0].astype(np.int64)
        self.items = full_data[:, 1].astype(np.int64)
        self.labels = full_data[:, 2].astype(np.float32)

        logger.info(f"Ranker Eval Dataset ({mode}): {len(self.users)} samples")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def __getitems__(self, idxs):
        return self.users[idxs], self.items[idxs], self.labels[idxs]


class SequentialDataset(Dataset):
    def __init__(self, interactions_path: str):
        logger = get_logger()
        # ".../train.parquet" or ".../val.parquet"
        path_obj = Path(interactions_path)
        stem = path_obj.stem

        parent = path_obj.parent
        # 1 index
        seq_path = parent / f"seq_{stem}_sequences.npy"
        # 0 index, see src/nanoRecSys/data/build_dataset.py
        uid_path = parent / f"seq_{stem}_user_ids.npy"

        if not (seq_path.exists() and uid_path.exists()):
            logger.error(f"Pre-built sequence files not found for {stem} at {parent}")
            logger.error(
                "Please run: python src/nanoRecSys/data/build_dataset.py --task prebuild"
            )
            raise FileNotFoundError(f"Missing pre-built sequences for {stem}")

        logger.info(f"Loading pre-built sequences for {stem}...")
        self.sequences = np.load(seq_path)
        self.user_ids = np.load(uid_path)

        if stem == "val":
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(self.sequences))

            self.sequences = self.sequences[perm]
            self.user_ids = self.user_ids[perm]

        logger.info(f"Sequential Dataset ({stem}): {len(self.sequences)} samples")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.user_ids[idx]

    def __getitems__(self, idxs):
        return self.sequences[idxs], self.user_ids[idxs]
