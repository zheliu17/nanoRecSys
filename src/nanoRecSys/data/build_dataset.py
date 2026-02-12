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
import zipfile

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from nanoRecSys.config import settings
from nanoRecSys.utils.logging_config import get_logger


def download_and_extract():
    """Download MovieLens 20M dataset if not already present."""
    logger = get_logger()
    url = settings.ml_20m_url
    zip_path = settings.data_dir / "ml-20m.zip"

    # Check if data already exists (e.g. ratings.csv)
    ratings_path = settings.raw_data_dir / "ratings.csv"
    if ratings_path.exists():
        logger.info(f"Data already exists at {settings.raw_data_dir}")
        return

    logger.info(f"Downloading {url}...")
    response = requests.get(url, stream=True)  # type: ignore
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, leave=False
    )

    with open(zip_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    logger.info("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(settings.data_dir)

    # Clean up zip file
    os.remove(zip_path)
    logger.info("Download and extraction complete.")


def process_data(min_interactions=None):
    """
    Load raw data, encode IDs, and save processed files.

    Args:
        min_interactions (int, optional): If set, perform k-core filtering.
                                          Recursively remove users/items with fewer than k interactions.
    """
    logger = get_logger()
    ratings_path = settings.raw_data_dir / "ratings.csv"
    # movies_path = settings.raw_data_dir / "movies.csv"

    if not ratings_path.exists():
        download_and_extract()

    logger.info("Loading ratings.csv...")
    df = pd.read_csv(
        ratings_path,
        dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int},
    )

    logger.info(f"Raw interactions: {len(df)}")

    # Optional: k-core filtering
    if min_interactions is not None:
        logger.info(f"Applying {min_interactions}-core filtering (recursive)...")
        while True:
            start_len = len(df)

            # Filter users
            user_counts = df["userId"].value_counts()
            valid_users = user_counts[user_counts >= min_interactions].index
            df = df[df["userId"].isin(valid_users)]

            # Filter items
            item_counts = df["movieId"].value_counts()
            valid_items = item_counts[item_counts >= min_interactions].index
            df = df[df["movieId"].isin(valid_items)]

            if len(df) == start_len:
                break

            logger.info(f"Reduced to {len(df)} interactions...")

        logger.info(
            f"Final interactions after {min_interactions}-core filtering: {len(df)}"
        )

    # 1. Encode Users
    unique_users = df["userId"].unique()
    user2id = {u: i for i, u in enumerate(unique_users)}
    df["user_idx"] = df["userId"].map(user2id)

    # 2. Encode Items
    # df["movieId"].unique().shape: 26,744
    # movies_df["movieId"].unique().shape: 27,278 (some movies have no ratings)
    logger.info("Loading movies.csv...")
    # movies_df = pd.read_csv(movies_path)
    unique_movies = df["movieId"].unique()
    movie2id = {m: i for i, m in enumerate(unique_movies)}

    # Map movies in ratings
    df = df[df["movieId"].isin(unique_movies)]  # filter valid movies
    df["item_idx"] = df["movieId"].map(movie2id)

    logger.info(f"Num Users: {len(unique_users)}")
    logger.info(f"Num Items: {len(unique_movies)}")
    logger.info(f"Num Interactions: {len(df)}")

    # Save processed
    logger.info("Saving processed data...")
    df.to_parquet(settings.processed_data_dir / "interactions.parquet", index=False)

    np.save(settings.processed_data_dir / "user_map.npy", unique_users)
    np.save(settings.processed_data_dir / "item_map.npy", unique_movies)

    logger.info("Data processing complete.")


def prebuild_sequential_files():
    """
    Pre-build sequence data for training/validation and inference.
    Saves .npy files to processed_data_dir.
    """
    logger = get_logger()
    logger.info("Pre-building sequential datasets...")

    processed_dir = settings.processed_data_dir

    # Pre-load all data once to avoid redundant reads/sorts
    dfs_dict = {}
    logger.info(
        f"Filtering interactions with rating >= {settings.retrieval_threshold} for sequence building."
    )
    for split in ["train", "val", "test"]:
        parquet_path = processed_dir / f"{split}.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if settings.retrieval_threshold is not None:
                df = df[df["rating"] >= settings.retrieval_threshold]
            # Sort once here
            if "timestamp" in df.columns:
                df = df.sort_values(["user_idx", "timestamp"])
            else:
                logger.info("No timestamp column found; sorting by user_idx only.")
                df = df.sort_values(["user_idx"])
            dfs_dict[split] = df

    if not dfs_dict:
        logger.error("No train/val/test parquet files found.")
        return

    # 1. Build Sliding Window Sequences for Training/Validation
    max_seq_len = settings.max_seq_len
    min_seq_len = settings.min_seq_len
    step_size = settings.seq_step_size

    # Prepare user histories for easy access
    train_groups = (
        dfs_dict.get("train", pd.DataFrame())
        .groupby("user_idx")["item_idx"]
        .apply(list)
        .to_dict()
        if "train" in dfs_dict
        else {}
    )
    val_groups = (
        dfs_dict.get("val", pd.DataFrame())
        .groupby("user_idx")["item_idx"]
        .apply(list)
        .to_dict()
        if "val" in dfs_dict
        else {}
    )
    test_groups = (
        dfs_dict.get("test", pd.DataFrame())
        .groupby("user_idx")["item_idx"]
        .apply(list)
        .to_dict()
        if "test" in dfs_dict
        else {}
    )

    # Identify all users for inference
    all_users = (
        set(train_groups.keys()) | set(val_groups.keys()) | set(test_groups.keys())
    )

    for split in ["train", "val", "test", "inference"]:
        if split != "inference" and split not in dfs_dict:
            continue

        logger.info(f"Building sequences for {split}...")

        sequences_list = []
        user_ids_list = []

        # Train/Val/Test: Input + Target (Window + 1)
        # Inference: Only Input (Window)
        seq_window_len = max_seq_len + 1 if split != "inference" else max_seq_len

        if split == "train":
            # Single loop over users; generate either one last-window sequence
            # or multiple sliding-window sequences depending on setting.
            for user_id, items in tqdm(train_groups.items(), desc=f"{split} sequences"):
                items = [i + 1 for i in items]  # Padding shift
                if len(items) <= min_seq_len:
                    continue

                if settings.train_single_last_sequence:
                    user_seqs = [items[-seq_window_len:]]
                else:
                    user_seqs = []
                    n_items = len(items)
                    for i in range(min_seq_len, n_items, step_size):
                        end = i + 1
                        start = max(0, end - seq_window_len)
                        user_seqs.append(items[start:end])

                # Pad and append each sequence
                for seq in user_seqs:
                    pad_len = seq_window_len - len(seq)
                    if pad_len > 0:
                        seq = [0] * pad_len + seq

                    sequences_list.append(seq)
                    user_ids_list.append(int(user_id))

        elif split in ["val", "test"]:
            # For val/test, we perform per-item evaluation
            target_groups = val_groups if split == "val" else test_groups

            for user_id, target_items in tqdm(
                target_groups.items(), desc=f"{split} sequences"
            ):
                target_items = [i + 1 for i in target_items]

                # Build base history
                history = []
                if user_id in train_groups:
                    history.extend([i + 1 for i in train_groups[user_id]])

                if split == "test" and user_id in val_groups:
                    history.extend([i + 1 for i in val_groups[user_id]])

                # Generate one sequence per target item
                for item in target_items:
                    history.append(item)

                    # Window logic
                    seq = history[-seq_window_len:]
                    pad_len = seq_window_len - len(seq)
                    if pad_len > 0:
                        seq = [0] * pad_len + seq

                    sequences_list.append(seq)
                    user_ids_list.append(int(user_id))

        elif split == "inference":
            # For inference, one sequence per user representing full history
            for user_id in tqdm(sorted(list(all_users)), desc=f"{split} sequences"):
                # Reconstruct full history: Train -> Val -> Test
                history = []
                if user_id in train_groups:
                    history.extend(train_groups[user_id])
                if user_id in val_groups:
                    history.extend(val_groups[user_id])
                # Don't include test interactions in inference history
                # if user_id in test_groups:
                #     history.extend(test_groups[user_id])

                history = [i + 1 for i in history]

                # Take last max_seq_len
                seq = history[-seq_window_len:]
                pad_len = seq_window_len - len(seq)
                if pad_len > 0:
                    seq = [0] * pad_len + seq

                sequences_list.append(seq)
                user_ids_list.append(int(user_id))

        if not sequences_list:
            logger.warning(f"No sequences generated for {split}.")
            continue

        # Save
        np.save(
            processed_dir / f"seq_{split}_sequences.npy",
            np.array(sequences_list, dtype=np.int64),
        )
        # All user IDs are 0 indexed
        np.save(
            processed_dir / f"seq_{split}_user_ids.npy",
            np.array(user_ids_list, dtype=np.int64),
        )

        logger.info(f"Saved {split} sequences: {len(sequences_list)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build/Process MovieLens Dataset")
    parser.add_argument(
        "--k_core",
        type=int,
        default=None,
        help="Apply recursive k-core filtering (remove users/items with < k interactions). Default: None (no filtering).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="process",
        choices=["process", "prebuild"],
        help="Task to perform: 'process' (download & process raw data) or 'prebuild' (generate sequence npy files).",
    )
    args = parser.parse_args()

    if args.task == "process":
        process_data(min_interactions=args.k_core)
    elif args.task == "prebuild":
        prebuild_sequential_files()
