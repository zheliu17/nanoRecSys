"""
Mine hard negatives and random negatives for training the ranker.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nanoRecSys.config import settings
from nanoRecSys.utils.logging_config import get_logger


def load_all_positives() -> dict:
    """
    Load all positive interactions for each user across train, val, and test splits.
    Returns:
        dict: user_idx -> set of positive item_idxs
    """
    logger = get_logger()
    logger.info("Loading all interactions to build global positive sets...")

    dfs = []
    splits = ["train", "val", "test"]

    for split in splits:
        path = settings.processed_data_dir / f"{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Filter by ranker positive threshold
            df = df[df["rating"] >= settings.ranker_positive_threshold]
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


def mine_hard_negatives_for_split(
    split_name: str,
    df: pd.DataFrame,
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    all_positives: dict,
    top_k: int = 100,
    skip_top: int = 20,
    batch_size: int = 256,
    device: str = "cpu",
):
    """
    Mine hard negatives for a specific split.
    """
    logger = get_logger()
    logger.info(f"Mining hard negatives for {split_name} set...")

    # We need to process users present in this split
    # Since we need to assign a hard negative for EACH interaction,
    # we first pre-compute candidate negatives for each unique user in this split.
    unique_users = df["user_idx"].unique()
    user_candidates = {}  # user_idx -> list of hard negative items

    item_embs_tensor = torch.tensor(item_embeddings, device=device)

    # Process users in batches to find their top-k hard negatives
    logger.info(f"Retrieving candidates for {len(unique_users)} users...")

    for i in tqdm(range(0, len(unique_users), batch_size)):
        batch_users = unique_users[i : i + batch_size]

        # Get user embeddings
        batch_u_embs = torch.tensor(user_embeddings[batch_users], device=device)

        # Dot product
        scores = torch.matmul(batch_u_embs, item_embs_tensor.T)

        # Get Top K
        _, topk_indices = torch.topk(scores, k=top_k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        # Filter and select
        for u_idx, retrieved_items in zip(batch_users, topk_indices):
            known_pos = all_positives.get(u_idx, set())

            # Filter out known positives
            negatives = [item for item in retrieved_items if item not in known_pos]

            # Select from the "tail" of the top-k (skipping the "too hard" ones)
            # If we have enough negatives, skip the first 'skip_top' (e.g., 20)
            if len(negatives) > skip_top:
                candidates = negatives[skip_top:]
            else:
                # If we don't have enough, just use what we have (even if they are "too hard")
                candidates = negatives

            if not candidates:
                # Fallback: if all top-k are positive (unlikely with k=100), sample random
                # This should be extremely rare
                candidates = []

            user_candidates[u_idx] = candidates

    # Now assign one hard negative for each interaction in the dataframe
    n_items = item_embeddings.shape[0]
    hard_negatives = np.zeros(len(df), dtype=np.int32)

    logger.info(f"Assigning hard negatives to {len(df)} interactions...")
    user_pos_indices = df.groupby("user_idx").indices

    for u_idx, indices in tqdm(user_pos_indices.items()):
        candidates = user_candidates.get(u_idx, [])
        count = len(indices)

        if candidates:
            # Vectorized sampling with replacement
            choices = np.random.choice(candidates, size=count, replace=True)
            hard_negatives[indices] = choices
        else:
            # Fallback (rare): Random sampling excluding known positives
            known_pos = all_positives.get(u_idx, set())
            for idx in indices:
                while True:
                    neg = np.random.randint(0, n_items)
                    if neg not in known_pos:
                        hard_negatives[idx] = neg
                        break

    return hard_negatives


def mine_random_negatives_for_split(
    split_name: str,
    df: pd.DataFrame,
    all_positives: dict,
    n_items: int,
    num_negatives: int = 2,
):
    """
    Mine random negatives for a specific split.
    """
    logger = get_logger()
    logger.info(
        f"Mining {num_negatives} random negatives per interaction for {split_name}..."
    )

    n_interactions = len(df)
    random_negatives = np.zeros((n_interactions, num_negatives), dtype=np.int32)

    # Group interactions by user to sample in batches
    user_pos_indices = df.groupby("user_idx").indices

    for u_idx, indices in tqdm(user_pos_indices.items()):
        user_posids = all_positives.get(u_idx, set())
        n_rows = len(indices)
        n_needed = n_rows * num_negatives

        # Initial sampling with a small buffer for collisions
        n_buffer = int(n_needed * 1.1) + 10
        candidates = np.random.randint(0, n_items, size=n_buffer)

        # Filter positives efficiently
        if user_posids:
            valid_candidates = [x for x in candidates if x not in user_posids]
        else:
            valid_candidates = candidates.tolist()

        # Refill if insufficient (rejection sampling)
        while len(valid_candidates) < n_needed:
            shortfall = n_needed - len(valid_candidates)
            new_cands = np.random.randint(0, n_items, size=int(shortfall * 1.5) + 5)
            if user_posids:
                new_valid = [x for x in new_cands if x not in user_posids]
            else:
                new_valid = new_cands.tolist()
            valid_candidates.extend(new_valid)

        # Assign to result array
        selected = np.array(valid_candidates[:n_needed], dtype=np.int32)
        random_negatives[indices] = selected.reshape(n_rows, num_negatives)

    return random_negatives


def main(top_k=None, skip_top=None, batch_size=None):
    logger = get_logger()
    if top_k is None:
        top_k = settings.mining_top_k
    if skip_top is None:
        skip_top = settings.mining_skip_top
    if batch_size is None:
        batch_size = settings.batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(
        f"Parameters: batch_size={batch_size}, top_k={top_k}, skip_top={skip_top}"
    )

    # 1. Load item embeddings from disk
    item_emb_path = settings.artifacts_dir / "item_embeddings.npy"
    if not item_emb_path.exists():
        logger.error(f"Item embeddings not found at {item_emb_path}")
        logger.error(
            "Please generate embeddings first using: python src/indexing/build_embeddings.py --mode items"
        )
        return
    logger.info("Loading item embeddings...")
    item_embeddings = np.load(item_emb_path)

    n_items = item_embeddings.shape[0]

    # 2. Load user embeddings from disk
    user_emb_path = settings.artifacts_dir / "user_embeddings.npy"
    if not user_emb_path.exists():
        logger.error(f"User embeddings not found at {user_emb_path}")
        logger.error(
            "Please generate embeddings first using: python src/indexing/build_embeddings.py --mode users"
        )
        return
    logger.info("Loading user embeddings...")
    user_embeddings = np.load(user_emb_path)

    # 3. Load Global Positives
    all_positives = load_all_positives()

    # Process Train and Val splits
    for split in ["train", "val"]:
        path = settings.processed_data_dir / f"{split}.parquet"
        if not path.exists():
            logger.info(f"Skipping {split}, file not found.")
            continue

        logger.info(f"Processing {split} split...")
        df = pd.read_parquet(path)

        # Filter for positive interactions only
        df_pos = df[df["rating"] >= settings.ranker_positive_threshold].copy()
        logger.info(f"Found {len(df_pos)} positive interactions in {split}.")

        # 4. Mine Hard Negatives
        hard_negs = mine_hard_negatives_for_split(
            split,
            df_pos,
            user_embeddings,
            item_embeddings,
            all_positives,
            top_k=top_k,
            skip_top=skip_top,
            batch_size=batch_size,
            device=device,
        )

        # Save Hard Negatives
        # We save aligned with the filtered dataframe
        save_path_hard = settings.processed_data_dir / f"{split}_negatives_hard.parquet"
        df_hard = pd.DataFrame(
            {
                "user_idx": df_pos["user_idx"].values,
                # "item_idx": df_pos["item_idx"].values,  # The positive item
                "neg_item_idx": hard_negs,
            }
        )

        if split == "val":
            df_hard = df_hard.sample(frac=1).reset_index(drop=True)

        df_hard.to_parquet(save_path_hard)
        print(f"Saved hard negatives to {save_path_hard}")

        # 5. Mine Random Negatives
        random_negs = mine_random_negatives_for_split(
            split,
            df_pos,
            all_positives,
            n_items,
            num_negatives=settings.mining_num_negatives,
        )

        # Save Random Negatives
        save_path_random = (
            settings.processed_data_dir / f"{split}_negatives_random.parquet"
        )

        # Dynamic dict construction
        data_dict = {"user_idx": df_pos["user_idx"].values}
        # Assuming random_negs is (N, num_negatives)
        for i in range(random_negs.shape[1]):
            data_dict[f"neg_item_idx_{i + 1}"] = random_negs[:, i]

        df_random = pd.DataFrame(data_dict)

        if split == "val":
            df_random = df_random.sample(frac=1).reset_index(drop=True)

        df_random.to_parquet(save_path_random)
        print(f"Saved random negatives to {save_path_random}")


if __name__ == "__main__":
    main()
