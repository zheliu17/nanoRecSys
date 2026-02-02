import pandas as pd
import numpy as np
from nanoRecSys.config import settings
from nanoRecSys.utils.logging_config import get_logger


def load_processed_data() -> pd.DataFrame:
    """Load the processed interactions data."""
    path = settings.processed_data_dir / "interactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_dataset.py first.")
    return pd.read_parquet(path)


def create_global_time_split(val_ratio: float = 0.1, test_ratio: float = 0.1):
    """
    Split interactions globally by timestamp.
    Train: [0, 1 - val - test)
    Val:   [1 - val - test, 1 - test)
    Test:  [1 - test, 1.0]
    """
    logger = get_logger()
    logger.info("Loading data for splitting...")
    df = load_processed_data()

    # Sort by timestamp
    logger.info("Sorting by timestamp...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    test_start_idx = int(n * (1 - test_ratio))
    val_start_idx = int(n * (1 - val_ratio - test_ratio))

    logger.info(f"Total interactions: {n}")
    logger.info(f"Train end index: {val_start_idx}")
    logger.info(f"Val end index: {test_start_idx}")

    # Create splits
    train = df.iloc[:val_start_idx].copy()
    val = df.iloc[val_start_idx:test_start_idx].copy()
    test = df.iloc[test_start_idx:].copy()

    logger.info("Shuffling rows within each split...")
    # train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    # Shuffle val and test (In-Batch Negatives rely on randomness)
    val = val.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)

    # Stats
    logger.info(f"Train size: {len(train)} ({len(train) / n:.2%})")
    logger.info(f"Val size:   {len(val)} ({len(val) / n:.2%})")
    logger.info(f"Test size:  {len(test)} ({len(test) / n:.2%})")

    # Time boundaries
    logger.info(
        f"Train time range: {pd.to_datetime(train['timestamp'].min(), unit='s')} -> {pd.to_datetime(train['timestamp'].max(), unit='s')}"
    )
    logger.info(
        f"Val time range:   {pd.to_datetime(val['timestamp'].min(), unit='s')} -> {pd.to_datetime(val['timestamp'].max(), unit='s')}"
    )
    logger.info(
        f"Test time range:  {pd.to_datetime(test['timestamp'].min(), unit='s')} -> {pd.to_datetime(test['timestamp'].max(), unit='s')}"
    )

    # Save splits
    logger.info("Saving splits...")
    train.to_parquet(settings.processed_data_dir / "train.parquet", index=False)
    val.to_parquet(settings.processed_data_dir / "val.parquet", index=False)
    test.to_parquet(settings.processed_data_dir / "test.parquet", index=False)

    # Also save unique train users/items for matrix shapes
    np.save(settings.processed_data_dir / "train_users.npy", train["user_idx"].unique())
    np.save(settings.processed_data_dir / "train_items.npy", train["item_idx"].unique())


def create_user_time_split(val_k: int = 5, test_k: int = 5):
    """
    Split interactions by user, selecting the last k interactions for test/val.
    For each user:
      Test: Last test_k interactions
      Val:  Preceding val_k interactions
      Train: The rest (must have at least 1 interaction)
    """
    logger = get_logger()
    logger.info("Loading data for splitting (User-Based)...")
    df = load_processed_data()

    logger.info("Sorting by user and timestamp...")
    # Sort for rank calculation: User Ascending, Timestamp Descending (newest first)
    df = df.sort_values(["user_idx", "timestamp"], ascending=[True, False]).reset_index(
        drop=True
    )

    # Rank: 0 is newest (Test candidate), increases for older items
    # method='first' ensures unique ranks even if timestamps match
    logger.info("Ranking interactions per user...")
    df["rank"] = df.groupby("user_idx").cumcount()

    # Define masks
    is_test = df["rank"] < test_k
    is_val = (df["rank"] >= test_k) & (df["rank"] < (test_k + val_k))
    is_train = df["rank"] >= (test_k + val_k)

    # Filter checks
    n_users = df["user_idx"].nunique()
    logger.info(f"Total Users: {n_users}")

    train = df[is_train].drop(columns=["rank"])
    val = df[is_val].drop(columns=["rank"])
    test = df[is_test].drop(columns=["rank"])

    # Basic stats
    n = len(df)
    logger.info(f"Train size: {len(train)} ({len(train) / n:.2%})")
    logger.info(f"Val size:   {len(val)} ({len(val) / n:.2%})")
    logger.info(f"Test size:  {len(test)} ({len(test) / n:.2%})")

    # Warn if train is small
    train_users = train["user_idx"].nunique()
    logger.info(f"Users in Train: {train_users} (Missing: {n_users - train_users})")

    # Shuffle for saving
    logger.info("Shuffling splits...")
    # train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    val = val.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save splits
    logger.info("Saving splits...")
    train.to_parquet(settings.processed_data_dir / "train.parquet", index=False)
    val.to_parquet(settings.processed_data_dir / "val.parquet", index=False)
    test.to_parquet(settings.processed_data_dir / "test.parquet", index=False)

    # Also save unique train users/items for matrix shapes
    np.save(settings.processed_data_dir / "train_users.npy", train["user_idx"].unique())
    np.save(settings.processed_data_dir / "train_items.npy", train["item_idx"].unique())


if __name__ == "__main__":
    # Switching default to User-Based Split as requested
    # create_global_time_split()
    create_user_time_split(val_k=5, test_k=5)
