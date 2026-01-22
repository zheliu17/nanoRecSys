import pandas as pd
import numpy as np
from src.config import settings


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
    print("Loading data for splitting...")
    df = load_processed_data()

    # Sort by timestamp
    print("Sorting by timestamp...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    test_start_idx = int(n * (1 - test_ratio))
    val_start_idx = int(n * (1 - val_ratio - test_ratio))

    print(f"Total interactions: {n}")
    print(f"Train end index: {val_start_idx}")
    print(f"Val end index: {test_start_idx}")

    # Create splits
    train = df.iloc[:val_start_idx].copy()
    val = df.iloc[val_start_idx:test_start_idx].copy()
    test = df.iloc[test_start_idx:].copy()

    # Stats
    print(f"Train size: {len(train)} ({len(train) / n:.2%})")
    print(f"Val size:   {len(val)} ({len(val) / n:.2%})")
    print(f"Test size:  {len(test)} ({len(test) / n:.2%})")

    # Time boundaries
    print(
        f"Train time range: {pd.to_datetime(train['timestamp'].min(), unit='s')} -> {pd.to_datetime(train['timestamp'].max(), unit='s')}"
    )
    print(
        f"Val time range:   {pd.to_datetime(val['timestamp'].min(), unit='s')} -> {pd.to_datetime(val['timestamp'].max(), unit='s')}"
    )
    print(
        f"Test time range:  {pd.to_datetime(test['timestamp'].min(), unit='s')} -> {pd.to_datetime(test['timestamp'].max(), unit='s')}"
    )

    # Save splits
    print("Saving splits...")
    train.to_parquet(settings.processed_data_dir / "train.parquet", index=False)
    val.to_parquet(settings.processed_data_dir / "val.parquet", index=False)
    test.to_parquet(settings.processed_data_dir / "test.parquet", index=False)

    # Also save unique train users/items for matrix shapes
    np.save(settings.processed_data_dir / "train_users.npy", train["user_idx"].unique())
    np.save(settings.processed_data_dir / "train_items.npy", train["item_idx"].unique())


if __name__ == "__main__":
    create_global_time_split()
