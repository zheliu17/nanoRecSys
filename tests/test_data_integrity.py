import pytest
import pandas as pd
from src.config import settings


@pytest.fixture
def data_splits():
    try:
        train = pd.read_parquet(settings.processed_data_dir / "train.parquet")
        val = pd.read_parquet(settings.processed_data_dir / "val.parquet")
        test = pd.read_parquet(settings.processed_data_dir / "test.parquet")
        return train, val, test
    except FileNotFoundError:
        pytest.skip("Data splits not found. Run dataset build first.")


@pytest.mark.data
def test_split_temporal_ordering(data_splits):
    train, val, test = data_splits

    train_max_ts = train["timestamp"].max()
    val_min_ts = val["timestamp"].min()
    val_max_ts = val["timestamp"].max()
    test_min_ts = test["timestamp"].min()

    # Check that Train ends before Val starts
    assert train_max_ts <= val_min_ts, (
        "Train data overlaps with Validation data (Time Leakage)"
    )

    # Check that Val ends before Test starts
    assert val_max_ts <= test_min_ts, (
        "Validation data overlaps with Test data (Time Leakage)"
    )


@pytest.mark.data
def test_split_counts_sanity(data_splits):
    train, val, test = data_splits

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    total = len(train) + len(val) + len(test)
    assert total > 1000, "Dataset seems too small"
