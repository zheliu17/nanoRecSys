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

import pytest
import pandas as pd
from nanoRecSys.config import settings


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

    # We check for temporal ordering PER USER.
    # This logic is valid for both Global Time Split and User-Based Time Split.
    # Condition: For each user, max(Train) <= min(Val) <= max(Val) <= min(Test)

    # Train max timestamp per user
    train_max = train.groupby("user_idx")["timestamp"].max()

    # Val min/max timestamp per user
    val_min = val.groupby("user_idx")["timestamp"].min()
    val_max = val.groupby("user_idx")["timestamp"].max()

    # Test min timestamp per user
    test_min = test.groupby("user_idx")["timestamp"].min()

    # Check Train vs Val
    # Only for users present in both partitions
    common_tv = train_max.index.intersection(val_min.index)
    if len(common_tv) > 0:
        # Check if any user has train_max > val_min
        violations = train_max[common_tv] > val_min[common_tv]
        assert not violations.any(), (
            f"Found {violations.sum()} users where Train interactions occur after Validation interactions."
        )

    # Check Val vs Test
    common_vt = val_max.index.intersection(test_min.index)
    if len(common_vt) > 0:
        violations = val_max[common_vt] > test_min[common_vt]
        assert not violations.any(), (
            f"Found {violations.sum()} users where Validation interactions occur after Test interactions."
        )


@pytest.mark.data
def test_split_counts_sanity(data_splits):
    train, val, test = data_splits

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    total = len(train) + len(val) + len(test)
    assert total > 1000, "Dataset seems too small"
