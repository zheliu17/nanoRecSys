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

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from nanoRecSys.models.losses import InfoNCELoss
from nanoRecSys.models.ranker import RankerModel
from nanoRecSys.models.towers import Tower
from nanoRecSys.utils.utils import compute_item_probabilities


# -------------------------------------------------------------------
# 1. Tower Tests
# -------------------------------------------------------------------
class TestTower:
    def test_tower_output_shape_and_norm(self):
        BATCH_SIZE = 4
        VOCAB_SIZE = 10
        EMBED_DIM = 8
        OUTPUT_DIM = 8
        HIDDEN_DIMS = [16, 12]

        tower = Tower(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            output_dim=OUTPUT_DIM,
            hidden_dims=HIDDEN_DIMS,
        )

        # Input: Batch of indices
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE,))

        output = tower(x)

        # Check shape
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

        # Check normalization (L2 norm should be close to 1.0)
        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# -------------------------------------------------------------------
# 2. Loss Tests
# -------------------------------------------------------------------
class TestLosses:
    def test_infonce_loss_forward(self):
        BATCH_SIZE = 4
        EMBED_DIM = 8

        loss_fn = InfoNCELoss(temperature=0.1)

        user_embs = torch.randn(BATCH_SIZE, EMBED_DIM)
        item_embs = torch.randn(BATCH_SIZE, EMBED_DIM)

        # Normalize inputs as the loss expects raw dot products / temp
        user_embs = torch.nn.functional.normalize(user_embs, p=2, dim=1)
        item_embs = torch.nn.functional.normalize(item_embs, p=2, dim=1)

        loss = loss_fn(user_embs, item_embs)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss > 0

    def test_infonce_loss_collision_masking(self):
        # Setup case where user 0 and user 1 are the SAME user ID
        BATCH_SIZE = 3
        EMBED_DIM = 4

        user_embs = torch.randn(BATCH_SIZE, EMBED_DIM)
        item_embs = torch.randn(BATCH_SIZE, EMBED_DIM)
        user_ids = torch.tensor([101, 101, 102])  # 101 is duplicated

        loss_fn = InfoNCELoss(temperature=1.0)

        # Masking should prevent negative mining between same user instances
        loss = loss_fn(user_embs, item_embs, user_ids=user_ids)
        assert loss > 0


# -------------------------------------------------------------------
# 3. Ranker Tests
# -------------------------------------------------------------------
class TestRanker:
    def test_ranker_forward_shape(self):
        BATCH_SIZE = 5
        INPUT_DIM = 8
        NUM_GENRES = 4
        NUM_YEARS = 10

        model = RankerModel(
            input_dim=INPUT_DIM,
            hidden_dims=[16, 8],
            num_genres=NUM_GENRES,
            genre_dim=4,
            num_years=NUM_YEARS,
            year_dim=4,
        )

        # Create mock inputs
        user_emb = torch.randn(BATCH_SIZE, INPUT_DIM)
        item_emb = torch.randn(BATCH_SIZE, INPUT_DIM)

        # Multi-hot genres (random 0s and 1s)
        genre_multihot = torch.randint(0, 2, (BATCH_SIZE, NUM_GENRES)).float()

        # Years
        year_idx = torch.randint(0, NUM_YEARS, (BATCH_SIZE,))

        # Popularity
        popularity = torch.rand(BATCH_SIZE, 1)

        output = model(user_emb, item_emb, genre_multihot, year_idx, popularity)

        # RankerModel.forward applies .squeeze(), so shape is (BATCH_SIZE,)
        assert output.shape == (BATCH_SIZE,)


# -------------------------------------------------------------------
# 4. Utils Tests
# -------------------------------------------------------------------
class TestUtils:
    @patch("nanoRecSys.utils.utils.pd.read_parquet")
    @patch("nanoRecSys.utils.utils.settings")
    def test_compute_item_probabilities(self, mock_settings, mock_read_parquet):
        # Mock settings pathes
        mock_settings.processed_data_dir = MagicMock()

        # Mock Data: 5 items total (0-4), but only 0, 1, 2 appear in train
        # Item 0 appears 2 times
        # Item 1 appears 1 time
        # Item 2 appears 1 time
        data = {"item_idx": [0, 0, 1, 2]}
        df = pd.DataFrame(data)
        mock_read_parquet.return_value = df

        n_items = 5

        # 1. Test Raw Frequencies (smooth=False)
        # Counts: 0->2, 1->1, 2->1, 3->0, 4->0. Total=4.
        # Probs: 0->0.5, 1->0.25, 2->0.25, 0, 0
        probs = compute_item_probabilities(
            n_items, return_log_probs=False, smooth=False
        )

        assert len(probs) == n_items
        assert probs[0].item() == 0.5
        assert probs[3].item() == 0.0

        # 2. Test Smoothing
        # Formula: (count + 1) / (total + n_items) = (count + 1) / (4 + 5) = (count + 1) / 9
        # Item 0: 3/9 = 0.333
        # Item 3: 1/9 = 0.111
        probs_smooth = compute_item_probabilities(
            n_items, return_log_probs=False, smooth=True
        )
        assert probs_smooth[0].item() == pytest.approx(3 / 9)
        assert probs_smooth[3].item() == pytest.approx(1 / 9)
