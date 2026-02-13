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

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from nanoRecSys.config import settings
from nanoRecSys.models.ranker import MLPRanker
from nanoRecSys.utils.utils import compute_item_probabilities


@pytest.mark.data
def test_ranker_overfits_real_data():
    """
    Overfit check using REAL training data artifacts and REAL model structure.
    """
    # 0. paths
    data_dir = settings.processed_data_dir
    artifacts_dir = settings.artifacts_dir

    if not (data_dir / "train.parquet").exists():
        pytest.skip("Real training data not found, skipping...")

    # 1. Load Real Embeddings & Metadata
    try:
        user_embs = np.load(artifacts_dir / "user_embeddings.npy")
        item_embs = np.load(artifacts_dir / "item_embeddings.npy")
        genre_matrix = np.load(data_dir / "genre_matrix_binned.npy")
        year_indices = np.load(data_dir / "year_indices_binned.npy")
    except FileNotFoundError:
        pytest.skip("Artifacts (embeddings/metadata) not found, skipping...")

    # 2. Get 1 Positive and 1 Negative Sample from Interactions
    # Read a small chunk of train.parquet to avoid memory issues
    df = pd.read_parquet(data_dir / "train.parquet")

    # Positive sample (Rating >= 4.0 just to be safe)
    # Ensure we get a user index that is within bounds of user_embs (usually it matches)
    pos_interactions = df[df["rating"] >= 4.0]
    if pos_interactions.empty:
        pytest.fail("No positive interactions found in train.parquet")

    pos_interaction = pos_interactions.iloc[0]
    u_idx_pos = int(pos_interaction["user_idx"])
    i_idx_pos = int(pos_interaction["item_idx"])

    # Negative sample
    # Try to find a low rating in the same df
    neg_interaction_df = df[df["rating"] <= 2.0]
    if not neg_interaction_df.empty:
        neg_interaction = neg_interaction_df.iloc[0]
        u_idx_neg = int(neg_interaction["user_idx"])
        i_idx_neg = int(neg_interaction["item_idx"])
    else:
        # Fallback: pick same user, random item that isn't the positive one
        u_idx_neg = u_idx_pos
        i_idx_neg = (i_idx_pos + 1) % len(item_embs)

    print(f"\nPositive Sample: User={u_idx_pos}, Item={i_idx_pos}")
    print(f"Negative Sample: User={u_idx_neg}, Item={i_idx_neg}")

    # 3. Model Setup (Infer dims from data)
    input_dim = user_embs.shape[1]
    num_genres = genre_matrix.shape[1]
    num_years = int(year_indices.max()) + 1

    print(
        f"Model Config: input_dim={input_dim}, num_genres={num_genres}, num_years={num_years}"
    )

    torch.manual_seed(42)
    model = MLPRanker(
        input_dim=input_dim,
        hidden_dims=settings.ranker_hidden_dims,
        num_genres=num_genres,
        genre_dim=16,
        num_years=num_years,
        year_dim=8,
    )

    # 4. Prepare Batch
    # Indices
    b_u_indices = [u_idx_pos, u_idx_neg]
    b_i_indices = [i_idx_pos, i_idx_neg]

    # Fetch Features
    b_user_emb = torch.tensor(user_embs[b_u_indices], dtype=torch.float32)
    b_item_emb = torch.tensor(item_embs[b_i_indices], dtype=torch.float32)

    # Metadata
    b_genre_multihot = torch.tensor(genre_matrix[b_i_indices], dtype=torch.float32)
    b_year_idx = torch.tensor(year_indices[b_i_indices], dtype=torch.long)

    # Popularity (REALISTIC: Log Probs + Normalization)
    # We compute global stats to match training logic exactly
    num_items = len(item_embs)
    print("Computing global item probabilities for normalization...")
    item_pop_log_probs = compute_item_probabilities(
        num_items, return_log_probs=True, device="cpu"
    )
    pop_mean = item_pop_log_probs.mean()
    pop_std = item_pop_log_probs.std()

    # Lookup for our batch
    b_pop_raw = item_pop_log_probs[b_i_indices]

    # Normalize! (Exact same logic as RankerPL.forward)
    b_popularity = (b_pop_raw - pop_mean) / (pop_std + 1e-6)
    b_popularity = b_popularity.unsqueeze(1)  # (B, 1)

    # Labels
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)

    # 5. Training Loop
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    initial_loss = None
    final_loss = None

    print("Starting Training...")
    # Increase epochs for convergence with small batch BN
    for i in range(500):
        optimizer.zero_grad()

        logits = model(
            user_emb=b_user_emb,
            item_emb=b_item_emb,
            genre_multihot=b_genre_multihot,
            year_idx=b_year_idx,
            popularity=b_popularity,
        )

        loss = criterion(logits, labels)

        if i == 0:
            initial_loss = loss.item()
            print(f"Initial Loss: {initial_loss:.4f}")

        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if i % 100 == 0:
            print(f"Msg: Epoch {i}, Loss: {final_loss:.6f}")

    print(f"Final Loss: {final_loss:.6f}")

    # 6. Assertions
    # Loss should be very low on the training batch
    assert final_loss < 0.05, f"Real-data model failed to overfit. Loss: {final_loss}"  # type: ignore

    # Check predictions in Eval mode
    model.eval()
    with torch.inference_mode():
        preds_eval = model.predict(
            b_user_emb, b_item_emb, b_genre_multihot, b_year_idx, b_popularity
        )
        print(f"Predictions (Eval): {preds_eval}")

    # Check predictions in Train mode (to bypass BN running stats lag issues)
    model.train()
    with torch.inference_mode():
        preds_train = model.predict(
            b_user_emb, b_item_emb, b_genre_multihot, b_year_idx, b_popularity
        )
        print(f"Predictions (Train): {preds_train}")

    # Assert roughly correct
    # We prioritize the fact that loss went to ~0, which means the model captured the data.
    # Eval mode might lag due to BN statistics on such a tiny batch.
    assert preds_train[0] > 0.9 and preds_train[1] < 0.1, (
        "Train mode predictions should be perfect"
    )

    # Relaxed Eval assertion
    assert preds_eval[0] > 0.7 and preds_eval[1] < 0.3, (
        "Eval mode predictions should be reasonable"
    )


if __name__ == "__main__":
    test_ranker_overfits_real_data()
