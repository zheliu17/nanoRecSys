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

from typing import List, Union

import numpy as np


def recall_at_k(
    predictions: np.ndarray, targets: List[Union[List[int], np.ndarray]], k: int
) -> float:
    """
    Compute Mean Recall@K.

    Args:
        predictions: (N, K_max) array of predicted item indices, sorted by score.
        targets: List of length N, where each element is a list/array of ground truth item indices.
        k: Cutoff for recall.

    Returns:
        Mean Recall@K over the batch.
    """
    recalls = []
    # Truncate predictions to k
    preds_k = predictions[:, :k]

    for user_preds, user_targets in zip(preds_k, targets):
        if len(user_targets) == 0:
            continue

        # Intersection
        hits = np.isin(user_preds, user_targets).sum()
        recalls.append(hits / len(user_targets))

    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k(
    predictions: np.ndarray, targets: List[Union[List[int], np.ndarray]], k: int
) -> float:
    """
    Compute Mean NDCG@K (assuming binary relevance).

    DCG@K = sum(rel_i / log2(i + 2))
    IDCG@K = sum(1 / log2(i + 2)) for i in 0..min(K, |rel|)
    """
    ndcgs = []
    preds_k = predictions[:, :k]

    for user_preds, user_targets in zip(preds_k, targets):
        if len(user_targets) == 0:
            continue

        # Binary relevance vector for the top K predictions
        # 1 if item in targets, 0 otherwise
        relevance = np.isin(user_preds, user_targets).astype(int)

        # DCG
        discounts = 1 / np.log2(np.arange(len(relevance)) + 2)
        dcg = np.sum(relevance * discounts)

        # IDCG (best possible ordering: all relevant items first)
        n_relevant = len(user_targets)
        idcg_len = min(k, n_relevant)
        idcg_discounts = 1 / np.log2(np.arange(idcg_len) + 2)
        idcg = np.sum(idcg_discounts)

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def hit_rate_at_k(
    predictions: np.ndarray, targets: List[Union[List[int], np.ndarray]], k: int
) -> float:
    """
    Compute Mean Hit Rate @ K.
    1 if at least one relevant item is in the top K, 0 otherwise.
    """
    hit_rates = []
    preds_k = predictions[:, :k]

    for user_preds, user_targets in zip(preds_k, targets):
        if len(user_targets) == 0:
            continue

        # Check if any prediction is in targets
        is_hit = np.isin(user_preds, user_targets).any()
        hit_rates.append(1.0 if is_hit else 0.0)

    return float(np.mean(hit_rates)) if hit_rates else 0.0


def mrr_at_k(
    predictions: np.ndarray, targets: List[Union[List[int], np.ndarray]], k: int
) -> float:
    """
    Compute Mean Reciprocal Rank @ K.
    """
    mrrs = []
    preds_k = predictions[:, :k]

    for user_preds, user_targets in zip(preds_k, targets):
        if len(user_targets) == 0:
            continue

        # Find first relevant item
        hits = np.where(np.isin(user_preds, user_targets))[0]
        if len(hits) > 0:
            # 1-based rank of first hit
            first_hit_rank = hits[0] + 1
            mrrs.append(1.0 / first_hit_rank)
        else:
            mrrs.append(0.0)

    return float(np.mean(mrrs)) if mrrs else 0.0


def compute_batch_metrics(
    predictions: np.ndarray,
    targets: List[Union[List[int], np.ndarray]],
    k_list: List[int],
) -> dict:
    """
    Compute Recall, NDCG, MRR, and HitRate for multiple K values efficiently in a single pass.
    Returns dictionary with keys like "Recall@10", "NDCG@50" containing the SUM of metrics for the batch.
    """
    results = {
        f"{m}@{k}": 0.0 for k in k_list for m in ["HitRate", "Recall", "NDCG", "MRR"]
    }
    max_k = max(k_list)

    # Precompute log discounts for NDCG
    # discounts[i] = 1 / log2(i + 2)
    # i goes from 0 to max_k-1
    discounts = np.log2(np.arange(max_k) + 2)
    discounts = 1.0 / discounts

    for user_preds, user_targets in zip(predictions, targets):
        if len(user_targets) == 0:
            continue

        # Use set for O(1) lookups
        target_set = set(user_targets)
        n_relevant = len(target_set)

        # Boolean hit vector (using list comprehension is fast for small K)
        # We only need up to max_k predictions
        # 1.0 for hit, 0.0 for miss
        hits = [1.0 if item in target_set else 0.0 for item in user_preds[:max_k]]

        for k in k_list:
            hits_k = hits[:k]
            num_hits = sum(hits_k)

            # --- HitRate ---
            if num_hits > 0:
                results[f"HitRate@{k}"] += 1.0

            # --- Recall ---
            results[f"Recall@{k}"] += num_hits / n_relevant

            # --- MRR ---
            # Find first non-zero hit
            try:
                first_hit_idx = hits_k.index(1.0)
                results[f"MRR@{k}"] += 1.0 / (first_hit_idx + 1)
            except ValueError:
                pass  # No hit, MRR is 0

            # --- NDCG ---
            if num_hits > 0:
                # DCG: sum of discounts where hit is true
                # slice discounts to k
                dcg = sum(h * d for h, d in zip(hits_k, discounts[:k]))

                # IDCG: sum of first n_relevant discounts
                idcg_len = min(k, n_relevant)
                idcg = sum(discounts[:idcg_len])

                results[f"NDCG@{k}"] += dcg / idcg

    return results
