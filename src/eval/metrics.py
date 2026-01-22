import numpy as np
from typing import List, Union


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

    return np.mean(recalls) if recalls else 0.0


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

    return np.mean(ndcgs) if ndcgs else 0.0


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

    return np.mean(mrrs) if mrrs else 0.0
