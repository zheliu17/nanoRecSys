import numpy as np
import pytest
from src.eval.metrics import recall_at_k, ndcg_at_k, mrr_at_k


@pytest.fixture
def mock_predictions():
    # 3 users, top-5 predictions
    return np.array(
        [
            [10, 20, 30, 40, 50],  # User 0
            [11, 22, 33, 44, 55],  # User 1
            [1, 2, 3, 4, 5],  # User 2
        ]
    )


@pytest.fixture
def mock_targets():
    # Ground truth items for 3 users
    return [
        [10, 30, 99],  # User 0: 10 (rank 1), 30 (rank 3) -> 2 hits
        [99, 88],  # User 1: No hits
        [1, 5],  # User 2: 1 (rank 1), 5 (rank 5) -> 2 hits
    ]


def test_recall_at_k(mock_predictions, mock_targets):
    # k=3
    # User 0: preds=[10, 20, 30], targets=[10, 30, 99]. Hits: 10, 30. Recall: 2/3 = 0.666
    # User 1: preds=[11, 22, 33], targets=[99, 88]. Hits: 0. Recall: 0
    # User 2: preds=[1, 2, 3], targets=[1, 5]. Hits: 1. Recall: 1/2 = 0.5
    # Mean: (0.666 + 0 + 0.5) / 3 = 0.3888

    score = recall_at_k(mock_predictions, mock_targets, k=3)
    assert score == pytest.approx(0.3888, rel=1e-3)


def test_mrr_at_k(mock_predictions, mock_targets):
    # k=3
    # User 0: First hit is 10 at rank 1. Reciprocal rank: 1/1 = 1.0
    # User 1: No hits. RR: 0
    # User 2: First hit is 1 at rank 1. RR: 1/1 = 1.0
    # Mean: (1 + 0 + 1) / 3 = 0.666

    score = mrr_at_k(mock_predictions, mock_targets, k=3)
    assert score == pytest.approx(0.6666, rel=1e-3)


def test_ndcg_at_k_simple():
    # Manual simple case
    preds = np.array([[10, 20]])  # 1 user
    targets = [[10, 30]]  # 2 relevant

    # k=2
    # Preds: [10, 20]. Relevance: [1, 0] (since 10 is relevant, 20 is not)
    # DCG = 1/log2(1+1) + 0 = 1.0
    # IDCG: Best ordering for top-2 is [10, 30] (both relevant) -> Relevance [1, 1]
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
    # NDCG = 1.0 / 1.6309 = 0.613

    score = ndcg_at_k(preds, targets, k=2)
    assert score == pytest.approx(1.0 / 1.6309, rel=1e-3)


def test_metrics_empty_targets():
    preds = np.array([[1, 2, 3]])
    targets = [[]]  # Empty target

    assert recall_at_k(preds, targets, k=3) == 0.0
    assert ndcg_at_k(preds, targets, k=3) == 0.0
    assert mrr_at_k(preds, targets, k=3) == 0.0
