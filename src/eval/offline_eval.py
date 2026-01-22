import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import settings
from src.eval.metrics import recall_at_k, ndcg_at_k, mrr_at_k


def get_popularity_scores(train_df: pd.DataFrame) -> pd.Series:
    """Compute popularity counts from training data."""
    return train_df["item_idx"].value_counts()


def evaluate_baseline(model_name: str = "popularity", k_list=[10, 20, 50, 100]):
    print(f"Evaluating {model_name} baseline...")

    # 1. Load Data
    print("Loading splits...")
    train_df = pd.read_parquet(settings.processed_data_dir / "train.parquet")
    test_df = pd.read_parquet(settings.processed_data_dir / "test.parquet")

    # 2. Prepare Ground Truth (Test)
    # Group by user -> set of items
    print("Grouping test data by user...")
    test_user_groups = test_df.groupby("user_idx")["item_idx"].apply(list)
    test_users = test_user_groups.index.values
    test_targets = test_user_groups.values

    # 3. Train Baseline
    print("Training baseline...")
    if model_name == "popularity":
        # Global top items
        pop_counts = train_df["item_idx"].value_counts()
        # Sort items by count desc
        top_items = pop_counts.index.values  # These are item_idx

        max_k = max(k_list)
        global_top_k = top_items[:max_k]

        print(f"Generating predictions for {len(test_users)} users in test set...")

        # We can compute metrics in batches to save memory if N is huge (2M interactions in test).
        # Test users might be subset of total users.

        batch_size = 1000
        n_users = len(test_users)

        metrics_sum = {k: {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0} for k in k_list}

        for i in tqdm(range(0, n_users, batch_size)):
            batch_targets = test_targets[i : i + batch_size]
            current_batch_size = len(batch_targets)

            # Predict global top K for everyone in batch
            # Shape (batch, max_k)
            batch_preds = np.tile(global_top_k, (current_batch_size, 1))

            for k in k_list:
                metrics_sum[k]["recall"] += (
                    recall_at_k(batch_preds, batch_targets, k) * current_batch_size
                )
                metrics_sum[k]["ndcg"] += (
                    ndcg_at_k(batch_preds, batch_targets, k) * current_batch_size
                )
                metrics_sum[k]["mrr"] += (
                    mrr_at_k(batch_preds, batch_targets, k) * current_batch_size
                )

        # Aggregation
        results = {}
        for k in k_list:
            results[f"Recall@{k}"] = metrics_sum[k]["recall"] / n_users
            results[f"NDCG@{k}"] = metrics_sum[k]["ndcg"] / n_users
            results[f"MRR@{k}"] = metrics_sum[k]["mrr"] / n_users

        return results

    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


if __name__ == "__main__":
    results = evaluate_baseline("popularity")

    print("\n--- Offline Evaluation Results ---")
    df_results = pd.DataFrame([results])
    print(df_results.T)
    # create table with metrics as rows
