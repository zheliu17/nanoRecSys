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

import time
import numpy as np
import pandas as pd
import faiss
from nanoRecSys.config import settings
from nanoRecSys.eval.metrics import recall_at_k

# Constants for ablation
N_LISTS = [64, 128, 256, 512]
N_PROBES = [1, 4, 8, 16, 32]
MS = [8, 16]


def load_test_data():
    print("Loading test data...")
    try:
        test_df = pd.read_parquet(settings.processed_data_dir / "test.parquet")
        test_df = test_df[test_df["rating"] >= settings.retrieval_threshold]
        # Group by user to get ground truth
        ground_truth = test_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
        return ground_truth
    except Exception as e:
        print(f"Error loading test data: {e}")
        return {}


def load_user_embeddings():
    path = settings.artifacts_dir / "user_embeddings.npy"
    if path.exists():
        return np.load(path).astype("float32")
    return None


def load_item_embeddings():
    path = settings.artifacts_dir / "item_embeddings.npy"
    if path.exists():
        return np.load(path).astype("float32")
    return None


def run_ablation():
    ground_truth = load_test_data()
    user_embs = load_user_embeddings()
    item_embs = load_item_embeddings()

    if not ground_truth or user_embs is None or item_embs is None:
        print("Missing data for ablation. check artifacts and processed data.")
        return

    d = item_embs.shape[1]

    test_users = list(ground_truth.keys())
    # Filter test_users that have embeddings
    test_users = [u for u in test_users if u < len(user_embs)]

    if not test_users:
        print("No test users found within embedding range.")
        return

    results = []

    # 1. Baseline: Flat Index
    print("\n--- Running Baseline: Flat Index ---")
    index_flat = faiss.IndexFlatIP(d)
    index_flat.add(item_embs)  # type: ignore

    # We use a sample of users for benchmarking
    SAMPLE_SIZE = 1000
    if len(test_users) > SAMPLE_SIZE:
        query_users = np.random.choice(test_users, SAMPLE_SIZE, replace=False)
    else:
        query_users = test_users

    query_vectors = user_embs[query_users]

    k_retrieval = 100

    start_time = time.time()
    D, I = index_flat.search(query_vectors, k_retrieval)  # type: ignore  # noqa: E741
    elapsed = time.time() - start_time
    latency = (elapsed / len(query_users)) * 1000  # ms per query

    # Prepare targets for recall_at_k (once)
    targets = [ground_truth[u_idx] for u_idx in query_users]
    avg_recall = recall_at_k(I, targets, k_retrieval)
    print(f"Flat Index: Recall@100={avg_recall:.4f}, Latency={latency:.3f} ms")
    results.append(
        {
            "type": "Flat",
            "nlist": "-",
            "m": "-",
            "nprobe": "-",
            "recall@100": avg_recall,
            "latency_ms": latency,
        }
    )

    # 2. ANN Sweep
    print("\n--- Running ANN Sweep (IVF-PQ) ---")
    for nlist in N_LISTS:
        for m in MS:
            if d % m != 0:
                print(f"Skipping m={m} (d={d} not divisible)")
                continue

            print(f"Building Index: IVF{nlist}, PQ{m}")
            try:
                # Build Index
                index = faiss.index_factory(
                    d, f"IVF{nlist},PQ{m}", faiss.METRIC_INNER_PRODUCT
                )
                index.train(item_embs)
                index.add(item_embs)

                for nprobe in N_PROBES:
                    index.nprobe = nprobe

                    # Measure latency and metrics
                    start_time = time.time()
                    D, I = index.search(query_vectors, k_retrieval)  # noqa: E741
                    elapsed = time.time() - start_time
                    latency = (elapsed / len(query_users)) * 1000

                    avg_recall = recall_at_k(I, targets, k_retrieval)
                    print(
                        f"IVF{nlist}, PQ{m}, nprobe={nprobe}: Recall@100={avg_recall:.4f}, Latency={latency:.3f} ms"
                    )

                    results.append(
                        {
                            "type": "ANN",
                            "nlist": nlist,
                            "m": m,
                            "nprobe": nprobe,
                            "recall@100": avg_recall,
                            "latency_ms": latency,
                        }
                    )
            except Exception as e:
                print(f"Failed configuration IVF{nlist}, PQ{m}: {e}")

    # Save results
    df = pd.DataFrame(results)
    output_csv = settings.artifacts_dir / "ann_ablation_results.csv"
    print("\nSummary Results:")
    print(df)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    run_ablation()
