import time
import numpy as np
import faiss
import csv
import json
import argparse
import os
from nanoRecSys.config import settings


def benchmark_synthetic(
    config_path="benchmark_config.json",
    custom_configs=None,
    embeddings_filename=None,
    target_count=None,
):
    """
    Benchmarks multiple FAISS index configurations on a synthetic dataset.

    Args:
        config_path (str): Path to a JSON file containing configurations.
        custom_configs (list or dict, optional): Direct list of dictionaries defining configs.
                                                 If provided, ignores config_path.
        embeddings_filename (str, optional): Explicit embeddings file name to use. If provided,
                                             this will be used directly (relative to artifacts dir).
        target_count (int, optional): If provided, constructs a filename of the form
                                      'synthetic_{N}M_embeddings.npy' where N = target_count//1_000_000.
    """
    # Allow selecting desired embeddings by filename or by target_count
    if target_count is not None:
        embeddings_filename = f"synthetic_{target_count // 1000000}M_embeddings.npy"

    if embeddings_filename is None:
        embeddings_filename = "synthetic_1M_embeddings.npy"

    embeddings_path = settings.artifacts_dir / embeddings_filename
    user_emb_path = settings.artifacts_dir / "user_embeddings.npy"
    results_csv = settings.artifacts_dir / "benchmark_results.csv"

    if not embeddings_path.exists():
        print(f"Error: {embeddings_filename} not found at {embeddings_path}.")
        return

    # --- 1. Load Data ---
    print(f"Loading {embeddings_filename} with memory mapping...")
    xb = np.load(embeddings_path, mmap_mode="r")
    n_total, d = xb.shape

    print("Normalizing database vectors...")
    try:
        # Try normalizing in-place if array is small enough
        if xb.flags["WRITEABLE"]:
            faiss.normalize_L2(xb)
        else:
            # For mmap, normalization must be done on a RAM copy
            xb = np.array(xb, dtype="float32", copy=True)
            faiss.normalize_L2(xb)
    except Exception as e:
        print(
            f"Warning: normalization failed or was skipped due to memory constraints: {e}"
        )
        return

    # Queries
    n_queries = 1000
    if user_emb_path.exists():
        all_users = np.load(user_emb_path).astype("float32")
        indices = np.random.choice(len(all_users), n_queries, replace=False)
        xq = all_users[indices]
    else:
        xq = np.random.randn(n_queries, d).astype("float32")
        xq = xq / np.linalg.norm(xq, axis=1, keepdims=True)

    k = 10
    results = []  # List to store dicts for CSV

    # --- 2. Ground Truth (Flat Index) ---
    print("\n--- 1. Baseline: Flat Index (Calculate Ground Truth) ---")
    index_flat = faiss.IndexFlatIP(d)
    start_time = time.time()
    index_flat.add(xb)  # type: ignore
    print(f"Flat Build Time: {time.time() - start_time:.4f}s")

    start_time = time.time()
    D_gt, I_gt = index_flat.search(xq, k)  # type: ignore
    search_time = time.time() - start_time
    flat_latency = (search_time * 1000) / n_queries

    print(f"Flat Latency: {flat_latency:.3f} ms/query")

    results.append(
        {
            "config_name": "Flat (Exact)",
            "nlist": 0,
            "m": 0,
            "nprobe": 0,
            "latency_ms": flat_latency,
            "recall": 1.0,
        }
    )

    # --- 3. Test Configurations ---
    configs = []

    # Priority: 1. Direct Argument -> 2. JSON File -> 3. Defaults
    if custom_configs is not None:
        if isinstance(custom_configs, dict):
            configs = [custom_configs]
        elif isinstance(custom_configs, list):
            configs = custom_configs
        print(f"\nUsing {len(configs)} configuration(s) provided directly.")

    elif os.path.exists(config_path):
        print(f"\nLoading configurations from {config_path}...")
        try:
            with open(config_path, "r") as f:
                configs = json.load(f)
        except Exception as e:
            print(f"Error reading config file: {e}")
            return
    else:
        print(f"\nConfig file '{config_path}' not found. Using defaults.")
        configs = [
            {"nlist": 4096, "m": 32, "name": "IVF4096_PQ32"},
            {"nlist": 2048, "quantizer_type": "SQ8", "name": "IVF2048_SQ8 (Scalar)"},
            # {"nlist": 2048, "m": d, "name": "IVF2048_Flat"},
        ]

    for cfg in configs:
        nlist = cfg["nlist"]
        m = cfg.get("m", 0)
        name = cfg["name"]
        nbits = 8

        print(f"\n--- Testing Config: {name} ---")

        quantizer = faiss.IndexFlatL2(d)

        if "quantizer_type" in cfg:
            qt_name = cfg["quantizer_type"]
            print(f"  Note: Using IndexIVFScalarQuantizer ({qt_name})")

            if qt_name == "SQ8":
                qtype = faiss.ScalarQuantizer.QT_8bit
            elif qt_name == "SQ4":
                qtype = faiss.ScalarQuantizer.QT_4bit
            elif qt_name == "SQfp16":
                qtype = faiss.ScalarQuantizer.QT_fp16
            else:
                print(f"Unknown quantizer type {qt_name}, using SQ8")
                qtype = faiss.ScalarQuantizer.QT_8bit

            ivf_index = faiss.IndexIVFScalarQuantizer(
                quantizer, d, nlist, qtype, faiss.METRIC_L2
            )

        elif m == d:
            print(f"  Note: m={m} (matches dim), using IndexIVFFlat (No Compression)")
            ivf_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        else:
            ivf_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_L2)

        index = ivf_index

        # min_train_points = 40 * nlist
        # train_size = min(n_total, max(50000, min_train_points))
        # print(f"Training on {train_size} items (required ~{min_train_points})...")

        # train_indices = np.random.choice(n_total, train_size, replace=False)
        # index.train(xb[train_indices])  # type: ignore

        # Train on full dataset for better clustering
        index.train(xb)  # type: ignore
        index.add(xb)  # type: ignore

        # Sweep nprobe from 1 (fastest) to 100 (most accurate)
        nprobes_to_test = [1, 5, 10, 20, 50, 100]

        for nprobe in nprobes_to_test:
            ivf_index.nprobe = nprobe

            start_time = time.time()
            D_ann, I_ann = index.search(xq, k)  # type: ignore
            duration = time.time() - start_time
            latency_ms = (duration * 1000) / n_queries

            # Calculate Recall@K
            matches = 0
            for i in range(n_queries):
                gt_set = set(I_gt[i].tolist())
                ann_set = set(I_ann[i].tolist())
                matches += len(gt_set.intersection(ann_set))

            recall = matches / (n_queries * k)

            print(
                f"  nprobe={nprobe:<3} | Latency={latency_ms:.3f}ms | Recall={recall:.4f}"
            )

            results.append(
                {
                    "config_name": name,
                    "nlist": nlist,
                    "m": m,
                    "nprobe": nprobe,
                    "latency_ms": latency_ms,
                    "recall": recall,
                }
            )

    # --- 4. Save CSV ---
    if not results:
        print("No results generated.")
        return

    keys = list(results[0].keys())
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FAISS ANN performance")
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Explicit embeddings filename to use (relative to artifacts dir)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target total number of items (e.g., 1000000)",
    )
    args = parser.parse_args()

    benchmark_synthetic(
        config_path=args.config,
        embeddings_filename=args.embeddings,
        target_count=args.target,
    )
