# System Benchmarks & Performance Analysis

This document details the performance characteristics of the **nanoRecSys** system, covering Approximate Nearest Neighbor (ANN) accuracy, index scalability, and end-to-end serving latency.

## 1. ANN Algorithm Ablation

We benchmarked different Approximate Nearest Neighbor (ANN) algorithms using `faiss`.

* **Dataset:** MovieLens-20M (27k items)
* **Dimension:** 256
* **Hardware:** Colab Standard CPU (<2 cores @ 2.2GHz>)

```bash
python -m nanoRecSys.indexing.benchmark_ivfpq
```

For a corpus of this size (~27k items), **Flat Index (Exact Search)** is the recommended default. The latency overhead of exact search vs. approximate search is negligible (<1ms difference) at this scale.

![ANN_Ablation_Results](../docs/images/ann_ablation_results.png)

## 2. Synthetic Scale-Up (3M Items)

To validate the system's readiness for production-scale data, we synthetically expanded the item corpus from 27k to 3 million items. The synthetic generation preserves the original embedding distribution (mean and variance) to ensure realistic retrieval behavior.

```bash
# 1. Expand items artificially
python -m nanoRecSys.indexing.synthetic_expand
# 2. Benchmark Indexing at scale
python -m nanoRecSys.indexing.benchmark_synthetic --target_count 3_000_000
```

The graph below compares `IVF-PQ`, `IVF-SQ` against `FlatIP` (Baseline, Recall=1.0):

![Synthetic_Expansion_Results](../docs/images/synthetic_expansion_results.png)

## 3. Production Serving Optimization

We performed end-to-end load testing to validate the system's throughput and latency (30% Cold Users / 70% Cache Hits). We compared a **Baseline (PyTorch)** implementation against an **Optimized (Quantized ONNX + AsyncIO)** implementation.

### 3.1 Optimization Results

We improved the serving stack by:

1. **Model Quantization/Export:** Migrating from PyTorch Eager mode to **ONNX Runtime (Quantized)**.
2. **Concurrency:** Moving from synchronous blocking calls to **AsyncIO** with thread-pool offloading for CPU-bound tasks.
3. **Resource Constraints:** The optimized test was run on restricted hardware (2 vCPUs) to mimic a realistic production container, whereas the baseline had access to the full host (16 Cores).

| Metric | Baseline (PyTorch) | Optimized (ONNX) | Improvement |
| --- | --- | --- | --- |
| **Resources** | 16 Cores (Laptop) | 2 vCPUs (Docker Limit) | **8x Less Compute** |
| **Throughput** | 24 RPS | ~144 RPS | **~6x Higher** |
| **Embedding Latency** | 75 ms | 23 ms | **3.2x Faster** |
| **Efficiency** | ~1.5 RPS / Core | ~72 RPS / Core | **~48x Efficiency Gain** |

### 3.2 Detailed Latency Breakdown (Optimized)

**Setup:**

```bash
python -m scripts.export_onnx
docker-compose up --build -d
# Load testing with 30% cold users, 70% cache hits
locust -f locustfile.py --host http://localhost:8000
# Offline metrics
python -m scripts.benchmarks.test_API_offline_metrics
```

Even under restricted hardware (2 vCPUs), the system maintains sub-60ms median latency. The heavy lifting (Transformer Inference) is handled efficiently by ONNX Runtime.

| Metric | Component | Median (ms) | P95 (ms) | Note |
| --- | --- | --- | --- | --- |
| **Total Request** | **POST /recommend** | **54** | **180** | **End-to-End Latency** |
| Internal | Embedding Gen (ONNX, Quantized) | 23 | 39 | Cold Path Only (Transformer) |
| Internal | Retrieval (FAISS) | 2 | 4 | Nearest Neighbor Search |
| Internal | Ranking (ONNX) | 2 | 4 | Ranker Inference |
| Internal | Total Compute (Cold Path) | 28 | 47 | |

> **Note on Tail Latency (P99):**
> While the execution time (Internal metrics) remains stable, the End-to-End P99 latency spikes to ~340ms. When the request rate (~144 RPS) saturates the workers, incoming requests queue up in the executor before processing begins. The delta between Total Latency and Total Compute (Cold Path) represents this **Queuing Time**.

![Load_Testing_Latency](../docs/images/load_testing_latency.png)
