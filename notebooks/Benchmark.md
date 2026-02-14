# System Benchmarks & Performance Analysis

This document details the performance characteristics of the **nanoRecSys** system, covering Approximate Nearest Neighbor (ANN) accuracy, index scalability, and end-to-end serving latency.

## 1. ANN Algorithm Ablation

We benchmarked different Approximate Nearest Neighbor (ANN) algorithms using `faiss`.

* **Dataset:** MovieLens-20M (27k items)
* **Dimension:** 256
* **Hardware:** Colab Standard CPU (<2 cores @ 2.2GHz>)

```bash
python -m src.nanoRecSys.indexing.benchmark_ivfpq
```

For a corpus of this size (~27k items), **Flat Index (Exact Search)** is the recommended default. The latency overhead of exact search vs. approximate search is negligible (<1ms difference) at this scale.

![ANN_Ablation_Results](../docs/images/ann_ablation_results.png)

## 2. Synthetic Scale-Up (3M Items)

To validate the system's readiness for production-scale data, we synthetically expanded the item corpus from 27k to 3 million items. The synthetic generation preserves the original embedding distribution (mean and variance) to ensure realistic retrieval behavior.

```bash
# 1. Expand items artificially
python -m src.nanoRecSys.indexing.synthetic_expand
# 2. Benchmark Indexing at scale
python -m src.nanoRecSys.indexing.benchmark_synthetic --target_count 3_000_000
```

The graph below compares `IVF-PQ`, `IVF-SQ` against `FlatIP` (Baseline, Recall=1.0):

![Synthetic_Expansion_Results](../docs/images/synthetic_expansion_results.png)

## 3. Production Latency (Load Testing)

We performed end-to-end load testing on the full serving stack, mimicking a real-world production environment.

**Setup:**

* **Index:** FAISS Flat Index (27k items)
* **Workload:** Hybrid traffic with a mix of cache hits and misses.
  * The default `locustfile.py` configuration simulates **~30% cold users** (cache-miss path). Adjusting the hot/cold ratio changes the end-to-end latency distribution.

**Command:**

```bash
# 1. Build Index
python -m src.nanoRecSys.indexing.build_faiss_flat

# 2. Start Stack
docker-compose up --build -d

# 3. Run Load Test
locust -f locustfile.py
```

**Results:** (CPU-only laptop, example run)

* **Throughput:** 24 RPS sustained.
* **Latency:** 53ms Median / 180ms P95.

Even on CPU-only hardware, the system delivers sub-200ms tail latency. The Cold-Path (Transformer Inference) is the dominant cost (~90ms).

**Detailed Breakdown:**

| Metric | Component | Median (ms) | P95 (ms) | P99 (ms) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Request** | **POST /recommend** | **53** | **180** | **360** | **End-to-End Latency** |
| Internal | Embedding Generation | 75 | 170 | 380 | Transformer Inference (Cold Path) |
| Internal | Ranking | 6 | 18 | 42 | Ranker Inference |
| Internal | Retrieval (FAISS) | 2 | 4 | 7 | Nearest Neighbor Search |
| Internal | Server Processing | 1 | 140 | 320 | Total Python Time |

![Load_Testing_Latency](../docs/images/load_testing_latency.png)
