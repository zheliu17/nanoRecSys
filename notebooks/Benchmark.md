# Benchmarking Results

This document contains ablation studies and latency benchmarks for the system.

## 1. ANN Algorithm Ablation

We benchmarked different Approximate Nearest Neighbor (ANN) algorithms using `faiss`.

* **Dataset:** MovieLens-20M (27k items)
* **Dimension:** 256
* **Hardware:** Colab Standard CPU (<2 cores @ 2.2GHz>)

**Command:**

```bash
python -m src.nanoRecSys.indexing.benchmark_ivfpq
```

**Conclusion (for ~27k items):** **Flat** (exact search) is a reasonable default because the observed latency overhead was small in this setting.

![ANN_Ablation_Results](../docs/images/ann_ablation_results.png)

## 2. Synthetic Scale-Up (3M Items)

To probe scalability, we synthetically expanded the item corpus from 27k to 3 million items while matching the original embedding distribution (mean/variance).

**Command:**

```bash
# 1. Expand items
python -m src.nanoRecSys.indexing.synthetic_expand
# 2. Benchmark Indexing
python -m src.nanoRecSys.indexing.benchmark_synthetic --target_count 3_000_000
```

**Results:**
Comparing `IVF-PQ` against `FlatIP` (Baseline, Recall=1.0):

![Synthetic_Expansion_Results](../docs/images/synthetic_expansion_results.png)

(Note: Graphs indicate trade-offs between `nlist` / `m` parameters and outcome metrics).

## 3. Production Latency (Load Testing)

We load tested the full serving stack (FastAPI + Redis + FAISS + transformer-based user embedding + MLP ranker).

**Setup:**

* **Index:** FAISS Flat (27k items)
* **Traffic:** Generated using `locust`.
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

**Results (CPU-only laptop, example run):**

* **Throughput:** 24 RPS (Requests Per Second) sustained.
* **Latency:** 53ms Median / 180ms P95.

**Detailed Breakdown:**

| Type | Name | Median (ms) | P95 (ms) | P99 (ms) | Average (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **POST** | **/recommend** | **53** | **180** | **360** | **77.58** |
| DB | Embedding Lookup | 75 | 170 | 380 | 89.28 |
| DB | Ranking | 6 | 18 | 42 | 7.96 |
| DB | Retrieval | 2 | 4 | 7 | 2.41 |
| DB | Server Processing* | 1 | 140 | 320 | 46.12 |

*> `Server_Processing` denotes Python-level processing overhead excluding network transfer.*

![Load_Testing_Latency](../docs/images/load_testing_latency.png)

**Note:** During cache misses, the system performs real-time Sequential Transformer inference (generating user embedding from history) and MLP re-ranking in **<90ms**.
