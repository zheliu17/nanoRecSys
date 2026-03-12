# Operations Guide

## Purpose

This document is a practical runbook for operating `nanoRecSys` locally.

It focuses on:

- bringing the stack up,
- regenerating artifacts,
- checking health and readiness,
- debugging common failure modes,
- understanding degraded modes,
- and interpreting serving behavior during load testing.

This is a local-development / single-node operations guide, not a cloud deployment manual.

---

## Services

The default deployment uses Docker Compose with three services:

- `api`
- `frontend`
- `redis`

### API

The API service:

- serves recommendation requests,
- reads model/index artifacts from `artifacts/`,
- reads supporting data from `data/`,
- connects to Redis for caching,
- exposes health/readiness/debug endpoints,
- and can run either in normal artifact-backed mode or stub mode.

### Frontend

The frontend is a Streamlit app that calls the API service through `API_URL=http://api:8000`.

### Redis

Redis is used as a serving-time cache.

If Redis is unavailable, the API can still serve requests, but cache hit rate will drop and end-to-end latency will increase.

---

## Required Local State

Before the serving stack can work correctly, the API container needs access to:

- generated artifacts in `./artifacts`
- required data in `./data`

These directories are mounted into the API container through Docker Compose.

If these directories are missing or incomplete, the API may:

- fail startup,
- report `not ready` via `/readyz`,
- or return fallback recommendations in specific edge cases.

---

## First-Time Setup

### 1. Install dependencies

```bash
pip install -e .[all]
```

### 2. Prepare data

```bash
make data
```

### 3. Generate artifacts

Fast path using pretrained retriever checkpoints:

```bash
python pipeline.py run --skip-retriever
```

Full path from scratch:

```bash
python pipeline.py run
```

### 4. Start the serving stack

```bash
make serve
```

### 5. Stop the serving stack

```bash
make stop
```

---

## Core Health and Debug Endpoints

The API exposes several operational endpoints.

## `/health` and `/healthz`

These are lightweight liveness checks.

Use them to confirm that:

- the FastAPI process is up,
- the container is reachable,
- and the app has started serving HTTP traffic.

Example:

```bash
curl http://localhost:8000/healthz
```

Expected response:

```json
{"status":"ok"}
```

## `/readyz`

This is the readiness check.

Use it to confirm that the service is actually ready to handle recommendation requests.

It reflects:

- startup success,
- artifact loading state,
- FAISS readiness,
- Redis status,
- backend selection,
- and service warnings/errors.

Example:

```bash
curl http://localhost:8000/readyz
```

Typical success response includes fields like:

- `ready`
- `stub_mode`
- `redis`
- `faiss`
- `user_tower_backend`
- `ranker_backend`
- `artifacts`
- `warnings`

If the service is not ready, `/readyz` returns HTTP `503`.

## `/debug/status`

This returns a more detailed internal service snapshot.

Use it for local debugging when:

- startup succeeded but behavior looks wrong,
- Redis seems degraded,
- artifact loading is uncertain,
- or you want to inspect in-memory counters/state.

Example:

```bash
curl http://localhost:8000/debug/status
```

This endpoint is mainly for development and debugging, not for public exposure.

## Send a recommendation request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "k": 10, "explain": false, "include_history": true}'
```

---

## Serving Modes

## Live mode

This is the normal mode.

In live mode, the service loads:

- maps,
- FAISS index,
- user tower,
- ranker,
- metadata,
- cached popularity tensors,
- and user history.

This mode requires valid artifacts and processed data.

## Stub mode

The API also supports a stub mode controlled by:

```bash
NANORECSYS_STUB=1
```

In stub mode:

- artifacts are not required,
- the service returns deterministic fallback-style recommendations,
- and the stack can be smoke-tested quickly without full artifact generation.

This is useful for:

- UI testing,
- API contract testing,
- container smoke tests,
- and demoing the service skeleton when model artifacts are unavailable.

---

## Environment Variables

The API supports several useful environment variables.

## Redis / cache settings

```bash
REDIS_HOST=redis
REDIS_PORT=6379
CACHE_NAMESPACE=v1
```

### `CACHE_NAMESPACE`

Used to isolate cache entries.

This is useful when:

- you change response shape,
- you change ranking logic,
- you switch artifacts,
- or you want to invalidate old cache keys without flushing Redis globally.

If recommendation behavior changes significantly, bump the namespace.

---

## Debug / observability settings

```bash
NANORECSYS_LOG_REQUESTS=1
NANORECSYS_INCLUDE_DEBUG_FIELDS=1
```

### `NANORECSYS_LOG_REQUESTS`

When enabled, the API logs request-level timing information such as:

- request id,
- cache hit/miss,
- embedding latency,
- retrieval latency,
- ranking latency,
- total internal inference latency.

Useful for:

- local profiling,
- debugging latency regressions,
- and validating cache behavior.

### `NANORECSYS_INCLUDE_DEBUG_FIELDS`

When enabled, API responses may include extra debug metadata such as:

- request id,
- cache source,
- mode,
- debug timing info.

Useful for:

- local inspection,
- frontend debugging,
- and verifying cache behavior in development.

For benchmark-style runs, you may choose to disable these, although on some machines the overhead is negligible.

---

## Day-2 Operations

## Rebuild the FAISS index

If embeddings or retriever checkpoints have changed, rebuild the index:

```bash
make build-index
```

If you want to regenerate embeddings and then rebuild everything downstream:

```bash
make post-train
```

---

## Re-train the ranker only

If the retriever artifacts are already available and you want to iterate only on ranking:

```bash
make mine-negatives
make train-ranker
```

This is useful when:

- retriever checkpoints are stable,
- candidate generation is unchanged,
- and only ranker logic or thresholds are being tuned.

---

## Re-run the full pipeline

```bash
make data
python pipeline.py run
```

Use this when:

- raw data was refreshed,
- preprocessing logic changed,
- retriever architecture changed,
- artifact compatibility is uncertain,
- or you want a completely fresh rebuild.

---

## Practical Notes

## Artifact compatibility

The serving stack assumes that:

- the retriever checkpoint,
- materialized embeddings,
- FAISS index,
- ranker artifacts,
- and ONNX exports

are mutually compatible.

If you change the retriever architecture, embedding dimension, or response assumptions, do not blindly reuse old downstream artifacts or cache keys.

## CPU-first serving

The default production-style path is CPU-oriented and based on ONNX export. The experimental LLM reranker is not part of this online operations flow.

## Local deployment model

This repo currently targets a local Docker Compose deployment. It is not yet a full production deployment framework with orchestration, autoscaling, remote observability, or multi-node rollout logic.

## Internal timing vs benchmark timing

Internal timing and load-test timing should not be expected to match exactly.

- Internal timing is useful for model/system optimization.
- Load-test timing is useful for end-user experience and throughput tuning.

Interpret them together, not competitively.

---

## Future operational improvements

Natural next steps for this repo include:

- artifact manifests with explicit version compatibility,
- `/metrics` or Prometheus-style counters,
- startup validation summaries in the frontend,
- structured JSON logging,
- request tracing across workers,
- cache hit-rate dashboards,
- and model/index rollback support.

These are intentionally outside the current lightweight scope.
