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

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from nanoRecSys.config import settings
from nanoRecSys.models.ranker import MLPRanker
from nanoRecSys.models.towers import ItemTower, TransformerUserTower, UserTower
from nanoRecSys.utils.utils import compute_item_probabilities

from .exceptions import (
    ArtifactMissingError,
    InvalidRecommendationRequestError,
    ServiceNotReadyError,
)
from .faiss_store import FaissStore
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

RETRIEVAL_K = 100


class ONNXRanker:
    def __init__(self, path):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")
        sess_options = ort.SessionOptions()  # type: ignore[attr-defined]
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(  # type: ignore[attr-defined]
            str(path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

    def predict(self, user_emb, item_emb, genre_multihot, year_idx, popularity):
        def to_numpy(x):
            return x.cpu().detach().numpy() if hasattr(x, "cpu") else x

        if hasattr(popularity, "dim") and popularity.dim() == 1:
            popularity = popularity.unsqueeze(1)
        elif isinstance(popularity, np.ndarray) and popularity.ndim == 1:
            popularity = popularity[:, np.newaxis]

        inputs = {
            "user_emb": to_numpy(user_emb),
            "item_emb": to_numpy(item_emb),
            "genre_multihot": to_numpy(genre_multihot),
            "year_idx": to_numpy(year_idx),
            "popularity": to_numpy(popularity),
        }
        out = self.session.run(["score"], inputs)
        return torch.from_numpy(out[0]).squeeze()

    def eval(self):
        pass

    def to(self, device):
        pass


class ONNXUserTower:
    def __init__(self, path):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")
        sess_options = ort.SessionOptions()  # type: ignore[attr-defined]
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(  # type: ignore[attr-defined]
            str(path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

    def encode(self, item_seq):
        def to_numpy(x):
            return x.cpu().detach().numpy() if hasattr(x, "cpu") else x

        inputs = {"item_seq": to_numpy(item_seq)}
        out = self.session.run(["user_embedding"], inputs)
        return torch.from_numpy(out[0])

    def eval(self):
        pass

    def to(self, device):
        pass


def _is_truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


class RecommendationService:
    def __init__(self):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.device = "cpu"
        self.stub_mode = _is_truthy_env("NANORECSYS_STUB")
        self.log_requests = _is_truthy_env("NANORECSYS_LOG_REQUESTS")
        self.include_debug_fields = _is_truthy_env("NANORECSYS_INCLUDE_DEBUG_FIELDS")
        self.cache = RedisCache()

        self.ready = False
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_errors": 0,
            "errors": 0,
        }
        self.user_tower_backend = "unknown"
        self.ranker_backend = "unknown"
        self.index_type = os.getenv("FAISS_INDEX_TYPE", "flat")
        self.artifacts: dict[str, bool] = {}

        if self.stub_mode:
            self.user_id_to_idx = {1: 0}
            self.fallback_movie_ids = list(range(1, 1001))
            self.fallback_scores = [
                1.0 / (i + 1) for i in range(len(self.fallback_movie_ids))
            ]
            self.user_history = {0: [10, 20, 30]}
            self.ready = True
            logger.warning(
                "NANORECSYS_STUB=1 enabled: serving stub recommendations (no artifacts required)."
            )
            return

        self._load_live_service()

    async def post_init(self):
        redis_ok = await self.cache.ping()
        if not redis_ok:
            self.warnings.append("Redis unavailable; continuing without cache.")
            logger.warning("Redis unavailable at startup; continuing without cache.")

    def _mark_artifact(self, name: str, path: Path) -> bool:
        exists = path.exists()
        self.artifacts[name] = exists
        return exists

    def _require_file(self, name: str, path: Path):
        if not self._mark_artifact(name, path):
            raise ArtifactMissingError(f"Missing required artifact: {path}")

    def _load_live_service(self):
        self.faiss_store = FaissStore(
            settings.artifacts_dir, index_type=self.index_type
        )
        if not self.faiss_store.is_ready():
            raise ArtifactMissingError(
                f"No FAISS index found in {settings.artifacts_dir}."
            )

        logger.info("Loading maps...")
        user_map_path = settings.processed_data_dir / "user_map.npy"
        item_map_path = settings.processed_data_dir / "item_map.npy"
        self._require_file("user_map", user_map_path)
        self._require_file("item_map", item_map_path)

        self.user_map = np.load(user_map_path)
        self.user_id_to_idx = {uid: i for i, uid in enumerate(self.user_map)}
        self.item_map = np.load(item_map_path)

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        logger.info("Loading User Tower (%s)...", settings.user_tower_type)
        onnx_tower_path = settings.artifacts_dir / "user_tower.quant.onnx"
        user_tower_ckpt_path = settings.artifacts_dir / "user_tower.pth"
        self.artifacts["user_tower.quant.onnx"] = onnx_tower_path.exists()
        self.artifacts["user_tower.pth"] = user_tower_ckpt_path.exists()

        if ONNX_AVAILABLE and onnx_tower_path.exists():
            logger.info("Loading User Tower from ONNX: %s", onnx_tower_path)
            self.user_tower = ONNXUserTower(onnx_tower_path)
            self.user_tower_backend = "onnx"
        elif settings.user_tower_type == "transformer":
            dummy_item_tower = ItemTower(
                vocab_size=n_items,
                embed_dim=settings.embed_dim,
                output_dim=settings.tower_out_dim,
                hidden_dims=settings.towers_hidden_dims,
                use_projection=settings.use_projection,
            )

            self.user_tower = TransformerUserTower(
                vocab_size=n_items,
                embed_dim=settings.tower_out_dim,
                output_dim=settings.tower_out_dim,
                max_seq_len=settings.max_seq_len,
                n_heads=settings.transformer_heads,
                n_layers=settings.transformer_layers,
                dropout=settings.transformer_dropout,
                swiglu_hidden_dim=settings.swiglu_hidden_dim,
                shared_embedding=dummy_item_tower,
            )
            self.user_tower_backend = "pytorch"
        else:
            self.user_tower = UserTower(
                vocab_size=n_users,
                embed_dim=settings.embed_dim,
                output_dim=settings.tower_out_dim,
            )
            self.user_tower_backend = "pytorch"

        if not isinstance(self.user_tower, ONNXUserTower):
            if user_tower_ckpt_path.exists():
                self.user_tower.load_state_dict(
                    torch.load(
                        user_tower_ckpt_path,
                        map_location=self.device,
                        weights_only=False,
                    )
                )
            else:
                raise ArtifactMissingError(
                    f"User Tower checkpoint not found: {user_tower_ckpt_path}"
                )
            self.user_tower.to(self.device)
            self.user_tower.eval()

        logger.info("Loading metadata...")
        genre_path = settings.processed_data_dir / "genre_matrix_binned.npy"
        year_path = settings.processed_data_dir / "year_indices_binned.npy"
        self._require_file("genre_matrix_binned", genre_path)
        self._require_file("year_indices_binned", year_path)

        self.genre_matrix = (
            torch.from_numpy(np.load(genre_path)).float().to(self.device)
        )
        self.year_indices = torch.from_numpy(np.load(year_path)).long().to(self.device)

        num_genres = self.genre_matrix.shape[1]
        num_years = 6

        logger.info("Loading Ranker...")
        onnx_ranker_path = settings.artifacts_dir / "ranker_model.onnx"
        ranker_ckpt_path = settings.artifacts_dir / "ranker_model.pth"
        self.artifacts["ranker_model.onnx"] = onnx_ranker_path.exists()
        self.artifacts["ranker_model.pth"] = ranker_ckpt_path.exists()

        if ONNX_AVAILABLE and onnx_ranker_path.exists():
            logger.info("Loading Ranker from ONNX: %s", onnx_ranker_path)
            self.ranker_model = ONNXRanker(onnx_ranker_path)
            self.ranker_backend = "onnx"
        else:
            self.ranker_model = MLPRanker(
                input_dim=settings.tower_out_dim,
                hidden_dims=settings.ranker_hidden_dims,
                num_genres=num_genres,
                num_years=num_years,
                genre_dim=16,
                year_dim=8,
            )
            if ranker_ckpt_path.exists():
                self.ranker_model.load_state_dict(
                    torch.load(
                        ranker_ckpt_path,
                        map_location=self.device,
                        weights_only=False,
                    )
                )
            else:
                self.warnings.append(
                    "Ranker checkpoint missing; service may not function correctly."
                )
                logger.warning("Ranker checkpoint not found: %s", ranker_ckpt_path)
            self.ranker_model.to(self.device)
            self.ranker_model.eval()
            self.ranker_backend = "pytorch"

        logger.info("Loading Ranker Assets...")
        item_emb_path = settings.artifacts_dir / "ranker_item_embeddings.pt"
        pop_stats_path = settings.artifacts_dir / "ranker_pop_stats.pt"
        popularity_path = settings.artifacts_dir / "popularity_cache.pt"

        self._require_file("ranker_item_embeddings", item_emb_path)
        self.ranker_item_embeddings = torch.load(
            item_emb_path, map_location=self.device, weights_only=False
        )

        self.artifacts["ranker_pop_stats.pt"] = pop_stats_path.exists()
        if pop_stats_path.exists():
            pop_stats = torch.load(
                pop_stats_path, map_location=self.device, weights_only=False
            )
            self.pop_mean = pop_stats["mean"]
            self.pop_std = pop_stats["std"]
        else:
            self.warnings.append("Ranker popularity stats missing; using defaults.")
            logger.warning("Pop stats not found. Using defaults.")
            self.pop_mean = 0.0
            self.pop_std = 1.0

        logger.info("Loading Popularity Tensor...")
        self.artifacts["popularity_cache.pt"] = popularity_path.exists()
        if popularity_path.exists():
            self.popularity = torch.load(
                popularity_path, map_location=self.device, weights_only=False
            )
        else:
            logger.info("Computing Popularity Tensor from interactions...")
            train_path = settings.processed_data_dir / "train.parquet"
            self.artifacts["train.parquet"] = train_path.exists()
            if train_path.exists():
                popularity = compute_item_probabilities(n_items, return_log_probs=True)
                popularity = (popularity - self.pop_mean) / (self.pop_std + 1e-6)
                self.popularity = popularity.to(self.device)
                torch.save(self.popularity, popularity_path)
                logger.info("Saved popularity tensor to %s", popularity_path)
            else:
                self.warnings.append(
                    "train.parquet missing; using zero popularity fallback."
                )
                logger.warning("train.parquet not found. Using zeros for popularity.")
                self.popularity = torch.zeros(
                    n_items, dtype=torch.float32, device=self.device
                )

        pop_vals, pop_indices = torch.topk(self.popularity, k=min(n_items, RETRIEVAL_K))
        self.fallback_indices = pop_indices.cpu().numpy()
        self.fallback_scores = torch.sigmoid(pop_vals).cpu().numpy().tolist()
        self.fallback_movie_ids = [
            int(self.item_map[idx]) for idx in self.fallback_indices
        ]

        logger.info("Loading User History...")
        self.user_history = {}
        history_cache_path = settings.artifacts_dir / "user_history_cache.pkl"
        self.artifacts["user_history_cache.pkl"] = history_cache_path.exists()

        if history_cache_path.exists():
            import pickle

            try:
                with open(history_cache_path, "rb") as f:
                    self.user_history = pickle.load(f)
                logger.info(
                    "Loaded cached history for %s users.", len(self.user_history)
                )
            except Exception as e:
                logger.error("Error loading cached history: %s. Recomputing.", e)
                self.user_history = {}

        if not self.user_history:
            history_path = settings.processed_data_dir / "train.parquet"
            val_path = settings.processed_data_dir / "val.parquet"
            self.artifacts["val.parquet"] = val_path.exists()

            if history_path.exists():
                try:
                    import pickle

                    dfs = []
                    columns = ["user_idx", "item_idx", "timestamp"]
                    df_train = pd.read_parquet(history_path, columns=columns)
                    dfs.append(df_train)

                    if val_path.exists():
                        df_val = pd.read_parquet(val_path, columns=columns)
                        dfs.append(df_val)

                    full_df = pd.concat(dfs, ignore_index=True)
                    full_df = full_df.sort_values(by=["user_idx", "timestamp"])
                    self.user_history = (
                        full_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
                    )

                    with open(history_cache_path, "wb") as f:
                        pickle.dump(self.user_history, f)

                    logger.info(
                        "Loaded and cached history for %s users (train+val).",
                        len(self.user_history),
                    )
                except Exception as e:
                    self.warnings.append(f"Failed to compute user history cache: {e}")
                    logger.error("Error loading history: %s", e)
            else:
                self.warnings.append("train.parquet missing; user history is empty.")
                logger.warning("train.parquet not found. History will be empty.")

        self.ready = True
        logger.info(
            "Recommendation service ready. user_tower_backend=%s ranker_backend=%s index_type=%s",
            self.user_tower_backend,
            self.ranker_backend,
            self.faiss_store.loaded_index_type,
        )

    def get_status(self) -> dict:
        return {
            "ready": self.ready,
            "stub_mode": self.stub_mode,
            "warnings": self.warnings,
            "errors": self.errors,
            "redis": self.cache.status(),
            "faiss": self.faiss_store.status() if hasattr(self, "faiss_store") else {},
            "user_tower_backend": self.user_tower_backend,
            "ranker_backend": self.ranker_backend,
            "artifacts": self.artifacts,
            "stats": self.stats,
            "mode": "stub" if self.stub_mode else "live",
        }

    def _ensure_ready(self):
        if not self.ready:
            raise ServiceNotReadyError(
                "Recommendation service is not ready. Check /readyz for details."
            )

    def _fallback_response(
        self,
        k: int,
        explain: bool,
        reason: str,
        include_history: bool = False,
        request_id: str | None = None,
    ):
        limit = min(k, len(self.fallback_movie_ids))
        result = {
            "movie_ids": self.fallback_movie_ids[:limit],
            "scores": self.fallback_scores[:limit],
            "explanations": [reason] * limit if explain else [],
            "history": [] if include_history else None,
        }
        if self.include_debug_fields:
            result["debug_timing"] = {"fallback_reason": reason}
            result["request_id"] = request_id
            result["served_from_cache"] = False
            result["mode"] = "stub" if self.stub_mode else "live"
        return result

    def _compute_inference(
        self,
        user_id: int,
        k: int,
        explain: bool,
        include_history: bool,
        request_id: str,
    ):
        if getattr(self, "stub_mode", False):
            limit = max(0, min(int(k), len(self.fallback_movie_ids)))
            return {
                "movie_ids": self.fallback_movie_ids[:limit],
                "scores": self.fallback_scores[:limit],
                "explanations": ["Stub recommendation"] * limit if explain else [],
                "debug_timing": {"source": "stub"},
                "history": [10, 20, 30] if include_history else None,
                "request_id": request_id,
                "served_from_cache": False,
                "mode": "stub",
            }

        t0 = time.perf_counter()
        timings = {}

        if user_id not in self.user_id_to_idx:
            return self._fallback_response(
                k=k,
                explain=explain,
                reason="Popularity fallback (User Unknown)",
                include_history=include_history,
                request_id=request_id,
            )

        u_idx = self.user_id_to_idx[user_id]

        t_emb_start = time.perf_counter()
        with torch.inference_mode():
            if settings.user_tower_type == "transformer":
                history = self.user_history.get(u_idx, [])
                if not history:
                    return self._fallback_response(
                        k=k,
                        explain=explain,
                        reason="Popularity fallback (No History)",
                        include_history=include_history,
                        request_id=request_id,
                    )

                seq = history[-settings.max_seq_len :]
                seq = [item + 1 for item in seq]
                pad_len = settings.max_seq_len - len(seq)
                if pad_len > 0:
                    seq = [0] * pad_len + seq
                seq_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
                user_emb = self.user_tower.encode(seq_tensor)
            else:
                u_tensor = torch.tensor([u_idx], device=self.device)
                user_emb = self.user_tower(u_tensor)  # type: ignore[operator]

        timings["embedding_ms"] = (time.perf_counter() - t_emb_start) * 1000.0

        t_ret_start = time.perf_counter()
        candidate_indices, _ = self.faiss_store.search(user_emb, k=RETRIEVAL_K)
        timings["retrieval_ms"] = (time.perf_counter() - t_ret_start) * 1000.0

        history_items = self.user_history.get(u_idx, [])
        if history_items:
            watched = set(history_items)
            filtered = [int(ci) for ci in candidate_indices if int(ci) not in watched]
            candidate_indices = filtered

        if len(candidate_indices) == 0:
            return self._fallback_response(
                k=k,
                explain=explain,
                reason="Popularity fallback (No Candidates)",
                include_history=include_history,
                request_id=request_id,
            )

        candidate_indices = torch.tensor(
            candidate_indices, dtype=torch.long, device=self.device
        )

        t_rank_start = time.perf_counter()
        with torch.inference_mode():
            user_emb_expanded = user_emb.repeat(len(candidate_indices), 1)
            item_embs = self.ranker_item_embeddings[candidate_indices]
            genres = self.genre_matrix[candidate_indices]
            years = self.year_indices[candidate_indices]
            pops = self.popularity[candidate_indices]

            scores = self.ranker_model.predict(
                user_emb=user_emb_expanded,
                item_emb=item_embs,
                genre_multihot=genres,
                year_idx=years,
                popularity=pops,
            )

        topk_scores, topk_indices = torch.topk(scores, k=min(k, len(scores)))

        final_indices = candidate_indices[topk_indices].cpu().numpy()
        final_scores = topk_scores.cpu().numpy().tolist()
        final_movie_ids = [int(self.item_map[idx]) for idx in final_indices]

        timings["ranking_ms"] = (time.perf_counter() - t_rank_start) * 1000.0
        timings["total_ms"] = (time.perf_counter() - t0) * 1000.0

        result = {
            "movie_ids": final_movie_ids,
            "scores": final_scores,
            "explanations": [],
        }

        if self.include_debug_fields:
            result["debug_timing"] = timings
            result["request_id"] = request_id
            result["served_from_cache"] = False
            result["mode"] = "live"

        if explain:
            result["explanations"] = ["Explanation not implemented"] * len(
                final_movie_ids
            )

        if include_history:
            if u_idx in self.user_history:
                hist_indices = self.user_history[u_idx]
                if len(hist_indices) > 20:
                    hist_indices = hist_indices[-20:]
                result["history"] = [int(self.item_map[idx]) for idx in hist_indices]
            else:
                result["history"] = []

        if self.log_requests:
            logger.info(
                "recommend request_id=%s user_id=%s k=%s cache_hit=false embedding_ms=%.2f retrieval_ms=%.2f ranking_ms=%.2f total_ms=%.2f",
                request_id,
                user_id,
                k,
                timings.get("embedding_ms", 0.0),
                timings.get("retrieval_ms", 0.0),
                timings.get("ranking_ms", 0.0),
                timings.get("total_ms", 0.0),
            )

        return result

    async def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        explain: bool = False,
        include_history: bool = False,
    ):
        self._ensure_ready()
        self.stats["requests"] += 1

        if k < 1 or k > 100:
            self.stats["errors"] += 1
            raise InvalidRecommendationRequestError("k must be between 1 and 100")

        request_id = uuid.uuid4().hex[:12]
        cache_namespace = os.getenv("CACHE_NAMESPACE", "v1")
        cache_key = (
            f"{cache_namespace}:user:{user_id}:k:{k}:"
            f"h:{int(include_history)}:e:{int(explain)}"
        )

        cached = await self.cache.get(cache_key)
        if cached:
            self.stats["cache_hits"] += 1
            if self.include_debug_fields:
                cached["debug_timing"] = {"source": "cache"}
                cached["served_from_cache"] = True
                cached["request_id"] = request_id
                cached["mode"] = "stub" if self.stub_mode else "live"
            if self.log_requests:
                logger.info(
                    "recommend request_id=%s user_id=%s k=%s cache_hit=true",
                    request_id,
                    user_id,
                    k,
                )
            return cached

        self.stats["cache_misses"] += 1

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                self._compute_inference,
                user_id,
                k,
                explain,
                include_history,
                request_id,
            )
        except Exception:
            self.stats["errors"] += 1
            logger.exception("Error during threaded inference")
            raise

        await self.cache.set(cache_key, result)
        if not self.cache.available:
            self.stats["cache_errors"] = self.cache.error_count

        return result
