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

from .faiss_store import FaissStore
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

RETRIEVAL_K = 100  # Number of candidates


class ONNXRanker:
    def __init__(self, path):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")
        sess_options = ort.SessionOptions()  # type: ignore
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(  # type: ignore
            str(path), sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

    def predict(self, user_emb, item_emb, genre_multihot, year_idx, popularity):
        # Convert torch tensors to numpy if needed
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
        pass  # No-op for ONNX

    def to(self, device):
        pass  # No-op for ONNX (unless using GPU provider)


class ONNXUserTower:
    def __init__(self, path):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")
        sess_options = ort.SessionOptions()  # type: ignore
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(  # type: ignore
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
        # Force single thread for torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.stub_mode = _is_truthy_env("NANORECSYS_STUB")
        if self.stub_mode:
            self.device = "cpu"
            self.cache = RedisCache()
            self.user_id_to_idx = {1: 0}
            self.fallback_movie_ids = list(range(1, 1001))
            self.fallback_scores = [
                1.0 / (i + 1) for i in range(len(self.fallback_movie_ids))
            ]
            self.user_history = {0: [10, 20, 30]}
            logger.warning(
                "NANORECSYS_STUB=1 enabled: serving stub recommendations (no artifacts required)."
            )
            return

        self.device = "cpu"
        # Initialize Cache
        self.cache = RedisCache()

        # Initialize FAISS
        index_type = os.getenv("FAISS_INDEX_TYPE", "flat")
        self.faiss_store = FaissStore(settings.artifacts_dir, index_type=index_type)

        # Load Maps
        logger.info("Loading maps...")
        # user_map: index -> raw_id
        # We need raw_id -> index
        self.user_map = np.load(settings.processed_data_dir / "user_map.npy")
        self.user_id_to_idx = {uid: i for i, uid in enumerate(self.user_map)}

        self.item_map = np.load(settings.processed_data_dir / "item_map.npy")
        # For mapping back to raw IDs at output
        # item_map is index -> raw_id, which is what we want for output

        n_users = len(self.user_map)
        n_items = len(self.item_map)

        # Load User Tower
        logger.info(f"Loading User Tower ({settings.user_tower_type})...")
        onnx_tower_path = settings.artifacts_dir / "user_tower.quant.onnx"
        if ONNX_AVAILABLE and onnx_tower_path.exists():
            logger.info(f"Loading User Tower from ONNX: {onnx_tower_path}")
            self.user_tower = ONNXUserTower(onnx_tower_path)
        elif settings.user_tower_type == "transformer":
            # Transformer User Tower uses sequences of items

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
        else:
            # MLP User Tower uses user ID
            self.user_tower = UserTower(
                vocab_size=n_users,
                embed_dim=settings.embed_dim,
                output_dim=settings.tower_out_dim,
            )

        if not isinstance(self.user_tower, ONNXUserTower):
            user_tower_path = settings.artifacts_dir / "user_tower.pth"
            if user_tower_path.exists():
                self.user_tower.load_state_dict(
                    torch.load(
                        user_tower_path, map_location=self.device, weights_only=False
                    )
                )
            else:
                logger.error("User Tower checkpoint not found.")
                raise ValueError(
                    "User Tower checkpoint is required for the service to function."
                )
            self.user_tower.to(self.device)
            self.user_tower.eval()

        # Load Ranker Metadata
        logger.info("Loading metadata...")
        self.genre_matrix = (
            torch.from_numpy(
                np.load(settings.processed_data_dir / "genre_matrix_binned.npy")
            )
            .float()
            .to(self.device)
        )
        self.year_indices = (
            torch.from_numpy(
                np.load(settings.processed_data_dir / "year_indices_binned.npy")
            )
            .long()
            .to(self.device)
        )

        num_genres = self.genre_matrix.shape[1]
        # Year bin size is fixed at 6 (see datasets.py)
        num_years = 6

        # Load Ranker
        logger.info("Loading Ranker...")

        onnx_ranker_path = settings.artifacts_dir / "ranker_model.onnx"
        if ONNX_AVAILABLE and onnx_ranker_path.exists():
            logger.info(f"Loading Ranker from ONNX: {onnx_ranker_path}")
            self.ranker_model = ONNXRanker(onnx_ranker_path)
        else:
            self.ranker_model = MLPRanker(
                input_dim=settings.tower_out_dim,
                hidden_dims=settings.ranker_hidden_dims,
                num_genres=num_genres,
                num_years=num_years,
                genre_dim=16,
                year_dim=8,
            )
            ranker_path = settings.artifacts_dir / "ranker_model.pth"
            if ranker_path.exists():
                self.ranker_model.load_state_dict(
                    torch.load(
                        ranker_path, map_location=self.device, weights_only=False
                    )
                )
            else:
                logger.warning("Ranker checkpoint not found.")
            self.ranker_model.to(self.device)
            self.ranker_model.eval()

        # Load Ranker Assets
        logger.info("Loading Ranker Assets...")
        item_emb_path = settings.artifacts_dir / "ranker_item_embeddings.pt"
        if item_emb_path.exists():
            self.ranker_item_embeddings = torch.load(
                item_emb_path, map_location=self.device, weights_only=False
            )
        else:
            logger.warning("Ranker item embeddings not found.")
            raise ValueError(
                "Ranker item embeddings are required for the service to function."
            )

        pop_stats_path = settings.artifacts_dir / "ranker_pop_stats.pt"
        if pop_stats_path.exists():
            pop_stats = torch.load(
                pop_stats_path, map_location=self.device, weights_only=False
            )
            self.pop_mean = pop_stats["mean"]
            self.pop_std = pop_stats["std"]
        else:
            logger.warning("Pop stats not found.")
            self.pop_mean = 0.0
            self.pop_std = 1.0

        # Load or compute Popularity Tensor
        logger.info("Loading Popularity Tensor...")
        popularity_path = settings.artifacts_dir / "popularity_cache.pt"
        if popularity_path.exists():
            self.popularity = torch.load(
                popularity_path, map_location=self.device, weights_only=False
            )
        else:
            logger.info("Computing Popularity Tensor (from interactions)...")
            if (settings.processed_data_dir / "train.parquet").exists():
                popularity = compute_item_probabilities(n_items, return_log_probs=True)
                popularity = (popularity - self.pop_mean) / (self.pop_std + 1e-6)
                self.popularity = popularity.to(self.device)
                # Save for future use
                torch.save(self.popularity, popularity_path)
                logger.info(f"Saved popularity tensor to {popularity_path}")
            else:
                logger.warning("train.parquet not found. Using zeros for popularity.")
                self.popularity = torch.zeros(
                    n_items, dtype=torch.float32, device=self.device
                )

        # Pre-compute fallback (top-k popular items)
        # self.popularity is (N_items,)
        pop_vals, pop_indices = torch.topk(self.popularity, k=min(n_items, RETRIEVAL_K))

        self.fallback_indices = pop_indices.cpu().numpy()
        # Heuristic score: sigmoid of normalized popularity => [0, 1] range like ranker
        self.fallback_scores = torch.sigmoid(pop_vals).cpu().numpy().tolist()
        self.fallback_movie_ids = [
            int(self.item_map[idx]) for idx in self.fallback_indices
        ]

        # Load User History
        logger.info("Loading User History...")
        self.user_history = {}
        history_cache_path = settings.artifacts_dir / "user_history_cache.pkl"

        if history_cache_path.exists():
            # Load from cache
            import pickle

            try:
                with open(history_cache_path, "rb") as f:
                    self.user_history = pickle.load(f)
                logger.info(
                    f"Loaded cached history for {len(self.user_history)} users."
                )
            except Exception as e:
                logger.error(f"Error loading cached history: {e}. Will recompute.")
                self.user_history = {}

        if not self.user_history:
            # Recompute and cache
            history_path = settings.processed_data_dir / "train.parquet"
            if history_path.exists():
                try:
                    import pickle

                    # Load train and val for complete history
                    dfs = []
                    # Load user_idx, item_idx, and timestamp
                    columns = ["user_idx", "item_idx", "timestamp"]
                    df_train = pd.read_parquet(history_path, columns=columns)
                    dfs.append(df_train)

                    val_path = settings.processed_data_dir / "val.parquet"
                    if val_path.exists():
                        df_val = pd.read_parquet(val_path, columns=columns)
                        dfs.append(df_val)

                    full_df = pd.concat(dfs, ignore_index=True)
                    full_df = full_df.sort_values(by=["user_idx", "timestamp"])

                    self.user_history = (
                        full_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
                    )
                    # Save for future use
                    with open(history_cache_path, "wb") as f:
                        pickle.dump(self.user_history, f)
                    logger.info(
                        f"Loaded and cached history for {len(self.user_history)} users (train+val)."
                    )
                except Exception as e:
                    logger.error(f"Error loading history: {e}")
            else:
                logger.warning("train.parquet not found. History will be empty.")

    def _compute_inference(
        self, user_id: int, k: int, explain: bool, include_history: bool
    ):
        """
        Synchronous method containing CPU-bound tasks (Torch, FAISS, Numpy).
        This method should be run in a separate thread to avoid blocking the asyncio loop.
        """
        # Handle Stub Mode (Quick return, but good to keep in the offloaded function for consistency)
        if getattr(self, "stub_mode", False):
            limit = max(0, min(int(k), len(self.fallback_movie_ids)))
            response = {
                "movie_ids": self.fallback_movie_ids[:limit],
                "scores": self.fallback_scores[:limit],
                "explanations": ["Stub recommendation"] * limit if explain else None,
                "debug_timing": {"source": "stub"},
                "history": [10, 20, 30] if include_history else None,
            }
            return response

        t0 = time.time()
        timings = {}

        # 2. Map User ID
        t_emb_start = time.time()
        if user_id not in self.user_id_to_idx:
            # Cold user handling: Return popular items
            limit = min(k, len(self.fallback_movie_ids))
            return {
                "movie_ids": self.fallback_movie_ids[:limit],
                "scores": self.fallback_scores[:limit],
                "explanations": ["Popularity fallback (User Unknown)"] * limit
                if explain
                else None,
            }

        u_idx = self.user_id_to_idx[user_id]

        # 3. Retrieve Candidates (User Tower + FAISS)
        with torch.inference_mode():
            if settings.user_tower_type == "transformer":
                # Get user history
                history = self.user_history.get(u_idx, [])
                if not history:
                    # Cold user (known ID but no history) - Fallback
                    limit = min(k, len(self.fallback_movie_ids))
                    return {
                        "movie_ids": self.fallback_movie_ids[:limit],
                        "scores": self.fallback_scores[:limit],
                        "explanations": ["Popularity fallback (No History)"] * limit
                        if explain
                        else None,
                    }

                # Truncate to max_seq_len
                seq = history[-settings.max_seq_len :]
                seq = [item + 1 for item in seq]  # Shift by 1 for padding idx=0
                pad_len = settings.max_seq_len - len(seq)
                if pad_len > 0:
                    seq = [0] * pad_len + seq
                # Transformer expects (B, SeqLen)
                seq_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)

                # Use .encode() for transformer tower
                user_emb = self.user_tower.encode(seq_tensor)  # (1, emb_dim)
            else:
                u_tensor = torch.tensor([u_idx], device=self.device)
                user_emb = self.user_tower(u_tensor)  # type: ignore # (1, emb_dim)

        timings["embedding"] = (time.time() - t_emb_start) * 1000

        t_ret_start = time.time()
        candidate_indices, _ = self.faiss_store.search(user_emb, k=RETRIEVAL_K)
        timings["retrieval"] = (time.time() - t_ret_start) * 1000

        history_items = self.user_history.get(u_idx, [])
        if history_items:
            watched = set(history_items)
            filtered = [int(ci) for ci in candidate_indices if int(ci) not in watched]
            candidate_indices = filtered

        if len(candidate_indices) == 0:
            limit = min(k, len(self.fallback_movie_ids))
            return {
                "movie_ids": self.fallback_movie_ids[:limit],
                "scores": self.fallback_scores[:limit],
                "explanations": ["Popularity fallback (No Candidates)"] * limit
                if explain
                else None,
            }

        candidate_indices = torch.tensor(
            candidate_indices, dtype=torch.long, device=self.device
        )

        # 4. Re-Rank
        t_rank_start = time.time()
        with torch.inference_mode():
            # Prepare Ranker Inputs
            # User emb expanded
            user_emb_expanded = user_emb.repeat(len(candidate_indices), 1)

            # Item embs
            item_embs = self.ranker_item_embeddings[candidate_indices]

            # Metadata
            genres = self.genre_matrix[candidate_indices]
            years = self.year_indices[candidate_indices]
            pops = self.popularity[candidate_indices]

            # Predict
            scores = self.ranker_model.predict(
                user_emb=user_emb_expanded,
                item_emb=item_embs,
                genre_multihot=genres,
                year_idx=years,
                popularity=pops,
            )
            # scores is (B,) probabilities

        # 5. Sort and Select Top-K
        topk_scores, topk_indices = torch.topk(scores, k=min(k, len(scores)))

        # Map back to raw IDs
        final_indices = candidate_indices[topk_indices].cpu().numpy()
        final_scores = topk_scores.cpu().numpy().tolist()

        final_movie_ids = [int(self.item_map[idx]) for idx in final_indices]

        timings["ranking"] = (time.time() - t_rank_start) * 1000
        timings["total"] = (time.time() - t0) * 1000

        result = {
            "movie_ids": final_movie_ids,
            "scores": final_scores,
            "explanations": None,
            "debug_timing": timings,
        }

        # 6. Explanations
        if explain:
            result["explanations"] = ["Explanation not implemented"] * len(
                final_movie_ids
            )

        # 7. History
        if include_history:
            if u_idx in self.user_history:
                hist_indices = self.user_history[u_idx]
                # Limit to last 20 items to save bandwidth
                if len(hist_indices) > 20:
                    hist_indices = hist_indices[-20:]

                result["history"] = [int(self.item_map[idx]) for idx in hist_indices]
            else:
                result["history"] = []

        return result

    async def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        explain: bool = False,
        include_history: bool = False,
    ):
        """
        Async entry point. Handles Cache (I/O) and offloads Compute (CPU) to a thread.
        """
        # 1. Check Redis Cache (Async I/O - Keep this in the main loop!)
        cache_key = f"user:{user_id}:k:{k}:h:{int(include_history)}:e:{int(explain)}"
        cached = await self.cache.get(cache_key)
        if cached:
            # Add basic timing info for cache hit
            cached["debug_timing"] = {
                "source": "cache",
            }
            return cached

        # 2. Cache Miss? Run inference in a separate thread.
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                self._compute_inference,  # Function to run
                user_id,
                k,
                explain,
                include_history,  # Args
            )
        except Exception as e:
            logger.exception("Error during threaded inference")
            raise e

        # 3. Save to Redis
        await self.cache.set(cache_key, result)

        return result
