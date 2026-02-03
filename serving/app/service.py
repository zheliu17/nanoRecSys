import logging
import torch
import numpy as np
import time
import os
import pandas as pd

from nanoRecSys.models.towers import UserTower
from nanoRecSys.models.ranker import RankerModel
from nanoRecSys.utils.utils import compute_item_probabilities
from nanoRecSys.config import settings

from .faiss_store import FaissStore
from .redis_cache import RedisCache

logger = logging.getLogger(__name__)

RETRIEVAL_K = 100  # Number of candidates


class RecommendationService:
    def __init__(self):
        self.device = "cpu"
        # Initialize Cache
        self.cache = RedisCache()

        # Initialize FAISS
        index_type = os.getenv("FAISS_INDEX_TYPE", "ivf")
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
        logger.info("Loading User Tower...")
        self.user_tower = UserTower(
            vocab_size=n_users,
            embed_dim=settings.embed_dim,
            output_dim=settings.tower_out_dim,
        )
        user_tower_path = settings.artifacts_dir / "user_tower.pth"
        if user_tower_path.exists():
            self.user_tower.load_state_dict(
                torch.load(
                    user_tower_path, map_location=self.device, weights_only=False
                )
            )
        else:
            logger.warning("User Tower checkpoint not found.")
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
        self.ranker_model = RankerModel(
            input_dim=settings.tower_out_dim,
            num_genres=num_genres,
            num_years=num_years,
        )
        ranker_path = settings.artifacts_dir / "ranker_model.pth"
        if ranker_path.exists():
            self.ranker_model.load_state_dict(
                torch.load(ranker_path, map_location=self.device, weights_only=False)
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
            logger.warning("Ranker item embeddings not found. Using random.")
            self.ranker_item_embeddings = torch.randn(
                n_items, settings.tower_out_dim, device=self.device
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

                    # Load only necessary columns to save memory
                    df = pd.read_parquet(history_path, columns=["user_idx", "item_idx"])
                    self.user_history = (
                        df.groupby("user_idx")["item_idx"].apply(list).to_dict()
                    )
                    # Save for future use
                    with open(history_cache_path, "wb") as f:
                        pickle.dump(self.user_history, f)
                    logger.info(
                        f"Loaded and cached history for {len(self.user_history)} users."
                    )
                except Exception as e:
                    logger.error(f"Error loading history: {e}")
            else:
                logger.warning("train.parquet not found. History will be empty.")

    def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        explain: bool = False,
        include_history: bool = False,
    ):
        t0 = time.time()
        timings = {}

        # 1. Check Redis Cache
        # Include history and explain in key to differentiate responses
        cache_key = f"user:{user_id}:k:{k}:h:{int(include_history)}:e:{int(explain)}"
        cached = self.cache.get(cache_key)
        if cached:
            # Add basic timing info for cache hit
            cached["debug_timing"] = {
                "total": (time.time() - t0) * 1000,
                "source": "cache",
            }
            return cached

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
        u_tensor = torch.tensor([u_idx], device=self.device)

        # 3. Retrieve Candidates (User Tower + FAISS)
        with torch.no_grad():
            user_emb = self.user_tower(u_tensor)  # (1, 128)

        timings["embedding"] = (time.time() - t_emb_start) * 1000

        t_ret_start = time.time()
        # Retrieve more candidates for re-ranking (e.g. 100)
        candidate_indices, _ = self.faiss_store.search(user_emb, k=RETRIEVAL_K)
        timings["retrieval"] = (time.time() - t_ret_start) * 1000

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
        with torch.no_grad():
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

        self.cache.set(cache_key, result)

        return result
