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

import torch
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from nanoRecSys.config import settings
from nanoRecSys.eval.metrics import compute_batch_metrics
from nanoRecSys.models.ranker import RankerModel
from nanoRecSys.data.datasets import load_item_metadata
from nanoRecSys.utils.utils import get_vocab_sizes, compute_item_probabilities
from nanoRecSys.utils.logging_config import get_logger
from nanoRecSys.training.mine_negatives import load_all_positives


class OfflineEvaluator:
    def __init__(
        self,
        batch_size=256,
        k_list=[10, 20, 50, 100],
        sampled=False,
        sample_strategy="uniform",
    ):
        logger = get_logger()
        self.batch_size = batch_size
        self.k_list = k_list
        self.max_k = max(k_list)
        self.sampled = sampled
        self.sample_strategy = sample_strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.n_users, self.n_items = get_vocab_sizes()
        logger.info(f"Vocab: Users={self.n_users}, Items={self.n_items}")

        self.test_users, self.test_targets = self._load_test_data()

        if self.sampled:
            # Check constraints
            lengths = np.array([len(x) for x in self.test_targets], dtype=int)
            if np.any(lengths != 1):
                logger.error(
                    f"Sampled evaluation requires exactly 1 interaction per user. Found max {lengths.max()}."
                )
                sys.exit(1)

            # Prepare negatives
            self._prepare_sampled_candidates()

        # Cache for embeddings and models
        self.user_embs = None
        self.item_embs = None
        self.ranker = None

        # Cache for metadata
        self.genre_matrix = None
        self.year_indices = None
        self.item_popularity = None

    def _prepare_sampled_candidates(self):
        logger = get_logger()
        logger.info(
            f"Preparing sampled candidates (1 positive + 100 negatives) using {self.sample_strategy} sampling..."
        )
        # Ensure we have all positives loaded
        logger.warning(
            f"Filtering known positives based on ranker_positive_threshold ({settings.ranker_positive_threshold}). "
            "If you want to adjust what is considered a positive for exclusion, change ranker_positive_threshold."
        )
        all_positives = load_all_positives()

        n_neg = 100
        n_samples = len(self.test_users)
        candidates = np.zeros((n_samples, n_neg + 1), dtype=int)

        # Fill positive at index 0
        positives = np.array([x[0] for x in self.test_targets])
        candidates[:, 0] = positives

        # Define strategies
        if self.sample_strategy == "mixed":
            # 50 random, 50 popularity
            strategies = [("uniform", 50), ("popularity", 50)]
        elif self.sample_strategy == "popularity":
            strategies = [("popularity", 100)]
        else:
            strategies = [("uniform", 100)]

        # Pre-calculate negatives pools
        pools = {}
        pool_indices = {}
        probs = None

        # Check if we need popularity probs
        if any(s == "popularity" for s, _ in strategies):
            logger.info("Computing item probabilities for sampling...")
            probs_tensor = compute_item_probabilities(
                self.n_items, return_log_probs=False, smooth=False
            )
            probs = probs_tensor.cpu().numpy()
            probs /= probs.sum()

        # Generate initial pools
        # We estimate usage: n_samples * (avg items needed)
        # Just use a large buffer.
        est_pool_size = n_samples * 120

        for strategy_name, _ in strategies:
            if strategy_name not in pools:
                logger.info(f"Generating initial pool for '{strategy_name}'...")
                if strategy_name == "uniform":
                    pools[strategy_name] = np.random.randint(
                        0, self.n_items, size=est_pool_size
                    )
                elif strategy_name == "popularity":
                    pools[strategy_name] = np.random.choice(
                        self.n_items, size=est_pool_size, p=probs
                    )
                pool_indices[strategy_name] = 0

        logger.info("Assigning negatives...")
        for i, u_idx in tqdm(enumerate(self.test_users), total=n_samples):
            u_pos = all_positives.get(u_idx, set())

            current_user_negs = []
            current_user_negs_set = set()  # To ensure uniqueness within row

            for strategy_name, count_needed in strategies:
                count_collected = 0
                pool = pools[strategy_name]
                idx = pool_indices[strategy_name]

                while count_collected < count_needed:
                    # Fetch a chunk
                    remaining = count_needed - count_collected
                    chunk_size = int(remaining * 2) + 5

                    # Refill check
                    if idx + chunk_size >= len(pool):
                        logger.info(f"Refilling '{strategy_name}' pool...")
                        refill_size = est_pool_size // 10
                        if strategy_name == "uniform":
                            new_pool = np.random.randint(
                                0, self.n_items, size=refill_size
                            )
                        else:
                            new_pool = np.random.choice(
                                self.n_items, size=refill_size, p=probs
                            )
                        pool = np.concatenate((pool[idx:], new_pool))
                        pools[strategy_name] = pool
                        idx = 0

                    batch = pool[idx : idx + chunk_size]
                    idx += chunk_size

                    for item in batch:
                        if item not in u_pos and item not in current_user_negs_set:
                            current_user_negs.append(item)
                            current_user_negs_set.add(item)
                            count_collected += 1
                            if count_collected == count_needed:
                                break

                pool_indices[strategy_name] = idx

            candidates[i, 1:] = current_user_negs

        self.test_candidates = torch.from_numpy(candidates).long().to(self.device)
        logger.info(f"Candidates prepared: {self.test_candidates.shape}")

    def _load_test_data(self):
        """Load and filter test data."""
        logger = get_logger()
        logger.info("Loading test splits...")
        test_df = pd.read_parquet(settings.processed_data_dir / "test.parquet")
        test_df = test_df[test_df["rating"] >= settings.retrieval_threshold]

        test_user_groups = test_df.groupby("user_idx")["item_idx"].apply(list)
        # Copy arrays to ensure they are writable (avoids PyTorch warnings during indexing)
        return test_user_groups.index.values.copy(), test_user_groups.values.copy()

    def _load_embeddings(self):
        """Load pre-computed embeddings from disk."""
        logger = get_logger()
        logger.info("Loading embeddings from disk...")
        u_path = settings.artifacts_dir / "user_embeddings.npy"
        i_path = settings.artifacts_dir / "item_embeddings.npy"

        if not u_path.exists() or not i_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {settings.artifacts_dir}. "
                "Please run `python src/indexing/build_embeddings.py --mode all` first."
            )

        # Copy numpy arrays to ensure writability before conversion
        self.user_embs = (
            torch.from_numpy(np.load(u_path).copy()).to(self.device).float()
        )
        self.item_embs = (
            torch.from_numpy(np.load(i_path).copy()).to(self.device).float()
        )
        logger.info(
            f"Loaded Users: {self.user_embs.shape}, Items: {self.item_embs.shape}"
        )

    def _load_ranker(self):
        """Load Ranker model and Metadata."""
        logger = get_logger()
        logger.info("Loading Ranker Model and Metadata...")

        # 1. Metadata
        item_map_path = settings.processed_data_dir / "item_map.npy"
        movies_path = settings.raw_data_dir / "movies.csv"
        g_mat, y_idx, n_genres, n_years = load_item_metadata(
            item_map_path, movies_path, cache_dir=str(settings.processed_data_dir)
        )
        self.genre_matrix = g_mat.to(self.device)
        self.year_indices = y_idx.to(self.device)
        self.num_genres = n_genres
        self.num_years = n_years

        # 2. Popularity (Log Probs)
        self.item_popularity = compute_item_probabilities(
            self.n_items, return_log_probs=True, device=self.device
        )

        # 2.1 Load Popularity Stats (if available)
        pop_stats_path = settings.artifacts_dir / "ranker_pop_stats.pt"
        if pop_stats_path.exists():
            logger.info("Loading Popularity Stats...")
            stats = torch.load(pop_stats_path)
            self.pop_mean = stats["mean"]
            self.pop_std = stats["std"]
        else:
            print(
                "Warning: Pop stats not found. Ranker might expect normalized inputs."
            )
            self.pop_mean = 0.0
            self.pop_std = 1.0

        # 2.2 Load Fine-Tuned Item Embeddings (if available)
        ft_item_path = settings.artifacts_dir / "ranker_item_embeddings.pt"
        if ft_item_path.exists():
            print("Loading Fine-Tuned Item Embeddings for Ranker...")
            self.ranker_item_embs = torch.load(ft_item_path, map_location=self.device)
        else:
            print(
                "Warning: Fine-tuned item embeddings not found, using base embeddings."
            )
            self.ranker_item_embs = self.item_embs

        # 3. Model
        self.ranker = RankerModel(
            input_dim=settings.tower_out_dim,
            hidden_dims=settings.ranker_hidden_dims,
            num_genres=n_genres,
            num_years=n_years,
            genre_dim=16,
            year_dim=8,
        ).to(self.device)

        ranker_path = settings.artifacts_dir / "ranker_model.pth"
        if not ranker_path.exists():
            raise FileNotFoundError(f"Ranker model not found at {ranker_path}")

        self.ranker.load_state_dict(torch.load(ranker_path, map_location=self.device))
        self.ranker.eval()

    def _batch_metrics(self, preds_batch, targets_batch):
        """Compute metrics for a batch of predictions efficiently."""
        return compute_batch_metrics(preds_batch, targets_batch, self.k_list)

    def _accumulate_metrics(self, global_metrics, batch_metrics, prefix=""):
        """Add batch metrics to global sum."""
        for k, v in batch_metrics.items():
            key = f"{prefix}{k}"
            global_metrics[key] = global_metrics.get(key, 0.0) + v

    def eval_popularity(self):
        print("Evaluating Popularity Baseline...")
        train_df = pd.read_parquet(settings.processed_data_dir / "train.parquet")
        pop_counts = train_df["item_idx"].value_counts()

        if self.sampled:
            print("Running Sampled Popularity Eval...")
            # Create dense popularity vector
            pop_vector = np.zeros(self.n_items, dtype=float)
            if not pop_counts.empty:
                pop_vector[pop_counts.index] = pop_counts.values
            pop_tensor = torch.from_numpy(pop_vector).float().to(self.device)

            metrics_sum = {}
            for i in tqdm(range(0, len(self.test_users), self.batch_size)):
                batch_targets = self.test_targets[i : i + self.batch_size]
                batch_candidates = self.test_candidates[
                    i : i + self.batch_size
                ]  # (B, 101)

                # Scores
                scores = pop_tensor[batch_candidates]  # (B, 101)

                # Sort
                _, sorted_indices = torch.topk(scores, k=self.max_k, dim=1)

                # Map to Item IDs
                preds = torch.gather(batch_candidates, 1, sorted_indices).cpu().numpy()

                res = self._batch_metrics(preds, batch_targets)
                self._accumulate_metrics(metrics_sum, res)

            return {k: v / len(self.test_users) for k, v in metrics_sum.items()}

        global_top_k = pop_counts.index.values[: self.max_k]

        metrics_sum = {}
        for i in tqdm(range(0, len(self.test_users), self.batch_size)):
            batch_targets = self.test_targets[i : i + self.batch_size]
            batch_preds = np.tile(global_top_k, (len(batch_targets), 1))

            res = self._batch_metrics(batch_preds, batch_targets)
            self._accumulate_metrics(metrics_sum, res)

        return {k: v / len(self.test_users) for k, v in metrics_sum.items()}

    def eval_retrieval(self):
        """Evaluate Two-Tower Retrieval (Standard vs LogQ)."""
        self._load_embeddings()
        log_p = compute_item_probabilities(
            self.n_items, device=self.device, return_log_probs=True
        )

        metrics_sum = {}

        print("Scoring users (Retrieval)...")
        for i in tqdm(range(0, len(self.test_users), self.batch_size)):
            idx_end = i + self.batch_size
            batch_u_idx = self.test_users[i:idx_end]
            batch_targets = self.test_targets[i:idx_end]

            # Embeddings
            batch_u_emb = self.user_embs[batch_u_idx]  # type: ignore # (B, Dim)

            if self.sampled:
                batch_candidates = self.test_candidates[i:idx_end]  # (B, 101)

                # Item embs for candidates
                batch_i_emb = self.item_embs[batch_candidates]  # type: ignore # (B, 101, Dim)

                # Compute Score: (B, 1, Dim) * (B, 101, Dim) -> sum -> (B, 101)
                scores = (batch_u_emb.unsqueeze(1) * batch_i_emb).sum(dim=2)

                # 1. Standard
                _, topk_std = torch.topk(scores, k=self.max_k, dim=1)
                preds_std = torch.gather(batch_candidates, 1, topk_std).cpu().numpy()
                res_std = self._batch_metrics(preds_std, batch_targets)
                self._accumulate_metrics(metrics_sum, res_std, prefix="Standard_")

                # 2. LogQ
                cand_log_p = log_p[batch_candidates]  # (B, 101)
                scores_logq = (scores / settings.temperature) + cand_log_p
                _, topk_logq = torch.topk(scores_logq, k=self.max_k, dim=1)
                preds_logq = torch.gather(batch_candidates, 1, topk_logq).cpu().numpy()
                res_logq = self._batch_metrics(preds_logq, batch_targets)
                self._accumulate_metrics(metrics_sum, res_logq, prefix="LogQ_")
                continue

            # Scores (B, N_items)
            scores = torch.matmul(batch_u_emb, self.item_embs.T)  # type: ignore

            # 1. Standard (No Correction)
            _, topk_std = torch.topk(scores, k=self.max_k, dim=1)
            preds_std = topk_std.cpu().numpy()
            res_std = self._batch_metrics(preds_std, batch_targets)
            self._accumulate_metrics(metrics_sum, res_std, prefix="Standard_")

            # 2. LogQ Correction
            scores_logq = (scores / settings.temperature) + log_p
            _, topk_logq = torch.topk(scores_logq, k=self.max_k, dim=1)
            preds_logq = topk_logq.cpu().numpy()
            res_logq = self._batch_metrics(preds_logq, batch_targets)
            self._accumulate_metrics(metrics_sum, res_logq, prefix="LogQ_")

        return {k: v / len(self.test_users) for k, v in metrics_sum.items()}

    def eval_ranker(self):
        """Evaluate Ranker (Re-rank Top-100 from Retrieval)."""
        self._load_embeddings()
        self._load_ranker()

        retrieve_k = 100  # Candidates to retrieve
        assert retrieve_k >= self.max_k, "Retrieval K must be >= Evaluation K"

        metrics_sum = {}

        print("Scoring users (Retrieval + Ranker)...")
        for i in tqdm(range(0, len(self.test_users), self.batch_size)):
            idx_end = i + self.batch_size
            batch_u_idx = self.test_users[i:idx_end]
            batch_targets = self.test_targets[i:idx_end]
            current_bs = len(batch_u_idx)
            batch_u_emb = self.user_embs[batch_u_idx]  # type: ignore

            if self.sampled:
                candidates = self.test_candidates[i:idx_end]
                k_cands = candidates.shape[1]
            else:
                # --- A. Retrieval (LogQ) ---
                scores = torch.matmul(batch_u_emb, self.item_embs.T)  # type: ignore
                scores_logq = (scores / settings.temperature) + self.item_popularity  # type: ignore

                # Get Top-100 Candidates
                _, candidates = torch.topk(scores_logq, k=retrieve_k, dim=1)  # (B, 100)
                k_cands = retrieve_k

                # A1. Record Retrieval Metrics (for comparison)
                # Take top max_k from the 100 candidates (since topk returns sorted, this is just slicing)
                preds_retrieval = candidates[:, : self.max_k].cpu().numpy()
                res_retrieval = self._batch_metrics(preds_retrieval, batch_targets)
                self._accumulate_metrics(
                    metrics_sum, res_retrieval, prefix="Retrieval_"
                )

            # --- B. Ranker Re-Ranking ---
            # Flatten for batch processing
            flat_candidates = candidates.view(-1)  # (B*100)

            # Expand User embeddings: (B, Dim) -> (B, 100, Dim) -> (B*100, Dim)
            flat_u_emb = (
                batch_u_emb.unsqueeze(1)
                .expand(-1, k_cands, -1)
                .reshape(-1, settings.tower_out_dim)
            )

            # Fetch Item Embeddings & Metadata
            # Use Fine-Tuned embeddings for Ranker scoring
            flat_i_emb = self.ranker_item_embs[flat_candidates]  # type: ignore
            flat_genre = self.genre_matrix[flat_candidates]  # type: ignore
            flat_year = self.year_indices[flat_candidates]  # type: ignore

            # Normalize Popularity
            raw_pop = self.item_popularity[flat_candidates]  # type: ignore
            flat_pop = (raw_pop - self.pop_mean) / (self.pop_std + 1e-6)

            with torch.no_grad():
                # self.ranker now returns logits (no sigmoid). Sorting by logits is equivalent to sorting by probs.
                ranker_scores = self.ranker(
                    user_emb=flat_u_emb,
                    item_emb=flat_i_emb,
                    genre_multihot=flat_genre,
                    year_idx=flat_year,
                    popularity=flat_pop,
                )  # type: ignore # (B*100)

            # Reshape scores to (B, 100)
            ranker_scores = ranker_scores.view(current_bs, k_cands)

            # Sort the 100 candidates by Ranker Score
            # We want the indices into the 'candidates' array (0..99)
            _, sorted_local_indices = torch.topk(ranker_scores, k=self.max_k, dim=1)

            # Map back to original Item IDs
            final_preds = (
                torch.gather(candidates, 1, sorted_local_indices).cpu().numpy()
            )

            # B1. Record Ranker Metrics
            res_ranker = self._batch_metrics(final_preds, batch_targets)
            self._accumulate_metrics(metrics_sum, res_ranker, prefix="Ranker_")

        return {k: v / len(self.test_users) for k, v in metrics_sum.items()}

    def formatted_results(self, results):
        """Format results"""
        df = pd.DataFrame(index=self.k_list)
        data_map = {}

        for k, v in results.items():
            if "@" not in k:
                continue
            parts = k.split("@")
            val_k = int(parts[1])
            name_parts = parts[0].split("_")
            metric = name_parts[-1]
            prefix = "_".join(name_parts[:-1])  # "Ranker", "Retrieval", or ""

            col_name = f"{prefix}_{metric}" if prefix else metric
            if col_name not in data_map:
                data_map[col_name] = {}
            data_map[col_name][val_k] = v

        df = pd.DataFrame(data_map).sort_index()
        return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, choices=["popularity", "twotower", "ranker"]
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Use Sampled Evaluation (1 positive + 100 negatives per user). "
        "Requires users to have exactly 1 interaction in test set.",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "popularity", "mixed"],
        help="Strategy for sampling negatives in sampled evaluation.",
    )
    args = parser.parse_args()

    evaluator = OfflineEvaluator(
        batch_size=args.batch_size,
        sampled=args.sampled,
        sample_strategy=args.sample_strategy,
    )

    results = {}
    if args.model == "popularity":
        results = evaluator.eval_popularity()
    elif args.model == "twotower":
        results = evaluator.eval_retrieval()
    elif args.model == "ranker":
        results = evaluator.eval_ranker()

    print(f"\n--- Results ({args.model}) ---")
    print(evaluator.formatted_results(results))
