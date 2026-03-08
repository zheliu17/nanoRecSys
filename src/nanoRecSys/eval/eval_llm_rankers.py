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


"""Unified LLM-ranking evaluation: API-based and local-model scorers on the same test set."""

from __future__ import annotations

try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None  # type: ignore

import argparse
import json
import os
import re
import time
from typing import Protocol

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

from nanoRecSys.config import settings
from nanoRecSys.eval.offline_eval import OfflineEvaluator
from nanoRecSys.utils.logging_config import get_logger
from nanoRecSys.utils.prompt_utils import (
    build_candidate_suffix,
    decode_ids_to_titles_and_keep_ids,
    prepare_prompt_prefix,
)
from nanoRecSys.utils.utils import format_results_to_dataframe


class Scorer(Protocol):
    """Minimal interface every ranker must implement."""

    name: str

    def setup(self, runner: LLMEvalRunner) -> None:
        """Called once after the shared runner has finished loading data/models."""
        ...

    def score(
        self,
        u_idx: int,  # mainly for logging purposes
        history_ids: list[int],
        candidates: list[int],
    ) -> list[tuple[int, float]]:
        """Return ``[(candidate_id, score), ...]`` in *any* order (runner sorts)."""
        ...


class LLMEvalRunner:
    """Loads all shared resources and drives the evaluation loop."""

    K_CANDS: int = 100  # retrieval budget fed to every ranker

    def __init__(self, num_users: int = 100) -> None:
        self.logger = get_logger()
        self.num_users = num_users

        self.movies_df: pd.DataFrame = pd.read_csv(settings.raw_data_dir / "movies.csv")
        self.movie_mapping: dict = self.movies_df.set_index("movieId")[
            "title"
        ].to_dict()
        self.item_map: np.ndarray = np.load(
            settings.processed_data_dir / "item_map.npy"
        )
        self._load_seq_dict()

        self.logger.info("Initialising OfflineEvaluator + two-tower embeddings…")
        self.evaluator = OfflineEvaluator(batch_size=1024, remove_history=True)
        self.evaluator._load_embeddings()
        self.evaluator._load_transformer_user_tower()

    def _load_seq_dict(self) -> None:
        test_seqs = np.load(settings.processed_data_dir / "seq_test_sequences.npy")
        test_seqs = test_seqs[:, :-1]  # drop target item
        seq_user_ids = np.load(settings.processed_data_dir / "seq_test_user_ids.npy")
        # 0-index
        self.seq_dict: dict[int, list[int]] = {
            int(u): (sq[sq != 0] - 1).tolist() for u, sq in zip(seq_user_ids, test_seqs)
        }

    @torch.inference_mode()
    def _retrieve_candidates(
        self, u_idx: int, history_ids: list[int]
    ) -> list[int] | None:
        """Run two-tower retrieval and return top-K candidate item IDs."""
        ev = self.evaluator

        seq = history_ids[-settings.max_seq_len :]
        seq = [item + 1 for item in seq]  # Shift by 1 for padding idx=0
        pad_len = settings.max_seq_len - len(seq)
        if pad_len > 0:
            seq = [0] * pad_len + seq
        input_seq = torch.tensor([seq], dtype=torch.long, device=ev.device)

        u_emb = ev.user_tower.encode(input_seq)
        scores = torch.matmul(u_emb, ev.item_embs.T)  # (1, n_items)

        scores = ev._mask_history(scores, [u_idx])

        _, topk_idx = torch.topk(scores, k=self.K_CANDS, dim=1)
        return topk_idx[0].cpu().tolist()

    def run(self, scorers: list[Scorer]) -> dict[str, dict[str, float]]:
        """Evaluate a list of *scorers* over :attr:`num_users` test users.

        Returns:
            Nested dict mapping scorer name to averaged metric dict.
        """
        for scorer in scorers:
            self.logger.info(f"[{scorer.name}] Setting up scorer…")
            scorer.setup(self)

        metrics_sum: dict[str, dict[str, float]] = {s.name: {} for s in scorers}
        processed = 0
        ev = self.evaluator

        self.logger.info(
            f"Evaluating {len(scorers)} scorers over up to {self.num_users} users"
        )

        pbar = tqdm(
            total=min(len(ev.test_users), self.num_users), desc="Evaluating Users"
        )
        for i in range(len(ev.test_users)):
            u_idx = int(ev.test_users[i])
            if u_idx not in self.seq_dict:
                continue

            history_ids = self.seq_dict[u_idx]
            if not history_ids:
                continue

            target_items = ev.test_targets[i]
            candidates = self._retrieve_candidates(
                u_idx, history_ids
            )  # Full history used for retrieval
            if candidates is None:
                continue
            history_ids = history_ids[-settings.llm_history_len :]

            for scorer in scorers:
                ranked = scorer.score(u_idx, history_ids, candidates)
                ranked.sort(key=lambda x: x[1], reverse=True)
                final_preds = np.array([[cid for cid, _ in ranked]])

                res = ev._batch_metrics(final_preds, [target_items])
                ev._accumulate_metrics(
                    metrics_sum[scorer.name], res, prefix=f"{scorer.name}_"
                )

            processed += 1
            pbar.update(1)
            if processed >= self.num_users:
                break
        pbar.close()

        if processed == 0:
            self.logger.warning("No users processed — check seq_dict.")
            return {}

        final_metrics: dict[str, dict[str, float]] = {}
        for scorer in scorers:
            final = {k: v / processed for k, v in metrics_sum[scorer.name].items()}
            final_metrics[scorer.name] = final

        return final_metrics


class RetrieverScorer:
    """Baseline scorer that simply returns the original retrieval order."""

    name = "Retriever"

    def setup(self, runner: LLMEvalRunner) -> None:
        pass

    def score(
        self, u_idx: int, history_ids: list[int], candidates: list[int]
    ) -> list[tuple[int, float]]:
        # Candidates are already sorted by retrieval score from _retrieve_candidates.
        # We assign descending scores to preserve their rank.
        return [(cid, float(len(candidates) - i)) for i, cid in enumerate(candidates)]


class APIScorer:
    """Rank candidates by querying an OpenAI-compatible chat completion API.

    Scores are the log-probability of the token ``Yes`` given the prompt.
    Results are persisted to a JSON cache so interrupted runs can resume.
    """

    name = "LLMAPIRanker"

    def __init__(self, clean_cache: bool = False) -> None:
        self.cache_path = settings.artifacts_dir / "api_responses_cache.json"
        self.cache: dict[str, float] = {}
        if clean_cache and os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            get_logger().info(f"Removed cache file {self.cache_path}")
        elif os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                self.cache = json.load(f)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            }
        )
        try:
            from requests.adapters import HTTPAdapter

            adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
        except Exception:
            pass

    def setup(self, runner: LLMEvalRunner) -> None:
        self.item_map = runner.item_map
        self.movie_mapping = runner.movie_mapping

    def _titles(self, ids: list[int]) -> list[str]:
        titles, _ = decode_ids_to_titles_and_keep_ids(
            ids, self.item_map, self.movie_mapping
        )
        return titles

    def _api_call(
        self, system_prompt: str, user_prompt: str, max_retries: int = 3
    ) -> tuple[float, bool]:
        """Return (score, success) where success indicates status_code == 200.

        Returns score of ``-999.0`` on any failure. Only cache when success is True.
        """
        logger = get_logger()
        data = {
            "model": settings.llm_api_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 1,
            "temperature": 0.0,
        }
        # Only include the explicit `enable_thinking` key when it's disabled
        # in settings. This avoids sending the key when it's enabled/default.
        if settings.llm_enable_thinking is not None:
            data["enable_thinking"] = settings.llm_enable_thinking
        for attempt in range(max_retries):
            try:
                resp = self.session.post(
                    settings.llm_api_endpoint,
                    json=data,
                    timeout=settings.llm_api_timeout,
                )
                if resp.status_code == 200:
                    try:
                        top_lps = resp.json()["choices"][0]["logprobs"]["content"][0][
                            "top_logprobs"
                        ]
                    except Exception as e:
                        logger.info(f"API response parsing failed: {e}")
                        return (-999.0, True)

                    yes_probs = []
                    no_probs = []
                    for lp in top_lps:
                        clean = re.sub(r"[^A-Za-z0-9]", "", lp.get("token", "")).lower()
                        if clean == "yes":
                            yes_probs.append(float(lp.get("logprob", -999.0)))
                        elif clean == "no":
                            no_probs.append(float(lp.get("logprob", -999.0)))

                    yes_prob = (
                        float(np.logaddexp.reduce(yes_probs)) if yes_probs else -999.0
                    )
                    no_prob = (
                        float(np.logaddexp.reduce(no_probs)) if no_probs else -999.0
                    )

                    if yes_prob != -999.0 and no_prob != -999.0:
                        return (yes_prob - no_prob, True)
                    elif yes_prob != -999.0:
                        return (yes_prob, True)

                    logger.info(
                        "API call succeeded but 'Yes' token not found in top_logprobs"
                    )
                    return (-999.0, True)

                if resp.status_code == 429:
                    # rate limited, retry after backoff
                    time.sleep(2**attempt)
                    continue

                # other non-success status codes
                logger.info(
                    f"API call failed: status={resp.status_code} body={resp.text[:200]}"
                )
                return (-999.0, False)
            except Exception as e:
                logger.info(f"API call exception: {e}")
                time.sleep(2**attempt)
        logger.info("API call failed after retries")
        return (-999.0, False)

    def score(
        self, u_idx: int, history_ids: list[int], candidates: list[int]
    ) -> list[tuple[int, float]]:
        history_titles = self._titles(history_ids)
        history_str = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(history_titles))

        results: list[tuple[int, float]] = []
        for cid in candidates:
            key = f"{u_idx}_{cid}"
            if key in self.cache:
                results.append((cid, self.cache[key]))
                continue

            cand_titles = self._titles([cid])
            cand_title = cand_titles[0] if cand_titles else "Unknown Movie"
            user_prompt = (
                f"User Viewing History:\n{history_str}\n\n"
                f"Candidate Movie: {cand_title}\n\nWill the user watch this next?"
            )
            score, got_200 = self._api_call(settings.llm_system_prompt_api, user_prompt)
            if got_200:
                self.cache[key] = score
                with open(self.cache_path, "w") as f:
                    json.dump(self.cache, f)
            results.append((cid, score))

        return results


class LocalLLMScorer:
    """Rank candidates using a locally loaded LLM with LoRA adapter.

    The history prefix embeddings are built once per user, then candidates are
    scored in mini-batches via full embedding-only forward passes.
    """

    name = "Local_LLM_Ranker"

    def __init__(
        self,
        model_name: str = settings.llm_model_name,
        adapter_path: str | None = None,
        batch_size: int = 8,
        use_lora: bool = True,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path or str(settings.llm_output_dir)
        self.batch_size = max(1, int(batch_size))
        self.use_lora = use_lora

    def setup(self, runner: LLMEvalRunner) -> None:
        from nanoRecSys.models.llm_ranker import LLMRanker

        logger = get_logger()
        self.item_map = runner.item_map
        self.movie_mapping = runner.movie_mapping
        self.device = runner.evaluator.device
        self.item_embs = runner.evaluator.item_embs.to(self.device)

        logger.info(
            "Loading base LLM + projection layer"
            + (" + LoRA adapter…" if self.use_lora else "…")
        )
        sasrec_dim = self.item_embs.shape[1]
        ranker = LLMRanker(
            sasrec_emb_dim=sasrec_dim,
            model_name=self.model_name,
            use_lora=self.use_lora,
        )
        self.tokenizer = ranker.tokenizer

        projection_loaded = ranker.load_checkpoint(
            self.adapter_path, device=self.device
        )
        if not projection_loaded:
            logger.warning(
                "Trained weights not found — running zero-shot with random projection. "
                "(Did you run train_llm_ranker.py?)"
            )

        if FastLanguageModel is not None:
            FastLanguageModel.for_inference(ranker.llm)
        else:
            logger.info("Unsloth not installed; using standard HF inference path.")
        ranker.eval()
        self.model = ranker

        # Use Hugging Face's standard get_input_embeddings() safely for base vs LoRA
        self._embed = ranker.llm.get_input_embeddings()
        self._target_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self._no_token_id = self.tokenizer.convert_tokens_to_ids("No")

    @torch.inference_mode()
    def _build_prefix_embeds(self, prefix_encoded, valid_history_ids):
        """Build prefix embeddings with history item vectors injected."""
        model = self.model
        input_ids = prefix_encoded["input_ids"]
        hist_embs = self.item_embs[
            torch.tensor(valid_history_ids, dtype=torch.long, device=self.device)
        ].unsqueeze(0)

        inputs_embeds = self._embed(input_ids)
        projected = model.projection(hist_embs).to(dtype=inputs_embeds.dtype)

        mask = input_ids == model.special_token_id
        cnt = min(mask[0].sum().item(), projected.size(1))
        if cnt > 0:
            indices = mask[0].nonzero(as_tuple=True)[0][:cnt]
            inputs_embeds[0, indices] = projected[0, :cnt]
        return inputs_embeds

    @torch.inference_mode()
    def _score_batch(
        self, cand_ids: list[int], cand_titles: list[str], prefix_embeds
    ) -> list[float]:
        model = self.model
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        suffix_id_list = []
        for cand_title in cand_titles:
            suffix_text = build_candidate_suffix(cand_title, settings.llm_special_token)
            suffix_ids = self.tokenizer(
                suffix_text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)["input_ids"][0]
            suffix_id_list.append(suffix_ids)

        suffix_lengths = torch.tensor(
            [int(x.size(0)) for x in suffix_id_list], device=self.device
        )
        suffix_ids = torch.nn.utils.rnn.pad_sequence(
            suffix_id_list, batch_first=True, padding_value=pad_id
        )
        suffix_embs = self._embed(suffix_ids)

        cand_embs = self.item_embs[
            torch.tensor(cand_ids, dtype=torch.long, device=self.device)
        ]
        cand_proj = model.projection(cand_embs).to(dtype=suffix_embs.dtype)

        suffix_mask = suffix_ids == model.special_token_id
        for i in range(suffix_ids.size(0)):
            idx = suffix_mask[i].nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                suffix_embs[i, idx[0]] = cand_proj[i]

        batch_size = suffix_ids.size(0)
        prefix_batch = prefix_embeds.expand(batch_size, -1, -1)
        full_embeds = torch.cat([prefix_batch, suffix_embs], dim=1)

        out = model.llm(
            inputs_embeds=full_embeds,
            use_cache=False,
        )

        prefix_len = prefix_embeds.size(1)
        last_pos = prefix_len + suffix_lengths - 1
        row_idx = torch.arange(batch_size, device=self.device)
        yes_scores = out.logits[row_idx, last_pos, self._target_token_id]
        no_scores = out.logits[row_idx, last_pos, self._no_token_id]
        scores = yes_scores - no_scores
        return scores.float().cpu().tolist()

    def score(
        self, u_idx: int, history_ids: list[int], candidates: list[int]
    ) -> list[tuple[int, float]]:
        prefix_encoded, valid_history_ids = prepare_prompt_prefix(
            history_ids,
            self.item_map,
            self.movie_mapping,
            self.tokenizer,
            settings.llm_system_prompt_local,
            self.device,
        )
        if prefix_encoded is None:
            return [(cid, -999.0) for cid in candidates]

        prefix_embeds = self._build_prefix_embeds(prefix_encoded, valid_history_ids)

        all_titles, valid_cands = decode_ids_to_titles_and_keep_ids(
            candidates, self.item_map, self.movie_mapping
        )
        title_map = dict(zip(valid_cands, all_titles))

        results: list[tuple[int, float]] = []
        for start in range(0, len(candidates), self.batch_size):
            chunk = candidates[start : start + self.batch_size]

            batch_ids = [cid for cid in chunk if cid in title_map]
            score_map: dict[int, float] = {}
            if batch_ids:
                batch_titles = [title_map[cid] for cid in batch_ids]
                batch_scores = self._score_batch(batch_ids, batch_titles, prefix_embeds)
                score_map = {cid: s for cid, s in zip(batch_ids, batch_scores)}

            for cid in chunk:
                results.append((cid, score_map.get(cid, -999.0)))

        return results


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate LLM-based re-rankers on the two-tower retrieval candidates."
    )
    p.add_argument(
        "--method",
        choices=["api", "local", "both", "retriever"],
        help="Which ranker(s) to evaluate (api/local/both/retriever).",
    )
    p.add_argument("--num_users", type=int, help="Number of test users to evaluate.")
    # API-scorer options
    p.add_argument(
        "--clean_cache",
        action="store_true",
        help="Delete the API response cache before running (API scorer only).",
    )
    # Local-scorer options
    p.add_argument(
        "--model_name",
        type=str,
        help="HuggingFace model name for the local LLM scorer.",
    )
    p.add_argument(
        "--adapter_path",
        type=str,
        help="Path to the LoRA adapter directory.",
    )
    p.add_argument(
        "--local_batch_size",
        type=int,
        help="Batch size for local LLM candidate scoring.",
    )
    p.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to load and apply a LoRA adapter (if False, evaluates base model + projection only).",
    )
    return p.parse_args()


def format_llm_results(
    metrics_by_scorer: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Convert nested scorer metrics to flat format compatible with formatted_results().

    Input: {"ScorerName": {"ScorerName_Recall@10": value, ...}, ...}
    Output: {"ScorerName_Recall@10": value, ...}
    """
    flat_metrics = {}
    for scorer_name, metrics in metrics_by_scorer.items():
        flat_metrics.update(metrics)
    return flat_metrics


def evaluate(
    method: str = "both",
    num_users: int = 100,
    clean_cache: bool = False,
    model_name: str | None = None,
    adapter_path: str | None = None,
    local_batch_size: int = 8,
    use_lora: bool = True,
) -> dict[str, float]:
    runner = LLMEvalRunner(num_users=num_users)

    # retriever is always included as a baseline
    scorers: list[Scorer] = [RetrieverScorer()]
    if method in ("api", "both"):
        scorers.append(APIScorer(clean_cache=clean_cache))
    if method in ("local", "both"):
        scorers.append(
            LocalLLMScorer(
                model_name=model_name or settings.llm_model_name,
                adapter_path=adapter_path,
                batch_size=local_batch_size,
                use_lora=use_lora,
            )
        )

    all_metrics = runner.run(scorers)

    return format_llm_results(all_metrics)


def main() -> dict[str, float]:
    args = _build_args()
    call_args = {
        k: v
        for k, v in {
            "method": args.method,
            "num_users": args.num_users,
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "local_batch_size": args.local_batch_size,
            "use_lora": args.use_lora,
        }.items()
        if v is not None
    }
    # clean_cache is a boolean flag; include it explicitly
    call_args["clean_cache"] = args.clean_cache

    all_metrics = evaluate(**call_args)

    # Format results to DataFrame for display (same as offline_eval)
    print("\n--- LLM Ranker Evaluation Results ---")
    df = format_results_to_dataframe(all_metrics, k_list=settings.evaluation_k_list)

    # Print all columns, 4 at a time
    n_cols = df.shape[1]
    col_chunks = [df.columns[i : i + 4] for i in range(0, n_cols, 4)]
    for chunk in col_chunks:
        print(df[chunk])
        print()

    return all_metrics


if __name__ == "__main__":
    main()
