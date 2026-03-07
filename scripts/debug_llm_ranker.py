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

import argparse
import random

import numpy as np
import pandas as pd
import torch

from nanoRecSys.config import settings
from nanoRecSys.eval.eval_llm_rankers import LocalLLMScorer
from nanoRecSys.utils.logging_config import get_logger
from nanoRecSys.utils.prompt_utils import (
    build_candidate_suffix,
    decode_ids_to_titles_and_keep_ids,
    prepare_prompt_prefix,
)


class MinimalRunnerMock:
    """Mock to satisfy LocalLLMScorer's data requirements without loading Faiss, metrics, full evaluator, etc."""

    def __init__(self, item_map, movies_df, device, item_embs):
        self.item_map = item_map
        self.movies_df = movies_df

        class MockEvaluator:
            pass

        self.evaluator = MockEvaluator()
        self.evaluator.device = device  # type: ignore
        self.evaluator.item_embs = item_embs  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description="Lightning Fast Minimal Debug Local LLM Ranker"
    )
    parser.add_argument(
        "--user-id", type=int, required=True, help="Raw user ID to debug"
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=1,
        help="Number of negative samples to evaluate",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--use-lora",
        dest="use_lora",
        action="store_true",
        help="Enable LoRA adapter (default).",
    )
    g.add_argument(
        "--no-lora",
        dest="use_lora",
        action="store_false",
        help="Disable LoRA adapter.",
    )
    parser.set_defaults(use_lora=True)

    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Path to LoRA adapter directory (overrides default).",
    )
    args = parser.parse_args()

    logger = get_logger()

    logger.info("1. Loading minimal data slices directly...")
    user_map_arr = np.load(settings.processed_data_dir / "user_map.npy")
    item_map_arr = np.load(settings.processed_data_dir / "item_map.npy")

    try:
        u_idx = np.where(user_map_arr == args.user_id)[0][0]
    except IndexError:
        logger.error(f"User ID {args.user_id} not found in user_map.")
        return

    logger.info(f"   User {args.user_id} mapped to internal user_idx {u_idx}")

    movies_df = (
        pd.read_csv(settings.raw_data_dir / "movies.csv")
        .drop_duplicates("movieId")
        .set_index("movieId")
    )

    seq_user_ids = np.load(settings.processed_data_dir / "seq_test_user_ids.npy")
    try:
        seq_row_idx = np.where(seq_user_ids == u_idx)[0][0]
    except IndexError:
        logger.error(f"User {args.user_id} (idx {u_idx}) not found in test sequences.")
        return

    test_seqs = np.load(settings.processed_data_dir / "seq_test_sequences.npy")
    user_seq = test_seqs[seq_row_idx, :-1]  # drop target item
    history_ids = (user_seq[user_seq != 0] - 1).tolist()

    if not history_ids:
        logger.info(f"User {args.user_id} has empty history.")
        return

    test_df = pd.read_parquet(settings.processed_data_dir / "test.parquet")
    gt_item_idxs = test_df[test_df["user_idx"] == u_idx]["item_idx"].tolist()

    if not gt_item_idxs:
        logger.info(
            f"No Ground Truth found for user {args.user_id} in the test set. Cannot evaluate positive sample."
        )
        return

    pos_item_idx = gt_item_idxs[0]

    # Get negative items
    candidates = []
    total_items = len(item_map_arr)
    history_set = set(history_ids)
    gt_set = set(gt_item_idxs)

    while len(candidates) < args.num_negatives:
        neg_cand = random.randint(1, total_items - 1)
        if (
            neg_cand not in history_set
            and neg_cand not in gt_set
            and neg_cand not in candidates
        ):
            candidates.append(neg_cand)

    eval_candidates = [pos_item_idx] + candidates

    logger.info("2. Initializing LocalLLMScorer with minimal mock dependencies...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("   Loading pre-computed item_embeddings.npy directly...")
    item_embs = torch.tensor(
        np.load(settings.artifacts_dir / "item_embeddings.npy"), dtype=torch.float32
    )

    runner_mock = MinimalRunnerMock(item_map_arr, movies_df, device, item_embs)
    scorer = LocalLLMScorer(adapter_path=args.adapter_path, use_lora=args.use_lora)
    scorer.setup(runner_mock)  # type: ignore

    history_ids_eval = history_ids[-settings.llm_history_len :]

    logger.info("3. Preparing prefix embeddings...")

    prefix_encoded, valid_history_ids = prepare_prompt_prefix(
        history_ids_eval,
        runner_mock.item_map,
        runner_mock.movies_df,
        scorer.tokenizer,
        settings.llm_system_prompt_local,
        scorer.device,
    )

    if prefix_encoded is None:
        logger.error("Failed to build prefix.")
        return

    prefix_embeds = scorer._build_prefix_embeds(prefix_encoded, valid_history_ids)

    logger.info(f"Prefix tokens shape: {prefix_encoded['input_ids'].shape}")
    logger.info(f"Prefix embeddings shape: {prefix_embeds.shape}")
    logger.info(
        f"Prefix text:\n{scorer.tokenizer.decode(prefix_encoded['input_ids'][0][:])}"
    )

    logger.info("4. Explicit forward pass per candidate...")

    score_map = {}
    target_token_str = scorer.tokenizer.decode([scorer._target_token_id]).strip()
    no_token_str = scorer.tokenizer.decode([scorer._no_token_id]).strip()

    for idx, cid in enumerate(eval_candidates):
        is_pos = idx == 0
        label = "POSITIVE" if is_pos else "NEGATIVE"
        titles, _ = decode_ids_to_titles_and_keep_ids(
            [cid], runner_mock.item_map, runner_mock.movies_df
        )
        cand_title = titles[0] if titles else str(cid)

        logger.info(f"[{label}] Candidate: {cand_title} (ID: {cid})")
        logger.info("-" * 50)

        cand_suffix_text = build_candidate_suffix(
            cand_title, settings.llm_special_token
        )
        logger.info(f"Suffix text:\n{cand_suffix_text}")

        suffix_encoded = scorer.tokenizer(
            cand_suffix_text, return_tensors="pt", add_special_tokens=False
        ).to(scorer.device)
        suffix_ids = suffix_encoded["input_ids"]
        logger.info(f"Suffix tokens shape: {suffix_ids.shape}")

        with torch.inference_mode():
            suffix_embs = scorer._embed(suffix_ids).clone()

            cand_emb = scorer.item_embs[
                torch.tensor([cid], dtype=torch.long, device=scorer.device)
            ]
            cand_proj = scorer.model.projection(cand_emb).to(dtype=suffix_embs.dtype)

            special_token_id = scorer.model.special_token_id
            special_idx = (suffix_ids == special_token_id).nonzero(as_tuple=True)[1]

            if special_idx.numel() > 0:
                pos = special_idx[0].item()
                suffix_embs[0, pos] = cand_proj[0]
                logger.info(
                    f"-> Injected item projection at suffix explicit relative index: {pos} (Token ID: {special_token_id})"
                )
            else:
                logger.warning("-> WARNING: Special token not found in suffix!")

            full_embeds = torch.cat([prefix_embeds, suffix_embs], dim=1)

        logger.info(f"-> Concatenated full input embeddings shape: {full_embeds.shape}")

        with torch.inference_mode():
            out = scorer.model.llm(inputs_embeds=full_embeds, use_cache=False)

        last_pos = full_embeds.shape[1] - 1
        logits_vec = out.logits[0, last_pos]

        # Top-5 tokens at final position
        topk_vals, topk_ids = torch.topk(logits_vec, k=5)
        topk_vals = topk_vals.cpu().tolist()
        topk_ids = topk_ids.cpu().tolist()
        topk_tokens = [scorer.tokenizer.decode([tid]).strip() for tid in topk_ids]
        logger.info("-> Top-5 tokens at final sequence position:")
        for tok, val, tid in zip(topk_tokens, topk_vals, topk_ids):
            logger.info(f"   '{tok}' (ID: {tid}) -> Logit: {val:.4f}")

        yes_score = logits_vec[scorer._target_token_id].item()
        no_score = logits_vec[scorer._no_token_id].item()
        score = yes_score - no_score
        logger.info(
            f"-> Logits at final sequence position: '{target_token_str}' ({yes_score:.4f}), '{no_token_str}' ({no_score:.4f})"
        )
        logger.info(f"-> Final Score (Yes - No): {score:.4f}")
        score_map[cid] = score

    logger.info("5. Final summary")

    for idx, cid in enumerate(eval_candidates):
        is_pos = idx == 0
        label = "POS" if is_pos else "NEG"
        titles, _ = decode_ids_to_titles_and_keep_ids(
            [cid], runner_mock.item_map, runner_mock.movies_df
        )
        cand_title = titles[0] if titles else str(cid)
        score = score_map.get(cid, -999.0)

        logger.info(f"[{label}] {cand_title} (ID: {cid}) -> Final Score: {score:.4f}")


if __name__ == "__main__":
    main()
