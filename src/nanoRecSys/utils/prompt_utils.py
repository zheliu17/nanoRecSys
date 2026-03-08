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


"""Shared prompt formatting utilities for LLM ranker training and evaluation."""

import numpy as np

from nanoRecSys.config import settings


def decode_ids_to_titles_and_keep_ids(
    item_ids: list,
    item_map: np.ndarray,
    movie_mapping: dict,
) -> tuple:
    """
    Resolve internal item indices to movie titles, skipping unknowns.

    Args:
        item_ids: Internal item indices
        item_map: Array mapping internal index -> original movieId
        movie_mapping: Pre-built ``{movieId: title}`` dict.

    Returns:
        Tuple of (titles, valid_ids) — parallel lists with only resolvable items.
    """
    if not item_ids:
        return [], []

    # We don't filter anything; item_map is usually 0-indexed
    original_ids = item_map[item_ids]
    pairs = [
        (movie_mapping[mid], iid)
        for mid, iid in zip(original_ids, item_ids)
        if mid in movie_mapping
    ]
    if not pairs:
        return [], []
    titles, valid_ids = zip(*pairs)
    return list(titles), list(valid_ids)


def build_history_string(
    history_ids: list,
    item_map: np.ndarray,
    movie_mapping: dict,
    special_token: str = settings.llm_special_token,
) -> tuple:
    """
    Build a numbered history string from item IDs.

    Returns:
        Tuple of (history_string, valid_history_ids) or (None, None) if empty.
    """
    history_titles, valid_history_ids = decode_ids_to_titles_and_keep_ids(
        history_ids, item_map, movie_mapping
    )

    if not valid_history_ids:
        return None, None

    if settings.llm_causal_prefixing:
        history_str = "\n".join(
            [
                f"{i + 1}. {special_token} {title}"
                for i, title in enumerate(history_titles)
            ]
        )
    else:
        history_str = "\n".join(
            [
                f"{i + 1}. {title} {special_token}"
                for i, title in enumerate(history_titles)
            ]
        )
    return history_str, valid_history_ids


def build_candidate_block(
    cand_title: str,
    special_token: str = settings.llm_special_token,
) -> str:
    if settings.llm_causal_prefixing:
        return f"{special_token} {cand_title}\n\n{settings.llm_candidate_question}"
    else:
        return f"{cand_title} {special_token}\n\n{settings.llm_candidate_question}"


def build_prefix_text(
    history_str: str,
    system_prompt: str,
) -> str:
    """Build the raw ChatML text prefix ending just before the candidate block.

    This is the shared string used by both:

    - :func:`prepare_prompt_prefix` (tokenizes it for KV-cache eval), and
    - :func:`build_sft_example_for_candidate` (concatenates suffix for training).
    """
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nUser Viewing History:\n{history_str}\n\nCandidate Movie: "
    )


def build_sft_example_for_candidate(
    history_str: str,
    cand_id: int,
    item_map: np.ndarray,
    movie_mapping: dict,
    system_prompt: str,
    assistant_response: str,
    special_token: str = settings.llm_special_token,
) -> tuple:
    """Build a complete ChatML SFT training example from the shared prefix + suffix structure."""
    cand_titles, valid_cands = decode_ids_to_titles_and_keep_ids(
        [cand_id], item_map, movie_mapping
    )
    if not valid_cands:
        return None, None
    prefix = build_prefix_text(history_str, system_prompt)
    suffix = build_candidate_suffix(cand_titles[0], special_token)
    prompt = f"{prefix}{suffix}"
    text = f"{prompt}{assistant_response}<|im_end|>"
    return (text, prompt), valid_cands[0]


def build_candidate_suffix(
    cand_title: str,
    special_token: str = settings.llm_special_token,
) -> str:
    """Return the tokenizable suffix appended to the cached prefix during KV-cache eval."""
    candidate_block = build_candidate_block(cand_title, special_token)
    return f"{candidate_block}<|im_end|>\n<|im_start|>assistant\n"


def prepare_prompt_prefix(
    history_ids: list,
    item_map: np.ndarray,
    movie_mapping: dict,
    tokenizer,
    system_prompt: str,
    device,
    special_token: str = settings.llm_special_token,
) -> tuple:
    """
    Tokenize the prompt prefix (system prompt + history) for KV-cache eval.

    The prefix ends just before the candidate block so that :func:`build_candidate_suffix`
    can be appended per-candidate without re-encoding the history.

    Returns:
        Tuple of (prefix_encoded, valid_history_ids) or (None, None) if no valid items.
    """
    history_str, valid_history_ids = build_history_string(
        history_ids, item_map, movie_mapping, special_token
    )

    if history_str is None:
        return None, None

    prefix_encoded = tokenizer(
        build_prefix_text(history_str, system_prompt), return_tensors="pt"
    ).to(device)
    return prefix_encoded, valid_history_ids
