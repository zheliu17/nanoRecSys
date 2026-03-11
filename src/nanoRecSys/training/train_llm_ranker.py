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

from unsloth import is_bfloat16_supported  # isort: skip

import argparse
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from nanoRecSys.config import settings
from nanoRecSys.models.llm_ranker import LLMRanker
from nanoRecSys.utils.logging_config import get_logger
from nanoRecSys.utils.prompt_utils import (
    build_history_string,
    build_sft_example_for_candidate,
)


@dataclass
class LLMTrainingConfig:
    """Configuration for LLM Ranker training."""

    model_name: str = field(default_factory=lambda: settings.llm_model_name)
    n_samples: int = 800_000
    batch_size: int = 64
    dataloader_num_workers: int = 6
    tokenize_num_proc: int | None = (
        None  # Defaults to dataloader_num_workers if not set
    )
    tokenize_batch_size: int = 1000
    tokenize_writer_batch_size: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    warmup_steps: float = 2000
    epochs: int = 1
    save_steps: int | None = 1000
    resume_from_checkpoint: str | bool | None = None
    projection_path: str | None = (
        None  # Optional path to pre-trained projection weights
    )
    report_to: str = "none"
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    positive_sample_ratio: float = 1.0
    random_neg_ratio: float = 1
    data_suffix: str = "llm_ranker"
    use_lora: bool = True


def load_data(sample_size, suffix="llm_ranker"):
    movies_path = settings.raw_data_dir / "movies.csv"
    item_map_path = settings.processed_data_dir / "item_map.npy"

    movies_df = pd.read_csv(movies_path)
    item_map = np.load(item_map_path)

    train_int_df = pd.read_parquet(
        settings.processed_data_dir / f"train_{suffix}.parquet"
    )
    train_hard_df = pd.read_parquet(
        settings.processed_data_dir / f"train_{suffix}_negatives_hard.parquet"
    )
    train_rand_df = pd.read_parquet(
        settings.processed_data_dir / f"train_{suffix}_negatives_random.parquet"
    )

    if sample_size > 0 and len(train_int_df) > sample_size:
        subset = np.random.choice(train_int_df.index, sample_size, replace=False)
        train_int_df = train_int_df.loc[subset].reset_index(drop=True)
        train_hard_df = train_hard_df.loc[subset].reset_index(drop=True)
        train_rand_df = train_rand_df.loc[subset].reset_index(drop=True)

    return (
        train_int_df,
        train_hard_df,
        train_rand_df,
        movies_df,
        item_map,
    )


def generate_sft_examples(
    int_df,
    hard_df,
    rand_df,
    movies_df,
    item_map,
    positive_sample_ratio=1.0,
    random_neg_ratio=0.1,
):
    system_prompt = settings.llm_system_prompt_local

    # Pre-build {movieId: title} dict once to avoid repeated set_index inside the loop
    movie_mapping = movies_df.set_index("movieId")["title"].to_dict()
    rand_cols = [col for col in rand_df.columns if col.startswith("neg_item_idx_")]

    int_rows = int_df.itertuples(index=False, name=None)
    hard_rows = hard_df.itertuples(index=False, name=None)
    rand_rows = rand_df.itertuples(index=False, name=None)

    int_columns = list(int_df.columns)
    hard_columns = list(hard_df.columns)
    rand_columns = list(rand_df.columns)

    history_idx = int_columns.index("history_sequence")
    pos_item_idx = int_columns.index("item_idx")
    hard_neg_idx = hard_columns.index("neg_item_idx")
    rand_neg_indices = [rand_columns.index(col) for col in rand_cols]

    for row, hard_row, rand_row in zip(int_rows, hard_rows, rand_rows):
        history_sequence = row[history_idx]
        pos_item = int(row[pos_item_idx])
        hard_neg = int(hard_row[hard_neg_idx])

        if (
            history_sequence is None
            or len(history_sequence) < settings.llm_min_history_len
        ):
            continue

        # The history sequence from mine_negatives_sasrec is 0-indexed
        history_ids = list(history_sequence)

        history_ids = history_ids[-settings.llm_history_len :]

        history_str, valid_history_ids = build_history_string(
            history_ids, item_map, movie_mapping
        )
        if history_str is None:
            continue

        # Positive
        result, valid_cand_pos = build_sft_example_for_candidate(
            history_str, pos_item, item_map, movie_mapping, system_prompt, "Yes"
        )
        if valid_cand_pos is not None and np.random.rand() < positive_sample_ratio:
            yield {
                "text": result[0],
                "prompt": result[1],
                "item_sequence": valid_history_ids + [valid_cand_pos],
                "target_label": 1,
            }

        # Hard Negative
        result, valid_cand_hard = build_sft_example_for_candidate(
            history_str, hard_neg, item_map, movie_mapping, system_prompt, "No"
        )
        if valid_cand_hard is not None:
            yield {
                "text": result[0],
                "prompt": result[1],
                "item_sequence": valid_history_ids + [valid_cand_hard],
                "target_label": 0,
            }

        # Random Negative
        if np.random.rand() < random_neg_ratio:
            # Load all pre-mined random negatives to prevent false negatives
            # Columns are named neg_item_idx_1, neg_item_idx_2, etc.
            for rand_neg_idx in rand_neg_indices:
                rand_neg = int(rand_row[rand_neg_idx])
                result, valid_cand_rand = build_sft_example_for_candidate(
                    history_str, rand_neg, item_map, movie_mapping, system_prompt, "No"
                )
                if valid_cand_rand is not None:
                    yield {
                        "text": result[0],
                        "prompt": result[1],
                        "item_sequence": valid_history_ids + [valid_cand_rand],
                        "target_label": 0,
                    }


def build_dataset(
    int_df,
    hard_df,
    rand_df,
    movies_df,
    item_map,
    positive_sample_ratio=1.0,
    random_neg_ratio=0.1,
):
    dataset = Dataset.from_generator(
        generate_sft_examples,
        # TODO: cache random dataset generation
        # cache_dir=str(dataset_cache_dir),
        gen_kwargs={
            "int_df": int_df,
            "hard_df": hard_df,
            "rand_df": rand_df,
            "movies_df": movies_df,
            "item_map": item_map,
            "positive_sample_ratio": positive_sample_ratio,
            "random_neg_ratio": random_neg_ratio,
        },
    )

    return dataset


def tokenize_sft_examples(examples, tokenizer, max_seq_length):
    # We tokenize the entire text ("prompt" + "completion string")
    tokenized = tokenizer(examples["text"], truncation=True, max_length=max_seq_length)

    # Generate `completion_mask` for the TRL collator so it can set prompt tokens to -100 in labels
    completion_masks = []
    for prompt, input_ids in zip(examples["prompt"], tokenized["input_ids"]):
        # Tokenize ONLY the prompt to find the index where the completion starts
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # If the tokenizer injected a BOS token into the main input_ids, adjust length
        bos_offset = (
            1 if len(input_ids) > 0 and input_ids[0] == tokenizer.bos_token_id else 0
        )

        p_len = min(len(p_ids) + bos_offset, len(input_ids))

        # 0 = do not train on these (prompt), 1 = train on these (completion)
        mask = [0] * p_len + [1] * (len(input_ids) - p_len)
        completion_masks.append(mask)

    tokenized["completion_mask"] = completion_masks
    return tokenized


class MultimodalDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator that adds item embeddings to the batch."""

    def __init__(self, item_embeddings, tokenizer, *args, **kwargs):
        # Let TRL collator handle completion_only_loss by setting pad_token_id directly
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        # Remove any strict parent kwargs that may conflict
        kwargs.pop("mlm", None)
        super().__init__(
            pad_token_id=pad_id, completion_only_loss=True, *args, **kwargs
        )
        self.item_embeddings = item_embeddings

    def __call__(
        self, examples: list[dict[str, Any]], return_tensors=None
    ) -> dict[str, torch.Tensor]:
        return self.torch_call(examples)

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Process a batch of examples and add item embeddings."""
        item_sequences = [example["item_sequence"] for example in examples]
        max_len = max(len(seq) for seq in item_sequences)
        # Right-pad, item_sequences is 0-indexed
        # ranker will check actual lengths, so the redundant padding shouldn't cause issues
        padded_seqs = [seq + [0] * (max_len - len(seq)) for seq in item_sequences]
        item_targets_tensor = torch.tensor(padded_seqs, dtype=torch.long)

        # Remove item_sequence and any other non-standard keys to let TRL collator process normally
        cleansed_examples = []
        for ex in examples:
            cleansed_examples.append(
                {
                    k: v
                    for k, v in ex.items()
                    if k
                    not in [
                        "item_sequence",
                        "text",
                        "target_label",
                        "prompt",
                        "prompt_len",
                    ]
                }
            )

        # The TRL collator will automatically use `completion_mask` (created in tokenize_fn) -> `labels`
        batch = super().torch_call(cleansed_examples)

        # Add item embeddings back to batch
        batch["sasrec_embs"] = self.item_embeddings[item_targets_tensor]
        return batch


class CustomSFTTrainer(SFTTrainer):
    def save_model(self, output_dir=None, _internal_call=False):
        """Override to use LLMRanker's custom checkpoint saving."""
        if output_dir is None:
            output_dir = self.args.output_dir

        self.model.save_checkpoint(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        logger = get_logger()
        logger.info(f"Loading custom weights from {resume_from_checkpoint}")

        # Load checkpoint via LLMRanker method (handles projection + LoRA adapter)
        try:
            self.model.load_checkpoint(
                resume_from_checkpoint,
                device="cpu",
                strict=False,
            )
        except Exception as e:
            logger.warning(
                f"Could not load checkpoint from {resume_from_checkpoint}: {e}"
            )

    def _load_optimizer_and_scheduler(self, checkpoint):
        logger = get_logger()
        # If we change architectures (e.g. going from freeze-LLM in Stage 1
        # to LoRA-LLM in Stage 2), the optimizer sizes won't match!
        # Thus, we should skip loading the optimizer/scheduler to avoid errors.
        if checkpoint is not None and getattr(
            self.model,
            "use_lora",
            getattr(getattr(self.model, "model", self.model), "use_lora", True),
        ):
            adapter_path = os.path.join(checkpoint, "adapter_model.safetensors")
            adapter_bin_path = os.path.join(checkpoint, "adapter_model.bin")
            # If we're starting stage 2 and loading stage 1, LoRA wasn't saved.
            # Don't load optimizer, treat it like a fresh start to stage 2.
            if not os.path.exists(adapter_path) and not os.path.exists(
                adapter_bin_path
            ):
                logger.info(
                    "Stage 1 -> Stage 2 transition detected. Restarting optimizer/scheduler."
                )
                warnings.warn(
                    "Use projection_path instead of loading from ckpt",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return
        super()._load_optimizer_and_scheduler(checkpoint)


def main(config: LLMTrainingConfig | None = None):
    if config is None:
        config = LLMTrainingConfig()

    logger = get_logger()

    logger.info("Loading LLMRanker (Base model + LoRA + Projection)...")

    i_path = settings.artifacts_dir / "item_embeddings.npy"
    if not i_path.exists():
        raise FileNotFoundError(
            f"Item embeddings not found at {i_path}. "
            "Please run `python src/indexing/build_embeddings.py --mode all` first."
        )

    sasrec_item_embs = torch.from_numpy(np.load(i_path).copy()).float().pin_memory()
    logger.info(f"Loaded Items: {sasrec_item_embs.shape}")
    sasrec_dim = sasrec_item_embs.shape[1]

    model = LLMRanker(
        sasrec_emb_dim=sasrec_dim,
        model_name=config.model_name,
        use_lora=config.use_lora,
    )
    if config.projection_path is not None:
        state_dict = torch.load(config.projection_path, map_location=model.device)
        logger.info(f"Loading projection weights from {config.projection_path}")
        model.projection.load_state_dict(state_dict, strict=True)

    tokenizer = model.tokenizer

    model.print_trainable_parameters()

    logger.info("Loading datasets...")
    train_int, train_hard, train_rand, movies_df, item_map = load_data(
        sample_size=config.n_samples,
        suffix=config.data_suffix,
    )
    dataset = build_dataset(
        train_int,
        train_hard,
        train_rand,
        movies_df,
        item_map,
        positive_sample_ratio=config.positive_sample_ratio,
        random_neg_ratio=config.random_neg_ratio,
    )

    tokenize_num_proc = config.tokenize_num_proc
    if tokenize_num_proc is None:
        tokenize_num_proc = config.dataloader_num_workers

    logger.info(
        "Preparing SFT dataset with tokenize_num_proc=%d, tokenize_batch_size=%d, tokenize_writer_batch_size=%d",
        tokenize_num_proc,
        config.tokenize_batch_size,
        config.tokenize_writer_batch_size,
    )

    dataset = dataset.map(
        tokenize_sft_examples,
        batched=True,
        batch_size=config.tokenize_batch_size,
        remove_columns=["text", "prompt"],
        writer_batch_size=config.tokenize_writer_batch_size,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": settings.llm_max_seq_length,
        },
        num_proc=tokenize_num_proc,
        desc="Tokenizing SFT dataset",
    )

    logger.info(f"Generated {len(dataset)} SFT training samples.")

    collator = MultimodalDataCollator(
        item_embeddings=sasrec_item_embs,
        tokenizer=tokenizer,
    )

    output_dir = str(settings.llm_output_dir)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs,
        logging_steps=10,
        save_strategy="steps",
        save_steps=config.save_steps,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        remove_unused_columns=False,
        report_to=config.report_to,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=settings.pin_memory,
        # We handle it manually; Keep this only for consistency
        completion_only_loss=True,  # Train on completion tokens only
    )

    trainer = CustomSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=settings.llm_max_seq_length,
        data_collator=collator,
        args=training_args,
    )

    logger.info("Starting SFT...")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    if config.use_lora:
        trainer.model.llm.save_pretrained(output_dir)
        logger.info(f"Saved LoRA to {output_dir}")

    torch.save(
        trainer.model.projection.state_dict(),
        os.path.join(output_dir, "projection.pth"),
    )
    logger.info(f"Saved Projection to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dataloader_num_workers", type=int)
    parser.add_argument("--tokenize_num_proc", type=int)
    parser.add_argument("--tokenize_batch_size", type=int)
    parser.add_argument("--tokenize_writer_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--projection_path", type=str)
    parser.add_argument("--report_to", type=str)
    parser.add_argument("--optim", type=str)
    parser.add_argument("--lr_scheduler_type", type=str)
    parser.add_argument("--positive_sample_ratio", type=float)
    parser.add_argument("--random_neg_ratio", type=float)
    parser.add_argument("--data_suffix", type=str)
    parser.add_argument(
        "--use_lora", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    config_kwargs = {k: v for k, v in vars(args).items() if v is not None}
    config = LLMTrainingConfig(**config_kwargs)

    main(config)
