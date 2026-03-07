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
from dataclasses import dataclass, field
from pathlib import Path

from nanoRecSys.config import settings
from nanoRecSys.training import train_ranker, train_retriever
from nanoRecSys.utils.logging_config import setup_logger
from nanoRecSys.utils.utils import get_vocab_sizes


@dataclass
class TrainingConfig:
    """Configuration for model training (retriever or ranker)."""

    mode: str = "retriever"
    epochs: int = field(default_factory=lambda: settings.epochs)
    batch_size: int = field(default_factory=lambda: settings.batch_size)
    lr: float = field(default_factory=lambda: settings.learning_rate)
    num_workers: int = field(default_factory=lambda: settings.num_workers)
    user_tower_type: str = field(default_factory=lambda: settings.user_tower_type)
    retrieval_threshold: float | None = field(
        default_factory=lambda: settings.retrieval_threshold
    )
    ranker_positive_threshold: float = field(
        default_factory=lambda: settings.ranker_positive_threshold
    )
    ranker_negative_threshold: float | None = field(
        default_factory=lambda: settings.ranker_negative_threshold
    )
    explicit_neg_weight: float = field(
        default_factory=lambda: settings.explicit_neg_weight
    )
    random_neg_ratio: float = field(
        default_factory=lambda: settings.ranker_random_neg_ratio
    )
    ranker_loss_type: str = field(default_factory=lambda: settings.ranker_loss_type)
    ranker_loss_margin: float = field(
        default_factory=lambda: settings.ranker_loss_margin
    )
    id_dropout: float = field(default_factory=lambda: settings.id_dropout)
    temperature: float = field(default_factory=lambda: settings.temperature)
    ckpt_path: str | Path | None = field(default_factory=lambda: settings.ckpt_path)
    wandb_run_name: str | None = None
    item_lr: float = field(default_factory=lambda: settings.item_embedding_lr)
    weight_decay: float = field(default_factory=lambda: settings.weight_decay)
    adam_beta1: float = field(default_factory=lambda: settings.adam_beta1)
    adam_beta2: float = field(default_factory=lambda: settings.adam_beta2)
    adam_eps: float = field(default_factory=lambda: settings.adam_eps)
    use_scheduler: bool = field(default_factory=lambda: settings.use_scheduler)
    warmup_steps: int = field(default_factory=lambda: settings.warmup_steps)
    check_val_every_n_epoch: int = field(
        default_factory=lambda: settings.check_val_every_n_epoch
    )
    build_embeddings: bool = field(default_factory=lambda: settings.build_embeddings)
    max_steps: int = field(default_factory=lambda: settings.max_steps)
    limit_train_batches: float = field(
        default_factory=lambda: settings.limit_train_batches
    )
    enable_progress_bar: bool = True

    # Additional attrs to be set by main() based on user_tower_type
    user_emb_path: Path | None = None
    train_interactions_path: Path | None = None
    train_hard_negatives_path: Path | None = None
    train_random_negatives_path: Path | None = None
    val_interactions_path: Path | None = None
    val_hard_negatives_path: Path | None = None
    val_random_negatives_path: Path | None = None


def optional_float(v):
    if v is None or str(v).lower() == "none":
        return None
    return float(v)


def optional_bool(v):
    if v is None or str(v).lower() == "none":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("Fasle", "false", "f", "0", "no", "n"):
        return False
    if s in ("True", "true", "t", "1", "yes", "y"):
        return True
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def main(config: TrainingConfig | None = None):
    if config is None:
        config = TrainingConfig()

    # Set up logging
    logger = setup_logger()

    # Construct paths based on user_tower_type
    if config.user_tower_type == "transformer":
        suffix = "_sasrec_combined"
        config.user_emb_path = (
            settings.sasrec_user_embs_npy_path / f"user_embeddings{suffix}.npy"
        )
    else:
        suffix = ""
        config.user_emb_path = settings.artifacts_dir / "user_embeddings.npy"

    config.train_interactions_path = (
        settings.processed_data_dir / f"train{suffix}.parquet"
    )
    config.train_hard_negatives_path = (
        settings.processed_data_dir / f"train{suffix}_negatives_hard.parquet"
    )
    config.train_random_negatives_path = (
        settings.processed_data_dir / f"train{suffix}_negatives_random.parquet"
    )

    config.val_interactions_path = settings.processed_data_dir / f"val{suffix}.parquet"
    config.val_hard_negatives_path = (
        settings.processed_data_dir / f"val{suffix}_negatives_hard.parquet"
    )
    config.val_random_negatives_path = (
        settings.processed_data_dir / f"val{suffix}_negatives_random.parquet"
    )

    vocab_sizes = get_vocab_sizes()
    logger.info(f"Vocab sizes: Users={vocab_sizes[0]}, Items={vocab_sizes[1]}")

    if config.mode == "retriever":
        train_retriever(config, vocab_sizes)
    elif config.mode == "ranker":
        train_ranker(config, vocab_sizes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["retriever", "ranker"], default="retriever"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for DataLoader."
    )
    parser.add_argument(
        "--user_tower_type",
        type=str,
        choices=["mlp", "transformer"],
        help="Type of User Tower model: 'mlp' (ID-based) or 'transformer' (sequence-based).",
    )
    parser.add_argument(
        "--retrieval_threshold",
        type=float,
        help="Minimum rating threshold for retrieval dataset loading.",
    )
    parser.add_argument(
        "--ranker_positive_threshold",
        type=float,
        help="Minimum rating to label as positive for ranker training.",
    )
    parser.add_argument(
        "--ranker_negative_threshold",
        type=optional_float,
        help="Maximum rating to label as negative for ranker training. Set to None to disable explicit negatives.",
    )
    parser.add_argument(
        "--explicit_neg_weight",
        type=float,
        help="Weight for explicit negatives in ranker training.",
    )
    parser.add_argument(
        "--random_neg_ratio",
        type=float,
        help="Ratio of random negatives to use in ranker training (0.0=None, 1.0=All).",
    )
    parser.add_argument(
        "--ranker_loss_type",
        type=str,
        choices=["bce", "margin_ranking", "bpr"],
        help="Loss function type for ranker training. Options: 'bce' (default), 'margin_ranking', 'bpr'.",
    )
    parser.add_argument(
        "--ranker_loss_margin",
        type=float,
        help="Margin parameter for MarginRankingLoss (only used if loss_type='margin_ranking').",
    )
    parser.add_argument(
        "--id_dropout",
        type=float,
        help="Probability of dropping ID features during training (Modality Dropout).",
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature for InfoNCE loss."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to checkpoint to resume from, or 'last' to resume from latest. If None, starts training from scratch.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="W&B run name. If None, WandB will auto-generate a name.",
    )
    parser.add_argument(
        "--item_lr",
        type=float,
        help="Learning rate for item embeddings (if trainable).",
    )
    parser.add_argument(
        "--weight_decay", type=float, help="Weight decay for AdamW optimizer."
    )
    parser.add_argument(
        "--adam_beta1", type=float, help="Beta1 parameter for AdamW optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, help="Beta2 parameter for AdamW optimizer."
    )
    parser.add_argument(
        "--adam_eps", type=float, help="Epsilon parameter for AdamW optimizer."
    )
    parser.add_argument(
        "--use_scheduler",
        type=optional_bool,
        help="Whether to use a learning rate scheduler (true/false).",
    )
    parser.add_argument(
        "--warmup_steps", type=int, help="Number of linear warmup steps."
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        help="Perform a validation loop every N training epochs.",
    )
    parser.add_argument(
        "--build_embeddings",
        type=optional_bool,
        help="Generate user and item embeddings into artifacts after training (true/false).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Stop training after this many steps. -1 to disable.",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        help="How much of training dataset to check (float = fraction, int = num_batches).",
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=optional_bool,
        help="Enable progress bar (true/false). Accepts True/False/true/false/1/0/yes/no.",
    )

    args = parser.parse_args()

    # Convert argparse Namespace to dataclass, only overriding provided values
    config_kwargs = {k: v for k, v in vars(args).items() if v is not None}
    config = TrainingConfig(**config_kwargs)

    main(config)
