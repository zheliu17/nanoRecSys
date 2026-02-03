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
from nanoRecSys.config import settings
from nanoRecSys.training import train_retriever, train_ranker
from nanoRecSys.utils.utils import get_vocab_sizes
from nanoRecSys.utils.logging_config import setup_logger


def optional_float(v):
    if v is None or str(v).lower() == "none":
        return None
    return float(v)


def main(args=None):
    # Set up logging
    logger = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["retriever", "ranker"], default="retriever"
    )
    parser.add_argument("--epochs", type=int, default=settings.epochs)
    parser.add_argument("--batch_size", type=int, default=settings.batch_size)
    parser.add_argument("--lr", type=float, default=settings.learning_rate)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=settings.num_workers,
        help="Number of workers for DataLoader.",
    )
    parser.add_argument(
        "--retrieval_threshold",
        type=float,
        default=settings.retrieval_threshold,
        help="Minimum rating threshold for retrieval dataset loading.",
    )
    parser.add_argument(
        "--ranker_positive_threshold",
        type=float,
        default=settings.ranker_positive_threshold,
        help="Minimum rating to label as positive for ranker training.",
    )
    parser.add_argument(
        "--ranker_negative_threshold",
        type=optional_float,
        default=settings.ranker_negative_threshold,
        help="Maximum rating to label as negative for ranker training. Set to None to disable explicit negatives.",
    )
    parser.add_argument(
        "--explicit_neg_weight",
        type=float,
        default=settings.explicit_neg_weight,
        help="Weight for explicit negatives in ranker training.",
    )
    parser.add_argument(
        "--random_neg_ratio",
        type=float,
        default=settings.ranker_random_neg_ratio,
        help="Ratio of random negatives to use in ranker training (0.0=None, 1.0=All).",
    )
    parser.add_argument(
        "--ranker_loss_type",
        type=str,
        default=settings.ranker_loss_type,
        choices=["bce", "margin_ranking", "bpr"],
        help="Loss function type for ranker training. Options: 'bce' (default), 'margin_ranking', 'bpr'.",
    )
    parser.add_argument(
        "--ranker_loss_margin",
        type=float,
        default=settings.ranker_loss_margin,
        help="Margin parameter for MarginRankingLoss (only used if loss_type='margin_ranking').",
    )
    parser.add_argument(
        "--id_dropout",
        type=float,
        default=settings.id_dropout,
        help="Probability of dropping ID features during training (Modality Dropout).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=settings.temperature,
        help="Temperature for InfoNCE loss.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=settings.ckpt_path,
        help="Path to checkpoint to resume from, or 'last' to resume from latest. If None, starts training from scratch.",
    )
    parser.add_argument(
        "--item_lr",
        type=float,
        default=settings.item_embedding_lr,
        help="Learning rate for item embeddings (if trainable).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=settings.weight_decay,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=settings.adam_beta1,
        help="Beta1 parameter for AdamW optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=settings.adam_beta2,
        help="Beta2 parameter for AdamW optimizer.",
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=settings.adam_eps,
        help="Epsilon parameter for AdamW optimizer.",
    )
    parser.add_argument(
        "--use_scheduler",
        type=bool,
        default=settings.use_scheduler,
        help="Whether to use a learning rate scheduler.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=settings.warmup_steps,
        help="Number of linear warmup steps.",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=settings.check_val_every_n_epoch,
        help="Perform a validation loop every N training epochs.",
    )
    parser.add_argument(
        "--build_embeddings",
        type=bool,
        default=settings.build_embeddings,
        help="Generate user and item embeddings into artifacts after training.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=settings.max_steps,
        help="Stop training after this many steps. -1 to disable.",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=settings.limit_train_batches,
        help="How much of training dataset to check (float = fraction, int = num_batches).",
    )

    if args is None:
        args = parser.parse_args()
    else:
        default_args = parser.parse_args([])
        provided_dict = {k: v for k, v in vars(args).items() if not k.startswith("__")}

        final_dict = vars(default_args)
        final_dict.update(provided_dict)
        args = argparse.Namespace(**final_dict)

    vocab_sizes = get_vocab_sizes()
    logger.info(f"Vocab sizes: Users={vocab_sizes[0]}, Items={vocab_sizes[1]}")

    if args.mode == "retriever":
        train_retriever(args, vocab_sizes)
    elif args.mode == "ranker":
        train_ranker(args, vocab_sizes)


if __name__ == "__main__":
    main()
