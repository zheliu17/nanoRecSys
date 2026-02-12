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

"""Ranker model training module."""

import os
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torchmetrics.classification import BinaryAUROC

from nanoRecSys.config import settings
from nanoRecSys.data.datasets import (
    RankerEvalDataset,
    RankerTrainDataset,
    load_item_metadata,
)
from nanoRecSys.models.losses import get_ranker_loss
from nanoRecSys.models.ranker import MLPRanker
from nanoRecSys.utils.logging_config import get_logger
from nanoRecSys.utils.utils import (
    OnDemandEmbeddings,
    collate_fn_with_embeddings,
    compute_item_probabilities,
    get_linear_warmup_scheduler,
)


def create_val_dataloader(
    dataset, batch_size, collate_fn, num_workers, pin_memory, persistent_workers
):
    """Helper function to create validation dataloader with consistent configuration."""
    val_sampler = SequentialSampler(dataset)
    val_batch_sampler = BatchSampler(
        val_sampler, batch_size=batch_size, drop_last=False
    )
    return DataLoader(
        dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


class RankerPL(pl.LightningModule):
    def __init__(
        self,
        embed_dim,
        # Metadata Assets
        genre_matrix,  # (N_items, n_genres)
        year_indices,  # (N_items,)
        popularity,  # (N_items,)
        pop_mean,
        pop_std,
        num_genres,
        num_years,
        # Hyperparams
        optimizer_params,
        ranker_params,
        pretrained_item_embeddings,
    ):
        super().__init__()

        self.optimizer_params = optimizer_params
        self.ranker_params = ranker_params
        self.item_lr = self.optimizer_params.get("item_lr")

        # Item embeddings: Trainable Parameter OR Frozen Buffer
        if self.item_lr > 0:
            # We wrap in nn.Parameter to ensure it's treated as a model parameter that needs gradients
            self.item_embeddings = torch.nn.Parameter(pretrained_item_embeddings)
        else:
            # Register as buffer (not a parameter, no gradients)
            self.register_buffer("item_embeddings", pretrained_item_embeddings)

        # Register metadata buffers
        self.register_buffer("genre_matrix", genre_matrix)
        self.register_buffer("year_indices", year_indices)
        self.register_buffer("popularity", popularity)
        # Register popularity normalization stats
        self.register_buffer("pop_mean", pop_mean)
        self.register_buffer("pop_std", pop_std)

        self.model = MLPRanker(
            input_dim=embed_dim,
            hidden_dims=settings.ranker_hidden_dims,
            num_genres=num_genres,
            num_years=num_years,
            genre_dim=16,
            year_dim=8,
        )
        # Create loss function based on loss_type
        self.criterion = get_ranker_loss(
            loss_type=ranker_params.get("loss_type"),
            margin=ranker_params.get("loss_margin"),
            reduction="none",
        )
        self.loss_type = ranker_params.get("loss_type")
        self.id_dropout_prob = ranker_params.get("id_dropout_prob")

        # Metrics
        self.val_auc_explicit = BinaryAUROC()
        self.val_auc_hard = BinaryAUROC()
        self.val_auc_random = BinaryAUROC()

        # Save params but ignore large objects/tensors
        self.save_hyperparameters(
            ignore=[
                "genre_matrix",
                "year_indices",
                "popularity",
                "pretrained_item_embeddings",
            ]
        )

    def forward(self, users, items):
        # items is the item indices
        u_emb = users  # Already loaded by collate_fn
        i_emb = self.item_embeddings[items]  # type: ignore

        # Look up metadata
        g_mat = self.genre_matrix[items]  # type: ignore # (B, NumGenres)
        y_idx = self.year_indices[items]  # type: ignore # (B,)

        # Normalize Popularity
        pop = self.popularity[items]  # type: ignore # (B,)
        pop = (pop - self.pop_mean) / (self.pop_std + 1e-6)  # type: ignore

        preds = self.model(
            user_emb=u_emb,
            item_emb=i_emb,
            genre_multihot=g_mat,
            year_idx=y_idx,
            popularity=pop,
            id_dropout_prob=self.id_dropout_prob,
        )
        return preds

    def training_step(self, batch, batch_idx):
        users, items, labels, weights = batch
        logits = self(users, items)

        # Handle different loss types
        if self.loss_type == "margin_ranking":
            # MarginRankingLoss expects target as +1 or -1
            targets = 2 * labels.float() - 1  # Convert 0/1 to -1/+1
            loss_unreduced = self.criterion(logits, targets)
            loss = (loss_unreduced * weights).mean()
        elif self.loss_type == "bpr":
            # BPRLoss handles logits and labels directly
            loss = self.criterion(logits, labels.float(), weights)
        else:
            # BCEWithLogitsLoss expects labels as 0/1
            loss_unreduced = self.criterion(logits, labels.float())
            loss = (loss_unreduced * weights).mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        users, items, labels = batch
        logits = self(users, items)
        probs = torch.sigmoid(logits)

        # Calculate loss
        if self.loss_type == "margin_ranking":
            # MarginRankingLoss expects target as +1 or -1
            targets = 2 * labels.float() - 1  # Convert 0/1 to -1/+1
            loss = self.criterion(logits, targets).mean()
        elif self.loss_type == "bpr":
            # BPRLoss
            loss = self.criterion(logits, labels.float())
        else:
            # BCEWithLogitsLoss
            loss = self.criterion(logits, labels.float()).mean()

        if dataloader_idx == 0:
            self.val_auc_hard(probs, labels)
            self.log(
                "val_auc_hard",
                self.val_auc_hard,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
            self.log(
                "val_loss_hard",
                loss,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
        elif dataloader_idx == 1:
            self.val_auc_random(probs, labels)
            self.log(
                "val_auc_random",
                self.val_auc_random,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
            self.log(
                "val_loss_random",
                loss,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
        elif dataloader_idx == 2:
            self.val_auc_explicit(probs, labels)
            self.log(
                "val_auc_explicit",
                self.val_auc_explicit,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )
            self.log(
                "val_loss_explicit",
                loss,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )

    def configure_optimizers(self):  # type: ignore
        # Separate parameter groups
        # 1. Item Embeddings (custom LR)
        # 2. Rest (model + other params) (default LR)

        # Note: self.parameters() includes self.item_embeddings because it is nn.Parameter and assigned to self.
        # We need to filter it out from the main group.

        main_params = []
        embedding_params = []

        for name, param in self.named_parameters():
            if name == "item_embeddings":
                embedding_params.append(param)
            else:
                main_params.append(param)

        optimizer_groups = [
            {"params": main_params, "lr": self.optimizer_params.get("lr")}
        ]
        if len(embedding_params) > 0:
            optimizer_groups.append(
                {"params": embedding_params, "lr": self.optimizer_params.get("item_lr")}
            )

        optimizer = optim.AdamW(
            optimizer_groups,
            weight_decay=self.optimizer_params.get("weight_decay"),
            betas=(
                self.optimizer_params.get("adam_beta1"),
                self.optimizer_params.get("adam_beta2"),
            ),
            eps=self.optimizer_params.get("adam_eps"),
        )

        if self.optimizer_params.get("use_scheduler", False):
            # Note: LambdaLR applies to all groups or list of lambdas.
            # If one lambda, it applies to all.
            scheduler = get_linear_warmup_scheduler(
                optimizer, self.optimizer_params.get("warmup_steps", 0)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer


def train_ranker(args, vocab_sizes):
    logger = get_logger()
    num_workers = getattr(args, "num_workers", 0)
    n_users, n_items = vocab_sizes
    logger.info(f"Initializing Ranker Training (Users: {n_users}, Items: {n_items})")

    # 1. Load Embeddings from Disk
    logger.info("Checking for pre-computed user/item embeddings...")
    user_emb_path = getattr(args, "user_emb_path")
    item_emb_path = settings.artifacts_dir / "item_embeddings.npy"

    # Check User Embeddings
    if not user_emb_path.exists():
        logger.error(f"User embeddings not found at {user_emb_path}")
        logger.error(
            "Please generate embeddings first using: python src/indexing/build_embeddings.py --mode users"
        )
        return

    # Check Item Embeddings
    if not item_emb_path.exists():
        logger.error(f"Item embeddings not found at {item_emb_path}")
        logger.error(
            "Please generate embeddings first using: python src/indexing/build_embeddings.py --mode items"
        )
        return

    logger.info("Loading embeddings from disk...")
    # Load item embeddings fully (they're smaller and needed for scoring)
    # User embeddings will be loaded on-demand in DataLoader collate function
    try:
        item_embeddings = torch.from_numpy(np.load(item_emb_path))
        user_path_str = str(user_emb_path)
        try:
            size_bytes = os.path.getsize(user_path_str)
        except Exception:
            size_bytes = None

        LOAD_THRESHOLD = 8 * 1024**3  # 8 GB

        if size_bytes is not None and size_bytes <= LOAD_THRESHOLD:
            arr = np.load(user_path_str, mmap_mode=None)
            user_embeddings = torch.from_numpy(arr).float()
            try:
                user_embeddings.share_memory_()
            except Exception:
                # share_memory_ may fail on some platforms/configs; ignore and continue
                pass
            logger.info(
                f"User Embeddings (in-memory, shared): {user_embeddings.shape}, Item Embeddings: {item_embeddings.shape}"
            )
        else:
            # This won't crash. But will be extremely slow if accessed randomly.
            user_embeddings = OnDemandEmbeddings(user_path_str)
            logger.info(
                f"User Embeddings (on-demand): {user_embeddings.shape}, Item Embeddings: {item_embeddings.shape}"
            )
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return

    # 3. Build/Load Metadata
    item_map_path = settings.processed_data_dir / "item_map.npy"
    movies_path = settings.raw_data_dir / "movies.csv"

    logger.info("Loading Item Metadata (Genres + Years)...")
    genre_matrix, year_indices, num_genres, num_years = load_item_metadata(
        item_map_path, movies_path, cache_dir=str(settings.processed_data_dir)
    )

    logger.info("Computing Item Popularity...")
    # Use log probability for better numerical stability in neural nets
    item_popularity = compute_item_probabilities(
        n_items, return_log_probs=True, device="cpu"
    )

    # 4. Data
    logger.info("Loading training data for Ranker...")
    train_dataset = RankerTrainDataset(
        interactions_path=getattr(args, "train_interactions_path"),
        hard_neg_path=getattr(args, "train_hard_negatives_path"),
        random_neg_path=getattr(args, "train_random_negatives_path"),
        pos_threshold=args.ranker_positive_threshold,
        neg_threshold=args.ranker_negative_threshold,
        explicit_neg_weight=getattr(
            args, "explicit_neg_weight", settings.explicit_neg_weight
        ),
        random_neg_ratio=getattr(
            args, "random_neg_ratio", settings.ranker_random_neg_ratio
        ),
    )

    # Use BatchSampler for efficient numpy slicing
    train_sampler = RandomSampler(train_dataset)
    train_batch_sampler = BatchSampler(
        train_sampler, batch_size=args.batch_size, drop_last=True
    )

    # Create collate function that loads embeddings on-demand
    collate_fn_train = partial(
        collate_fn_with_embeddings, user_embeddings=user_embeddings
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn_train,
        # batch_size=None, # batch_sampler implies batch_size=None
        num_workers=num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # Val Datasets
    val_dataloaders = []
    collate_fn_val = partial(
        collate_fn_with_embeddings, user_embeddings=user_embeddings
    )

    # 1. Hard (Idx 0)
    val_dataset_hard = RankerEvalDataset(
        interactions_path=getattr(args, "val_interactions_path"),
        negatives_path=getattr(args, "val_hard_negatives_path"),
        mode="hard",
        pos_threshold=args.ranker_positive_threshold,
    )
    val_loader_hard = create_val_dataloader(
        val_dataset_hard,
        args.batch_size,
        collate_fn_val,
        num_workers,
        settings.pin_memory,
        num_workers > 0,
    )
    val_dataloaders.append(val_loader_hard)

    # 2. Random (Idx 1)
    val_dataset_random = RankerEvalDataset(
        interactions_path=getattr(args, "val_interactions_path"),
        negatives_path=getattr(args, "val_random_negatives_path"),
        mode="random",
        pos_threshold=args.ranker_positive_threshold,
    )
    val_loader_random = create_val_dataloader(
        val_dataset_random,
        args.batch_size,
        collate_fn_val,
        num_workers,
        settings.pin_memory,
        num_workers > 0,
    )
    val_dataloaders.append(val_loader_random)

    # 3. Explicit (Idx 2 - Optional)
    if (
        args.ranker_negative_threshold is not None
        and args.ranker_negative_threshold > 0
    ):
        val_dataset_explicit = RankerEvalDataset(
            interactions_path=getattr(args, "val_interactions_path"),
            negatives_path=None,
            mode="explicit",
            pos_threshold=args.ranker_positive_threshold,
            neg_threshold=args.ranker_negative_threshold,
        )
        val_loader_explicit = create_val_dataloader(
            val_dataset_explicit,
            args.batch_size,
            collate_fn_val,
            num_workers,
            settings.pin_memory,
            num_workers > 0,
        )
        val_dataloaders.append(val_loader_explicit)

    # 4. Model
    # Normalize Popularity Stats Calculation
    pop_mean = item_popularity.mean()
    pop_std = item_popularity.std()
    logger.info(f"Popularity Stats -- Mean: {pop_mean:.4f}, Std: {pop_std:.4f}")

    if args.item_lr > 0:
        logger.info(f"Item Embeddings will be FINE-TUNED (LR={args.item_lr}).")
    else:
        logger.info("Item Embeddings will be FROZEN (LR=0).")

    model = RankerPL(
        embed_dim=settings.tower_out_dim,
        genre_matrix=genre_matrix,
        year_indices=year_indices,
        popularity=item_popularity,
        pop_mean=pop_mean,
        pop_std=pop_std,
        num_genres=num_genres,
        num_years=num_years,
        optimizer_params={
            "lr": args.lr,
            "item_lr": args.item_lr,
            "weight_decay": args.weight_decay,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_eps": args.adam_eps,
            "use_scheduler": args.use_scheduler,
            "warmup_steps": args.warmup_steps,
        },
        ranker_params={
            "id_dropout_prob": args.id_dropout,
            "loss_type": args.ranker_loss_type,
            "loss_margin": args.ranker_loss_margin,
        },
        pretrained_item_embeddings=item_embeddings,
    )

    # 5. Trainer
    wandb_logger = WandbLogger(
        project="nanoRecSys",
        name=getattr(args, "wandb_run_name", None) or "ranker_run",
        config=vars(args),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.artifacts_dir,
        filename="ranker-{epoch:02d}-{val_auc_hard:.4f}",
        save_top_k=1,
        monitor="val_auc_hard",
        mode="max",
        save_last=False,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    # https://github.com/Lightning-AI/pytorch-lightning/issues/19325
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=settings.artifacts_dir,
        filename="latest",
        save_top_k=1,
        save_last=True,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=getattr(args, "max_steps", -1),
        limit_train_batches=getattr(args, "limit_train_batches", 1.0),
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, last_checkpoint_callback],
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_progress_bar=args.enable_progress_bar,
    )

    ckpt_path = getattr(args, "ckpt_path", None)
    trainer.fit(
        model,
        train_loader,
        val_dataloaders,
        ckpt_path=ckpt_path,
    )
    wandb.finish()

    # Save Inner Model (MLP)
    torch.save(model.model.state_dict(), settings.artifacts_dir / "ranker_model.pth")

    # Save Fine-Tuned Item Embeddings
    torch.save(
        model.item_embeddings.detach().cpu(),
        settings.artifacts_dir / "ranker_item_embeddings.pt",
    )

    # Save Popularity Stats
    pop_stats = {"mean": model.pop_mean.cpu().item(), "std": model.pop_std.cpu().item()}  # type: ignore
    torch.save(pop_stats, settings.artifacts_dir / "ranker_pop_stats.pt")
    logger.info(
        "Saved ranker_model.pth, ranker_item_embeddings.pt, ranker_pop_stats.pt"
    )
