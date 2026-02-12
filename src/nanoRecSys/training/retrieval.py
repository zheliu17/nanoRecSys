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

"""Retrieval model training module."""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from nanoRecSys.config import settings
from nanoRecSys.data.datasets import (
    InteractionsDataset,
    SequentialDataset,
)
from nanoRecSys.eval.metrics import recall_at_k
from nanoRecSys.models.losses import DCLLoss, InfoNCELoss
from nanoRecSys.models.towers import (
    ItemTower,
    TransformerUserTower,
    UserTower,
)
from nanoRecSys.utils.logging_config import get_logger
from nanoRecSys.utils.utils import (
    collate_fn_numpy_to_tensor,
    compute_item_probabilities,
    compute_seq_item_probabilities,
    get_linear_warmup_scheduler,
)


class RetrievalPL(pl.LightningModule):
    def __init__(
        self,
        n_users,
        n_items,
        embed_dim,  # MLP tower initial embedding dimension
        output_dim,  # Tower output dimension
        hidden_dims,
        use_projection,
        optimizer_params,
        item_probs,
        temperature,
        learnable_temperature,
        val_users,
        val_ground_truth,
        user_tower_type,
        transformer_params,
        negative_sampling_strategy,
        num_negatives,
        loss_type,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["item_probs", "val_ground_truth", "val_users"]
        )
        if item_probs is not None:
            self.register_buffer("item_probs", item_probs, persistent=False)

        self.val_users = val_users
        self.val_ground_truth = val_ground_truth
        self.user_tower_type = user_tower_type
        self.optimizer_params = optimizer_params
        self.negative_sampling_strategy = negative_sampling_strategy
        self.num_negatives = num_negatives
        self.loss_type = loss_type

        self.learnable_temperature = learnable_temperature
        if self.learnable_temperature:
            self.log_temperature = torch.nn.Parameter(torch.tensor(np.log(temperature)))
            self.temperature = None  # Not used directly if learnable
        else:
            self.temperature = temperature

        self.item_tower = ItemTower(
            n_items,
            embed_dim=embed_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            use_projection=use_projection,
        )

        if user_tower_type == "transformer":
            self.user_tower = TransformerUserTower(
                vocab_size=n_items,
                # We share the same embedding dimension as item tower
                # input dim here = tower output dim
                embed_dim=output_dim,
                output_dim=output_dim,
                max_seq_len=transformer_params.get("max_seq_len"),
                n_heads=transformer_params.get("n_heads"),
                n_layers=transformer_params.get("n_layers"),
                dropout=transformer_params.get("dropout"),
                swiglu_hidden_dim=transformer_params.get("swiglu_hidden_dim"),
                shared_embedding=self.item_tower,
            )
        elif user_tower_type == "mlp":
            self.user_tower = UserTower(
                n_users,
                embed_dim=embed_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
            )
        else:
            raise ValueError(f"Unknown user_tower_type: {user_tower_type}")

        if self.loss_type == "info_nce":
            self.criterion = InfoNCELoss(temperature=temperature)
        elif self.loss_type == "dcl":
            self.criterion = DCLLoss(temperature=temperature)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        self.val_criterion = InfoNCELoss(temperature=temperature)

    @property
    def current_temperature(self):
        if self.learnable_temperature:
            log_t = torch.clamp(self.log_temperature, min=np.log(1e-3), max=np.log(10))
            return torch.exp(log_t)
        return self.temperature

    def forward(self, users, items):
        # MLP: 0 index input
        # Transformer: 1 index sequence input
        u_emb = self.user_tower.encode(users)
        # 0 index input
        i_emb = self.item_tower.encode(items)
        return u_emb, i_emb

    def _compute_in_batch_loss(self, batch):
        if self.user_tower_type == "transformer":
            sequences, user_ids = batch

            inputs = sequences[:, :-1]
            targets = sequences[:, 1:]  # The actual items to predict

            # Forward pass through User Tower (Transformer)
            # Returns (B, L, D) embeddings for each position
            u_emb = self.user_tower(inputs)

            # u_emb: (B, L, D) -> Flatten to (N, D)
            B, L, D = u_emb.shape
            u_emb_flat = u_emb.reshape(-1, D)
            targets_flat = targets.reshape(-1)

            # Filter out padding targets (0 is padding)
            valid_mask = targets_flat != 0

            # Prepare inputs for loss
            final_u_emb = u_emb_flat[valid_mask]
            final_targets = targets_flat[valid_mask]
            final_targets -= 1  # Shift back to 0-indexed items

            # Don't mask user collisions for sequence targets
            final_user_ids = None
            # Expand user_ids to match sequence length for collision masking
            # user_ids: (B,) -> (B, L)
            # user_ids_expanded = user_ids.unsqueeze(1).expand(-1, L)
            # final_user_ids = user_ids_expanded.reshape(-1)[valid_mask]

            # Get Item Embeddings for targets
            final_i_emb = self.item_tower(final_targets)

        else:
            users_input, items, _ = batch
            final_u_emb, final_i_emb = self(users_input, items)
            final_targets = items
            final_user_ids = users_input

        # LogQ Correction
        batch_probs = None
        if hasattr(self, "item_probs") and self.item_probs is not None:
            batch_probs = getattr(self, "item_probs")[final_targets]

        loss = self.criterion(
            final_u_emb,
            final_i_emb,
            candidate_probs=batch_probs,
            # final_user_ids used to mask collisions; Doesn't matter 0 or 1 index
            # In fact, for transformer, user_ids are 0-indexed already
            user_ids=final_user_ids,
            item_ids=final_targets,
            temperature=self.current_temperature,
        )
        return loss

    def _compute_sampled_loss(self, batch):
        if self.user_tower_type != "transformer":
            raise ValueError(
                "Sampled negative sampling is only supported for Transformer architecture"
            )

        sequences, user_ids = batch

        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        # User embeddings: (B, L, D)
        u_emb = self.user_tower(inputs)
        B, L, D = u_emb.shape

        # Positives
        # targets are 1-based (0 is padding).
        valid_mask = targets != 0
        targets_idx = targets.clone()
        targets_idx[~valid_mask] = 1  # dummy to avoid index error
        targets_idx = targets_idx - 1  # 0-based

        pos_i_emb = self.item_tower(targets_idx)  # (B, L, D)

        # Negatives (Sampled)
        n_items = self.item_tower.embedding.num_embeddings
        neg_ids = torch.randint(0, n_items, (B, self.num_negatives), device=self.device)
        neg_i_emb = self.item_tower(neg_ids)  # (B, K, D)

        # Calculate Scores
        # pos: (B, L)
        pos_scores = (u_emb * pos_i_emb).sum(dim=-1)

        # neg: (B, L, D) x (B, D, K) -> (B, L, K)
        neg_scores = torch.matmul(u_emb, neg_i_emb.transpose(1, 2))

        # Collision Masking: Mask negatives that match the exact target at each step.
        # targets: (B, L) -> (B, L, 1)
        # neg_ids: (B, K) -> (B, 1, K) -> +1 to match 1-based targets
        # collision_mask: (B, L, K)
        collision_mask = targets.unsqueeze(2) == (neg_ids.unsqueeze(1) + 1)
        neg_scores = neg_scores.masked_fill(collision_mask, -1e9)

        # Concat Logits: (B, L, 1+K)
        logits = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
        logits = logits / self.current_temperature  # type: ignore

        # Loss on valid positions
        valid_logits = logits[valid_mask]
        labels = torch.zeros(valid_logits.size(0), dtype=torch.long, device=self.device)

        return F.cross_entropy(valid_logits, labels)

    def training_step(self, batch, batch_idx):
        if self.negative_sampling_strategy == "in-batch":
            loss = self._compute_in_batch_loss(batch)
        elif self.negative_sampling_strategy == "sampled":
            loss = self._compute_sampled_loss(batch)
        else:
            raise ValueError(
                f"Unknown negative sampling strategy: {self.negative_sampling_strategy}"
            )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _compute_in_batch_hit_rate(self, u_emb, i_emb):
        """
        In-Batch HitRate@10: Checks if the true item (diagonal) is in the top-K
        scores for the user against other items in the batch.
        """
        # 1. Compute similarity matrix for the validation batch (B x B)
        sim_matrix = torch.matmul(u_emb, i_emb.t())

        # 2. The "true" item for user_i is at index i (diagonal)
        targets = torch.arange(u_emb.size(0), device=u_emb.device)

        # 3. Check if the true item is in the Top-10 highest scores for that user
        k = min(10, u_emb.size(0))
        _, top_indices = torch.topk(sim_matrix, k=k, dim=1)

        # top_indices is (B, 10). Check if target[i] exists in top_indices[i]
        hits = (top_indices == targets.unsqueeze(1)).any(dim=1).float()
        return hits.mean()

    def _compute_sampled_hit_rate(self, batch_size, u_emb, i_emb):
        """
        Sampled HitRate@10: For each user, compares the positive item score against
        99 randomly sampled negative items (1 pos vs 99 negs).

        This metric is not recommended for evaluation purposes.
        See https://dl.acm.org/doi/10.1145/3535335
        """
        n_neg = 99
        # Randomly sample negatives
        n_items = self.item_tower.embedding.num_embeddings
        neg_items = torch.randint(0, n_items, (batch_size * n_neg,), device=self.device)

        # Get embeddings
        neg_emb = self.item_tower(neg_items)  # (B*99, D)
        neg_emb = neg_emb.view(batch_size, n_neg, -1)  # (B, 99, D)

        # Calculate scores
        u_emb_exp = u_emb.unsqueeze(1)  # (B, 1, D)

        # Positive scores: (u . i_pos)
        pos_scores = (u_emb_exp * i_emb.unsqueeze(1)).sum(dim=-1)  # (B, 1)

        # Negative scores: (u . i_negs)
        neg_scores = (u_emb_exp * neg_emb).sum(dim=-1)  # (B, 99)

        # Rank = 1 + count(neg_score > pos_score)
        # Hit@10 means Rank <= 10, i.e., count < 10
        greater_count = (neg_scores > pos_scores).float().sum(dim=1)
        sampled_hits = (greater_count < 10).float()
        return sampled_hits.mean()

    def validation_step(self, batch, batch_idx):
        # Per-type extraction: compute final_u_emb, final_i_emb, final_targets, final_user_ids
        if self.user_tower_type == "transformer":
            sequences, user_ids = batch
            inputs = sequences[:, :-1]
            final_targets = sequences[:, -1]
            final_targets -= 1  # Shift back to 0-indexed items

            final_u_emb, final_i_emb = self(inputs, final_targets)
            # Usually, we have unique user_ids in the validation batch
            final_user_ids = None
            # final_user_ids = user_ids
        else:
            users, items, _ = batch
            final_u_emb, final_i_emb = self(users, items)
            final_targets = items
            final_user_ids = users

        # Common validation logic: loss, logs, and metrics
        if final_u_emb.size(0) == 0:
            return None

        batch_probs = None
        if hasattr(self, "item_probs") and self.item_probs is not None:
            batch_probs = getattr(self, "item_probs")[final_targets]

        loss = self.val_criterion(
            final_u_emb,
            final_i_emb,
            candidate_probs=batch_probs,
            # final_user_ids used to mask collisions; Doesn't matter 0 or 1 index
            # In fact, for transformer, user_ids are 0-indexed already
            user_ids=final_user_ids,
            item_ids=final_targets,
            temperature=self.current_temperature,
        )

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        hit_rate = self._compute_in_batch_hit_rate(final_u_emb, final_i_emb)
        self.log("val_hit_rate", hit_rate, prog_bar=True, on_step=False, on_epoch=True)

        sampled_hit_rate = self._compute_sampled_hit_rate(
            final_u_emb.size(0), final_u_emb, final_i_emb
        )
        self.log(
            "val_sampled_hit10",
            sampled_hit_rate,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def on_validation_epoch_end(self):
        if self.val_users is not None and self.val_ground_truth is not None:
            # 1. Get embeddings for validation users
            val_users_tensor = torch.tensor(self.val_users, device=self.device)

            if self.user_tower_type == "transformer":
                # Use encode for inference (last step embedding)
                u_emb = self.user_tower.encode(val_users_tensor)
                # val_ground_truth is list of ints, wrap in list for recall_at_k
                targets = [[t] for t in self.val_ground_truth]
            else:
                u_emb = self.user_tower(val_users_tensor)  # (500, D)
                targets = [self.val_ground_truth[u] for u in self.val_users]

            # 2. Get embeddings for ALL items
            # Assuming n_items fits in memory.
            n_items = self.item_tower.embedding.num_embeddings
            # Note: We can reuse the item tower embeddings if they are static,
            # but they update during training.
            all_items = torch.arange(n_items, device=self.device)
            i_emb = self.item_tower(all_items)  # (N_items, D)

            # 3. Compute Scores (500, N_items)
            # Both are normalized, so matmul is cosine similarity
            scores = torch.matmul(u_emb, i_emb.t())

            # 4. Top-k
            k = 100
            _, top_indices = torch.topk(scores, k=k, dim=1)
            top_indices_np = top_indices.cpu().numpy()

            # 5. Calculate Metrics
            r10 = recall_at_k(top_indices_np, targets, 10)  # type: ignore
            r50 = recall_at_k(top_indices_np, targets, 50)  # type: ignore
            r100 = recall_at_k(top_indices_np, targets, 100)  # type: ignore

            self.log("val_recall_at_10", r10, prog_bar=True, on_epoch=True)
            self.log("val_recall_at_50", r50, prog_bar=True, on_epoch=True)
            self.log("val_recall_at_100", r100, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):  # type: ignore
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.optimizer_params.get("lr"),
            weight_decay=self.optimizer_params.get("weight_decay"),
            betas=(
                self.optimizer_params.get("adam_beta1"),
                self.optimizer_params.get("adam_beta2"),
            ),
            eps=self.optimizer_params.get("adam_eps"),
        )

        if self.optimizer_params.get("use_scheduler"):
            scheduler = get_linear_warmup_scheduler(
                optimizer, self.optimizer_params.get("warmup_steps")
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer


def train_retriever(args, vocab_sizes):
    logger = get_logger()
    num_workers = getattr(args, "num_workers")
    n_users, n_items = vocab_sizes
    logger.info(f"Initializing Retriever Training (Users: {n_users}, Items: {n_items})")

    user_tower_type = getattr(args, "user_tower_type")
    assert user_tower_type in ["mlp", "transformer"], (
        "user_tower_type must be 'mlp' or 'transformer'"
    )

    # 1. Data
    train_path = settings.processed_data_dir / "train.parquet"
    val_path = settings.processed_data_dir / "val.parquet"

    # Calculate item popularity for LogQ correction
    item_probs = None
    if settings.use_logq_correction:
        logger.info("Calculating item popularity...")
        if user_tower_type == "transformer":
            item_probs = compute_seq_item_probabilities(n_items)
        else:
            item_probs = compute_item_probabilities(n_items)

    val_users_for_pl = None
    val_ground_truth = None

    if user_tower_type == "transformer":
        logger.info("Using Transformer User Tower with SequentialDataset")
        train_dataset = SequentialDataset(str(train_path))
        val_dataset = SequentialDataset(str(val_path))

        # Sample validation sequences for metrics
        all_sequences = val_dataset.sequences
        N_val = len(all_sequences)
        sample_size = min(500, N_val)
        rng = np.random.default_rng(42)
        indices = rng.choice(N_val, size=sample_size, replace=False)

        subset_sequences = all_sequences[indices]
        # input: all except last, target: last
        val_inputs = subset_sequences[:, :-1]
        val_targets_arr = subset_sequences[:, -1]
        val_targets_arr -= 1  # Shift back to 0-indexed items

        val_users_for_pl = val_inputs.tolist()
        val_ground_truth = val_targets_arr.tolist()

    elif user_tower_type == "mlp":
        logger.info("Loading training data for MLP...")
        train_dataset = InteractionsDataset(
            str(train_path), positive_threshold=args.retrieval_threshold
        )

        logger.info("Loading validation data for metrics...")
        df_val = pd.read_parquet(val_path)
        if args.retrieval_threshold is not None:
            df_val = df_val[df_val["rating"] >= args.retrieval_threshold]

        unique_val_users = df_val["user_idx"].unique()
        rng = np.random.default_rng(42)
        sample_size = min(500, len(unique_val_users))
        val_users = rng.choice(unique_val_users, size=sample_size, replace=False)

        # Ground truth for sampled users
        df_val_subset = df_val[df_val["user_idx"].isin(val_users)]
        val_ground_truth = (
            df_val_subset.groupby("user_idx")["item_idx"].apply(list).to_dict()
        )

        # Convert to list for passing
        val_users_for_pl = val_users.tolist()
        del df_val

        val_dataset = InteractionsDataset(
            str(val_path), positive_threshold=args.retrieval_threshold
        )

    train_loader = DataLoader(
        train_dataset,  # type: ignore
        batch_sampler=BatchSampler(
            RandomSampler(train_dataset),  # type: ignore
            batch_size=args.batch_size,
            drop_last=True,
        ),
        collate_fn=collate_fn_numpy_to_tensor,
        num_workers=num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,  # type: ignore
        batch_sampler=BatchSampler(
            SequentialSampler(val_dataset),  # type: ignore
            batch_size=args.batch_size,
            drop_last=True,
        ),  #
        collate_fn=collate_fn_numpy_to_tensor,
        num_workers=num_workers,
        # Ensure batch size is consistent for in-batch consistency metrics if needed
        pin_memory=settings.pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # 2. Model
    model = RetrievalPL(
        n_users,
        n_items,
        settings.embed_dim,
        settings.tower_out_dim,
        settings.towers_hidden_dims,
        settings.use_projection,
        optimizer_params={
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_eps": args.adam_eps,
            "use_scheduler": args.use_scheduler,
            "warmup_steps": args.warmup_steps,
        },
        item_probs=item_probs,
        temperature=args.temperature,
        learnable_temperature=settings.learnable_temperature,
        val_users=val_users_for_pl,
        val_ground_truth=val_ground_truth,
        user_tower_type=user_tower_type,
        transformer_params={
            "max_seq_len": settings.max_seq_len,
            "n_heads": settings.transformer_heads,
            "n_layers": settings.transformer_layers,
            "dropout": settings.transformer_dropout,
            "swiglu_hidden_dim": settings.swiglu_hidden_dim,
        },
        negative_sampling_strategy=settings.negative_sampling_strategy,
        num_negatives=settings.num_negatives,
        loss_type=settings.retrieval_in_batch_loss_type,
    )

    # 3. Trainer
    wandb_logger = WandbLogger(
        project="nanoRecSys",
        name=getattr(args, "wandb_run_name", None) or "retriever_run",
        config=vars(args),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.artifacts_dir,
        filename="retriever-{epoch:02d}-{val_recall_at_10:.4f}",
        save_top_k=1,
        monitor="val_recall_at_10",
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
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    wandb.finish()

    # Save artifacts manually for inference usage
    logger.info("Saving Retriever artifacts...")
    torch.save(model.user_tower.state_dict(), settings.artifacts_dir / "user_tower.pth")
    torch.save(model.item_tower.state_dict(), settings.artifacts_dir / "item_tower.pth")

    if getattr(args, "build_embeddings", False):
        logger.info("Generating embeddings from trained model...")
        from nanoRecSys.indexing.build_embeddings import (
            build_item_embeddings,
            build_user_embeddings,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        build_item_embeddings(
            model=model.item_tower, device=device, batch_size=args.batch_size
        )
        build_user_embeddings(
            model=model.user_tower,
            device=device,
            batch_size=args.batch_size,
            user_tower_type=args.user_tower_type,
        )
