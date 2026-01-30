"""Retrieval model training module."""

import torch
import torch.optim as optim
import pandas as pd
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR

from ..config import settings
from ..data.datasets import InteractionsDataset, UniqueUserDataset
from ..models.towers import TwoTowerModel, UserTower, ItemTower
from ..models.losses import InfoNCELoss
from ..eval.metrics import recall_at_k
from ..utils.utils import (
    compute_item_probabilities,
    collate_fn_numpy_to_tensor,
)


def get_linear_warmup_scheduler(optimizer, warmup_steps):
    """Create a linear warmup scheduler."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class RetrievalPL(pl.LightningModule):
    def __init__(
        self,
        n_users,
        n_items,
        embed_dim,
        output_dim,
        hidden_dims,
        lr,
        weight_decay,
        adam_beta1,
        adam_beta2,
        adam_eps,
        use_scheduler,
        warmup_steps,
        item_probs,
        temperature,
        val_users,
        val_ground_truth,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["item_probs", "val_ground_truth", "val_users"]
        )
        if item_probs is not None:
            self.register_buffer("item_probs", item_probs)

        self.val_users = val_users
        self.val_ground_truth = val_ground_truth

        self.user_tower = UserTower(
            n_users, embed_dim=embed_dim, output_dim=output_dim, hidden_dims=hidden_dims
        )
        self.item_tower = ItemTower(
            n_items, embed_dim=embed_dim, output_dim=output_dim, hidden_dims=hidden_dims
        )
        self.model = TwoTowerModel(self.user_tower, self.item_tower)
        self.criterion = InfoNCELoss(temperature=temperature)
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.use_scheduler = use_scheduler
        self.warmup_steps = warmup_steps

    def forward(self, users, items):
        return self.model(users, items)

    def training_step(self, batch, batch_idx):
        users, items, _ = batch
        u_emb, i_emb = self(users, items)

        batch_probs = None
        if hasattr(self, "item_probs") and self.item_probs is not None:
            batch_probs = getattr(self, "item_probs")[items]

        loss = self.criterion(u_emb, i_emb, candidate_probs=batch_probs, user_ids=users)
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

    def _compute_sampled_hit_rate(self, users, u_emb, i_emb):
        """
        Sampled HitRate@10: For each user, compares the positive item score against
        99 randomly sampled negative items (1 pos vs 99 negs).

        This metric is not recommended for evaluation purposes.
        See https://dl.acm.org/doi/10.1145/3535335
        """
        B = users.size(0)
        n_neg = 99
        # Randomly sample negatives
        n_items = self.item_tower.embedding.num_embeddings
        neg_items = torch.randint(0, n_items, (B * n_neg,), device=self.device)

        # Get embeddings
        neg_emb = self.item_tower(neg_items)  # (B*99, D)
        neg_emb = neg_emb.view(B, n_neg, -1)  # (B, 99, D)

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
        users, items, _ = batch
        u_emb, i_emb = self(users, items)

        batch_probs = None
        if (
            hasattr(self, "item_probs")
            and self.item_probs is not None
            and isinstance(self.item_probs, torch.Tensor)
        ):
            batch_probs = self.item_probs[items]

        loss = self.criterion(u_emb, i_emb, candidate_probs=batch_probs)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Metrics
        hit_rate = self._compute_in_batch_hit_rate(u_emb, i_emb)
        self.log("val_hit_rate", hit_rate, prog_bar=True, on_step=False, on_epoch=True)

        sampled_hit_rate = self._compute_sampled_hit_rate(users, u_emb, i_emb)
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
            # 1. Get embeddings for validaton users
            val_users_tensor = torch.tensor(self.val_users, device=self.device)
            u_emb = self.user_tower(val_users_tensor)  # (500, D)

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
            targets = [self.val_ground_truth[u] for u in self.val_users]

            r10 = recall_at_k(top_indices_np, targets, 10)
            r50 = recall_at_k(top_indices_np, targets, 50)
            r100 = recall_at_k(top_indices_np, targets, 100)

            self.log("val_recall_at_10", r10, prog_bar=True, on_epoch=True)
            self.log("val_recall_at_50", r50, prog_bar=True, on_epoch=True)
            self.log("val_recall_at_100", r100, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):  # type: ignore
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

        if self.use_scheduler:
            scheduler = get_linear_warmup_scheduler(optimizer, self.warmup_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer


def train_retriever(args, vocab_sizes):
    num_workers = getattr(args, "num_workers", 0)
    n_users, n_items = vocab_sizes
    print(f"Initializing Retriever Training (Users: {n_users}, Items: {n_items})")

    # 1. Data
    train_path = settings.processed_data_dir / "train.parquet"
    val_path = settings.processed_data_dir / "val.parquet"

    print("Loading training data...")
    df_train = pd.read_parquet(train_path)
    if args.retrieval_threshold is not None:
        df_train = df_train[df_train["rating"] >= args.retrieval_threshold]

    # Calculate item popularity for LogQ correction
    print("Calculating item popularity...")
    item_probs = compute_item_probabilities(n_items)

    # Data Source Toggle
    use_interactions_dataset = True  # Set to True to use InteractionsDataset

    if use_interactions_dataset:
        del df_train
        train_dataset = InteractionsDataset(
            str(train_path), positive_threshold=args.retrieval_threshold
        )
    else:
        user_histories = df_train.groupby("user_idx")["item_idx"].apply(list).to_dict()
        del df_train
        train_dataset = UniqueUserDataset(user_histories)

    print("Loading validaton data for metrics...")
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
    val_users = val_users.tolist()
    del df_val

    val_dataset = InteractionsDataset(
        str(val_path), positive_threshold=args.retrieval_threshold
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BatchSampler(
            RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=True
        ),
        collate_fn=collate_fn_numpy_to_tensor,
        num_workers=num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=BatchSampler(
            SequentialSampler(val_dataset), batch_size=args.batch_size, drop_last=True
        ),
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
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        use_scheduler=args.use_scheduler,
        warmup_steps=args.warmup_steps,
        item_probs=item_probs,
        temperature=args.temperature,
        val_users=val_users,
        val_ground_truth=val_ground_truth,
    )

    # 3. Trainer
    wandb_logger = WandbLogger(
        project="nanoRecSys", name="retriever_run", config=vars(args)
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.artifacts_dir,
        filename="retriever-{epoch:02d}-{val_recall_at_50:.4f}",
        save_top_k=1,
        monitor="val_recall_at_50",
        mode="max",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=getattr(args, "max_steps", -1),
        limit_train_batches=getattr(args, "limit_train_batches", 1.0),
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    ckpt_path = getattr(args, "ckpt_path", None)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    wandb.finish()

    # Save artifacts manually for inference usage
    print("Saving Retriever artifacts...")
    torch.save(model.user_tower.state_dict(), settings.artifacts_dir / "user_tower.pth")
    torch.save(model.item_tower.state_dict(), settings.artifacts_dir / "item_tower.pth")

    if getattr(args, "build_embeddings", False):
        print("Generating embeddings from trained model...")
        from ..indexing.build_embeddings import (
            build_item_embeddings,
            build_user_embeddings,
        )

        build_item_embeddings(
            model=model.item_tower, device=model.device, batch_size=args.batch_size
        )
        build_user_embeddings(
            model=model.user_tower, device=model.device, batch_size=args.batch_size
        )
