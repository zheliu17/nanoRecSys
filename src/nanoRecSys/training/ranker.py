"""Ranker model training module."""

import torch
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import numpy as np
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import LambdaLR

from nanoRecSys.config import settings
from nanoRecSys.data.datasets import (
    RankerTrainDataset,
    RankerEvalDataset,
    load_item_metadata,
)
from nanoRecSys.models.ranker import RankerModel
from nanoRecSys.models.losses import get_ranker_loss
from nanoRecSys.utils.utils import (
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
        id_dropout_prob,
        lr,
        item_lr,
        weight_decay,
        adam_beta1,
        adam_beta2,
        adam_eps,
        use_scheduler,
        warmup_steps,
        pretrained_user_embeddings,
        pretrained_item_embeddings,
        loss_type,
        loss_margin,
    ):
        super().__init__()
        # User embeddings: Frozen Buffer
        self.register_buffer("user_embeddings", pretrained_user_embeddings)

        # Item embeddings: Trainable Parameter OR Frozen Buffer
        if item_lr > 0:
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

        self.model = RankerModel(
            input_dim=embed_dim,
            hidden_dims=settings.ranker_hidden_dims,
            num_genres=num_genres,
            num_years=num_years,
            genre_dim=16,
            year_dim=8,
        )
        # Create loss function based on loss_type
        self.criterion = get_ranker_loss(
            loss_type=loss_type, margin=loss_margin, reduction="none"
        )
        self.loss_type = loss_type
        self.lr = lr
        self.item_lr = item_lr
        self.id_dropout_prob = id_dropout_prob
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.use_scheduler = use_scheduler
        self.warmup_steps = warmup_steps

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
                "pretrained_user_embeddings",
                "pretrained_item_embeddings",
            ]
        )

    def forward(self, users, items):
        u_emb = self.user_embeddings[users]  # type: ignore
        i_emb = self.item_embeddings[items]  # type: ignore (Accesses Parameter)

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

        optimizer_groups = [{"params": main_params, "lr": self.lr}]
        if len(embedding_params) > 0:
            optimizer_groups.append({"params": embedding_params, "lr": self.item_lr})

        optimizer = optim.AdamW(
            optimizer_groups,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

        if self.use_scheduler:
            # Note: LambdaLR applies to all groups or list of lambdas.
            # If one lambda, it applies to all.
            scheduler = get_linear_warmup_scheduler(optimizer, self.warmup_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer


def train_ranker(args, vocab_sizes):
    num_workers = getattr(args, "num_workers", 0)
    n_users, n_items = vocab_sizes
    print(f"Initializing Ranker Training (Users: {n_users}, Items: {n_items})")

    # 1. Load Embeddings from Disk
    print("Checking for pre-computed user/item embeddings...")
    user_emb_path = settings.artifacts_dir / "user_embeddings.npy"
    item_emb_path = settings.artifacts_dir / "item_embeddings.npy"

    # Check User Embeddings
    if not user_emb_path.exists():
        print(f"Error: User embeddings not found at {user_emb_path}")
        print(
            "Please generate embeddings first using: python src/indexing/build_embeddings.py --mode users"
        )
        return

    # Check Item Embeddings
    if not item_emb_path.exists():
        print(f"Error: Item embeddings not found at {item_emb_path}")
        print(
            "Please generate embeddings first using: python src/indexing/build_embeddings.py --mode items"
        )
        return

    print("Loading embeddings from disk...")
    # Load as numpy then convert to tensor
    try:
        user_embeddings = torch.from_numpy(np.load(user_emb_path))
        item_embeddings = torch.from_numpy(np.load(item_emb_path))
        print(
            f"Loaded User Embeddings: {user_embeddings.shape}, Item Embeddings: {item_embeddings.shape}"
        )
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    # 3. Build/Load Metadata
    item_map_path = settings.processed_data_dir / "item_map.npy"
    movies_path = settings.raw_data_dir / "movies.csv"

    print("Loading Item Metadata (Genres + Years)...")
    genre_matrix, year_indices, num_genres, num_years = load_item_metadata(
        item_map_path, movies_path, cache_dir=str(settings.processed_data_dir)
    )

    print("Computing Item Popularity...")
    # Use log probability for better numerical stability in neural nets
    item_popularity = compute_item_probabilities(
        n_items, return_log_probs=True, device="cpu"
    )

    # 4. Data
    print("Loading training data for Ranker...")
    train_dataset = RankerTrainDataset(
        interactions_path=str(settings.processed_data_dir / "train.parquet"),
        hard_neg_path=str(settings.processed_data_dir / "train_negatives_hard.parquet"),
        random_neg_path=str(
            settings.processed_data_dir / "train_negatives_random.parquet"
        ),
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

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=collate_fn_numpy_to_tensor,
        # batch_size=None, # batch_sampler implies batch_size=None
        num_workers=num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # Val Datasets
    val_dataloaders = []

    # 1. Hard (Idx 0)
    val_dataset_hard = RankerEvalDataset(
        interactions_path=str(settings.processed_data_dir / "val.parquet"),
        negatives_path=str(settings.processed_data_dir / "val_negatives_hard.parquet"),
        mode="hard",
        pos_threshold=args.ranker_positive_threshold,
    )
    val_sampler_hard = SequentialSampler(val_dataset_hard)
    val_batch_sampler_hard = BatchSampler(
        val_sampler_hard, batch_size=args.batch_size, drop_last=False
    )
    val_loader_hard = DataLoader(
        val_dataset_hard,
        batch_sampler=val_batch_sampler_hard,
        collate_fn=collate_fn_numpy_to_tensor,
        num_workers=num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_dataloaders.append(val_loader_hard)

    # 2. Random (Idx 1)
    val_dataset_random = RankerEvalDataset(
        interactions_path=str(settings.processed_data_dir / "val.parquet"),
        negatives_path=str(
            settings.processed_data_dir / "val_negatives_random.parquet"
        ),
        mode="random",
        pos_threshold=args.ranker_positive_threshold,
    )
    val_sampler_random = SequentialSampler(val_dataset_random)
    val_batch_sampler_random = BatchSampler(
        val_sampler_random, batch_size=args.batch_size, drop_last=False
    )
    val_loader_random = DataLoader(
        val_dataset_random,
        batch_sampler=val_batch_sampler_random,
        collate_fn=collate_fn_numpy_to_tensor,
        num_workers=num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_dataloaders.append(val_loader_random)

    # 3. Explicit (Idx 2 - Optional)
    if args.ranker_negative_threshold is not None:
        val_dataset_explicit = RankerEvalDataset(
            interactions_path=str(settings.processed_data_dir / "val.parquet"),
            negatives_path=None,
            mode="explicit",
            pos_threshold=args.ranker_positive_threshold,
            neg_threshold=args.ranker_negative_threshold,
        )
        val_sampler_explicit = SequentialSampler(val_dataset_explicit)
        val_batch_sampler_explicit = BatchSampler(
            val_sampler_explicit, batch_size=args.batch_size, drop_last=False
        )
        val_loader_explicit = DataLoader(
            val_dataset_explicit,
            batch_sampler=val_batch_sampler_explicit,
            collate_fn=collate_fn_numpy_to_tensor,
            num_workers=num_workers,
            pin_memory=settings.pin_memory,
            persistent_workers=(num_workers > 0),
        )
        val_dataloaders.append(val_loader_explicit)

    # 4. Model
    # Normalize Popularity Stats Calculation
    pop_mean = item_popularity.mean()
    pop_std = item_popularity.std()
    print(f"Popularity Stats -- Mean: {pop_mean:.4f}, Std: {pop_std:.4f}")

    if args.item_lr > 0:
        print(f"Item Embeddings will be FINE-TUNED (LR={args.item_lr}).")
    else:
        print("Item Embeddings will be FROZEN (LR=0).")

    model = RankerPL(
        embed_dim=settings.tower_out_dim,
        genre_matrix=genre_matrix,
        year_indices=year_indices,
        popularity=item_popularity,
        pop_mean=pop_mean,
        pop_std=pop_std,
        num_genres=num_genres,
        num_years=num_years,
        id_dropout_prob=args.id_dropout,
        lr=args.lr,
        item_lr=args.item_lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        use_scheduler=args.use_scheduler,
        warmup_steps=args.warmup_steps,
        pretrained_user_embeddings=user_embeddings,
        pretrained_item_embeddings=item_embeddings,
        loss_type=args.ranker_loss_type,
        loss_margin=args.ranker_loss_margin,
    )

    # 5. Trainer
    # Pass all args to wandb config for tracking
    wandb_logger = WandbLogger(
        project="nanoRecSys", name="ranker_run", config=vars(args)
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.artifacts_dir,
        filename="ranker-{epoch:02d}-{val_auc_hard:.4f}",
        save_top_k=1,
        monitor="val_auc_hard",
        mode="max",
        save_last=True,
        every_n_epochs=args.check_val_every_n_epoch,
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
    print("Saved ranker_model.pth, ranker_item_embeddings.pt, ranker_pop_stats.pt")
