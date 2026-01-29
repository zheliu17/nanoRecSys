from typing import Union
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # Paths
    project_root: Path = Path(__file__).parent.parent.parent.absolute()
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "ml-20m"
    processed_data_dir: Path = data_dir / "processed"
    artifacts_dir: Path = project_root / "artifacts"
    ckpt_path: Union[str, Path, None] = None

    # Dataset
    ml_20m_url: str = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    retrieval_threshold: float = 3.5  # Minimum rating for retrieval dataset loading
    min_user_interactions: int = 5  # Filter users with too few interactions
    ranker_positive_threshold: float = (
        3.5  # Rating >= this is labeled positive for ranker
    )
    # Set explicit_neg_weight=0 to ignore explicit negatives in ranker training
    ranker_negative_threshold: Union[float, None] = (
        2.5  # Rating <= this is labeled negative for ranker
        # Will be used in both training and validation datasets
    )

    # Model params
    embed_dim: int = 64
    id_dropout: float = 0.1  # Default probability for ID modality dropout
    temperature: float = 0.1  # InfoNCE loss temperature

    # Training params
    # Parameters below are not optimally tuned; adjust as needed.
    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_steps: int = -1
    limit_train_batches: Union[float, int] = 1.0
    num_workers: int = 0
    pin_memory: bool = True
    check_val_every_n_epoch: int = 1

    # Optimizer params (AdamW)
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # Learning rate scheduler params
    use_scheduler: bool = True  # Enable/disable scheduler
    warmup_steps: int = 500  # Number of warmup steps for linear warmup

    # Negative mining params
    mining_top_k: int = 100  # Number of top items to retrieve per user
    mining_skip_top: int = 20  # Number of top items to skip (too hard negatives)
    mining_num_negatives: int = 2  # Number of random negatives per positive interaction

    # Ranker training params
    explicit_neg_weight: float = (
        4.0  # Weight for explicit negatives in ranker training, 0 to ignore
    )
    ranker_random_neg_ratio: float = (
        1.0  # Ratio of random negatives to use (0.0 to 1.0)
    )
    ranker_hidden_dims: list[int] = [256, 128, 64]
    ranker_loss_type: str = (
        "bce"  # Loss function type: 'bce' (default), 'margin_ranking', or 'bpr'
    )
    ranker_loss_margin: float = (
        1.0  # Margin for MarginRankingLoss (only used if loss_type='margin_ranking')
    )
    item_embedding_lr: float = 1e-5
    user_embedding_lr: float = (
        0.0  # Effectively frozen if 0 or not included in optimizer
    )

    # Embedding generation
    build_embeddings: bool = False

    @field_validator("ranker_negative_threshold", mode="before")
    @classmethod
    def validate_ranker_negative_threshold(cls, v):
        if isinstance(v, str) and v == "None":
            return None
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
