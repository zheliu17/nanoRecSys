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

"""Configuration for nanoRecSys.

Provides a pydantic Settings class with sensible defaults and helpers for
common project paths (data, processed, artifacts).
"""

from pathlib import Path
from typing import Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with sensible defaults.

    Override values via environment variables or a local `.env` file.
    """

    # --- Paths ---
    project_root: Path = Path(__file__).parent.parent.parent.absolute()
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "ml-20m"
    processed_data_dir: Path = data_dir / "processed"
    artifacts_dir: Path = project_root / "artifacts"
    # Parent path for generated embeddings (src/nanoRecSys/training/mine_negatives_sasrec.py)
    # This file can be large (10GB+) and you may want to set it to a different location
    sasrec_user_embs_npy_path: Path = artifacts_dir
    ckpt_path: Union[str, Path, None] = None  # Optional checkpoint path

    # --- Dataset ---
    ml_20m_url: str = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    # Minimum rating to include when building the retrieval dataset. Set to 0 or None to include all.
    retrieval_threshold: Union[float, None] = 0
    min_user_interactions: int = 5  # Exclude users with too few interactions
    evaluation_positive_threshold: float = (
        0  # Rating >= this is positive during evaluation
    )
    # Note for the sequential retriever, the current implementation treats all generated interactions as positives (rating=5)
    # see src/nanoRecSys/training/mine_negatives_sasrec.py
    ranker_positive_threshold: float = (
        0  # Rating >= this is positive for ranker training
    )
    # Rating <= this is treated as an explicit negative for ranker; set to None to ignore
    # This will affect both training and validation steps.
    # For disable training only, see `explicit_neg_weight` in the training config below.
    ranker_negative_threshold: Union[float, None] = None

    # --- MLP tower parameters ---
    embed_dim: int = 256
    tower_out_dim: int = 256
    towers_hidden_dims: list[int] = []  # Hidden dims for MLP towers
    use_projection: bool = False  # Add final projection layer to towers
    user_tower_type: str = "transformer"  # "mlp" or "transformer"

    # --- Transformer tower settings ---
    max_seq_len: int = 200
    _rope_default_max_seq_len: int = 512  # Only used for RoPE cache initialization
    min_seq_len: int = 0  # Minimum sequence length to keep a user
    seq_step_size: int = 96  # Sliding-window step size (used for data generation)
    # If True, train using one sequence per user formed from the last `max_seq_len` items
    train_single_last_sequence: bool = True

    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dropout: float = 0.2
    positional_embedding_type: str = "rope"  # "rope" or "absolute"
    rope_base: int = 500
    # None to use default, rounded 8/3* embed_dim
    swiglu_hidden_dim: Union[int, None] = 1024

    # --- Loss & negative sampling ---
    # Use logQ correction for in-batch loss; affects evaluation and in-batch training negatives.
    use_logq_correction: bool = True
    temperature: float = 0.05  # Softmax temperature (initial value if learnable)
    learnable_temperature: bool = False  # Make temperature a trainable parameter
    # Negative sampling strategy during training: "in-batch" or "sampled".
    # Note: logQ correction is not applied when using "sampled".
    negative_sampling_strategy: str = "in-batch"  # "in-batch" or "sampled"
    retrieval_in_batch_loss_type: str = "info_nce"  # "info_nce" or "dcl"
    num_negatives: int = 3000  # Used only when negative_sampling_strategy == "sampled"
    id_dropout: float = 0.1  # Probability of ID-modality dropout

    # --- Training ---
    # Defaults are not optimally tuned
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_steps: int = -1  # -1 disables a global step cap
    limit_train_batches: Union[float, int] = 1.0
    num_workers: int = 0
    pin_memory: bool = True
    check_val_every_n_epoch: int = 3

    # --- Optimizer (AdamW) ---
    weight_decay: float = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98  # Transformer
    adam_eps: float = 1e-8

    # --- Scheduler ---
    use_scheduler: bool = True  # Enable/disable scheduler
    warmup_steps: int = 2000  # Number of warmup steps for linear warmup

    # --- Negative mining ---
    mining_top_k: int = 100  # Number of top items to retrieve per user
    mining_skip_top: int = 20  # Number of top items to skip (too-hard negatives)
    mining_num_negatives: int = 2  # Random negatives per positive interaction

    # --- Ranker training ---
    explicit_neg_weight: float = (
        4.0  # Weight for explicit negatives in ranker training, 0 to ignore
    )
    ranker_random_neg_ratio: float = (
        1.0  # Ratio of random negatives to use (0.0 to 1.0)
    )
    ranker_hidden_dims: list[int] = [512, 256, 128]
    ranker_loss_type: str = "bce"  # 'bce' (default), 'margin_ranking', or 'bpr'
    ranker_loss_margin: float = 1.0  # Margin for MarginRankingLoss
    item_embedding_lr: float = 1e-5
    user_embedding_lr: float = 0.0  # Effectively frozen if 0

    # --- Embedding generation ---
    build_embeddings: bool = False

    @field_validator(
        "ckpt_path",
        "retrieval_threshold",
        "ranker_negative_threshold",
        "swiglu_hidden_dim",
        mode="before",
    )
    def _none_str_to_none(cls, v):
        if isinstance(v, str) and v.strip().lower() in ("", "none", "null"):
            return None
        return v


model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
settings: Settings = Settings()

__all__ = ["Settings", "settings"]

# Create required directories if missing
for _path in (
    settings.data_dir,
    settings.raw_data_dir,
    settings.processed_data_dir,
    settings.artifacts_dir,
):
    _path.mkdir(parents=True, exist_ok=True)
