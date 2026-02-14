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

"""Build and save item embeddings for all items in the dataset."""

import numpy as np
import torch
from tqdm import tqdm

from nanoRecSys.config import settings
from nanoRecSys.models.towers import ItemTower, TransformerUserTower, UserTower
from nanoRecSys.utils.logging_config import get_logger
from nanoRecSys.utils.utils import get_vocab_sizes


def generate_embeddings(
    tower, vocab_size, batch_size=256, device=None, description="embeddings"
):
    """
    Generic function to generate embeddings using a tower model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    tower = tower.to(device)
    tower.eval()

    all_embeddings = np.zeros((vocab_size, settings.tower_out_dim), dtype=np.float32)

    logger = get_logger()
    logger.info(f"Generating {description} for {vocab_size} items...")

    with torch.inference_mode():
        for i in tqdm(range(0, vocab_size, batch_size)):
            batch_end = min(i + batch_size, vocab_size)
            batch_ids = torch.arange(i, batch_end, device=device)

            batch_emb = tower(batch_ids)
            all_embeddings[i:batch_end] = batch_emb.cpu().numpy()
    tqdm.write("")

    embeddings = all_embeddings

    # Verify shape and normalization
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(
        f"Sample embedding norms (should be ~1.0): {np.linalg.norm(embeddings[:5], axis=1)}"
    )

    return embeddings


def build_item_embeddings(
    output_file: str = "item_embeddings.npy",
    batch_size: int = 256,
    device=None,
    model=None,
):
    """
    Generate embeddings for all items using the trained ItemTower model.

    Args:
        output_file: Name of the output file to save embeddings (saved in artifacts_dir)
        batch_size: Batch size for inference
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        model: Optional pre-loaded ItemTower model. If None, loads from checkpoint.

    Returns:
        Numpy array of shape (n_items, tower_out_dim) containing L2-normalized embeddings
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    logger = get_logger()
    logger.info(f"Using device: {device}")

    # Get vocabulary sizes
    n_users, n_items = get_vocab_sizes()
    logger.info(f"Vocab: Users={n_users}, Items={n_items}")

    item_tower = model
    if item_tower is None:
        # Load trained ItemTower model
        logger.info("Loading trained ItemTower model...")
        item_tower = ItemTower(
            n_items,
            embed_dim=settings.embed_dim,
            output_dim=settings.tower_out_dim,
            hidden_dims=settings.towers_hidden_dims,
            use_projection=settings.use_projection,
        ).to(device)

        model_path = settings.artifacts_dir / "item_tower.pth"
        try:
            item_tower.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            logger.error(f"Model checkpoint not found at {model_path}")
            logger.error("Please run train.py first to train the model.")
            return None

    embeddings = generate_embeddings(
        item_tower, n_items, batch_size, device, "item embeddings"
    )

    # Save embeddings
    output_path = settings.artifacts_dir / output_file
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")

    return embeddings


def generate_sequence_embeddings(
    tower, n_users, batch_size=256, device=None, description="seq embeddings"
):
    """
    Generate embeddings for TransformerUserTower by feeding the latest interaction history.
    Values are loaded from pre-built numpy files (seq_inference_*.npy).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    tower = tower.to(device)
    tower.eval()

    logger = get_logger()
    processed_dir = settings.processed_data_dir

    seq_path = processed_dir / "seq_inference_sequences.npy"
    uid_path = processed_dir / "seq_inference_user_ids.npy"

    if not (seq_path.exists() and uid_path.exists()):
        logger.error(f"Pre-built inference sequences not found at {processed_dir}")
        logger.error(
            "Please run: python src/nanoRecSys/data/build_dataset.py --task prebuild"
        )
        raise FileNotFoundError("Missing pre-built inference sequences")

    logger.info("Loading pre-built inference sequences...")
    seq_list = np.load(seq_path)
    idx_list = np.load(uid_path)

    all_embeddings = np.zeros((n_users, settings.tower_out_dim), dtype=np.float32)

    logger.info(f"Generating {description} for {len(idx_list)} users...")
    seq_tensor = torch.from_numpy(seq_list).long()

    with torch.inference_mode():
        for i in tqdm(range(0, len(idx_list), batch_size), desc="Inference"):
            batch_end = min(i + batch_size, len(idx_list))
            batch_seqs = seq_tensor[i:batch_end].to(device)

            emb = tower.encode(batch_seqs)
            emb_np = emb.cpu().numpy()

            uid_slice = idx_list[i:batch_end]
            all_embeddings[uid_slice] = emb_np

    return all_embeddings


def build_user_embeddings(
    output_file: str = "user_embeddings.npy",
    batch_size: int = 512,
    device=None,
    model=None,
    user_tower_type: str = "transformer",
):
    """
    Generate embeddings for all users using the trained UserTower model.

    Args:
        output_file: Name of the output file to save embeddings (saved in artifacts_dir)
        batch_size: Batch size for inference
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        model: Optional pre-loaded UserTower model. If None, loads from checkpoint.
        user_tower_type: 'mlp' or 'transformer'. Required if model is None to instantiate correct class.

    Returns:
        Numpy array of shape (n_users, tower_out_dim) containing L2-normalized embeddings
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    logger = get_logger()
    logger.info(f"Using device: {device}")

    # Get vocabulary sizes
    n_users, n_items = get_vocab_sizes()
    logger.info(f"Vocab: Users={n_users}, Items={n_items}")

    user_tower = model
    if user_tower is None:
        logger.info(f"Loading trained UserTower model (type={user_tower_type})...")
        model_path = settings.artifacts_dir / "user_tower.pth"

        if user_tower_type == "transformer":
            # Use shared embedding structure (size N) to match training
            dummy_item_tower = ItemTower(
                vocab_size=n_items,
                embed_dim=settings.embed_dim,
                output_dim=settings.tower_out_dim,
                hidden_dims=settings.towers_hidden_dims,
                use_projection=settings.use_projection,
            )

            user_tower = TransformerUserTower(
                vocab_size=n_items,
                embed_dim=settings.embed_dim,
                output_dim=settings.tower_out_dim,
                max_seq_len=settings.max_seq_len,
                n_heads=settings.transformer_heads,
                n_layers=settings.transformer_layers,
                dropout=settings.transformer_dropout,
                swiglu_hidden_dim=settings.swiglu_hidden_dim,
                shared_embedding=dummy_item_tower,
            ).to(device)
        else:
            user_tower = UserTower(
                n_users,
                embed_dim=settings.embed_dim,
                output_dim=settings.tower_out_dim,
                hidden_dims=settings.towers_hidden_dims,
                use_projection=settings.use_projection,
            ).to(device)

        try:
            user_tower.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            logger.error(
                f"Ensure user_tower_type='{user_tower_type}' matches the trained checkpoint."
            )
            return None

    # Check model type to choose generation strategy
    if isinstance(user_tower, TransformerUserTower):
        logger.info(
            "Detected TransformerUserTower. Generating embeddings from sequences..."
        )
        embeddings = generate_sequence_embeddings(
            user_tower, n_users, batch_size, device, "transformer user embeddings"
        )
    else:
        logger.info("Detected MLP UserTower. Generating embeddings from IDs...")
        embeddings = generate_embeddings(
            user_tower, n_users, batch_size, device, "user embeddings"
        )

    # Save embeddings
    output_path = settings.artifacts_dir / output_file
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")

    return embeddings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and save embeddings for users/items"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["items", "users", "all"],
        default="items",
        help="Which embeddings to build: 'items', 'users', or 'all'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--user_tower_type",
        type=str,
        choices=["mlp", "transformer"],
        default="transformer",
        help="Type of user tower to load (if mode includes users)",
    )

    args = parser.parse_args()

    logger = get_logger()
    if args.mode in ["items", "all"]:
        logger.info("=" * 60)
        logger.info("Building Item Embeddings")
        logger.info("=" * 60)
        build_item_embeddings(batch_size=args.batch_size, device=args.device)

    if args.mode in ["users", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("Building User Embeddings")
        logger.info("=" * 60)
        build_user_embeddings(
            batch_size=args.batch_size,
            device=args.device,
            user_tower_type=args.user_tower_type,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)
