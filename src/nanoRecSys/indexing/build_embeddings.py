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

import torch
import numpy as np
from tqdm import tqdm
from nanoRecSys.config import settings
from nanoRecSys.models.towers import UserTower, ItemTower
from nanoRecSys.utils.utils import get_vocab_sizes
from nanoRecSys.utils.logging_config import get_logger


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

    all_embeddings = []

    logger = get_logger()
    logger.info(f"Generating {description} for {vocab_size} items...")

    with torch.no_grad():
        for i in tqdm(range(0, vocab_size, batch_size)):
            batch_end = min(i + batch_size, vocab_size)
            batch_ids = torch.arange(i, batch_end, device=device)

            batch_emb = tower(batch_ids)
            all_embeddings.append(batch_emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

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


def build_user_embeddings(
    output_file: str = "user_embeddings.npy",
    batch_size: int = 512,
    device=None,
    model=None,
):
    """
    Generate embeddings for all users using the trained UserTower model.

    Args:
        output_file: Name of the output file to save embeddings (saved in artifacts_dir)
        batch_size: Batch size for inference
        device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        model: Optional pre-loaded UserTower model. If None, loads from checkpoint.

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
        # Load trained UserTower model
        logger.info("Loading trained UserTower model...")
        user_tower = UserTower(
            n_users,
            embed_dim=settings.embed_dim,
            output_dim=settings.tower_out_dim,
            hidden_dims=settings.towers_hidden_dims,
        ).to(device)

        model_path = settings.artifacts_dir / "user_tower.pth"
        try:
            user_tower.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            logger.error(f"Model checkpoint not found at {model_path}")
            logger.error("Please run train.py first to train the model.")
            return None

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
        build_user_embeddings(batch_size=args.batch_size, device=args.device)

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)
