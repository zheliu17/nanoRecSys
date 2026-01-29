"""Build and save item embeddings for all items in the dataset."""

import torch
import numpy as np
from tqdm import tqdm
from nanoRecSys.config import settings
from nanoRecSys.models.towers import UserTower, ItemTower
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

    all_embeddings = []

    print(f"Generating {description} for {vocab_size} items...")

    with torch.no_grad():
        for i in tqdm(range(0, vocab_size, batch_size)):
            batch_end = min(i + batch_size, vocab_size)
            batch_ids = torch.arange(i, batch_end, device=device)

            batch_emb = tower(batch_ids)
            all_embeddings.append(batch_emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Verify shape and normalization
    print(f"Embeddings shape: {embeddings.shape}")
    print(
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
        Numpy array of shape (n_items, embed_dim) containing L2-normalized embeddings
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    print(f"Using device: {device}")

    # Get vocabulary sizes
    n_users, n_items = get_vocab_sizes()
    print(f"Vocab: Users={n_users}, Items={n_items}")

    item_tower = model
    if item_tower is None:
        # Load trained ItemTower model
        print("Loading trained ItemTower model...")
        item_tower = ItemTower(n_items, embed_dim=settings.embed_dim).to(device)

        model_path = settings.artifacts_dir / "item_tower.pth"
        try:
            item_tower.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            print(f"Error: Model checkpoint not found at {model_path}")
            print("Please run train.py first to train the model.")
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
        Numpy array of shape (n_users, embed_dim) containing L2-normalized embeddings
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    print(f"Using device: {device}")

    # Get vocabulary sizes
    n_users, n_items = get_vocab_sizes()
    print(f"Vocab: Users={n_users}, Items={n_items}")

    user_tower = model
    if user_tower is None:
        # Load trained UserTower model
        print("Loading trained UserTower model...")
        user_tower = UserTower(n_users, embed_dim=settings.embed_dim).to(device)

        model_path = settings.artifacts_dir / "user_tower.pth"
        try:
            user_tower.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            print(f"Error: Model checkpoint not found at {model_path}")
            print("Please run train.py first to train the model.")
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

    if args.mode in ["items", "all"]:
        print("=" * 60)
        print("Building Item Embeddings")
        print("=" * 60)
        build_item_embeddings(batch_size=args.batch_size, device=args.device)

    if args.mode in ["users", "all"]:
        print("\n" + "=" * 60)
        print("Building User Embeddings")
        print("=" * 60)
        build_user_embeddings(batch_size=args.batch_size, device=args.device)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
