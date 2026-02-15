import numpy as np
import torch
import torch.nn as nn

from nanoRecSys.config import settings
from nanoRecSys.models.ranker import MLPRanker
from nanoRecSys.models.towers import TransformerUserTower


class UserTowerWrapper(nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def forward(self, item_seq):
        # Return only the last embedding
        out = self.tower(item_seq)
        return out[:, -1, :]


class RankerWrapper(nn.Module):
    def __init__(self, ranker):
        super().__init__()
        self.ranker = ranker

    def forward(self, user_emb, item_emb, genre_multihot, year_idx, popularity):
        return self.ranker.predict(
            user_emb, item_emb, genre_multihot, year_idx, popularity
        )


def main():
    print("Starting ONNX export...")

    # Paths
    artifacts_dir = settings.artifacts_dir
    processed_dir = settings.processed_data_dir

    # Load metadata
    print("Loading metadata...")
    item_map = np.load(processed_dir / "item_map.npy")
    genre_matrix = np.load(processed_dir / "genre_matrix_binned.npy")
    year_indices = np.load(processed_dir / "year_indices_binned.npy")

    n_items = len(item_map)
    num_items_genres, num_genres = genre_matrix.shape
    num_years = year_indices.max() + 1

    # 1. Export User Tower (Transformer)
    print("Exporting User Tower...")
    user_tower = TransformerUserTower(
        vocab_size=n_items,
        embed_dim=settings.tower_out_dim,
        output_dim=settings.tower_out_dim,
        max_seq_len=settings.max_seq_len,
        n_heads=settings.transformer_heads,
        n_layers=settings.transformer_layers,
        dropout=settings.transformer_dropout,
        swiglu_hidden_dim=settings.swiglu_hidden_dim,
    )

    tower_path = artifacts_dir / "user_tower.pth"
    if tower_path.exists():
        state_dict = torch.load(tower_path, map_location="cpu")

        if "embedding.embedding.weight" in state_dict:
            print("Detected wrapped embedding in checkpoint. Adjusting keys...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("embedding.embedding."):
                    new_key = k.replace("embedding.embedding.", "embedding.")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict

        user_tower.load_state_dict(state_dict)
        print("Loaded user tower weights.")
    else:
        raise FileNotFoundError(
            f"User tower weights not found at {tower_path}. Please train the model first."
        )

    user_tower.eval()

    # Create wrapper for encode behavior
    user_tower_wrapper = UserTowerWrapper(user_tower)
    user_tower_wrapper.eval()

    # Dummy input for user tower
    # Shape: (Batch, SeqLen)
    dummy_seq = torch.randint(0, n_items, (1, settings.max_seq_len), dtype=torch.long)

    torch.onnx.export(
        user_tower_wrapper,
        dummy_seq,  # type: ignore
        artifacts_dir / "user_tower.onnx",
        input_names=["item_seq"],
        output_names=["user_embedding"],
        dynamic_axes={
            "item_seq": {0: "batch_size"},
            "user_embedding": {0: "batch_size"},
        },
        opset_version=18,  # Use newer opset to avoid conversion issues
    )
    print(f"User Tower exported to {artifacts_dir / 'user_tower.onnx'}")

    # 2. Export Ranker
    print("Exporting Ranker...")
    ranker = MLPRanker(
        input_dim=settings.tower_out_dim,
        hidden_dims=settings.ranker_hidden_dims,
        num_genres=num_genres,
        genre_dim=16,
        num_years=num_years,
        year_dim=8,
    )

    ranker_path = artifacts_dir / "ranker_model.pth"
    if ranker_path.exists():
        state_dict = torch.load(ranker_path, map_location="cpu")
        ranker.load_state_dict(state_dict)
        print("Loaded ranker weights.")
    else:
        print("Warning: Ranker weights not found, using initialized weights.")

    ranker.eval()

    ranker_wrapper = RankerWrapper(ranker)
    ranker_wrapper.eval()

    # Dummy inputs for ranker
    batch_size = 1
    dummy_user_emb = torch.randn(batch_size, settings.tower_out_dim)
    dummy_item_emb = torch.randn(batch_size, settings.tower_out_dim)
    dummy_genre = torch.randn(batch_size, num_genres)  # float
    dummy_year = torch.zeros(batch_size, dtype=torch.long)
    dummy_pop = torch.randn(batch_size, 1)

    torch.onnx.export(
        ranker_wrapper,
        (dummy_user_emb, dummy_item_emb, dummy_genre, dummy_year, dummy_pop),
        artifacts_dir / "ranker_model.onnx",
        input_names=[
            "user_emb",
            "item_emb",
            "genre_multihot",
            "year_idx",
            "popularity",
        ],
        output_names=["score"],
        dynamic_axes={
            "user_emb": {0: "batch_size"},
            "item_emb": {0: "batch_size"},
            "genre_multihot": {0: "batch_size"},
            "year_idx": {0: "batch_size"},
            "popularity": {0: "batch_size"},
            "score": {0: "batch_size"},
        },
        opset_version=18,
    )
    print(f"Ranker exported to {artifacts_dir / 'ranker_model.onnx'}")


if __name__ == "__main__":
    main()
