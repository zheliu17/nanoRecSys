import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Tower(nn.Module):
    """
    Generic Tower (Encoder): Embedding -> MLP -> Output
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        output_dim: int = 128,
        hidden_dims: List[int] = [256, 192],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        layers = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final projection
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size,)
        emb = self.embedding(x)  # (batch, embed)
        out = self.mlp(emb)  # (batch, tower_out_dim)
        return F.normalize(out, p=2, dim=1)  # L2 normalize for cosine similarity


class UserTower(Tower):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)


class ItemTower(Tower):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)


class TwoTowerModel(nn.Module):
    """
    Combined model for training convenience
    """

    def __init__(self, user_tower: UserTower, item_tower: ItemTower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_emb = self.user_tower(users)
        i_emb = self.item_tower(items)
        return u_emb, i_emb
