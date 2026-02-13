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

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoRecSys.config import settings
from nanoRecSys.models.layers import RMSNorm, RotaryEmbedding, TransformerBlock


class Tower(nn.Module):
    """
    Generic Tower (Encoder): Embedding -> MLP -> Output
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        use_projection: bool = True,
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

        # Final projection (optional)
        self.use_projection = use_projection
        if self.use_projection:
            layers.append(nn.Linear(in_dim, output_dim))
            self.output_dim = output_dim
        else:
            # No final projection; output dim equals last hidden dim (or embed_dim if no hidden)
            self.output_dim = in_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size,) or (batch_size, num_items)
        emb = self.embedding(x)  # (..., embed)
        out = self.mlp(emb)  # (..., tower_out_dim)
        return F.normalize(out, p=2, dim=-1)  # L2 normalize for cosine similarity

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference helper: Returns (Batch, OutputDim).
        For MLP/ID-based towers, this is just the forward pass.
        """
        return self(x)


class UserTower(Tower):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)


class TransformerUserTower(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        output_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        swiglu_hidden_dim: Optional[int] = None,
        shared_embedding: Optional[nn.Module] = None,
        pos_embedding_type: str = settings.positional_embedding_type,
    ):
        super().__init__()

        if shared_embedding is not None:
            self.embedding = shared_embedding
            self.has_shared_embedding = True
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.has_shared_embedding = False

        # No absolute positional embedding forRoPE
        # self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.pos_embedding_type = pos_embedding_type
        if self.pos_embedding_type == "absolute":
            self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
            self.rope = None
        else:
            head_dim = embed_dim // n_heads
            self.rope = RotaryEmbedding(head_dim, max_seq_len)

        self.emb_dropout = nn.Dropout(dropout)

        # SwiGLU: round up to nearest multiple of 256
        if swiglu_hidden_dim is None:
            swiglu_hidden_dim = int(embed_dim * 8 / 3)
            swiglu_hidden_dim = (swiglu_hidden_dim + 255) // 256 * 256

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=embed_dim,
                    n_heads=n_heads,
                    dim_feedforward=swiglu_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Causal Mask Cache
        # Shape: (1, 1, max_seq_len, max_seq_len)
        # 1 means Masked (Future), 0 means Visible
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer(
            "causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False
        )

        # Final Norm for Pre-Norm Architecture
        self.norm = RMSNorm(embed_dim)

        self.projection = nn.Linear(embed_dim, output_dim)

        self.apply(self._init_weights)
        self._apply_residual_scaling(n_layers)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # elif isinstance(module, nn.Embedding):
        #     nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()

        # elif isinstance(module, (nn.LayerNorm, RMSNorm)):
        #     if hasattr(module, "bias") and module.bias is not None:
        #         nn.init.zeros_(module.bias)  # type: ignore
        #     if hasattr(module, "weight") and module.weight is not None:
        #         nn.init.ones_(module.weight)

    def _apply_residual_scaling(self, n_layers):
        """
        Scales the weights of the residual projection layers by 1/sqrt(2 * L).
        """
        scale_factor = 1.0 / math.sqrt(2.0 * n_layers)

        for block in self.blocks:
            # 1. Attention Output Projection (attn.out_proj)
            # We scale the existing Xavier weights down
            block.attn.out_proj.weight.data.mul_(scale_factor)  # type: ignore

            # 2. FeedForward Output Projection (mlp.w_2 in SwiGLU)
            block.mlp.w_2.weight.data.mul_(scale_factor)  # type: ignore

    def forward(self, item_seq: torch.Tensor) -> torch.Tensor:
        # item_seq: (batch, seq_len)
        batch_size, seq_len = item_seq.shape

        # Padding Mask: (B, 1, 1, L) - Mask keys that are pad
        # True = Pad/Ignore
        pad_mask = (item_seq == 0).view(batch_size, 1, 1, seq_len)

        # Get Causal Mask from Cache
        # (1, 1, L, L)
        causal_mask_slice = self.causal_mask[:, :, :seq_len, :seq_len]  # type: ignore

        # Combined Mask: True if either pad or future
        mask = pad_mask | causal_mask_slice

        valid_vals = item_seq != 0
        lookup_indices = torch.where(
            valid_vals, item_seq - 1, torch.zeros_like(item_seq)
        )
        x = self.embedding(lookup_indices)

        if self.pos_embedding_type == "absolute":
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)

        # Manually zero out padding positions
        x = x * valid_vals.unsqueeze(-1).type_as(x)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x, mask=mask, rope=self.rope)

        x = self.norm(x)

        # Return full sequence embeddings for dense training
        # x: (B, L, D)
        out = self.projection(x)
        return F.normalize(out, p=2, dim=-1)

    def encode(self, item_seq: torch.Tensor) -> torch.Tensor:
        """
        Inference helper: Returns only the last step's embedding.
        Input: (Batch, SeqLen)
        Output: (Batch, OutputDim)
        """
        out = self(item_seq)
        return out[:, -1, :]


class ItemTower(Tower):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__(vocab_size, **kwargs)
