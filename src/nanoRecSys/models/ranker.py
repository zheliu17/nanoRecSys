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

import torch
import torch.nn as nn


class RankerModel(nn.Module):
    def __init__(
        self,
        input_dim=128,
        hidden_dims=[256, 128, 64],
        num_genres=20,
        genre_dim=16,
        num_years=100,
        year_dim=8,
    ):
        super().__init__()

        # Embeddings
        self.genre_embeddings = nn.Embedding(num_genres, genre_dim)
        self.year_embeddings = nn.Embedding(num_years, year_dim)

        nn.init.kaiming_uniform_(self.genre_embeddings.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.year_embeddings.weight, nonlinearity="relu")

        # Example input features:
        # User (128) + Item (128) + Prod (128) + Pop (1) + Year (8) + Genre (16) + IsUnknown (1)
        self.concat_dim = (
            input_dim + input_dim + input_dim + 1 + year_dim + genre_dim + 1
        )

        # The MLP (Dense Layers) with Batch Normalization
        layers = []
        curr_dim = self.concat_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity="relu")

            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            curr_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # 3. Final Output (Logits)
        self.output = nn.Linear(curr_dim, 1)
        nn.init.kaiming_uniform_(
            self.output.weight, nonlinearity="relu"
        )  # optional for last layer
        # Remove Sigmoid from forward training path

    def forward(
        self,
        user_emb,
        item_emb,
        genre_multihot,
        year_idx,
        popularity,
        id_dropout_prob=0.0,
        new_item_mask=None,
    ):
        """
        Args:
            user_emb: (B, input_dim)
            item_emb: (B, input_dim)
            genre_multihot: (B, num_genres)
            year_idx: (B,)
            popularity: (B, 1) or (B,)
            id_dropout_prob: float, probability of masking item embedding during training (for cold-start simulation)
            new_item_mask: (B, 1) or (B,), binary mask where 1 indicates new/unknown items.
                Overrides id_dropout_prob if provided. Used for evaluation with new items.
        """
        batch_size = user_emb.shape[0]
        device = user_emb.device

        # Handle new items: explicit mask takes priority
        if new_item_mask is not None:
            # Convert to proper shape if needed
            if new_item_mask.dim() == 1:
                new_item_mask = new_item_mask.unsqueeze(1)
            # new_item_mask is 1 for unknown items, 0 for known items
            # mask is 1 for known items, 0 for unknown items (inverse)
            mask = 1.0 - new_item_mask
            is_unknown_item = new_item_mask.float()
        else:
            # Training-time cold-start simulation via dropout
            mask = torch.ones((batch_size, 1), device=device)
            is_unknown_item = torch.zeros((batch_size, 1), device=device)

            if self.training and id_dropout_prob > 0.0:
                mask = torch.bernoulli(
                    torch.full((batch_size, 1), 1.0 - id_dropout_prob, device=device)
                )
                is_unknown_item = 1.0 - mask

        i_emb_masked = item_emb * mask

        # Element-wise product (with MASKED item)
        # If item is unknown (masked), dot product will be 0, which is correct
        # Explicit Product https://arxiv.org/abs/1708.05031
        dot_product = user_emb * i_emb_masked  # (B, input_dim)

        # Ensure popularity is (B, 1)
        if popularity.dim() == 1:
            popularity = popularity.unsqueeze(1)

        # Year Embedding
        year_emb = self.year_embeddings(year_idx)  # (B, 8)

        # Genre Embedding (Mean Pooling)
        # (B, n_genres) @ (n_genres, 16) -> (B, 16) sum
        genre_sum = torch.matmul(genre_multihot, self.genre_embeddings.weight)
        # Counts per item
        genre_counts = genre_multihot.sum(dim=1, keepdim=True).clamp_min(1.0)
        genre_emb = genre_sum / genre_counts  # (B, 16)

        # Concatenate
        x = torch.cat(
            [
                user_emb,  # input_dim
                i_emb_masked,  # input_dim
                dot_product,  # input_dim
                popularity,  # 1
                year_emb,  # 8
                genre_emb,  # 16
                is_unknown_item,  # 1
            ],
            dim=1,
        )

        x = self.mlp(x)
        # Return logits (no sigmoid)
        return self.output(x).squeeze()

    def predict(
        self,
        user_emb,
        item_emb,
        genre_multihot,
        year_idx,
        popularity,
        new_item_mask=None,
    ):
        """Helper for inference producing probabilities.

        Args:
            new_item_mask: (B, 1) or (B,), binary mask where 1 indicates new/unknown items
        """
        logits = self.forward(
            user_emb,
            item_emb,
            genre_multihot,
            year_idx,
            popularity,
            id_dropout_prob=0.0,
            new_item_mask=new_item_mask,
        )
        return torch.sigmoid(logits)
