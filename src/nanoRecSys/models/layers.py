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
import torch.nn.functional as F

from nanoRecSys.config import settings


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        rope_base = settings.rope_base
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cached_cos", emb.cos())
        self.register_buffer("cached_sin", emb.sin())

    def forward(self, x, seq_len):
        if (
            self.cached_cos is None
            or self.cached_cos.shape[0] < seq_len
            or self.cached_cos.device != x.device
        ):
            t = torch.arange(
                max(seq_len, self.max_seq_len),
                device=x.device,
                dtype=self.inv_freq.dtype,  # type: ignore
            )
            freqs = torch.outer(t, self.inv_freq)  # type: ignore
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cached_cos = emb.cos()
            self.cached_sin = emb.sin()

        return (
            self.cached_cos[:seq_len].to(x.dtype),  # type: ignore
            self.cached_sin[:seq_len].to(x.dtype),  # type: ignore
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, SeqLen, Heads, HeadDim) or (B, SeqLen, Dim)
    # cos, sin: (SeqLen, Dim) -> (1, SeqLen, 1, Dim) or compatible
    if x.ndim == 4:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif x.ndim == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)


class RoPEAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        assert self.head_dim * n_heads == d_model, (
            "d_model must be divisible by n_heads"
        )

        # Merge Q, K, V projections for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rope=None):
        # x: (B, SeqLen, D)
        B, L, D = x.shape

        # Combined QKV projection
        qkv = self.qkv_proj(x)
        # Split into Q, K, V
        # (B, L, 3, n_heads, head_dim)
        qkv = qkv.view(B, L, 3, self.n_heads, self.head_dim)

        q = qkv[:, :, 0]  # (B, L, H, D_h)
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        if rope is not None:
            cos, sin = rope(x, L)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Prepare for SDPA: (B, H, L, D_h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None and mask.ndim == 2:  # (B, L)
            mask = mask.view(B, 1, 1, L)

        # Input mask has True for masked positions
        # SDPA expects True for allowed positions
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~mask if mask is not None else None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        # Output
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        # SwiGLU: (Swish(xW_g) * (xW_1)) W_2

        # Merge Gate and Value projections
        self.w_gate_val = nn.Linear(d_model, 2 * dim_feedforward, bias=False)
        self.w_2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)

        # Projected
        gate_val = self.w_gate_val(x)  # (B, L, 2 * dim_feedforward)
        gate, val = gate_val.chunk(2, dim=-1)

        # Activation
        gate = F.silu(gate)

        # Element-wise mult
        x = gate * val
        x = self.dropout(x)

        # Output project
        return self.w_2(x)


class TransformerBlockWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = RoPEAttentionLayer(d_model, n_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Replace standard MLP with SwiGLU
        self.mlp = SwiGLU(d_model, dim_feedforward, dropout)

    def forward(self, x, mask=None, rope=None):
        # Pre-Norm implementation: x = x + f(norm(x))

        # Self Attention
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, mask, rope)

        # FFN
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)

        return x
