"""Standalone SpecFormer architecture for inference.

Bundled from AstroCLIP (https://github.com/PolymathicAI/AstroCLIP) to avoid
a transitive dependency on dinov2 which pins torch==2.0.0.  Only the modules
required for forward-pass inference are included; training helpers
(training_step, validation_step, LR schedulers) are omitted.

Original authors: Liam Parker, Leopoldo Sarra, Francois Lanusse,
                  Siavash Golkar, Miles Cranmer (Flatiron Institute / Cambridge)
License: MIT
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Transformer building blocks (from astroclip.modules)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-5, bias=True, dtype=None):
        super().__init__()
        self.eps = eps
        if isinstance(shape, int):
            self.normalized_shape = (shape,)
        else:
            self.normalized_shape = tuple(shape)
        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, causal, dropout, bias=True):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim should be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.attention = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.uses_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attention(x).split(self.embedding_dim, dim=2)
        nh, hs = self.num_heads, C // self.num_heads
        k = k.view(B, T, nh, hs).transpose(1, 2)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        if self.uses_flash:
            dropout_p = self.dropout if self.training else 0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=self.causal
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            if self.causal:
                mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) == 0
                att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attention_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.residual_dropout(self.projection(y))


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, activation=None, dropout=0.0, bias=True):
        super().__init__()
        self.activation = activation if activation is not None else nn.GELU()
        self.encoder = nn.Linear(in_features, hidden_features, bias=bias)
        self.decoder = nn.Linear(hidden_features, in_features, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.activation(self.encoder(x))
        x = self.decoder(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, causal, dropout, bias=True, mlp_expansion=4):
        super().__init__()
        self.layernorm1 = LayerNorm(embedding_dim, bias=bias)
        self.attention = SelfAttention(embedding_dim, num_heads, bias=bias, dropout=dropout, causal=causal)
        self.layernorm2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = MLP(embedding_dim, mlp_expansion * embedding_dim, nn.GELU(), dropout=dropout, bias=bias)

    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x


def _init_by_depth(module, depth):
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        std = 1 / math.sqrt(2 * fan_in * depth)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# SpecFormer (from astroclip.models.specformer, rewritten as plain nn.Module)
# ---------------------------------------------------------------------------

class SpecFormer(nn.Module):
    """1-D Transformer for galaxy spectra (inference-only, plain nn.Module).

    Architecture and weight-loading are compatible with the original
    AstroCLIP ``SpecFormer`` Lightning module so that checkpoints produced
    by AstroCLIP training can be loaded directly.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 50,
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()

        # Store hyper-parameters for preprocessing helpers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len
        self.mask_num_chunks = mask_num_chunks
        self.mask_chunk_width = mask_chunk_width
        self.slice_section_length = slice_section_length
        self.slice_overlap = slice_overlap

        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    causal=False,
                    dropout=dropout,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)

        self._reset_parameters_datapt()

    # ---- forward pass ----------------------------------------------------

    def forward(self, x: Tensor):
        x = self.preprocess(x)
        return self.forward_without_preprocessing(x)

    def forward_without_preprocessing(self, x: Tensor):
        t = x.shape[1]
        if t > self.max_len:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.max_len}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)
        data_emb = self.data_embed(x)
        pos_emb = self.position_embed(pos)
        x = self.dropout(data_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.final_layernorm(x)
        reconstructions = self.head(x)
        return {"reconstructions": reconstructions, "embedding": x}

    def forward_layerwise(self, x: Tensor):
        """Forward pass collecting per-layer representations."""
        x = self.preprocess(x)
        t = x.shape[1]
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)
        data_emb = self.data_embed(x)
        pos_emb = self.position_embed(pos)
        x = self.dropout(data_emb + pos_emb)

        layer_embeddings = [x]
        for block in self.blocks:
            x = block(x)
            layer_embeddings.append(x)
        x = self.final_layernorm(x)
        layer_embeddings.append(x)

        return {"layer_embeddings": layer_embeddings, "embedding": x}

    # ---- preprocessing ---------------------------------------------------

    def preprocess(self, x):
        std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)
        x = (x - mean) / std
        x = self._slice(x)
        x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
        x[:, 0, 0] = (mean.squeeze() - 2) / 2
        x[:, 0, 1] = (std.squeeze() - 2) / 8
        return x

    def _slice(self, x):
        start_indices = np.arange(
            0,
            x.shape[1] - self.slice_overlap,
            self.slice_section_length - self.slice_overlap,
        )
        sections = [
            x[:, start : start + self.slice_section_length].transpose(1, 2)
            for start in start_indices
        ]
        if sections[-1].shape[1] < self.slice_section_length:
            sections.pop(-1)
        return torch.cat(sections, 1)

    # ---- initialisation --------------------------------------------------

    def _reset_parameters_datapt(self):
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)
        self.blocks.apply(lambda m: _init_by_depth(m, self.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))
