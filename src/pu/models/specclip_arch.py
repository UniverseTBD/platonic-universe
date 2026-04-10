"""Standalone SpecCLIP LRS encoder architecture for inference.

Bundled from SpecCLIP (https://github.com/Xiaosheng-Zhao/SpecCLIP) to avoid
a transitive dependency on Lightning.  Only the modules required for
forward-pass inference are included.

Original authors: Xiaosheng Zhao et al.
License: MIT
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pu.models.specformer_arch import LayerNorm, TransformerBlock, _init_by_depth


class SpecCLIPEncoder(nn.Module):
    """1-D masked transformer for LAMOST LRS spectra (inference-only).

    Architecture and weight-loading are compatible with the original
    SpecCLIP ``SpecFormerControl20_wstd`` Lightning module so that
    checkpoints can be loaded directly.

    Key differences from AstroCLIP SpecFormer:
    - Padding: ``pad=(1, 0, 1, 0)`` → tokens have ``section_length + 1`` features
    - Stats token: stores ``log10(std)`` at position ``[0, 0]`` (no mean)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int = 400,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 20,
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len
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

    def preprocess(self, x):
        std, mean = x.std(1, keepdim=True), x.mean(1, keepdim=True)
        x = (x - mean) / std
        x = self._slice(x)
        x = F.pad(x, pad=(1, 0, 1, 0), mode="constant", value=0)
        x[:, 0, 0] = torch.log10(std.squeeze())
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

    def _reset_parameters_datapt(self):
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)
        self.blocks.apply(lambda m: _init_by_depth(m, self.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))
