"""Normalization layers for DiT point cloud model.

This module provides various normalization techniques including RMS normalization
and adaptive layer normalization with timestep conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


class MultiHeadRMSNorm(nn.Module):
    """Multi-head RMS normalization layer. 

    Ref: 
        https://github.com/lucidrains/mmdit/blob/main/mmdit/mmdit_pytorch.py

    Args:
        dim: Feature dimension.
        heads: Number of attention heads.
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head RMS normalization."""
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization with timestep conditioning.
        
    Args:
        embedding_dim: Dimension of embeddings.
        num_embeddings: Number of embeddings for timestep conditioning.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.timestep_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.activation = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive layer normalization.
        
        Args:
            x: Input tensor of shape (n_points, embedding_dim).
            timestep: Timestep tensor of shape (n_valid_parts,).
            batch: Batch indices of shape (n_valid_parts,).
            
        Returns:
            Normalized tensor conditioned on timestep.
        """
        emb = self.linear(
            self.activation(self.timestep_embedder(self.timestep_proj(timestep)))
        )
        scale, shift = emb.chunk(2, dim=1)
        scale = scale[batch]
        shift = shift[batch]
        return self.norm(x) * (1 + scale) + shift