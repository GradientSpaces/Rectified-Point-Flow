"""Normalization layers for DiT point cloud model.

This module provides various normalization techniques including RMS normalization
and adaptive layer normalization with timestep conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


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
    """Adaptive layer normalization with timestep conditioning."""

    def __init__(self, emb_dim: int, act_fn: nn.Module = nn.SiLU):
        """Initialize the adaptive layer normalization.
        
        Args:
            emb_dim (int): Dimension of embeddings.
            act_fn (nn.Module): Activation function. Default: nn.SiLU.
        """
        super().__init__()
        self.timestep_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=emb_dim
        )
        self.activation = act_fn()
        self.linear = nn.Linear(emb_dim, emb_dim * 2)  # for scale and shift
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive layer normalization.
        
        Args:
            x (n_points, emb_dim): Input tensor.
            timestep (n_valid_parts,): Timestep tensor.
            batch (n_valid_parts,): Batch indices.
            
        Returns:
            (n_points, emb_dim): Normalized tensor.
        """
        emb = self.linear(
            self.activation(self.timestep_embedder(self.timestep_proj(timestep)))
        )
        scale, shift = emb.chunk(2, dim=1)      # (n_valid_parts, emb_dim) for both
        # boardcast to the same shape as x
        scale = scale[batch]                    # (n_points, emb_dim)
        shift = shift[batch]                    # (n_points, emb_dim) 
        # apply layer norm
        return self.norm(x) * (1 + scale) + shift