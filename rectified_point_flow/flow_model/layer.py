"""Diffusion Transformer layer for Rectified Point Flow."""

import flash_attn
import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward

from .norm import AdaptiveLayerNorm, MultiHeadRMSNorm


class DiTLayer(nn.Module):
    """Diffusion Transformer layer for Rectified Point Flow.
    
    This layer includes:
        1. Part-wise attention, independent for points in each part.
        2. Global attention, across all parts.
        3. Feed-forward network.

    Ref: 
        Some codes are adapted from GARF https://github.com/ai4ce/GARF
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        qkv_proj_bias: bool = False,
        attn_dtype: torch.dtype = torch.float16,
    ):
        """Initialize the DiT layer.
        
        Args:
            dim (int): Feature dimension.
            num_attention_heads (int): Number of attention heads.
            attention_head_dim (int): Dimension of each attention head.
            dropout (float): Dropout probability. Default: 0.0.
            activation_fn (str): Activation function for feed-forward. Default: "geglu".
            qkv_proj_bias (bool): Whether to use bias in QKV projections. Default: False.
            attn_dtype (torch.dtype): Data type for attention. Default: torch.float16.
        """
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.attn_dtype = attn_dtype

        # Part-wise Attention
        self.self_prenorm = AdaptiveLayerNorm(dim)
        self.self_qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_proj_bias)
        self.self_out_proj = nn.Linear(dim, dim)
        self.self_q_norm = MultiHeadRMSNorm(
            self.attention_head_dim, heads=num_attention_heads
        )
        self.self_k_norm = MultiHeadRMSNorm(
            self.attention_head_dim, heads=num_attention_heads
        )

        # Global Attention
        self.global_prenorm = AdaptiveLayerNorm(dim)
        self.global_qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_proj_bias)
        self.global_out_proj = nn.Linear(dim, dim)
        self.global_q_norm = MultiHeadRMSNorm(
            self.attention_head_dim, heads=num_attention_heads
        )
        self.global_k_norm = MultiHeadRMSNorm(
            self.attention_head_dim, heads=num_attention_heads
        )

        # Feed-forward
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
    
    @staticmethod
    def _apply_qk_norm(qkv: torch.Tensor, q_norm: nn.Module, k_norm: nn.Module) -> torch.Tensor:
        """Apply query-key normalization and keep the dtype."""
        q, k, v = qkv.unbind(dim=1)
        q, k = q_norm(q).to(v.dtype), k_norm(k).to(v.dtype)
        return torch.stack([q, k, v], dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
        self_attn_cu_seqlens: torch.Tensor,
        self_attn_max_seqlen: torch.Tensor,
        global_attn_cu_seqlens: torch.Tensor,
        global_attn_max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the DiT layer.
        
        Args:
            hidden_states (n_points, embed_dim): Input tensor.
            timestep (n_valid_parts,): Integer tensor of timesteps.
            batch (n_valid_parts, ): Integer tensor of batch indices.
            self_attn_cu_seqlens (n_valid_parts,): Integer tensor of cumulative sequence lengths for part-wise attention.
            self_attn_max_seqlen (n_valid_parts,): Integer tensor of maximum sequence length for part-wise attention.
            global_attn_cu_seqlens (n_valid_parts,): Integer tensor of cumulative sequence lengths for global attention.
            global_attn_max_seqlen (n_valid_parts,): Integer tensor of maximum sequence length for global attention.
            
        Returns:
            hidden_states (n_points, embed_dim): Output tensor.
        """
        n_points, embed_dim = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Part-wise Attention
        x = self.self_prenorm(hidden_states, timestep, batch)
        qkv = self.self_qkv_proj(x).reshape(
            n_points, 3, self.num_attention_heads, self.attention_head_dim
        )
        qkv = self._apply_qk_norm(qkv, self.self_q_norm, self.self_k_norm)

        attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            cu_seqlens=self_attn_cu_seqlens,
            max_seqlen=self_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim).to(dtype)

        hidden_states = hidden_states + self.self_out_proj(attn_output)

        # 2. Global Attention
        x = self.global_prenorm(hidden_states, timestep, batch)
        qkv = self.global_qkv_proj(x).reshape(
            n_points, 3, self.num_attention_heads, self.attention_head_dim
        )
        qkv = self._apply_qk_norm(qkv, self.global_q_norm, self.global_k_norm)

        attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            cu_seqlens=global_attn_cu_seqlens,
            max_seqlen=global_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim).to(dtype)

        hidden_states = hidden_states + self.global_out_proj(attn_output)

        # 3. Feed-forward
        x = self.ff_norm(hidden_states)
        hidden_states = hidden_states + self.ff(x)

        return hidden_states