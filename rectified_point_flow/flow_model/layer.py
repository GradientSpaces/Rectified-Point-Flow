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
        dropout: float = 0,
        softcap: float = 0,
        activation_fn: str = "geglu",
        qkv_proj_bias: bool = False,
        qk_norm: bool = True,
        attn_dtype: torch.dtype = torch.float16,
    ):
        """Initialize the DiT layer.
        
        Args:
            dim (int): Feature dimension.
            num_attention_heads (int): Number of attention heads.
            attention_head_dim (int): Dimension of each attention head.
            dropout (float): Dropout probability. Default: 0.0.
            softcap (float): Soft cap for attention scores. Default: 0.0.
            activation_fn (str): Activation function for feed-forward. Default: "geglu".
            qkv_proj_bias (bool): Whether to use bias in QKV projections. Default: False.
            qk_norm (bool): Whether to use query-key normalization. Default: True.
            attn_dtype (torch.dtype): Data type for attention. Default: torch.float16.
        """
        super().__init__()

        assert dim == attention_head_dim * num_attention_heads, \
            "dim must be equal to attention_head_dim * num_attention_heads"
        
        self.dim = dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.softcap = softcap
        self.qk_norm = qk_norm
        self.attn_dtype = attn_dtype

        # Part-wise Attention
        self.self_prenorm = AdaptiveLayerNorm(dim)
        self.self_qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_proj_bias)
        self.self_out_proj = nn.Linear(dim, dim)
        if qk_norm:
            self.self_q_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)
            self.self_k_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)

        # Global Attention
        self.global_prenorm = AdaptiveLayerNorm(dim)
        self.global_qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_proj_bias)
        self.global_out_proj = nn.Linear(dim, dim)
        if qk_norm:
            self.global_q_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)
            self.global_k_norm = MultiHeadRMSNorm(self.head_dim, self.num_heads)

        # Feed-forward
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
    
    @staticmethod
    def _qk_norm(qkv: torch.Tensor, q_norm: nn.Module, k_norm: nn.Module) -> torch.Tensor:
        """Apply query-key normalization and keep the dtype."""
        q, k, v = qkv.unbind(dim=1)
        q, k = q_norm(q).to(v.dtype), k_norm(k).to(v.dtype)
        return torch.stack([q, k, v], dim=1)

    def _part_attention(self, x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        """Apply part-wise attention."""
        B, N, _ = x.shape
        qkv = self.self_qkv_proj(x)                                    # (B, N, 3 * embed_dim)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)        # (B*N, 3, num_heads, head_dim)
        
        if self.qk_norm:
            qkv = self._qk_norm(qkv, self.self_q_norm, self.self_k_norm)

        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            softcap=self.softcap,
        ).to(x.dtype)
        out = out.view(B, N, self.dim)                                  # (B, N, embed_dim)
        return self.self_out_proj(out)
    
    def _global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global attention."""
        B, N, _ = x.shape
        qkv = self.global_qkv_proj(x)                                   # (B, N, 3 * embed_dim)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)         # (B*N, 3, num_heads, head_dim)

        if self.qk_norm:
            qkv = self._qk_norm(qkv, self.global_q_norm, self.global_k_norm)

        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        out = flash_attn.flash_attn_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            softcap=self.softcap,
        ).to(x.dtype)
        out = out.view(B, N, self.dim)                                   # (B, N, embed_dim)
        return self.global_out_proj(out)

    # @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        part_cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Forward pass through the DiT layer.
        
        Args:
            hidden_states (B, N, dim): Input tensor.
            timestep (B, ): Timestep values.
            part_cu_seqlens (N_valid_parts + 1, ): Cumulative lengths for each part.
            max_seqlen (int): Maximum sequence length for part-wise attention.

        Returns:
            hidden_states (B, N, dim): Output tensor.
        """
        # 1. Part-wise Attention
        x = self.self_prenorm(hidden_states, timestep)
        part_attn = self._part_attention(x, part_cu_seqlens, max_seqlen)
        hidden_states = hidden_states + part_attn

        # 2. Global Attention
        x = self.global_prenorm(hidden_states, timestep)
        global_attn = self._global_attention(x)
        hidden_states = hidden_states + global_attn

        # 3. Feed-forward
        x = self.ff_norm(hidden_states)
        hidden_states = hidden_states + self.ff(x)

        return hidden_states
