"""Transformer layers for DiT point cloud model."""

import torch
import torch.nn as nn
import flash_attn
from diffusers.models.attention import FeedForward

from .norm import MultiHeadRMSNorm, AdaptiveLayerNorm


class DiTLayer(nn.Module):
    """Diffusion Transformer layer for point cloud processing.
    
    This layer includes:
    1. Part-wise attention, independent for points in each part.
    2. Global attention, across all parts.
    3. Feed-forward network.
    
    Args:
        dim: Feature dimension.
        num_attention_heads: Number of attention heads.
        attention_head_dim: Dimension of each attention head.
        dropout: Dropout probability.
        activation_fn: Activation function for feed-forward.
        num_embeds_ada_norm: Number of embeddings for adaptive normalization.
        attention_bias: Whether to use bias in attention projections.
        norm_elementwise_affine: Whether to use element-wise affine in norm.
        final_dropout: Whether to apply final dropout.
        attn_dtype: Data type for attention. Default: torch.float16.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        attn_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.attn_dtype = attn_dtype

        # Part-wise Attention
        self.norm1 = AdaptiveLayerNorm(dim, num_embeds_ada_norm)
        self.self_attn_to_qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.self_attn_to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.self_q_norm = MultiHeadRMSNorm(
            self.attention_head_dim, heads=num_attention_heads
        )
        self.self_k_norm = MultiHeadRMSNorm(
            self.attention_head_dim, heads=num_attention_heads
        )

        # Global Attention
        self.norm2 = AdaptiveLayerNorm(dim, num_embeds_ada_norm)
        self.global_attn_to_qkv = nn.Linear(dim, dim * 3, bias=attention_bias)
        self.global_attn_to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.global_q_norm = MultiHeadRMSNorm(
            self.attention_head_dim, heads=num_attention_heads
        )
        self.global_k_norm = MultiHeadRMSNorm(
            self.attention_head_dim, heads=num_attention_heads
        )

        # Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )
    
    @staticmethod
    def apply_qk_norm(qkv: torch.Tensor, q_norm: nn.Module, k_norm: nn.Module) -> torch.Tensor:
        """Apply query-key normalization to the qkv tensor.
        
        Args:
            qkv: QKV tensor of shape (n_points, 3, num_heads, head_dim).
            q_norm: Query normalization module.
            k_norm: Key normalization module.
            
        Returns:
            QKV tensor of shape (n_points, 3, num_heads, head_dim).
        """
        q, k, v = qkv.unbind(dim=1)
        q = q_norm(q).to(v.dtype)
        k = k_norm(k).to(v.dtype)
        qkv = torch.stack([q, k, v], dim=1)
        return qkv

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        batch: torch.Tensor,
        self_attn_seqlens: torch.Tensor,
        self_attn_cu_seqlens: torch.Tensor,
        self_attn_max_seqlen: torch.Tensor,
        global_attn_seqlens: torch.Tensor,
        global_attn_cu_seqlens: torch.Tensor,
        global_attn_max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the DiT layer.
        
        Args:
            hidden_states: Input tensor of shape (n_points, embed_dim).
            timestep: Timestep tensor of shape (n_valid_parts,).
            batch: Batch indices of shape (n_valid_parts,).
            self_attn_seqlens: Sequence lengths for self attention.
            self_attn_cu_seqlens: Cumulative sequence lengths for self attention.
            self_attn_max_seqlen: Maximum sequence length for self attention.
            global_attn_seqlens: Sequence lengths for global attention.
            global_attn_cu_seqlens: Cumulative sequence lengths for global attention.
            global_attn_max_seqlen: Maximum sequence length for global attention.
            
        Returns:
            Output tensor of shape (n_points, embed_dim).
        """
        n_points, embed_dim = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Part-wise Attention
        norm_hidden_states = self.norm1(hidden_states, timestep, batch)
        qkv = self.self_attn_to_qkv(norm_hidden_states).reshape(
            n_points, 3, self.num_attention_heads, self.attention_head_dim
        )
        qkv = self.apply_qk_norm(qkv, self.self_q_norm, self.self_k_norm)

        attn_output = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            cu_seqlens=self_attn_cu_seqlens,
            max_seqlen=self_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim).to(dtype)
        attn_output = self.self_attn_to_out(attn_output)
        hidden_states = hidden_states + attn_output

        # 2. Global Attention
        norm_hidden_states = self.norm2(hidden_states, timestep, batch)
        qkv = self.global_attn_to_qkv(norm_hidden_states).reshape(
            n_points, 3, self.num_attention_heads, self.attention_head_dim
        )
        qkv = self.apply_qk_norm(qkv, self.global_q_norm, self.global_k_norm)

        global_out_flash = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv=qkv.to(self.attn_dtype),
            cu_seqlens=global_attn_cu_seqlens,
            max_seqlen=global_attn_max_seqlen,
            dropout_p=0.0,
        ).view(n_points, embed_dim).to(dtype)
        global_out_flash = self.global_attn_to_out(global_out_flash)
        hidden_states = hidden_states + global_out_flash

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states

        return hidden_states