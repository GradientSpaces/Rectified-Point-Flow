"""Point Cloud Diffusion Transformer (DiT) model."""

import torch
import torch.nn as nn

from .embedding import PointCloudEncodingManager
from .layer import DiTLayer


class PointCloudDiT(nn.Module):
    """Point cloud Diffusion Transformer model.
    
    A simplified transformer-based diffusion model for processing structured point cloud data
    with part-based representations. The model uses hierarchical attention mechanisms
    with both local (within-part) and global (cross-part) attention.
    
    Query-key normalization is always enabled, and bidirectional processing is disabled
    for simplicity.
    
    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        embed_dim: Embedding dimension for the transformer.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attn_dtype: torch.dtype = torch.float16,
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attn_dtype = attn_dtype
        self.activation = activation

        # Reference part embedding for distinguishing anchor vs. moving parts
        self.ref_part_emb = nn.Embedding(2, self.embed_dim)

        # Point cloud encoding manager
        self.encoding_manager = PointCloudEncodingManager(
            in_dim=self.in_dim,
            embed_dim=self.embed_dim,
            multires=10
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            DiTLayer(
                dim=self.embed_dim,
                num_attention_heads=self.num_heads,
                attention_head_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                activation_fn="geglu",
                num_embeds_ada_norm=6 * self.embed_dim,
                attention_bias=False,
                norm_elementwise_affine=True,
                final_dropout=False,
                attn_dtype=self.attn_dtype,
            )
            for _ in range(self.num_layers)
        ])

        # MLP for final predictions
        self.final_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            self.activation(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            self.activation(),
            nn.Linear(self.embed_dim // 2, out_dim),
        )

    def add_reference_part_embedding(
        self,
        x_emb: torch.Tensor,      # (n_points, embed_dim)
        ref_part: torch.Tensor,   # (n_valid_parts,) boolean
        latent: dict,             # PointTransformer Point instance
    ) -> torch.Tensor:
        """Add reference part embeddings to distinguish reference from moving parts.
        
        Args:
            x_emb: Input embeddings of shape (n_points, embed_dim).
            ref_part: Boolean tensor indicating reference parts of shape (n_valid_parts,).
            latent: Dictionary containing point cloud features and metadata.
            
        Returns:
            Embeddings with reference part information added.
        """
        # ref_part_emb.weight[0] for non-ref part
        # ref_part_emb.weight[1] for ref part
        ref_part_broadcasted = ref_part[latent["batch"]]  # (n_points,)
        ref_part_emb = self.ref_part_emb.weight[0].repeat(ref_part_broadcasted.shape[0], 1)
        ref_part_emb[ref_part_broadcasted.to(torch.bool)] = self.ref_part_emb.weight[1]
        x_emb = x_emb + ref_part_emb
        return x_emb

    def forward(
        self,
        x: torch.Tensor,          # (n_points, 3)
        timesteps: torch.Tensor,  # (n_valid_parts,)
        latent: dict,             # PointTransformer Point instance
        part_valids: torch.Tensor,# (B, P)
        scale: torch.Tensor,      # (n_valid_parts, 1)
        ref_part: torch.Tensor,   # (n_valid_parts,)
    ) -> dict:
        """Forward pass through the PointCloudDiT model.
        
        Args:
            x: Input tensor of shape (n_points, 3).
            timesteps: Diffusion timesteps of shape (n_valid_parts,).
            latent: Dictionary containing point cloud features and metadata:
                - "coord": Point coordinates
                - "feat": Point features  
                - "normal": Point normals
                - "batch": Batch indices
            part_valids: Valid parts mask of shape (B, P).
            scale: Scale factors of shape (n_valid_parts, 1).
            ref_part: Reference part indicators of shape (n_valid_parts,).
            
        Returns:
            Dictionary containing predictions with key "pred".
        """

        # Encoding
        x = self.encoding_manager(x, latent, scale)
        x = self.add_reference_part_embedding(x, ref_part, latent)
        
        # Prepare attention metadata
        self_attn_seqlen = torch.bincount(latent["batch"])  # (n_valid_parts,)
        self_attn_max_seqlen = self_attn_seqlen.max()
        self_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(self_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)
        points_per_part = torch.zeros_like(part_valids, dtype=self_attn_seqlen.dtype)
        points_per_part[part_valids] = self_attn_seqlen
        global_attn_seqlen = points_per_part.sum(1)
        global_attn_max_seqlen = global_attn_seqlen.max()
        global_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(global_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(
                hidden_states=x,
                timestep=timesteps,
                batch=latent["batch"],
                self_attn_seqlens=self_attn_seqlen,
                self_attn_cu_seqlens=self_attn_cu_seqlens,
                self_attn_max_seqlen=self_attn_max_seqlen,
                global_attn_seqlens=global_attn_seqlen,
                global_attn_cu_seqlens=global_attn_cu_seqlens,
                global_attn_max_seqlen=global_attn_max_seqlen,
            )

        return self.final_mlp(x.float())
    

if __name__ == "__main__":
    model = PointCloudDiT(
        in_dim=512,
        out_dim=6,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
    )
    print(f"PointCloudDiT with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")