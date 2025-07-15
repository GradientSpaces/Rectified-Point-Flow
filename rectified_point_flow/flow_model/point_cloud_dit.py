"""Multi-part Point Cloud Diffusion Transformer (DiT) model."""

import torch
import torch.nn as nn

from .embedding import PointCloudEncodingManager
from .layer import DiTLayer


class PointCloudDiT(nn.Module):
    """A transformer-based diffusion model for multi-part point cloud data.

    Ref:
        DiT: https://github.com/facebookresearch/DiT
        mmdit: https://github.com/lucidrains/mmdit/tree/main/mmdit
        GARF: https://github.com/ai4ce/GARF
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        attn_dtype: torch.dtype = torch.float16,
        final_mlp_act: nn.Module = nn.SiLU,
    ):
        """
        Args:
            in_dim: Input dimension of the point features (e.g., 64).
            out_dim: Output dimension (e.g., 3 for velocity field).
            embed_dim: Hidden dimension of the transformer layers (e.g., 512).
            num_layers: Number of transformer layers (e.g., 6). 
            num_heads: Number of attention heads (e.g., 8).
            dropout_rate: Dropout rate, default 0.0.
            attn_dtype: Attention data type, default float16.
            final_mlp_act: Activation function for the final MLP, default SiLU.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attn_dtype = attn_dtype
        self.final_mlp_act = final_mlp_act

        # Reference part embedding for distinguishing anchor vs. moving parts
        self.anchor_part_emb = nn.Embedding(2, self.embed_dim)

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
                attn_dtype=self.attn_dtype,
            )
            for _ in range(self.num_layers)
        ])

        # MLP for final predictions
        self.final_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            self.final_mlp_act(),
            nn.Linear(self.embed_dim // 2, out_dim, bias=False)  # No bias for 3D coordinates
        )

    def _add_anchor_embedding(
        self,
        x_emb: torch.Tensor,
        anchor_part: torch.Tensor,
        latent: dict,
    ) -> torch.Tensor:
        """Add anchor part embeddings to distinguish anchor from moving parts.
        
        Args:
            x_emb (n_points, embed_dim): Point cloud features.
            anchor_part (n_valid_parts, ): Boolean tensor indicating anchor parts.
            latent: PointTransformer's Point instance of conditional point cloud.
            
        Returns:
            (n_points, embed_dim) Point cloud features with anchor part information added.
        """
        # anchor_part_emb.weight[0] for non-anchor part
        # anchor_part_emb.weight[1] for anchor part
        anchor_part_broadcasted = anchor_part[latent["batch"]]          # (n_points,)
        anchor_part_emb = self.anchor_part_emb.weight[0].repeat(
            anchor_part_broadcasted.shape[0], 1
        )                                                               # (n_points, embed_dim)
        anchor_part_emb[anchor_part_broadcasted.to(torch.bool)] = self.anchor_part_emb.weight[1]  # (n_points, embed_dim)
        x_emb = x_emb + anchor_part_emb
        return x_emb

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        latent: dict,
        part_valids: torch.Tensor,
        scale: torch.Tensor,
        anchor_part: torch.Tensor,
    ) -> dict:
        """Forward pass through the PointCloudDiT model.
        
        Args:
            x (n_points, 3): Noise point coordinates.
            timesteps (n_valid_parts, ): Float tensor of timesteps.
            latent: PointTransformer's Point instance of conditional point cloud:
                - "coord" (n_points, 3): Point coordinates
                - "normal" (n_points, 3): Point normals
                - "feat" (n_points, in_dim): Point features  
                - "batch" (n_points, ): Integer tensor of batch indices.
            part_valids (bs, max_parts): bool tensor. True => valid parts.
            scale (n_valid_parts, ): float tensor of part scale factor.
            anchor_part (n_valid_parts, ): bool tensor, True => anchor parts.
            
        Returns:
            Tensor of shape (n_points, out_dim)
        """

        # Encoding
        x = self.encoding_manager(x, latent, scale)                     # (n_points, embed_dim)
        x = self._add_anchor_embedding(x, anchor_part, latent)          # (n_points, embed_dim)
        
        # Prepare attention metadata
        self_attn_seqlen = torch.bincount(latent["batch"])              # (n_valid_parts, )
        self_attn_max_seqlen = self_attn_seqlen.max()                   # (1)
        self_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(self_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)                                               # (n_valid_parts + 1, )
        points_per_part = torch.zeros_like(
            part_valids, dtype=self_attn_seqlen.dtype
        )                                                               # (bs, max_parts)
        points_per_part[part_valids] = self_attn_seqlen                 # (bs, max_parts)
        global_attn_seqlen = points_per_part.sum(1)                     # (bs, )
        global_attn_max_seqlen = global_attn_seqlen.max()               # (1)
        global_attn_cu_seqlens = nn.functional.pad(
            torch.cumsum(global_attn_seqlen, 0), (1, 0)
        ).to(torch.int32)                                               # (bs + 1, )

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(
                hidden_states=x,
                timestep=timesteps,
                batch=latent["batch"],
                self_attn_cu_seqlens=self_attn_cu_seqlens,
                self_attn_max_seqlen=self_attn_max_seqlen,
                global_attn_cu_seqlens=global_attn_cu_seqlens,
                global_attn_max_seqlen=global_attn_max_seqlen,
            )                                                           # (n_points, embed_dim)

        # Use float32 for better numerical stability
        x = x.float()
        with torch.amp.autocast(x.device.type, torch.float32):
            out = self.final_mlp(x)                                     # (n_points, out_dim)
        return out


if __name__ == "__main__":
    model = PointCloudDiT(
        in_dim=64,
        out_dim=6,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.0,
    )
    print(f"PointCloudDiT with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")