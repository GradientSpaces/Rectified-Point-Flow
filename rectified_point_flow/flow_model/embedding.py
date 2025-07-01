"""Point embedding utilities."""

import torch
import torch.nn as nn


class PointCloudEmbedding:
    """Generate positional encodings for multi-part point clouds.
    
    Ref:
        Nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    """

    def __init__(
        self,
        include_input: bool = True,
        input_dims: int = 3,
        max_freq_log2: int = 9,
        num_freqs: int = 10,
        log_sampling: bool = True,
        periodic_fns: list = None,
    ):
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns or [torch.sin, torch.cos]
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        """Create the embedding function and compute output dimension."""
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Embed input tensor using sinusoidal encoding."""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class PointCloudEncodingManager(nn.Module):
    """Generate PointCloudEmbedding from the input.

    It includes PointCloudEmbedding for:
        - Coordinate of condition PCs
        - Coordinate of noise PCs
        - Normal vector of condition PCs
        - Scale of condition PCs

    Args:
        in_dim: Input feature dimension.
        embed_dim: Output embedding dimension.
        multires: Multiresolution level for frequency encoding.
    """

    def __init__(self, in_dim: int, embed_dim: int, multires: int = 10):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.multires = multires

        # Coordinate of condition PCs
        self.coord_embedding = PointCloudEmbedding(
            input_dims=3,
            max_freq_log2=multires - 1,
            num_freqs=multires,
        )
        
        # Coordinate of noise PCs
        self.noise_embedding = PointCloudEmbedding(
            input_dims=3,
            max_freq_log2=multires - 1,
            num_freqs=multires,
        )

        # Normal vector of condition PCs
        self.normal_embedding = PointCloudEmbedding(
            input_dims=3,
            max_freq_log2=multires - 1,
            num_freqs=multires,
        )

        # Scale factor of condition PCs
        self.scale_embedding = PointCloudEmbedding(
            input_dims=1,
            max_freq_log2=multires - 1,
            num_freqs=multires,
        )

        # Embedding projection
        embed_input_dim = (
            self.in_dim
            + self.coord_embedding.out_dim
            + self.noise_embedding.out_dim
            + self.normal_embedding.out_dim
            + self.scale_embedding.out_dim
        ) 
        self.emb_proj = nn.Linear(embed_input_dim, self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,       # (n_points, 3)
        latent: dict,          # PointTransformer's `Point` instance
        scale: torch.Tensor,   # (n_valid_parts, )
    ) -> torch.Tensor:
        """Generate PointCloudEmbedding from the input.
        
        Args:
            x: Input coordinates tensor of shape (n_points, 3).
            latent: Dictionary containing point cloud features and metadata.
            scale: Scale factors of shape (n_valid_parts, 1).
            
        Returns:
            Shape embeddings of shape (n_points, embed_dim).
        """
        # Coordinate of noise PCs
        x_pos_emb = self.noise_embedding.embed(x)                   # (n_points, emb_dim)

        # Coordinate of condition PCs
        c_pos_emb = self.coord_embedding.embed(latent["coord"])     # (n_points, emb_dim)
        
        # Normal of condition PCs
        normal_emb = self.normal_embedding.embed(latent["normal"])  # (n_points, emb_dim)

        # Scale of condition PCs
        scale_emb = self.scale_embedding.embed(scale.unsqueeze(-1)) # (n_valid_parts, emb_dim)
        scale_emb = scale_emb[latent["batch"]]                      # (n_points, emb_dim)

        # Concatenate with point features
        x = torch.cat([latent["feat"], c_pos_emb, x_pos_emb, normal_emb, scale_emb], dim=-1)
        return self.emb_proj(x)
