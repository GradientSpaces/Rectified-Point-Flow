"""Utility functions for point cloud reshaping."""

import torch


def split_parts(pointclouds: torch.Tensor, points_per_part: torch.Tensor) -> list[list[torch.Tensor]]:
    """Split a packed tensor into per-part point clouds.

    Args:
        pointclouds: Tensor of shape (B, N, 3).
        points_per_part: Tensor of shape (B, P) giving the number of points in each part.

    Returns: 
        parts: A list of length B, where parts[b] is itself a list of P (or fewer)
            tensors of shape (N_i, 3), where N_i is the number of points in the i-th part.
    """
    counts_per_batch = points_per_part.tolist()
    parts: list[list[torch.Tensor]] = []
    for b, counts in enumerate(counts_per_batch):
        assert sum(counts) == pointclouds[b].size(0), (
            f"Mismatch detected: sum(counts)={sum(counts)} does not equal "
            f"pointclouds[b].size(0)={pointclouds[b].size(0)} for batch {b}."
        )
        splits = torch.split(pointclouds[b], counts, dim=0)
        parts.append([s for s in splits if s.size(0) > 0])
    return parts


def ppp_to_ids(points_per_part: torch.Tensor) -> torch.Tensor:
    """Convert a points_per_part tensor into a part-IDs tensor.

    Args:
        points_per_part: Tensor of shape (B, P).

    Returns:
        Long tensor of shape (B, max_points), where for each batch b, the first
        N_b = points_per_part[b].sum() entries are the part-indices (0...P-1)
        repeated according to points_per_part[b], and any remaining positions
        (out to max_points) are zero.
    """
    B, P = points_per_part.shape
    device = points_per_part.device
    max_points = int(points_per_part.sum(dim=1).max().item())
    result = torch.zeros(B, max_points, dtype=torch.long, device=device)
    part_ids = torch.arange(P, device=device)

    for b in range(B):
        # Repeat each part index by its count in this batch
        id_repeated = torch.repeat_interleave(part_ids, points_per_part[b])  # (N_b,)
        result[b, : id_repeated.size(0)] = id_repeated
    return result


def flatten_valid_parts(x: torch.Tensor, points_per_part: torch.Tensor) -> torch.Tensor:
    """Flatten tensor by selecting only valid parts.
    
    Args:
        x: Batched tensor of shape (B, P, ...).
        points_per_part: Number of points per part of shape (B, P).
        
    Returns:
        Tensor of shape (valid_P, ...).
    """
    part_valids = points_per_part != 0                        # (B, P)
    return x[part_valids]
