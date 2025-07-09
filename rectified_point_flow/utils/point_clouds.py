"""Utility functions for point cloud reshaping."""

import torch
from typing import Optional


def ppp_to_ids(points_per_part: torch.Tensor) -> torch.Tensor:
    """Convert points_per_part tensor to part ID tensor.
    
    Args:
        points_per_part: Number of points per part of shape (B, P).
        
    Returns:
        Part IDs of shape (B, max_points), where each point is assigned 
        the ID of its corresponding part.
    """
    batch_size, num_parts = points_per_part.shape
    max_points = points_per_part.sum(dim=1).max().item()
    part_ids = torch.zeros(
        batch_size, max_points, device=points_per_part.device, dtype=torch.long
    )

    for batch_idx in range(batch_size):
        point_idx = 0
        for part_idx in range(num_parts):
            num_points = points_per_part[batch_idx, part_idx].item()
            if num_points > 0:
                part_ids[batch_idx, point_idx:point_idx + num_points] = part_idx
                point_idx += num_points
    
    return part_ids


def ids_to_ppp(part_ids: torch.Tensor, num_parts: Optional[int] = None) -> torch.Tensor:
    """Convert part ID tensor to points_per_part tensor.
    
    Args:
        part_ids: Part IDs of shape (B, N) where each point has a part ID.
        num_parts: Maximum number of parts. If None, inferred from part_ids.
        
    Returns:
        Tensor of shape (B, P), where P is num_parts.
    """
    batch_size = part_ids.shape[0]
    
    if num_parts is None:
        num_parts = part_ids.max().item() + 1
    
    points_per_part = torch.zeros(
        batch_size, num_parts, device=part_ids.device, dtype=torch.long
    )
    for batch_idx in range(batch_size):
        # Count occurrences of each part ID
        valid_ids = part_ids[batch_idx][part_ids[batch_idx] >= 0]  # Filter out padding (-1, etc.)
        if len(valid_ids) > 0:
            part_counts = torch.bincount(valid_ids, minlength=num_parts)
            points_per_part[batch_idx, :len(part_counts)] = part_counts
    
    return points_per_part


def split_parts(x: torch.Tensor, points_per_part: torch.Tensor) -> list[list[torch.Tensor]]:
    """Split a packed tensor into per-part point clouds.
    
    Args:
        x: Tensor of shape (B, N, 3).
        points_per_part: Number of points per part of shape (B, P).

    Returns:
        parts (list[list[torch.Tensor]]), where parts[b][p] is the p-th part of the b-th batch of shape (N_p, 3).
    """
    B, P = points_per_part.shape
    offsets = points_per_part.cumsum(dim=1) - points_per_part
    parts = []
    for b in range(B):
        part_in_batch = []
        for p in range(P):
            if points_per_part[b, p] == 0:
                continue
            part_in_batch.append(x[b, offsets[b, p] : offsets[b, p] + points_per_part[b, p], :])
        parts.append(part_in_batch)
    return parts


def broadcast_part_to_points(x: torch.Tensor, points_per_part: torch.Tensor) -> torch.Tensor:
    """Broadcast per-part information to per-point information.
    
    Args:
        x: Tensor of shape (valid_P, ...), where valid_P is the number of valid parts.
        points_per_part: Number of points per part of shape (B, P).
        
    Returns:
        Tensor of shape (total_points, ...).
    """
    part_valids = points_per_part != 0                          # (B, P)
    points_per_valid_part = points_per_part[part_valids]        # (valid_P,)
    return x.repeat_interleave(points_per_valid_part, dim=0)


def broadcast_batch_to_part(x: torch.Tensor, points_per_part: torch.Tensor) -> torch.Tensor:
    """Broadcast per-batch information to per-part information.
    
    Args:
        x: Tensor of shape (B, ...).
        points_per_part: Number of points per part of shape (B, P).

    Returns:
        Tensor of shape (valid_P, ...).
    """
    B, P = points_per_part.shape
    part_valids = points_per_part != 0                          # (B, P)
    expanded = x.unsqueeze(1).expand(B, P, *x.shape[1:])
    return expanded[part_valids]


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

