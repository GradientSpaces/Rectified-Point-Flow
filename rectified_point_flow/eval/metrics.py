"""Metrics for evaluation."""

import torch
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss.chamfer import chamfer_distance
from scipy.optimize import linear_sum_assignment


def compute_object_cd(
    pointclouds_gt: torch.Tensor,
    pointclouds_pred: torch.Tensor,
) -> torch.Tensor:
    """Compute object-level Chamfer distance.

    Args:
        pointclouds_gt (B, N, 3): Ground truth point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.

    Returns:
        Tensor of shape (B,) with Chamfer distance per batch.
    """
    object_cd, _ = chamfer_distance(
        x=pointclouds_gt,
        y=pointclouds_pred,
        single_directional=False,
        point_reduction="mean",
        batch_reduction=None,
    )  # (B,)
    return object_cd

def compute_part_acc(
    pointclouds_gt: torch.Tensor,
    pointclouds_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    threshold: float = 0.01,
) -> torch.Tensor:
    """Compute part accuracy over the Hungarian matching between GT and predicted part order.
    The Hungarian matching is necessary due to the interchangeability of parts.

    Args:
        pointclouds_gt (B, N, 3): Ground truth point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.
        points_per_part (B, P): Number of points in each part.
        threshold (float): Chamfer distance threshold.

    Returns:
        Tensor of shape (B,) with part accuracy per batch.
    """
    device = pointclouds_gt.device
    B, _ = points_per_part.shape
    part_acc = torch.zeros(B, device=device)

    # Compute part offsets
    seg_offsets = points_per_part.cumsum(dim=1) - points_per_part  # (B, P)

    for b in range(B):
        lengths = points_per_part[b]                     # (P,)
        valid = lengths > 0
        idx = valid.nonzero(as_tuple=False).squeeze(1)
        n_parts = idx.numel()

        offs = seg_offsets[b, idx]                       # (n_parts,)
        lens = lengths[idx]                              # (n_parts,)
        flat_gt = pointclouds_gt[b].view(-1, 3)          # (N, 3)
        flat_pred = pointclouds_pred[b].view(-1, 3)      # (N, 3)

        # Split into list of point clouds per part
        parts_gt = [flat_gt[o : o + l] for o, l in zip(offs.tolist(), lens.tolist())]
        parts_pred = [flat_pred[o : o + l] for o, l in zip(offs.tolist(), lens.tolist())]

        # Pad parts to same length
        pts_gt = pad_sequence(parts_gt, batch_first=True)    # (n_parts, max_len, 3)
        pts_pred = pad_sequence(parts_pred, batch_first=True)
        n_parts, max_len, _ = pts_gt.shape

        # Compute pairwise Chamfer distances between all parts
        gt_exp = pts_gt.unsqueeze(1).expand(n_parts, n_parts, max_len, 3).reshape(-1, max_len, 3)
        pred_exp = pts_pred.unsqueeze(0).expand(n_parts, n_parts, max_len, 3).reshape(-1, max_len, 3)
        len_x = lens.unsqueeze(1).expand(n_parts, n_parts).reshape(-1)
        len_y = lens.unsqueeze(0).expand(n_parts, n_parts).reshape(-1)
        cd_all, _ = chamfer_distance(
            x=gt_exp,
            y=pred_exp,
            x_lengths=len_x,
            y_lengths=len_y,
            single_directional=False,
            point_reduction="mean",
            batch_reduction=None,
        )
        cd_mat = cd_all.view(n_parts, n_parts)

        # Find best matching assignment
        cost_mat = (cd_mat >= threshold).float().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_mat)

        # Compute part accuracy
        matched = (cd_mat[row_ind, col_ind] < threshold).sum().item()
        part_acc[b] = matched / n_parts

    return part_acc
