"""Metrics for evaluation."""

import torch
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss.chamfer import chamfer_distance
from scipy.optimize import linear_sum_assignment

from .procrustes import fit_part_transformations


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
    return_matched_part_ids: bool = True,
) -> torch.Tensor:
    """Compute part accuracy over the Hungarian matching of the part order between GT and predicted point clouds.
    The Hungarian matching is necessary due to the interchangeability of parts.

    Args:
        pointclouds_gt (B, N, 3): Ground truth point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.
        points_per_part (B, P): Number of points in each part.
        threshold (float): Chamfer distance threshold.
        return_matched_part_ids (bool): Whether to return the matched part ids.

    Returns:
        Tensor of shape (B,) with part accuracy per batch.
        Tensor of shape (B, P): For each batch, the i-th part is matched to the j-th part if matched_part_ids[b, i] == j.

    """
    device = pointclouds_gt.device
    B, P = points_per_part.shape
    part_acc = torch.zeros(B, device=device)
    matched_part_ids = torch.zeros(B, P, device=device, dtype=torch.long)

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

        # Store matched part ids
        for i, j in zip(row_ind, col_ind):
            matched_part_ids[b, i] = j

    if return_matched_part_ids:
        return part_acc, matched_part_ids
    
    return part_acc

def compute_transform_errors(
    pointclouds: torch.Tensor,
    pointclouds_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    gt_rotations: torch.Tensor,
    gt_translations: torch.Tensor,
    anchor_part: torch.Tensor,
    matched_part_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute transform errors between GT and predicted point clouds.
    
    Args:
        pointclouds (B, N, 3): Condition point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.
        points_per_part (B, P): Number of points in each part.
        gt_rotations (B, P, 3, 3): Ground truth rotation matrices.
        gt_translations (B, P, 3): Ground truth translation vectors.
        anchor_part (B, P): Whether the part is an anchor part.
        matched_part_ids (B, P): Matched part ids per batch. If None, use the original part order.

    Returns:
        Tensor of shape (B,) with transform errors per batch. 
    """
    device = pointclouds.device
    B, P = points_per_part.shape

    # Fit per-part transformations
    rot_hats, trans_hats = fit_part_transformations(pointclouds, pointclouds_pred, points_per_part)

    # Re-order parts
    if matched_part_ids is not None:
        batch_idx = torch.arange(B, device=device)[:, None]
        rot_hats = rot_hats[batch_idx, matched_part_ids]
        trans_hats = trans_hats[batch_idx, matched_part_ids]

    # Compute transform errors  
    rot_errors = torch.zeros(B, device=device)
    trans_errors = torch.zeros(B, device=device)

    for b in range(B):
        for p in range(P):
            if anchor_part[b, p] or points_per_part[b, p] == 0:
                continue

            # Rotation error by Rodrigues' formula (in degrees)
            rot_error = torch.acos(
                torch.clamp(0.5 * (torch.trace(gt_rotations[b, p].t() @ rot_hats[b, p]) - 1), -1, 1)
            )
            rot_errors[b] += torch.rad2deg(rot_error)

            # Translation error (in meters)
            trans_errors[b] += torch.norm(gt_translations[b, p] - trans_hats[b, p])

        # Average over valid parts
        n_valid = torch.sum(points_per_part[b] > 0)
        rot_errors[b] /= n_valid
        trans_errors[b] /= n_valid

    return rot_errors, trans_errors