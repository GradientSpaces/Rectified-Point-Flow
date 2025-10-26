"""Metrics for evaluation."""

import torch
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.ops import iterative_closest_point
from scipy.optimize import linear_sum_assignment

from ..utils.point_clouds import split_parts


def align_anchor(
    pointclouds_gt: torch.Tensor,
    pointclouds_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    anchor_parts: torch.Tensor,
) -> torch.Tensor:
    """Align the predicted anchor parts to the ground truth anchor parts using ICP.

    Args:
        pointclouds_gt (B, N, 3): Ground truth point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.
        points_per_part (B, P): Number of points in each part.
        anchor_parts (B, P): Whether the part is an anchor part; we use the first part with the flag of True as the anchor part.

    Returns:
        pointclouds_pred_aligned (B, N, 3): Aligned sampled point clouds.
    """
    B, P = anchor_parts.shape
    device = pointclouds_pred.device
    pointclouds_pred_aligned = pointclouds_pred.clone()

    with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
        for b in range(B):
            pts_count = 0
            for p in range(P):
                if points_per_part[b, p] == 0:
                    continue
                if anchor_parts[b, p]:
                    ed = pts_count + points_per_part[b, p]
                    anchor_align_icp = iterative_closest_point(pointclouds_pred[b, pts_count:ed].unsqueeze(0), pointclouds_gt[b, pts_count:ed].unsqueeze(0)).RTs
                    break

            pts_count = 0
            for p in range(P):
                if points_per_part[b, p] == 0:
                    continue
                ed = pts_count + points_per_part[b, p]
                pointclouds_pred_aligned[b, pts_count:ed] = pointclouds_pred[b, pts_count:ed] @ anchor_align_icp.R[0].T + anchor_align_icp.T[0]
                pts_count = ed

    return pointclouds_pred_aligned


def compute_object_cd(
    pointclouds_gt: torch.Tensor,
    pointclouds_pred: torch.Tensor,
) -> torch.Tensor:
    """Compute the whole object Chamfer Distance (CD) between ground truth and predicted point clouds.

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
    """Compute Part Accuracy (PA), the ratio of successfully posed parts over the total number of parts.

    The success is defined as the Chamfer Distance (CD) between a predicted part and a ground truth part is 
    less than the threshold (0.01 meter by default). Here, we use Hungarian matching to find the best matching 
    between predicted and ground truth parts, which is necessary due to the part interchangeability.

    Args:
        pointclouds_gt (B, N, 3): Ground truth point clouds.
        pointclouds_pred (B, N, 3): Sampled point clouds.
        points_per_part (B, P): Number of points in each part.
        threshold (float): Chamfer distance threshold.
        return_matched_part_ids (bool): Whether to return the matched part ids.

    Returns:
        Tensor of shape (B,) with part accuracy per batch.
        Tensor of shape (B, P): For each batch, the i-th part is matched to the j-th part if 
            matched_part_ids[b, i] == j.

    """
    device = pointclouds_gt.device
    B, P = points_per_part.shape
    part_acc = torch.zeros(B, device=device)
    matched_part_ids = torch.zeros(B, P, device=device, dtype=torch.long)
    parts_gt = split_parts(pointclouds_gt, points_per_part)
    parts_pred = split_parts(pointclouds_pred, points_per_part)

    for b in range(B):
        lengths = points_per_part[b]                                # (P,)
        valid = lengths > 0
        idx = valid.nonzero(as_tuple=False).squeeze(1)
        n_parts = idx.numel()
        lens = lengths[idx]                                         # (n_parts,)
        pts_gt = pad_sequence(parts_gt[b], batch_first=True)        # (n_parts, max_len, 3)
        pts_pred = pad_sequence(parts_pred[b], batch_first=True)    # (n_parts, max_len, 3)
        n_parts, max_len, _ = pts_gt.shape

        # Compute pairwise Chamfer distances between all parts (n_parts^2, max_len, 3)
        pts_gt = pts_gt.unsqueeze(1).expand(n_parts, n_parts, max_len, 3).reshape(-1, max_len, 3)
        pts_pred = pts_pred.unsqueeze(0).expand(n_parts, n_parts, max_len, 3).reshape(-1, max_len, 3)
        len_x = lens.unsqueeze(1).expand(n_parts, n_parts).reshape(-1)
        len_y = lens.unsqueeze(0).expand(n_parts, n_parts).reshape(-1)
        cd_all, _ = chamfer_distance(
            x=pts_gt,
            y=pts_pred,
            x_lengths=len_x,
            y_lengths=len_y,
            single_directional=False,
            point_reduction="mean",
            batch_reduction=None,
        )
        cd_mat = cd_all.view(n_parts, n_parts)

        # Find best matching using Hungarian algorithm
        cost_mat = (cd_mat >= threshold).float().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        matched = (cd_mat[row_ind, col_ind] < threshold).sum().item()
        part_acc[b] = matched / n_parts

        for i, j in zip(row_ind, col_ind):
            matched_part_ids[b, i] = j

    if return_matched_part_ids:
        return part_acc, matched_part_ids
    
    return part_acc

def compute_transform_errors(
    pointclouds: torch.Tensor,
    pointclouds_gt: torch.Tensor,
    rotations_pred: torch.Tensor,
    translations_pred: torch.Tensor,
    points_per_part: torch.Tensor,
    anchor_part: torch.Tensor,
    matched_part_ids: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the per-part rotation and translation errors between ground truth and predicted point clouds.

    To factor out the symmetry of parts, we estimate the minimum transformation by ICP between the ground truth 
    and predicted parts. The rotation error (RE) is computed using the angular difference (Rodrigues formula). 
    The translation error (TE) is computed using the L2 norm of the translation vectors.

    Note that the scale of the point clouds is considered in the computation of the translation errors.
    
    Args:
        pointclouds (B, N, 3): Condition point clouds.
        rotations_gt (B, P, 3, 3): Ground truth rotation matrices.
        translations_gt (B, P, 3): Ground truth translation vectors.
        rotations_pred (B, P, 3, 3): Estimated rotation matrices.
        translations_pred (B, P, 3): Estimated translation vectors.
        points_per_part (B, P): Number of points in each part.
        anchor_part (B, P): Whether the part is an anchor part.
        matched_part_ids (B, P): Matched part ids per batch. If None, use the original part order.
        scale (B,): Scale of the point clouds. If None, use 1.0.
        return_transforms (bool): Whether to return the estimated rotation and translation matrices.

    Returns:
        rot_errors_mean (B,): Mean rotation errors per batch.
        trans_errors_mean (B,): Mean translation errors per batch.
        rotations_pred (B, P, 3, 3): Estimated rotation matrices, only returned if return_transforms is True.
        translations_pred (B, P, 3): Estimated translation vectors, only returned if return_transforms is True.
    """
    device = pointclouds.device
    B, P = points_per_part.shape
    parts_cond = split_parts(pointclouds, points_per_part)
    parts_gt = split_parts(pointclouds_gt, points_per_part)

    # Re-order parts
    if matched_part_ids is not None:
        batch_idx = torch.arange(B, device=device)[:, None]
        rotations_pred = rotations_pred[batch_idx, matched_part_ids]
        translations_pred = translations_pred[batch_idx, matched_part_ids]

    if scale is None:
        scale = torch.ones(B, device=device)

    rot_errors = torch.zeros(B, P, device=device)
    trans_errors = torch.zeros(B, P, device=device)
    for b in range(B):
        for p in range(P):
            if points_per_part[b, p] == 0 or anchor_part[b, p]:
                continue

            with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
                part_gt = parts_gt[b][p].unsqueeze(0)
                part_cond = parts_cond[b][p]
                part_transformed = (part_cond @ rotations_pred[b, p].T + translations_pred[b, p]).unsqueeze(0)
                error = iterative_closest_point(part_gt, part_transformed).RTs
                rot_errors[b, p] = torch.rad2deg(
                    torch.acos(torch.clamp(0.5 * (torch.trace(error.R[0]) - 1), -1, 1))
                )
                trans_errors[b, p] = torch.norm(error.T[0]) * scale[b]

    # Average over valid parts
    n_parts = (points_per_part != 0).sum(dim=1)
    rot_errors_mean = rot_errors.sum(dim=1) / n_parts
    trans_errors_mean = trans_errors.sum(dim=1) / n_parts
    return rot_errors_mean, trans_errors_mean
