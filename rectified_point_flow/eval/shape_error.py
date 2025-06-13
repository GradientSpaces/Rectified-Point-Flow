import torch
from pytorch3d.loss.chamfer import chamfer_distance
import pytorch3d.transforms as p3dt


def mean_over_valid_parts(error_per_part, valids, ref_part=None):
    """Average loss values over valid parts, excluding reference part if specified.

    Args:
        error_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts
        ref_part: [n_valid_parts], index of reference part for each valid part

    Returns:
        average error over valid parts [B,]
    """
    valid_mask = valids.float().detach()
    if ref_part is not None:
        for b in range(valid_mask.shape[0]):
            valid_mask[b, ref_part[b]] = 0.0
    
    # Handle NaN values
    nan_mask = torch.isnan(error_per_part)
    error_per_part = error_per_part.clone()
    error_per_part[nan_mask] = 0.0
    valid_mask[nan_mask] = 0.0
    return (error_per_part * valid_mask).sum(1) / valid_mask.sum(1)


def transform_pc(trans, rot, pts):
    """Transform point clouds using translation and rotation.
    
    Args:
        trans: [B, P, 3] translation
        rot: [B, P, 4] quaternion
        pts: [B, P, N, 3] point clouds
        
    Returns:
        [B, P, N, 3] transformed point clouds
    """
    return p3dt.quaternion_apply(rot.unsqueeze(-2), pts) + trans.unsqueeze(-2)


@torch.no_grad()
def compute_pose_errors(gt_trans, gt_rots, pred_trans, pred_rots, part_valids, ref_part=None, eps=1e-4):
    """Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        gt_trans: [B, P, 3] ground truth translations
        gt_rots: [B, P, 4] ground truth quaternions (xyzw order)
        pred_trans: [B, P, 3] predicted translations
        pred_rots: [B, P, 4] predicted quaternions (xyzw order)
        part_valids: [B, P] binary mask indicating valid parts
        ref_part: [B] optional index of reference part for each batch
        eps: small epsilon value for numerical stability

    Returns:
        rot_errors: [B] rotation errors averaged over valid parts
        trans_errors: [B] translation errors averaged over valid parts
    """
    # Compute translation errors
    trans_error_per_part = (pred_trans - gt_trans).pow(2).mean(dim=-1) ** 0.5  # [B, P]
    
    # Compute rotation errors
    pred_rots = pred_rots / (pred_rots.norm(dim=-1, keepdim=True) + eps)
    gt_rots = gt_rots / (gt_rots.norm(dim=-1, keepdim=True) + eps)
    R_pred = p3dt.quaternion_to_matrix(pred_rots)
    R_gt = p3dt.quaternion_to_matrix(gt_rots)
    
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_gt)
    cos_theta = (R_rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_theta = cos_theta.clamp(min=-1.0 + eps, max=1.0 - eps)
    theta = torch.acos(cos_theta)
    rot_error_per_part = torch.rad2deg(theta)  # [B, P]
    
    # Average errors over valid parts
    B, P = part_valids.shape
    rot_error_per_part_padded = torch.zeros(B, P, device=part_valids.device)
    rot_error_per_part_padded[part_valids] = rot_error_per_part
    trans_error_per_part_padded = torch.zeros(B, P, device=part_valids.device)
    trans_error_per_part_padded[part_valids] = trans_error_per_part

    rot_errors = mean_over_valid_parts(rot_error_per_part_padded, part_valids, ref_part)
    trans_errors = mean_over_valid_parts(trans_error_per_part_padded, part_valids, ref_part)
    return rot_errors, trans_errors


@torch.no_grad()
def compute_shape_errors(
    pts, gt_trans, gt_rots, pred_trans, pred_rots,
    points_per_part, part_valids, ref_part=None
):
    """Calculate all shape metrics (part accuracy, shape CD, object CD) at once.
    
    Args:
        pts: [B, N_sum, 3], input point cloud
        gt_trans: [B, P, 3], ground truth translation
        gt_rots: [B, P, 4], ground truth rotation (quaternion)
        pred_trans: [B, P, 3], predicted translation
        pred_rots: [B, P, 4], predicted rotation (quaternion)
        points_per_part: [B, P], number of points per part
        part_valids: [B, P], 1 for input parts, 0 for padded parts
        ref_part: [B], index of reference part for each batch
        
    Returns:
        part_acc: [B], part accuracy per data
        cd_per_part: [B, P], chamfer distance per part
        shape_cd: [B], shape chamfer distance per data
        object_cd: [B], object chamfer distance per data
    """
    B, P = part_valids.shape
    points_per_valid_part = points_per_part[part_valids]
    
    # Repeat transformations for each point
    gt_trans_point = gt_trans.repeat_interleave(points_per_valid_part, dim=0)
    gt_rots_point = gt_rots.repeat_interleave(points_per_valid_part, dim=0)
    pred_trans_point = pred_trans.repeat_interleave(points_per_valid_part, dim=0)
    pred_rots_point = pred_rots.repeat_interleave(points_per_valid_part, dim=0)
    
    # Transform points
    # breakpoint()
    pts_gt = (
        p3dt.quaternion_apply(gt_rots_point, pts.view(-1, 3)) + gt_trans_point
    ).detach()  # (B*N_sum, 3)
    pts_pred = (
        p3dt.quaternion_apply(pred_rots_point, pts.view(-1, 3)) + pred_trans_point
    ).detach()  # (B*N_sum, 3)

    # Pad to (n_valid_parts, N_max, 3) for part-level metrics
    N_max = points_per_valid_part.max()
    n_valid_parts = points_per_valid_part.shape[0]
    pts_gt_padded = torch.zeros(n_valid_parts, N_max, 3, device=pts.device)
    pts_pred_padded = torch.zeros(n_valid_parts, N_max, 3, device=pts.device)

    # Create indices for scatter
    row_idx = torch.arange(n_valid_parts, device=pts.device).unsqueeze(1).expand(-1, N_max)
    col_idx = torch.arange(N_max, device=pts.device).unsqueeze(0).expand(n_valid_parts, -1)
    mask = col_idx < points_per_valid_part.unsqueeze(1)
    source_idx = torch.arange(pts_gt.shape[0], device=pts.device)
    pts_gt_padded[row_idx[mask], col_idx[mask]] = pts_gt[source_idx]
    pts_pred_padded[row_idx[mask], col_idx[mask]] = pts_pred[source_idx]

    # Compute part chamfer distance
    cd_per_part, _ = chamfer_distance(
        x=pts_gt_padded,
        y=pts_pred_padded,
        x_lengths=points_per_valid_part,
        y_lengths=points_per_valid_part,
        single_directional=False,
        point_reduction="mean",
        batch_reduction=None,
    )

    # Compute part accuracy
    threshold = 0.01
    acc_per_part = (cd_per_part < threshold).float()

    # Recover to object (B, P)
    acc_per_part_padded = torch.zeros(B, P, device=part_valids.device)
    acc_per_part_padded[part_valids] = acc_per_part
    part_acc = mean_over_valid_parts(acc_per_part_padded, part_valids, ref_part)

    # Recover CD to object (B, P)
    cd_per_part_padded = torch.zeros(B, P, device=part_valids.device)
    cd_per_part_padded[part_valids] = cd_per_part

    # Compute shape CD (mean over parts)
    shape_cd = mean_over_valid_parts(cd_per_part_padded, part_valids, ref_part)

    # Compute object CD (treating all parts as one object)
    pts_gt = pts_gt.view(B, -1, 3)
    pts_pred = pts_pred.view(B, -1, 3)
    object_cd, _ = chamfer_distance(
        x=pts_gt,
        y=pts_pred,
        single_directional=False,
        point_reduction="mean",
        batch_reduction=None,
    )
    return part_acc, shape_cd, object_cd
