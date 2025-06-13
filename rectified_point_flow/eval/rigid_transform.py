import torch
import numpy as np
import pytorch3d.transforms as transforms


def solve_procrustes(source, target, source_mean, target_mean):
    """Solve Procrustes problem to find optimal rotation and translation."""

    source_centered = source - source_mean
    target_centered = target - target_mean
    
    # SVD for rotation
    H = source_centered.t() @ target_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.t() @ U.t()
    
    # Ensure proper rotation (det(R) = 1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.t() @ U.t()
    
    # Translation
    t = target_mean - source_mean @ R.t()
    return R, t.squeeze()


def fit_rigid_transform(pcd_0, pcd_1, points_per_part, sampling_weight=None, weight_bias=0.2):
    """Fit rigid transformation between point clouds."""
    device = pcd_0.device
    bs, n_parts = points_per_part.shape
    rot_hats = torch.zeros(bs, n_parts, 3, 3, device=device)
    trans_hats = torch.zeros(bs, n_parts, 3, device=device)

    for b_idx in range(bs):
        part_start = 0
        for n_p in range(n_parts):
            if points_per_part[b_idx, n_p] == 0:
                continue
            p = points_per_part[b_idx, n_p]
            source = pcd_0[b_idx, part_start:part_start + p]  # (n_points, 3)
            target = pcd_1[b_idx, part_start:part_start + p]  # (n_points, 3)

            # Convert to float32 for numerical stability
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                source_mean = source.mean(dim=0, keepdim=True)
                target_mean = target.mean(dim=0, keepdim=True)
                rot_mat, trans = solve_procrustes(source, target, source_mean, target_mean)

            part_start += p
            rot_hats[b_idx, n_p] = rot_mat
            trans_hats[b_idx, n_p] = trans
    return rot_hats, trans_hats 