"""Evaluation utilities for Rectified Point Flow."""

import os
import json
import torch
import pytorch3d.transforms as transforms
from typing import Dict, Any

from .shape_error import compute_shape_errors, compute_pose_errors
from .rigid_transform import fit_rigid_transform


class Evaluator:
    """Handles evaluation and result saving for Rectified Point Flow."""
    
    def __init__(self, model, num_steps: int = 20):
        self.model = model
        self.num_steps = num_steps
    
    def sample(self, data_dict: Dict[str, Any]) -> torch.Tensor:
        """Run rectified flow sampling."""
        points_per_part = data_dict["points_per_part"]
        part_valids = points_per_part != 0
        part_scale = data_dict["scale"][part_valids]
        ref_part = data_dict["ref_part"][part_valids]
        
        # Sample from rectified flow
        latent = self.model.extract_features(data_dict)
        x0, anchor_idx = self._initialize_sampling(data_dict, part_valids, latent['batch'])
        
        # Generate sample
        x_final = self.model.sample_rectified_flow(
            x0=x0,
            latent=latent,
            part_valids=part_valids,
            part_scale=part_scale,
            ref_part=ref_part,
            anchor_idx=anchor_idx,
            num_steps=self.num_steps,
        )
        return x_final
    
    def _initialize_sampling(self, data_dict: dict, part_valids: torch.Tensor, batch_indices: torch.Tensor):
        """Initialize sampling with random noise and anchor points."""
        x0 = torch.randn_like(data_dict["pointclouds_gt"])
        x0 = x0.reshape(-1, 3)
        anchor_idx = data_dict["ref_part"][part_valids][batch_indices]
        x0[anchor_idx] = data_dict["pointclouds_gt"].reshape(-1, 3)[anchor_idx]
        return x0, anchor_idx
    
    @staticmethod
    def unflatten_tensor(tensor: torch.Tensor, part_valids: torch.Tensor) -> torch.Tensor:
        """Unflatten tensor to original shape."""
        B, P = part_valids.shape
        unflattened = torch.zeros(B, P, *tensor.shape[1:], device=tensor.device)
        unflattened[part_valids] = tensor
        return unflattened

    def compute_metrics(
        self, 
        data_dict: Dict[str, Any], 
        x_final: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        points_per_part = data_dict["points_per_part"]
        part_valids = points_per_part != 0
        pts = data_dict["pointclouds"]
        gt_trans = data_dict["translations"][part_valids]
        gt_rots = data_dict["quaternions"][part_valids]
        B, N = pts.shape[:2]

        # Fit rigid transformation
        x_final = x_final.reshape(B, N, 3).detach()
        rot_hats, trans_hats = fit_rigid_transform(pts, x_final, points_per_part)
        pred_rots = transforms.matrix_to_quaternion(rot_hats[part_valids])
        pred_trans = trans_hats[part_valids]

        # Compute pose metrics
        rot_errors, trans_errors = compute_pose_errors(
            gt_trans, gt_rots, pred_trans, pred_rots, part_valids
        )
        # Compute shape metrics
        part_acc, shape_cd, object_cd = compute_shape_errors(
            pts, gt_trans, gt_rots, pred_trans, pred_rots, points_per_part, part_valids
        )
        return {
            "part_acc": part_acc,
            "rot_error": rot_errors,
            "trans_error": trans_errors,
            "shape_cd": shape_cd,
            "object_cd": object_cd,
        }
    
    def _save_single_result(self, data_dict: dict, metrics: dict, index: int):
        """Save a single result to JSON."""
        data = {
            "name": data_dict["name"][index],
            "num_parts": data_dict["num_parts"][index].item(),
            **{k: v[index].item() for k, v in metrics.items()},
        }
        save_dir = os.path.join(self.model.trainer.log_dir, "results")
        os.makedirs(save_dir, exist_ok=True)
        json.dump(data, open(os.path.join(save_dir, f"{data_dict['index'][index].item()}.json"), "w"))
    
    def run_evaluation(self, data_dict: dict, save_results: bool = False) -> dict:
        """Run evaluation with optional result saving."""
        x_final = self.sample(data_dict)
        metrics = self.compute_metrics(data_dict, x_final)

        if save_results:
            B = data_dict["points_per_part"].shape[0]
            for batch_idx in range(B):
                self._save_single_result(data_dict, metrics, batch_idx)
        return metrics
