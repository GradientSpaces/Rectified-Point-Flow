import os
import json
import torch
import pytorch3d.transforms as transforms
from typing import Dict, Any, Tuple

from .shape_error import compute_shape_errors, compute_pose_errors
from .rigid_transform import fit_rigid_transform


class EvaluationRunner:
    """Reusable evaluation logic for validation and test steps."""
    
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
    
    def _initialize_sampling(self, data_dict: Dict[str, Any], part_valids: torch.Tensor, batch_indices: torch.Tensor):
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
        
        # Compute pose metrics
        B, N = pts.shape[:2]
        x_final = x_final.reshape(B, N, 3).detach()
        
        # Fit rigid transformation
        rot_hats, trans_hats = fit_rigid_transform(pts, x_final, points_per_part)
        pred_rots = transforms.matrix_to_quaternion(rot_hats[part_valids])
        pred_trans = trans_hats[part_valids]

        # Compute pose metrics
        rot_errors, trans_errors = compute_pose_errors(
            gt_trans, gt_rots, pred_trans, pred_rots, part_valids
        )
        
        # Compute shape metrics
        part_acc, shape_cd, object_cd = compute_shape_errors(
            pts, gt_trans, gt_rots, pred_trans, pred_rots,
            points_per_part, part_valids
        )

        return {
            "part_acc": part_acc,
            "rot_error": rot_errors,
            "trans_error": trans_errors,
            "shape_cd": shape_cd,
            "object_cd": object_cd,
        }
    
    def _save_single_result(self, data_dict: Dict[str, Any], metrics: Dict[str, torch.Tensor], index: int):
        """Save a single result to JSON."""
        data = {
            "name": data_dict["name"][index],
            "num_parts": data_dict["num_parts"][index].item(),
            "part_acc": metrics["part_acc"][index].item(),
            "trans_error": metrics["trans_error"][index].item(),
            "rot_error": metrics["rot_error"][index].item(),
            "shape_cd": metrics["shape_cd"][index].item(),
            "object_cd": metrics["object_cd"][index].item(),
            "mesh_scale": data_dict["mesh_scale"][index].item(),
        }
        save_dir = os.path.join(self.model.trainer.log_dir, "json_results")
        os.makedirs(save_dir, exist_ok=True)
        json.dump(data, open(os.path.join(save_dir, f"{data_dict['index'][index].item()}.json"), "w"))
    
    def save_json_results(self, data_dict: Dict[str, Any], metrics: Dict[str, torch.Tensor]):
        """Save results to JSON files."""
        if not self.model.inference_config.get("write_to_json", True):
            return
            
        B = data_dict["points_per_part"].shape[0]
        for b in range(B):
            self._save_single_result(data_dict, metrics, b)
    
    def run_evaluation(self, data_dict: Dict[str, Any], save_results: bool = False) -> Dict[str, torch.Tensor]:
        """Run evaluation with optional result saving."""
        x_final = self.sample(data_dict)
        metrics = self.compute_metrics(data_dict, x_final)
        if save_results:
            self.save_json_results(data_dict, metrics)
        return metrics
    
    def run_validation_evaluation(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run validation evaluation."""
        metrics = self.run_evaluation(data_dict, save_results=False)
        return metrics
    
    def run_test_evaluation(self, data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run test evaluation with comprehensive metrics."""
        return self.run_evaluation(data_dict, save_results=True)
        