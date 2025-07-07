import json
from pathlib import Path
from typing import Any, Dict

import torch
import lightning as L

from .metrics import compute_object_cd, compute_part_acc, compute_transform_errors


class Evaluator:
    """Evaluator for Rectified Point Flow model. """
    
    def __init__(self, model: L.LightningModule):
        self.model = model

    def _compute_metrics(
        self,
        data: Dict[str, Any],
        pointclouds_pred: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute object Chamfer distance and part accuracy.

        Args:
            data: Dictionary containing:
                pointclouds_gt (B, N, 3): Ground truth point clouds.
                scale (B,): Scale factors.
                points_per_part (B, P): Points per part.
            pointclouds_pred (B, N, 3): Sampled point clouds.

        Returns:
            Dict with:
                object_cd (torch.Tensor): Object Chamfer distance, shape (B,).
                part_acc (torch.Tensor): Part accuracy, shape (B,).
        """
        pts = data["pointclouds"]                       # (B, N, 3)
        pts_gt = data["pointclouds_gt"]                 # (B, N, 3)
        gt_rotations = data["rotations"]                # (B, P, 3, 3)
        gt_translations = data["translations"]          # (B, P, 3)
        points_per_part = data["points_per_part"]       # (B, P)
        anchor_part = data["anchor_part"]               # (B, P)
        scale = data["scale"][:, 0]                     # (B,)

        B, N = pts_gt.shape[:2]
        pts_pred = pointclouds_pred.view(B, N, 3).detach()  # (B, N, 3)
        
        # Rescale to original scale (in meters)
        pts = pts * scale.view(B, 1, 1)
        pts_gt = pts_gt * scale.view(B, 1, 1)
        pts_pred = pts_pred * scale.view(B, 1, 1)

        # Compute metrics
        object_cd = compute_object_cd(pts_gt, pts_pred)
        part_acc, matched_parts = compute_part_acc(pts_gt, pts_pred, points_per_part)
        rot_errors, trans_errors = compute_transform_errors(
            pts, pts_pred, points_per_part, gt_rotations, gt_translations, anchor_part, matched_parts
        )

        return {
            "object_cd": object_cd,         # (B,)
            "part_acc": part_acc,           # (B,)
            "rot_errors": rot_errors,       # (B,)
            "trans_errors": trans_errors,   # (B,)
        }

    def _save_single_result(
        self,
        data: Dict[str, Any],
        metrics: Dict[str, torch.Tensor],
        idx: int,
    ) -> None:
        """Save a single evaluation result to JSON.

        Args:
            data: Input data dictionary.
            metrics: Computed metrics dictionary.
            idx: Index of the sample in the batch.
        """
        entry = {
            "name": data["name"][idx],
            "dataset": data["dataset_name"][idx],
            "num_parts": int(data["num_parts"][idx]),
        }
        entry.update({k: float(v[idx]) for k, v in metrics.items()})

        out_dir = Path(self.model.trainer.log_dir) / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{int(data['index'][idx])}.json"
        filepath.write_text(json.dumps(entry))

    def run(
        self,
        data: Dict[str, Any],
        pointclouds_pred: torch.Tensor,
        save_results: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run evaluation and optionally save results.

        Args:
            data: Input data dictionary, containing:
                pointclouds_gt (B, N, 3): Ground truth point clouds.
                scale (B,): Scale factors.
                points_per_part (B, P): Points per part.
                name (B,): Object names.
                dataset_name (B,): Dataset names.
                index (B,): Object indices.
                num_parts (B,): Number of parts.

            pointclouds_pred (B, N, 3): Model output samples.
            save_results (bool): If True, save each result to log_dir/results.

        Returns:
            Dict of computed metrics.
        """
        metrics = self._compute_metrics(data, pointclouds_pred)
        if save_results:
            for i in range(data["points_per_part"].size(0)):
                self._save_single_result(data, metrics, i)
        return metrics
