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
        rotations_pred: torch.Tensor,
        translations_pred: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        pts = data["pointclouds"]                       # (B, N, 3)
        pts_gt = data["pointclouds_gt"]                 # (B, N, 3)
        points_per_part = data["points_per_part"]       # (B, P)
        anchor_part = data["anchor_part"]               # (B, P)
        scale = data["scale"][:, 0]                     # (B,)
        
        # Rescale to original scale
        B, _, _ = pts_gt.shape
        pts_gt_rescaled = pts_gt * scale.view(B, 1, 1)
        pts_pred_rescaled = pointclouds_pred * scale.view(B, 1, 1)

        object_cd = compute_object_cd(pts_gt_rescaled, pts_pred_rescaled)
        part_acc, matched_parts = compute_part_acc(pts_gt_rescaled, pts_pred_rescaled, points_per_part)
        rot_errors, trans_errors = compute_transform_errors(
            pts, pts_gt, rotations_pred, translations_pred, points_per_part, anchor_part, matched_parts, scale,
        )

        return {
            "object_cd": object_cd,                     # (B,)
            "part_acc": part_acc,                       # (B,)
            "rot_errors": rot_errors,                   # (B,)
            "trans_errors": trans_errors,               # (B,)
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
        rotations_pred: torch.Tensor,
        translations_pred: torch.Tensor,
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
            rotations_pred (B, P, 3, 3): Estimated rotation matrices.
            translations_pred (B, P, 3): Estimated translation vectors.
            save_results (bool): If True, save each result to log_dir/results.

        Returns:
            A dictionary with:
                object_cd (torch.Tensor): Object Chamfer distance, shape (B,).
                part_acc (torch.Tensor): Part accuracy, shape (B,).
                rot_errors (torch.Tensor): Rotation errors, shape (B,).
                trans_errors (torch.Tensor): Translation errors, shape (B,).
        """
        metrics = self._compute_metrics(data, pointclouds_pred, rotations_pred, translations_pred)
        if save_results:
            for i in range(data["points_per_part"].size(0)):
                self._save_single_result(data, metrics, i)
        return metrics
