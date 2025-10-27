import json
from pathlib import Path
from typing import Any, Dict

import torch
import lightning as L

from .metrics import compute_object_cd, compute_part_acc, compute_transform_errors, align_anchor


class Evaluator:
    """Evaluator for Rectified Point Flow model. """
    
    def __init__(self, model: L.LightningModule):
        self.model = model

    def _compute_metrics(
        self,
        data: Dict[str, Any],
        pointclouds_pred: torch.Tensor,
        rotations_pred: torch.Tensor | None = None,
        translations_pred: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute evaluation metrics."""
        pts = data["pointclouds"]                       # (B, N, 3)
        pts_gt = data["pointclouds_gt"]                 # (B, N, 3)
        points_per_part = data["points_per_part"]       # (B, P)
        anchor_parts = data["anchor_parts"]             # (B, P)
        scales = data["scales"]                         # (B,)
        
        # Rescale to original scales
        B, _, _ = pts_gt.shape
        pointclouds_pred = pointclouds_pred.view(B, -1, 3)
        pts_gt_rescaled = pts_gt * scales.view(B, 1, 1)
        pts_pred_rescaled = pointclouds_pred * scales.view(B, 1, 1)

        # Align the predicted anchor parts to the ground truth anchor parts using ICP (only used in anchor-free mode)
        if self.model.anchor_free:
            pts_pred_rescaled = align_anchor(pts_gt_rescaled, pts_pred_rescaled, points_per_part, anchor_parts)

        object_cd = compute_object_cd(pts_gt_rescaled, pts_pred_rescaled)
        part_acc, matched_parts = compute_part_acc(pts_gt_rescaled, pts_pred_rescaled, points_per_part)
        metrics = {
            "part_accuracy": part_acc,
            "object_chamfer": object_cd,
        }
        
        if rotations_pred is not None and translations_pred is not None:
            rot_errors, trans_errors = compute_transform_errors(
                pts, pts_gt, rotations_pred, translations_pred, points_per_part, anchor_parts, matched_parts, scales,
            )
            rot_recalls = self._recall_at_thresholds(rot_errors, [5, 10])
            trans_recalls = self._recall_at_thresholds(trans_errors, [0.01, 0.05])
            metrics.update({
                "rotation_error": rot_errors,
                "translation_error": trans_errors,
                "recall_at_5deg": rot_recalls[0],
                "recall_at_10deg": rot_recalls[1],
                "recall_at_1cm": trans_recalls[0],
                "recall_at_5cm": trans_recalls[1],
            })

        return metrics
    
    @staticmethod
    def _recall_at_thresholds(metrics: torch.Tensor, thresholds: list[float]) -> list[torch.Tensor]:
        """Compute metrics of shape (B,) at thresholds."""
        return [(metrics <= threshold).float() for threshold in thresholds]

    def _save_single_result(
        self,
        data: Dict[str, Any],
        metrics: Dict[str, torch.Tensor],
        idx: int,
        generation_idx: int = 0,
    ) -> None:
        """Save a single evaluation result to JSON.

        Args:
            data: Input data dictionary.
            metrics: Computed metrics dictionary.
            idx: Index of the sample in the batch.
            generation_idx: Generation index for the result file name.
        """
        dataset_name = data["dataset_name"][idx]
        entry = {
            "name": data["name"][idx],
            "dataset": dataset_name,
            "num_parts": int(data["num_parts"][idx]),
            "generation_idx": generation_idx,
            "scales": float(data["scales"][idx]),
        }
        entry.update({k: float(v[idx]) for k, v in metrics.items()})

        out_dir = Path(self.model.trainer.log_dir) / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"{dataset_name}_sample{int(data['index'][idx]):05d}_generation{generation_idx:02d}.json"
        filepath.write_text(json.dumps(entry))

    def run(
        self,
        data: Dict[str, Any],
        pointclouds_pred: torch.Tensor,
        rotations_pred: torch.Tensor | None = None,
        translations_pred: torch.Tensor | None = None,
        save_results: bool = False,
        generation_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Run evaluation and optionally save results.

        Args:
            data: Input data dictionary, containing:
                pointclouds_gt (B, N, 3): Ground truth point clouds.
                scales (B,): scales factors.
                points_per_part (B, P): Points per part.
                name (B,): Object names.
                dataset_name (B,): Dataset names.
                index (B,): Object indices.
                num_parts (B,): Number of parts.

            pointclouds_pred (B, N, 3) or (B*N, 3): Model output samples.
            rotations_pred (B, P, 3, 3), optional: Estimated rotation matrices.
            translations_pred (B, P, 3), optional: Estimated translation vectors.
            save_results (bool): If True, save each result to log_dir/results.
            generation_idx (int): The index of the generation (mainly for best-of-n generations).

        Returns:
            A dictionary with:

                object_chamfer_dist (B,): Object Chamfer distance in meters.
                part_accuracy (B,): Part accuracy.

            If rotations_pred and translations_pred are provided, also return:

                rotation_error (B,): Rotation errors in degrees.
                translation_error (B,): Translation errors in meters.
                recall_at_5deg (B,): Recall at 5 degrees.
                recall_at_10deg (B,): Recall at 10 degrees.
                recall_at_1cm (B,): Recall at 1 cm.
                recall_at_5cm (B,): Recall at 5 cm.
        """
        metrics = self._compute_metrics(data, pointclouds_pred, rotations_pred, translations_pred)
        if save_results:
            B = data["points_per_part"].size(0)
            for i in range(B):
                self._save_single_result(data, metrics, i, generation_idx)
        return metrics
