from functools import partial
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics


class PointCloudEncoder(pl.LightningModule):
    """PyTorch Lightning module for overlap-aware pretraining on multi-part point clouds."""
    
    CHUNK_SIZE = 40               # Chunk size for computing overlap points. 0 means no chunking.
                                  # This is used to limit the peak memory usage.
                                  # This is a trade-off between memory usage and computation time.
    
    def __init__(
        self,
        pc_feat_dim: int,
        encoder: nn.Module,
        optimizer: partial,
        lr_scheduler: Optional[partial] = None,
        grid_size: float = 0.02,
        overlap_head_intermediate_dim: int = 16,
    ):
        super().__init__()
        self.pc_feat_dim = pc_feat_dim
        self.encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grid_size = grid_size
        self.overlap_head_intermediate_dim = overlap_head_intermediate_dim

        self._build_model()

    def _build_model(self):
        """Build the overlap-aware pretraining model components."""
        self.batch_norm = nn.BatchNorm1d(self.pc_feat_dim)
        self.overlap_head = nn.Sequential(
            nn.Linear(self.pc_feat_dim, self.overlap_head_intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.overlap_head_intermediate_dim, 1),
            nn.Flatten(0, 1),
        )

    def _init_weights(self):
        """Initialize model weights."""
        for layer in [self.overlap_head[0], self.overlap_head[2]]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _compute_overlap_points(self, batch: Dict[str, torch.Tensor], point_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute overlap points based on distance threshold."""
        pointclouds = batch["pointclouds_gt"]  # (B, N, 3)
        B, N, _ = pointclouds.shape
        part_ids = point_features.batch_level1.view(B, N)
        breakpoint()
        threshold = batch["threshold"].view(B, 1, 1)
        contact_mask = torch.zeros((B, N), dtype=torch.bool, device=pointclouds.device)
        
        # Process in chunks to limit memory usage
        chunk_size = self.CHUNK_SIZE
        if chunk_size > 0:
            assert N % chunk_size == 0, f"N ({N}) must be divisible by chunk_size ({chunk_size})"
            step_size = N // chunk_size
        else:
            step_size = N

        for start in range(0, N, step_size):
            end = min(N, start + step_size)
            sub_points = pointclouds[:, start:end, :]  # (B, M, 3)
            sub_part_ids = part_ids[:, start:end].unsqueeze(-1)  # (B, M, 1)
            
            # Pairwise distances
            distances = torch.cdist(sub_points, pointclouds, p=2)
            
            # Mask same-part points
            full_part_ids = part_ids.unsqueeze(1)  # (B, 1, N)
            different_parts = sub_part_ids != full_part_ids  # (B, M, N)
            distances[~different_parts] = float('inf')
            
            # Check for overlap points
            has_contact = (distances <= threshold).any(dim=2)  # (B, M)
            contact_mask[:, start:end] = has_contact

        return contact_mask.long().view(-1)

    def _extract_point_features(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Extract point features using the encoder."""
        pointclouds = batch["pointclouds"]  # (B, N, 3)
        normals = batch["pointclouds_normals"]  # (B, N, 3)
        points_per_part = batch["points_per_part"]  # (B, P)
        B, N, C = pointclouds.shape
        n_valid_partsarts = points_per_part != 0
        
        # Prepare inputs for encoder
        part_coords = pointclouds.view(-1, C)
        part_normals = normals.view(-1, C)
        points_offset = torch.cumsum(points_per_part[n_valid_partsarts], dim=-1)
        object_offset = torch.arange(1, B + 1, device=points_per_part.device) * N

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            super_point, point = self.encoder({
                "coord": part_coords,
                "offset": points_offset,
                "offset_level1": points_offset,
                "offset_level0": object_offset,
                "feat": torch.cat([part_coords, part_normals], dim=-1),
                "grid_size": torch.tensor(self.grid_size).to(part_coords.device),
            })
            point["normal"] = part_normals
            features = self.batch_norm(point["feat"])
            assert not features.isnan().any(), "Point features contain NaN values"
            
        return features, point, super_point, n_valid_partsarts

    def forward(self, batch: Dict[str, torch.Tensor], batch_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        # Extract features
        point_features, point_data, super_point_data, _ = self._extract_point_features(batch)
        
        # Overlap prediction
        seg_logits = self.overlap_head(point_features)
        seg_pred = torch.sigmoid(seg_logits)
        seg_pred_binary = seg_pred > 0.5

        # Prepare output dictionary
        output = {
            "seg_logits": seg_logits,
            "seg_pred": seg_pred,
            "seg_pred_binary": seg_pred_binary,
            "point": point_data,
            "super_point": super_point_data,
        }

        # Compute overlap points GT for pretraining stage
        if self.training:
            with torch.no_grad():
                output["seg_gt"] = self._compute_overlap_points(batch, point_data)
            
        return output
    
    def loss(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> tuple:
        """Compute loss and metrics for training/validation."""
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions["seg_logits"],
            predictions["seg_gt"].float(),
            reduction="mean",
        )
        
        # Compute metrics
        pred_binary = predictions["seg_pred_binary"]
        target = predictions["seg_gt"]
        metrics = {
            "accuracy": torchmetrics.functional.accuracy(pred_binary, target, task="binary"),
            "recall": torchmetrics.functional.recall(pred_binary, target, task="binary"),
            "precision": torchmetrics.functional.precision(pred_binary, target, task="binary"),
            "f1": torchmetrics.functional.f1_score(pred_binary, target, task="binary"),
            "bce_loss": bce_loss,
        }
        
        return bce_loss, metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        """Training step."""
        try:
            predictions = self(batch)
            loss, metrics = self.loss(predictions, batch)
            self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log_dict({f"train/{k}": v for k, v in metrics.items()}, on_step=True, on_epoch=False)
            return loss
        
        except Exception as e:
            print(f"Training step error: {e}")
            return None

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        """Validation step."""
        try:
            predictions = self(batch)
            loss, metrics = self.loss(predictions, batch)
            self.log("val/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            self.log_dict({f"val/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, sync_dist=True)
            return loss
        
        except Exception as e:
            print(f"Validation step error: {e}")
            return None

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        predictions = self.forward(batch)
        loss, metrics = self.loss(predictions, batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return {"optimizer": optimizer}
        
        lr_scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
