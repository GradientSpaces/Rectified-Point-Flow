"""Rectified Flow for Point Cloud Assembly."""

import math
from functools import partial

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eval.evaluator import Evaluator
from .utils.checkpoint import load_checkpoint_for_module, set_rng_state, get_rng_state
from .utils.metrics import MetricsHandler


class RectifiedPointFlow(L.LightningModule):
    """Rectified Flow model for point cloud assembly."""
    
    def __init__(
        self,
        feature_extractor: L.LightningModule,
        flow_model: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        inference_config: dict = None,
        feature_extractor_ckpt: str = None,
        flow_model_ckpt: str = None,
        timestep_sampling: str = "u-shaped",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.flow_model = flow_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.inference_config = inference_config or dict()
        self.timestep_sampling = timestep_sampling

        # Load checkpoints
        if feature_extractor_ckpt is not None:
            load_checkpoint_for_module(
                self.feature_extractor,
                feature_extractor_ckpt,
                prefix_to_remove="feature_extractor.",
                strict=False,
            )

        if flow_model_ckpt is not None:
            load_checkpoint_for_module(
                self.flow_model,
                flow_model_ckpt,
                prefix_to_remove="flow_model.",
                strict=False,
            )

        # Initialize
        self.freeze_feature_extractor()
        self.metrics_handler = MetricsHandler(self)
        self.evaluator = Evaluator(self)

    def freeze_feature_extractor(self):
        """Ensure feature extractor stays in eval mode."""
        self.feature_extractor.eval()
        for module in self.feature_extractor.modules():
            module.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def sample_timesteps(
        self,
        batch_size: int,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 2.0,
        a: float = 4.0,
    ):
        """Sample timesteps based on weighting scheme."""
        device = self.device
        if self.timestep_sampling == "u_shaped":
            u = torch.rand(batch_size, device=device) * 2 - 1
            u = torch.asinh(u * math.sinh(a)) / a
            u = (u + 1) / 2
        elif self.timestep_sampling == "logit_normal":
            u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device)
            u = torch.sigmoid(u)
        elif self.timestep_sampling == "mode":
            u = torch.rand(size=(batch_size,), device=device)
            u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        elif self.timestep_sampling == "uniform":
            u = torch.rand(size=(batch_size,), device=device)
        else:
            raise ValueError(f"Invalid timestep sampling mode: {self.timestep_sampling}")
        return u 
    
    def extract_features(self, data_dict: dict):
        """Extract features from input data using FP16."""
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                out_dict = self.feature_extractor(data_dict)
        points = out_dict["point"]
        points["batch"] = points["batch_level1"].clone()
        return points

    def get_targets(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple:
        """Compute learning targets for rectified flow:
            - linear interpolation point cloud (x_t),
            - velocity field (v_t)
        """
        t = t.view(-1, 1, 1)            # (B, 1, 1)
        x_t = (1 - t) * x0 + t * x1     # interpolated point cloud
        v_t = x1 - x0                   # velocity field
        return x_t, v_t

    def log_metrics(self, metrics: dict):
        """Log metrics."""
        if self.trainer.global_rank == 0:
            for metric_name, metric_value in metrics.items():
                self.log(
                    f"train/{metric_name}",
                    metric_value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    rank_zero_only=True,
                )

    def forward(self, data_dict: dict):
        """Forward pass for training using rectified flow."""
        B, P = data_dict["points_per_part"].shape
        part_valids = data_dict["points_per_part"] != 0

        # Extract features
        latent = self.extract_features(data_dict)

        # Compute learning targets
        t = self.sample_timesteps(batch_size=B)
        x1 = data_dict["pointclouds_gt"]
        x0 = torch.randn_like(x1)
        x_t, v_target = self.get_targets(x0, x1, t)
        
        # Apply anchor constraints
        x_t = x_t.reshape(-1, 3)
        v_target = v_target.reshape(-1, 3)
        anchor_idx = data_dict["ref_part"][part_valids][latent['batch']]
        v_target[anchor_idx] = 0.0
        t_expanded = t.repeat_interleave(P).view(-1)
        t_valid = t_expanded[part_valids.flatten()]  # Only valid parts
        
        # Predict velocity field
        v_pred = self.flow_model(
            x=x_t,
            timesteps=t_valid,
            latent=latent,
            part_valids=part_valids,
            scale=data_dict["scale"][part_valids],
            ref_part=data_dict["ref_part"][part_valids],
        )
        return {
            "v_pred": v_pred,
            "v_target": v_target,
            "x_t": x_t,
            "t": t_valid,
            "latent": latent,
        }

    def loss(self, output_dict: dict):
        """Compute rectified flow loss (simple MSE on velocity field)."""
        v_pred = output_dict["v_pred"]
        v_target = output_dict["v_target"]
        flow_loss = F.mse_loss(v_pred, v_target, reduction="mean")
        return {
            "mse_loss": flow_loss,
            "pred_norm": v_pred.norm(dim=-1).mean(),
            "target_norm": v_target.norm(dim=-1).mean(),
        }

    def training_step(self, data_dict: dict, batch_idx: int):
        """Training step."""
        loss_dict = self.loss(self(data_dict))
        self.log_metrics(loss_dict)
        return loss_dict["mse_loss"]

    def validation_step(self, data_dict: dict, batch_idx: int):
        """Validation step."""
        loss_dict = self.loss(self(data_dict))

        eval_results = self.evaluator.run_evaluation(data_dict, save_results=False)
        self.metrics_handler.add_metrics(dataset_names=data_dict['dataset_name'], **eval_results)
        loss_dict.update({k: v.mean() for k, v in eval_results.items()})
        return loss_dict

    def test_step(self, data_dict: dict, batch_idx: int):
        """Test step with comprehensive evaluation."""
        loss_dict = self.loss(self(data_dict))

        eval_results = self.evaluator.run_evaluation(data_dict, save_results=True)
        self.metrics_handler.add_metrics(dataset_names=data_dict['dataset_name'], **eval_results)
        loss_dict.update({k: v.mean() for k, v in eval_results.items()})
        return loss_dict

    def sample_rectified_flow(
        self, 
        x0: torch.Tensor, 
        latent: dict, 
        part_valids: torch.Tensor,
        part_scale: torch.Tensor,
        ref_part: torch.Tensor,
        anchor_idx: torch.Tensor,
        num_steps: int = 20,
        return_tarjectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Sample from rectified flow using Euler integration.
        
        Args:
            x0: Initial noise
            latent: Feature latent dictionary
            part_valids: Valid parts mask
            part_scale: Part scaling factors
            ref_part: Reference part indicators
            anchor_idx: Anchor indices for constraints
            num_steps: Number of integration steps
            return_tarjectory: Whether to return the trajectory
            
        Returns:
            Predicted point cloud or list of predicted point clouds if return_tarjectory is True
        """
        dt = 1.0 / num_steps
        x_t = x0.clone()
        trajectory = []
        for step in range(num_steps):
            t = step * dt
            t_tensor = torch.full((len(part_scale),), t, device=x_t.device)
            v_pred = self.flow_model(
                x=x_t,
                timesteps=t_tensor,
                latent=latent,
                part_valids=part_valids,
                scale=part_scale,
                ref_part=ref_part,
            )
            x_t = x_t + dt * v_pred
            x_t[anchor_idx] = x0[anchor_idx]
            if return_tarjectory:
                trajectory.append(x_t.clone())
    
        if return_tarjectory:
            return trajectory
        return x_t
    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.freeze_feature_extractor()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.freeze_feature_extractor()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.freeze_feature_extractor()

    def on_validation_epoch_end(self):
        """Aggregate and log validation results."""
        self.metrics_handler.log_on_epoch_end(prefix="eval")

    def on_test_epoch_end(self):
        """Aggregate and log test results."""
        self.metrics_handler.log_on_epoch_end(prefix="test")

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return {"optimizer": optimizer}

        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
    
    def on_save_checkpoint(self, checkpoint):
        """Save checkpoint with RNG states."""
        checkpoint["rng_state"] = get_rng_state()
        return super().on_save_checkpoint(checkpoint)
    
    def on_load_checkpoint(self, checkpoint):
        """Restore RNG states."""
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"])
        else:
            print("No RNG state found in checkpoint.")
        super().on_load_checkpoint(checkpoint)
    

if __name__ == "__main__":
    # Test the model
    from .encoder.pointtransformerv3 import PointTransformerV3Objcentric
    from .encoder import PointCloudEncoder
    from .flow_model import PointCloudDiT

    lr_scheduler = partial(torch.optim.lr_scheduler.StepLR, step_size=100, gamma=0.1)
    feature_extractor = PointCloudEncoder(
        pc_feat_dim=64,
        encoder=PointTransformerV3Objcentric(),
        optimizer=torch.optim.AdamW,
        lr_scheduler=lr_scheduler,
    )
    flow_model = PointCloudDiT(
        in_dim=64,
        out_dim=3,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
    )
    rectified_point_flow = RectifiedPointFlow(
        feature_extractor=feature_extractor,
        flow_model=flow_model,
        optimizer=torch.optim.AdamW,
        lr_scheduler=lr_scheduler,
    )
    print(rectified_point_flow)