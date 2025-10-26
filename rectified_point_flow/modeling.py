"""Rectified Flow for Point Cloud Assembly."""

import math
from functools import partial
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eval.evaluator import Evaluator
from .procrustes import fit_transformations
from .sampler import get_sampler
from .utils.checkpoint import get_rng_state, load_checkpoint_for_module, set_rng_state
from .utils.logging import MetricsMeter, log_metrics_on_step, log_metrics_on_epoch


class RectifiedPointFlow(L.LightningModule):
    """Rectified Flow model for point cloud assembly."""
    
    def __init__(
        self,
        feature_extractor: L.LightningModule,
        flow_model: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        encoder_ckpt: str = None,
        flow_model_ckpt: str = None,
        frozen_encoder: bool = False,
        anchor_free: bool = True,
        loss_type: str = "mse",
        timestep_sampling: str = "u_shaped",
        inference_sampling_steps: int = 20,
        inference_sampler: str = "euler",
        n_generations: int = 1,
        pred_proc_fn: Callable | None = None,
        save_results: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.flow_model = flow_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.frozen_encoder = frozen_encoder
        self.anchor_free = anchor_free
        self.loss_type = loss_type
        self.timestep_sampling = timestep_sampling
        self.inference_sampling_steps = inference_sampling_steps
        self.inference_sampler = inference_sampler
        self.n_generations = n_generations
        self.pred_proc_fn = pred_proc_fn
        self.save_results = save_results

        # Load checkpoints
        if encoder_ckpt is not None:
            load_checkpoint_for_module(
                self.feature_extractor,
                encoder_ckpt,
                keys_to_substitute={"feature_extractor.": ""},
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
        self.evaluator = Evaluator(self)
        self.meter = MetricsMeter(self)
        self._freeze_encoder()

    def _freeze_encoder(self, eval_mode: bool = False):
        if self.frozen_encoder or eval_mode:
            self.feature_extractor.eval()
            for module in self.feature_extractor.modules():
                module.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor.train()
            for module in self.feature_extractor.modules():
                module.train()
            for param in self.feature_extractor.parameters():
                param.requires_grad = True

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._freeze_encoder()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._freeze_encoder(eval_mode=True)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._freeze_encoder(eval_mode=True)

    def _sample_timesteps(
        self,
        batch_size: int,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 2.0,
        a: float = 4.0,
        eps: float = 0.01,
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
        
        # Clamp small t to reduce loss spikes
        u = u.clamp(eps, 1.0)
        return u
    
    def _encode(self, data_dict: dict):
        """Extract features from input data using FP16."""
        with torch.inference_mode(self.frozen_encoder):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                out_dict = self.feature_extractor(data_dict)
        points = out_dict["point"]
        points["batch"] = points["batch_level1"].clone()
        return points

    def _compute_flow_target(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> tuple:
        """Compute the learning target of rectified flow.

        Args:
            x0: Ground truth point cloud
            x1: Noise point cloud
            t: Timesteps

        Returns:
            x_t: Linear interpolation point cloud
            v_t: Velocity field
        """
        t = t.view(-1, 1, 1)              # (B, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1     # interpolated point cloud
        v_t = x_1 - x_0                   # velocity field
        return x_t, v_t

    def forward(self, data_dict: dict):
        """Forward pass for training using rectified flow."""
        x_0 = data_dict["pointclouds_gt"]                           # (B, N, 3)
        scales = data_dict["scales"]                                # (B, )
        anchor_indices = data_dict["anchor_indices"]                # (B, N)
        B, N, _ = x_0.shape

        # Encode point clouds
        latent = self._encode(data_dict)
        
        # Sample timesteps
        timesteps = self._sample_timesteps(batch_size=B)            # (B, )
        
        # Sample noise and compute flow target
        x_1 = torch.randn_like(x_0)                                 # (B, N, 3)
        x_t, v_t = self._compute_flow_target(x_0, x_1, timesteps)   # (B, N, 3) each
        
        # Apply anchor part constraints (only used in anchor-fixed mode)
        if not self.anchor_free:
            x_t[anchor_indices] = x_0[anchor_indices]
            v_t[anchor_indices] = 0.0
        
        # Predict velocity field
        v_pred = self.flow_model(
            x=x_t,
            timesteps=timesteps,
            latent=latent,
            scales=scales,
            anchor_indices=anchor_indices,
        )
        output_dict = {
            "t": timesteps,
            "v_pred": v_pred,
            "v_t": v_t,
            "x_0": x_0,
            "x_1": x_1,
            "x_t": x_t,
            "latent": latent,
        }

        if self.pred_proc_fn is not None:
            output_dict = self.pred_proc_fn(output_dict)
        return output_dict

    def loss(self, output_dict: dict):
        """Compute rectified flow loss."""
        v_pred = output_dict["v_pred"]
        v_t = output_dict["v_t"]

        if self.loss_type == "mse":
            loss = F.mse_loss(v_pred, v_t, reduction="mean")
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_pred, v_t, reduction="mean")
        elif self.loss_type == "huber":
            loss = F.huber_loss(v_pred, v_t, reduction="mean")
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        return {
            "loss": loss,
            "norm_v_pred": v_pred.norm(dim=-1).mean(),
            "norm_v_t": v_t.norm(dim=-1).mean(),
        }

    def training_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Training step."""
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict)
        log_metrics_on_step(self, loss_dict, prefix="train")
        return loss_dict["loss"]

    def validation_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Validation step."""
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict)
        pointclouds_pred = self.sample_rectified_flow(data_dict, output_dict["latent"])

        # Evaluate the final predicted point clouds
        eval_results = self.evaluator.run(data_dict, pointclouds_pred)
        self.meter.add_metrics(dataset_names=data_dict["dataset_name"], **eval_results)
        return loss_dict["loss"]

    def test_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Test step with support for multiple generations."""
        latent = self._encode(data_dict)
        n_trajectories = []
        n_rotations_pred = []
        n_translations_pred = []
        n_eval_results = []

        pointclouds_cond = data_dict["pointclouds"]
        points_per_part = data_dict["points_per_part"]

        for gen_idx in range(self.n_generations):
            trajs = self.sample_rectified_flow(data_dict, latent, return_tarjectory=True)
            pointclouds_pred = trajs[-1]      
            rotations_pred, translations_pred = fit_transformations(
                pointclouds_cond, pointclouds_pred, points_per_part
            )
            eval_results = self.evaluator.run(
                data_dict, 
                pointclouds_pred, 
                rotations_pred, 
                translations_pred, 
                save_results=self.save_results, 
                generation_idx=gen_idx,
            )
            n_trajectories.append(trajs)
            n_rotations_pred.append(rotations_pred)
            n_translations_pred.append(translations_pred)
            n_eval_results.append(eval_results)
        
        # Compute average metrics
        avg_results = {}
        for key in n_eval_results[0].keys():
            avg = sum(result[key] for result in n_eval_results) / len(n_eval_results)
            avg_results[f'avg/{key}'] = avg
        self.log_dict(avg_results, prog_bar=False)

        # Compute best of N (BoN) metrics
        if self.n_generations > 1:
            best_results = {}
            for key in n_eval_results[0].keys():
                values = [result[key] for result in n_eval_results]
                agg_fn = max if ('acc' in key or 'recall' in key) else min
                best_results[f'best_of_n/{key}'] = agg_fn(values)
            self.log_dict(best_results, prog_bar=False)
        
        return {
            'trajectories': n_trajectories,
            'rotations_pred': n_rotations_pred,
            'translations_pred': n_translations_pred,
        }
    
    @torch.inference_mode()
    def sample_rectified_flow(
        self, 
        data_dict: dict,
        latent: dict, 
        x_1: torch.Tensor | None = None,
        return_tarjectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Sample from rectified flow using configurable integration methods.
        
        Args:
            data_dict: Input data dictionary
            latent: Feature latent dictionary
            x_1: Optional initial noise. If None, generates random Gaussian noise.
            return_tarjectory: Whether to return the trajectory
            
        Returns:
            (num_points, 3) if return_tarjectory == False
            (num_steps, num_points, 3) if return_tarjectory == True
        """
        anchor_indices = data_dict["anchor_indices"]
        scales = data_dict["scales"]

        def _flow_model_fn(x: torch.Tensor, t: float) -> torch.Tensor:
            B = x.shape[0]
            timesteps = torch.full((B,), t, device=x.device)
            return self.flow_model(
                x=x,
                timesteps=timesteps,
                latent=latent,
                scales=scales,
                anchor_indices=anchor_indices,
            )

        x_0 = data_dict["pointclouds_gt"]
        x_1 = torch.randn_like(x_0) if x_1 is None else x_1
        
        result = get_sampler(self.inference_sampler)(
            flow_model_fn=_flow_model_fn,
            x_1=x_1,
            x_0=x_0,
            anchor_indices=anchor_indices if not self.anchor_free else None, # None => skip anchor constraints
            num_steps=self.inference_sampling_steps,
            return_trajectory=return_tarjectory,
        )
        return result

    def on_validation_epoch_end(self):
        metrics = self.meter.compute_average()
        log_metrics_on_epoch(self, metrics, prefix="val")
        return metrics
    
    def on_test_epoch_end(self):
        metrics = self.meter.compute_average()
        log_metrics_on_epoch(self, metrics, prefix="test")
        return metrics
        
    def on_save_checkpoint(self, checkpoint):
        checkpoint["rng_state"] = get_rng_state()
        return super().on_save_checkpoint(checkpoint)
    
    def on_load_checkpoint(self, checkpoint):
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"])
        else:
            print("No RNG state found in checkpoint.")
        super().on_load_checkpoint(checkpoint)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return {"optimizer": optimizer}

        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
    

if __name__ == "__main__":
    # Test the model
    from .encoder import PointCloudEncoder
    from .encoder.pointtransformerv3 import PointTransformerV3Objcentric
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
        inference_sampler="euler",  # Can be "euler", "rk2", or "rk4"
    )

    print(rectified_point_flow)
