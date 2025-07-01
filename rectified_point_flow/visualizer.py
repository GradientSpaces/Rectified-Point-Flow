"""Visualization utilities for point cloud assembly."""

from pathlib import Path
from typing import Any, Optional

import lightning as L
import matplotlib.cm as cm
import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds

from .utils import ppp_to_ids


def generate_part_colors(part_ids: torch.Tensor, colormap: str = "Set2") -> torch.Tensor:
    """Generate colors for parts based on part IDs.
    
    Args:
        part_ids: Tensor of shape (N,) containing part IDs for each point.
        colormap: Name of matplotlib colormap to use.
        
    Returns:
        RGB colors in float tensor of shape (N, 3).
    """
    device = part_ids.device
    unique_parts = torch.unique(part_ids)
    num_parts = len(unique_parts)
    cmap = cm.get_cmap(colormap)
    part_to_color = {}
    for i, part_id in enumerate(unique_parts):
        color_rgba = cmap(float(i) / max(1, num_parts - 1))
        part_to_color[part_id.item()] = torch.tensor(color_rgba[:3], device=device)
    
    colors = torch.stack([
        part_to_color[pid.item()] for pid in part_ids
    ], dim=0)
    return colors.float()


def img_tensor_to_pil(image_tensor: torch.Tensor) -> Image:
    """Tensor (C, H, W) to PIL Image (H, W, C) and scale to [0, 255]."""
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(image_np)


@torch.inference_mode()
def visualize_point_clouds(
    points: torch.Tensor,
    part_ids: Optional[torch.Tensor] = None,
    colormap: str = "Set2",
    center_points: bool = False,
    image_size: int = 512,
    point_radius: float = 0.015,
    camera_dist: float = 2.0,
    camera_elev: float = 1.0,
    camera_azim: float = 0.0,
    camera_fov: float = 45.0,
) -> torch.Tensor:
    """Render a single point cloud to an image using PyTorch3D.

    Args:
        points: Point cloud coordinates of shape (N, 3).
        part_ids: Part IDs for each point of shape (N,). If None, uses uniform gray color.
        colormap: Matplotlib colormap name for part coloring.
        center_points: If True, centers the point cloud around the origin.
        image_size: Output image resolution (square).
        point_radius: Radius of each rendered point in world units.
        camera_dist: Distance of camera from point cloud center.
        camera_elev: Camera elevation angle in degrees.
        camera_azim: Camera azimuth angle in degrees.
        camera_fov: Camera field of view in degrees.

    Returns:
        Rendered image of shape (3, H, W) with values in [0, 1].
    """
    device = points.device
    num_points = points.shape[0]

    if part_ids is not None:
        colors = generate_part_colors(part_ids, colormap)
    else:
        colors = torch.ones(num_points, 3, device=device) * 0.7
    
    if center_points:
        points = points - points.mean(dim=0, keepdim=True)
    pointclouds = Pointclouds(points=[points.float()], features=[colors.float()])
    R, T = look_at_view_transform(dist=camera_dist, elev=camera_elev, azim=camera_azim, device=device)
    cameras = FoVPerspectiveCameras(R=R.float(), T=T.float(), fov=camera_fov, device=device)
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=point_radius,
        points_per_pixel=40,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = AlphaCompositor(background_color=(1.0, 1.0, 1.0))
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    rendered_images = renderer(pointclouds)
    return rendered_images[0].permute(2, 0, 1)  # (3, H, W)


class VisualizationCallback(Callback):
    """Lightning callback for visualizing point cloud assemblies during evaluation."""

    def __init__(
        self,
        save_dir: Optional[str] = None,
        scale_to_original_size: bool = False,
        center_points: bool = False,
        colormap: str = "Set2",
        image_size: int = 512,
        point_radius: float = 0.015,
        camera_dist: float = 4.0,
        camera_elev: float = 20.0,
        camera_azim: float = 30.0,
        camera_fov: float = 45.0,
        max_samples_per_batch: Optional[int] = None,
        save_trajectory: bool = True,
        trajectory_gif_fps: int = 25,
        trajectory_gif_pause_last_frame: float = 1.0,
    ):
        """Initialize visualization callback.

        Args:
            save_dir (str): Directory to save images. If None, uses trainer.log_dir/visualizations.
            scale_to_original_size (bool): If True, scales the point clouds to the original size. 
                Otherwise, keep the scaling, i.e. [-1, 1]. Default: False.
            center_points: If True, centers the point cloud around the origin. Default: False.
            colormap (str): Matplotlib colormap name for coloring parts. Default: "Set2".
            image_size (int): Output image resolution (square). Default: 512.
            point_radius (float): Radius of each rendered point in world units. Default: 0.01.
            camera_dist (float): Distance (m) of camera from origin. Default: 2.0.
            camera_elev (float): Elevation angle (deg). Default: 20.0.
            camera_azim (float): Azimuth angle (deg). Default: 30.0.
            camera_fov (float): Field of view (deg). Default: 45.0.
            max_samples_per_batch (int): Maximum samples to visualize per batch. None means all.
            save_trajectory (bool): Whether to save trajectory as GIF. Default: True.
            trajectory_gif_fps (int): Frames per second for the GIF.
            trajectory_gif_pause_last_frame (float): Pause time for the last frame in seconds.
        """
        super().__init__()
        self.save_dir = save_dir
        self.scale_to_original_size = scale_to_original_size
        self.max_samples_per_batch = max_samples_per_batch
        self.save_trajectory = save_trajectory
        self.trajectory_gif_fps = trajectory_gif_fps
        self.trajectory_gif_pause_last_frame = trajectory_gif_pause_last_frame

        self.vis_dir = None
        self._vis_kwargs = {
            "colormap": colormap,
            "center_points": center_points,
            "image_size": image_size,
            "point_radius": point_radius,
            "camera_dist": camera_dist,
            "camera_elev": camera_elev,
            "camera_azim": camera_azim,
            "camera_fov": camera_fov,
        }

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage == "test":
            if self.save_dir is None:
                self.vis_dir = Path(trainer.log_dir) / "visualizations"
            else:
                self.vis_dir = Path(self.save_dir) / "visualizations"
            self.vis_dir.mkdir(parents=True, exist_ok=True)
            print(f"Visualization saved to: {self.vis_dir}")

    def _save_sample_images(
        self,
        input_points: torch.Tensor,
        generated_points: torch.Tensor,
        part_ids: torch.Tensor,
        sample_idx: int,
        sample_name: Optional[str] = None,
    ) -> None:
        """Save visualization images for a single sample.
        
        Args:
            input_points (torch.Tensor): Input point cloud of shape (N, 3).
            generated_points (torch.Tensor): Generated point cloud of shape (N, 3).
            part_ids (torch.Tensor): Part IDs of shape (N,).
            sample_idx (int): Global sample index for naming.
            sample_name (str): Optional sample name for filename.
        """
        try:
            input_image = visualize_point_clouds(
                points=input_points,
                part_ids=part_ids,
                **self._vis_kwargs
            )
            generated_image = visualize_point_clouds(
                points=generated_points,
                part_ids=part_ids,
                **self._vis_kwargs
            )
            input_pil = img_tensor_to_pil(input_image)
            generated_pil = img_tensor_to_pil(generated_image)

            base_name = f"{sample_idx:05d}"
            if sample_name:
                base_name += f"_{sample_name.replace('/', '-')}"

            input_pil.save(self.vis_dir / f"{base_name}_condition.png")
            generated_pil.save(self.vis_dir / f"{base_name}_generated.png")

        except Exception as e:
            print(f"Error saving visualization for sample {sample_idx}: {e}")

    def _save_trajectory_gif(
        self,
        trajectory: list[torch.Tensor],
        part_ids: torch.Tensor,
        sample_idx: int,
        sample_name: Optional[str] = None,
    ) -> None:
        """Save trajectory as GIF.
        
        Args:
            trajectory: List of point clouds representing the trajectory steps.
            part_ids: Part IDs of shape (N,) for coloring.
            sample_idx: Global sample index for naming.
            sample_name: Optional sample name for filename.
        """
        try:
            base_name = f"{sample_idx:05d}"
            if sample_name:
                base_name += f"_{sample_name.replace('/', '-')}"
            gif_path = self.vis_dir / f"{base_name}_trajectory.gif"

            frames = []
            for step in range(trajectory.shape[0]):
                rendered_image = visualize_point_clouds(
                    points=trajectory[step],
                    part_ids=part_ids,
                    **self._vis_kwargs,
                )
                frame_pil = img_tensor_to_pil(rendered_image)
                frames.append(frame_pil)
            
            # Frame duration and pause on last frame
            duration = int(1000 / self.trajectory_gif_fps)
            durations = [duration] * len(frames)
            durations[-1] = int(duration + self.trajectory_gif_pause_last_frame * 1000)
            
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=0,  # Infinite loop
                optimize=True
            )
            
        except Exception as e:
            print(f"Error saving trajectory GIF for sample {sample_idx}: {e}")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Save visualizations at the end of each test batch."""
        if self.vis_dir is None:
            return

        points_per_part = batch["points_per_part"]                            # (bs, max_parts)
        B, _ = points_per_part.shape
        part_ids = ppp_to_ids(points_per_part)                                # (bs, N)
        input_points = batch["pointclouds"].reshape(B, -1, 3)                 # (bs, N, 3)
        pointclouds_pred = outputs['pointclouds_pred'].reshape(B, -1, 3)      # (bs, N, 3)

        if self.scale_to_original_size:
            scale = batch["scale"][:, 0]                                      # (bs,)
            input_points = input_points * scale[:, None, None]                # (bs, N, 3)
            pointclouds_pred = pointclouds_pred * scale[:, None, None]        # (bs, N, 3)

        for i in range(B):
            sample_idx = batch_idx * B + i
            sample_name = None
            if "name" in batch and batch["name"][i] is not None:
                sample_name = batch["name"][i]
            
            self._save_sample_images(
                input_points=input_points[i],
                generated_points=pointclouds_pred[i],
                part_ids=part_ids[i],
                sample_idx=sample_idx,
                sample_name=sample_name,
            )

            if self.save_trajectory:
                trajectory = outputs['trajectory']
                num_steps = trajectory.shape[0]
                trajectory = trajectory.reshape(num_steps, B, -1, 3).permute(1, 0, 2, 3)  # (bs, num_steps, N, 3)

                if self.scale_to_original_size:
                    trajectory = trajectory * scale[:, None, None, None]        # (bs, num_steps, N, 3)

                self._save_trajectory_gif(
                    trajectory=trajectory[i],
                    part_ids=part_ids[i],
                    sample_idx=sample_idx,
                    sample_name=sample_name,
                )

            if self.max_samples_per_batch is not None and i >= self.max_samples_per_batch:
                break
