import torch
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import PyTorch3D modules for rendering
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizer,
    PointsRenderer,
    PointsRasterizationSettings,
    AlphaCompositor,
    look_at_view_transform
)
from torchvision.transforms import ToPILImage, ToTensor
from pytorch3d.structures import Pointclouds


def add_text_to_image(img_tensor: torch.Tensor, text: str, position: tuple = (10, 10)) -> torch.Tensor:
    """
    Adds text to a single image tensor.

    Args:
        img_tensor (torch.Tensor): A tensor of shape (3, H, W) with values in [0,1].
        text (str): Text to overlay.
        position (tuple): (x, y) position for the text.

    Returns:
        torch.Tensor: Annotated image tensor of shape (3, H, W) with values in [0,1].
    """
    # Convert tensor to a NumPy image (H, W, C) in uint8.
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil_img)
    
    # Use a default font (or specify a TTF file if needed)
    try:
        font = ImageFont.load_default(16)
    except:
        font = None

    # Draw the text in black.
    draw.text(position, text, fill=(0, 0, 0), font=font)
    
    # Convert back to tensor
    annotated_img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    return annotated_img


@torch.inference_mode()
def pcd_with_colormap(
    pcd: torch.Tensor,             # (B, N, 3) tensor with point cloud coordinates
    v: torch.Tensor,               # (B, N) tensor with scalar values used for the colormap
    vmin: float = 0.0,
    vmax: float = 1.0,
    colormap: str = "Reds",
    part_id: torch.Tensor = None,  # (B, N) long tensor with part IDs
    image_size: int = 512,
    point_radius: float = 0.01,   # Adjust based on the scale of your point clouds
    dist: float = 2.0,             # Distance of the camera from the point cloud
    elev: float = 1.0,             # Elevation angle of the camera
    azim: float = 0.0,              # Azimuth angle of the camera
    fov: float = 45.0,            # Field of view for the camera
    text: list[str] = [],         # List of text labels for each point cloud
    text_position: tuple = (10, 10),  # Position for the text overlay
    center = None,                  # Center of the point cloud (optional)
    color_legend: bool = False,         # Whether to show a color legend
) -> torch.Tensor:
    """
    Visualizes batched point clouds using PyTorch3D's point cloud rasterization.

    Each point cloud is first centered (so the camera is aimed at its center), then rasterized.
    Colors are computed via a Matplotlib colormap from the provided scalar values.
    
    Parameters:
      pcd (torch.Tensor): Batched point cloud tensor of shape (B, N, 3).
      v (torch.Tensor): Batched scalar values tensor of shape (B, N).
      vmin (float): Minimum value for normalization.
      vmax (float): Maximum value for normalization.
      colormap (str): Matplotlib colormap name.
      points_per_part (torch.Tensor): Ignored here.
      image_size (int): Output image resolution (image_size x image_size).
      point_radius (float): Radius for each point in world units.
    
    Returns:
      torch.Tensor: Rendered images of shape (B, 3, H, W).
    """
    device = pcd.device
    B, N, _ = pcd.shape

    # --- Map scalar values to colors ---
    v_np = v.detach().cpu().numpy()
    cmap = cm.get_cmap(colormap)
    if part_id is None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        normalized_values = norm(v_np)  # shape (B, N)
        colors_rgba = cmap(normalized_values)  # shape (B, N, 4)
        point_colors = torch.from_numpy(colors_rgba[..., :3]).to(device).float()  # (B, N, 3)
    else:
        # Use part_id to create a colormap for each part
        colors_rgbs = []
        for i in range(B):
            unique_parts = torch.unique(part_id[i])
            part_colors = {part.item(): cmap(float(i) / len(unique_parts)) for i, part in enumerate(unique_parts)}
            rgba = [torch.tensor(part_colors[pid.item()]) for pid in part_id[i]]
            
            rgba = torch.stack(rgba, dim=0)
            colors_rgbs.append(rgba)
        point_colors = torch.stack(colors_rgbs, dim=0)  # shape (B, N, 4)
        point_colors = point_colors[..., :3].to(device).float()  # (B, N, 3)

    # --- Center each point cloud ---
    if center is None:
        # Compute the center of each point cloud
        center = pcd.mean(dim=1, keepdim=True).float()
    pcd_centered = pcd - center  # Now each point cloud is centered at the origin

    # --- Set up the batched camera ---
    # We use look_at_view_transform to create a camera that looks from a distance of 2 meters
    # with zero elevation and azimuth (i.e. looking along the z-axis toward the origin).
    try:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device, batch_size=B)
    except TypeError:
        # Fallback for older versions: expand single camera parameters across the batch.
        R_single, T_single = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        R = R_single.expand(B, -1, -1)
        T = T_single.expand(B, -1)

    pointclouds = Pointclouds(points=pcd_centered.float(), features=point_colors.float())
    cameras = FoVPerspectiveCameras(R=R.float(), T=T.float(), fov=fov, device=device)

    # --- Set up the rasterizer and renderer ---
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=point_radius,
        points_per_pixel=40,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    # Use an AlphaCompositor with a white background
    compositor = AlphaCompositor(background_color=(1, 1, 1))
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    # --- Render the batched point clouds ---
    rendered_images = renderer(pointclouds)
    rendered_images = rendered_images.permute(0, 3, 1, 2)

    # --- Add text overlay if provided ---
    if len(text) > 0:
        annotated_images = []
        for i in range(rendered_images.shape[0]):
            annotated_img = add_text_to_image(rendered_images[i], text[i])
            annotated_images.append(annotated_img)
        rendered_images = torch.stack(annotated_images, dim=0)

    # --- Add color legend if requested ---
    if color_legend:
        imgs = []
        to_pil = ToPILImage()
        to_tensor = ToTensor()
        for i in range(B):
            img_pil = to_pil(rendered_images[i].cpu())
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.load_default(16)
            margin = 5
            if part_id is None:
                # Scalar colorbar
                cb_h = image_size
                cb_w = 20
                cb = np.linspace(vmax, vmin, cb_h)
                for y, val in enumerate(cb):
                    color = tuple((np.array(cmap((val - vmin) / (vmax - vmin))[:3]) * 255).astype(np.uint8))
                    draw.line([(margin, y), (margin + cb_w, y)], fill=color)
                # Labels
                draw.text((margin + cb_w + 2, 0), f"{vmax:.2f}", font=font, fill=(0, 0, 0))
                draw.text((margin + cb_w + 2, cb_h - 10), f"{vmin:.2f}", font=font, fill=(0, 0, 0))
            else:
                # Part legend
                unique_parts = torch.unique(part_id[i]).tolist()
                square_size = 20
                for j, part in enumerate(unique_parts):
                    color = tuple((np.array(cmap(j / len(unique_parts))[:3]) * 255).astype(np.uint8))
                    y0 = margin + j * (square_size + margin)
                    draw.rectangle([margin, y0, margin + square_size, y0 + square_size], fill=color)
                    draw.text((margin + square_size + margin, y0), str(part), font=font, fill=(0, 0, 0))
            imgs.append(to_tensor(img_pil).to(device))
        rendered_images = torch.stack(imgs, dim=0)

    return rendered_images