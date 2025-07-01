import logging
from typing import Tuple
import os

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

logger = logging.getLogger("Data")
trimesh.util.log.setLevel(logging.ERROR)

if os.environ.get("USE_PCU", "0") == "1":
    try:
        import point_cloud_utils as pcu
        use_pcu = True
        logger.info("Using point_cloud_utils for point sampling.")
    except ImportError:
        logger.warning("point_cloud_utils not found, using trimesh.sample instead.")
        use_pcu = False
else:
    logger.info("Using trimesh.sample for point sampling.")
    use_pcu = False


def sample_points_poisson(mesh: trimesh.Trimesh, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points using Poisson disk sampling."""
    if use_pcu:
        v = mesh.vertices
        f = mesh.faces
        idx, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=count)
        pts = pcu.interpolate_barycentric_coords(f, idx, bc, v)
    else:
        pts, idx = trimesh.sample.sample_surface_even(mesh, count=count)
    return pts, idx


def center_pcd(pcd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center point cloud at origin."""
    center = np.mean(pcd, axis=0)
    pcd = pcd - center
    return pcd, -center


def rotate_pcd(pcd: np.ndarray, normals: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly rotate point cloud."""
    rot = Rotation.random()
    pcd = rot.apply(pcd)
    
    # Convert to PyTorch3D convention (w, x, y, z) from scipy (x, y, z, w)
    # In later code, we use PyTorch3D for efficient point cloud transformations.
    quat_xyzw = rot.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    if normals is not None:
        normals = rot.apply(normals)
    return pcd, normals, quat_wxyz


def pad_data(input_data: np.ndarray, max_parts: int) -> np.ndarray:
    """Pad zeros to data of shape (N, ...) to (max_parts, ...)"""
    d = np.array(input_data)
    pad_shape = (max_parts,) + tuple(d.shape[1:])
    pad_data = np.zeros(pad_shape, dtype=np.float32)
    pad_data[: d.shape[0]] = d
    return pad_data