import trimesh
import numpy as np
from typing import List, Tuple

try:
    import point_cloud_utils as pcu
    use_pcu = True
except ImportError:
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

def pad_data(input_data: np.ndarray, max_parts: int) -> np.ndarray:
    """Pad data to max_parts."""
    d = np.array(input_data)
    pad_shape = (max_parts,) + tuple(d.shape[1:])
    pad_data = np.zeros(pad_shape, dtype=np.float32)
    pad_data[: d.shape[0]] = d
    return pad_data
