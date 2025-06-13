import numpy as np
from scipy.spatial.transform import Rotation


def center_pc(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center point cloud at origin."""
    center = np.mean(points, axis=0)
    points = points - center
    return points, center

def rotate_pc(points: np.ndarray, normals: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly rotate point cloud."""
    # Generate random rotation
    rot = Rotation.random()
    quaternion = rot.as_quat()
    points = rot.apply(points)
    if normals is not None:
        normals = rot.apply(normals)
    return points, normals, quaternion

def shuffle_pc(points: np.ndarray, normals: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle point cloud points."""
    points = points[order]
    if normals is not None:
        normals = normals[order]
    return points, normals
