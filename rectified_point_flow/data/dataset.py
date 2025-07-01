import os
import glob
import logging
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import trimesh
from trimesh.exchange.ply import load_ply
from torch.utils.data import Dataset

from .transform import sample_points_poisson, center_pcd, rotate_pcd, pad_data

logger = logging.getLogger("Data")


def _load_mesh_from_h5(group, part_name):
    """Load one mesh part from an HDF5 group."""
    sub_grp = group[part_name]
    verts = np.array(sub_grp["vertices"][:])
    norms = np.array(sub_grp["normals"][:]) if "normals" in sub_grp else None
    faces = np.array(sub_grp["faces"][:]) if "faces" in sub_grp else np.array([])
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, process=False)

def _load_mesh_from_ply(ply_path):
    """Load a mesh from a PLY file."""
    with open(ply_path, "rb") as f:
        ply_data = load_ply(f)
    verts = ply_data["vertices"]
    norms = ply_data["vertex_normals"] if "vertex_normals" in ply_data else None
    faces = ply_data["faces"] if "faces" in ply_data else None
    return trimesh.Trimesh(vertices=verts, vertex_normals=norms, faces=faces, process=False)
    

class PointCloudDataset(Dataset):
    """Dataset class for multi-part point clouds and apply part-level augmentation."""

    def __init__(
        self,
        split: str = "train",
        data_path: str = "data.hdf5",
        dataset_name: str = "",
        up_axis: str = "y",
        min_parts: int = 2,
        max_parts: int = 64,
        num_points_to_sample: int = 5000,
        min_points_per_part: int = 20,
        random_scale_range: tuple[float, float] | None = None,
        multi_anchor: bool = False,
        limit_val_samples: int = 0,
        min_dataset_size: int = 0,
        num_threads: int = 2,
    ):
        super().__init__()
        self.split = split
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.up_axis = up_axis.lower()
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.random_scale_range = random_scale_range
        self.multi_anchor = multi_anchor
        self.limit_val_samples = limit_val_samples
        self.min_dataset_size = min_dataset_size

        self.use_folder = os.path.isdir(self.data_path)
        self.pool = ThreadPoolExecutor(max_workers=num_threads)
        self._h5_file = None

        self.min_part_count = self.max_parts + 1
        self.max_part_count = 0
        self.fragments = self._build_fragment_list()
        logger.info(
            f"| {self.dataset_name:16s} | {self.split:8s} | {len(self.fragments):8d} "
            f"| [{int(self.min_part_count):2d}, {int(self.max_part_count):2d}] |"
        )

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, index):
        frag = self.fragments[index]
        if self.use_folder:
            sample = self._load_from_folder(frag, index)
        else:
            sample = self._load_from_h5(frag, index)
        return self._transform(sample)

    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.data_path, "r", libver='latest', swmr=True)
        return self._h5_file

    def _build_fragment_list(self) -> list[str]:
        """Read and filter fragment keys from hdf5 or folder."""
        if self.use_folder:
            split_file = os.path.join(self.data_path, "data_split", f"{self.split}.txt")
            with open(split_file, 'r') as f:
                frags = [line.strip() for line in f if line.strip()]
            fragments = []
            for frag in frags:
                parts = glob.glob(
                    os.path.join(self.data_path, frag, "*.ply")
                )
                n_parts = len(parts)
                if self.min_parts <= n_parts <= self.max_parts:
                    self.min_part_count = min(self.min_part_count, n_parts)
                    self.max_part_count = max(self.max_part_count, n_parts)
                    fragments.append(frag)
            return fragments

        elif self.data_path.endswith('.hdf5'):
            h5 = self._get_h5_file()
            raw = h5["data_split"][self.dataset_name][self.split]
            frags = [r.decode() for r in raw[:]]
            fragments = []
            for name in frags:
                try:
                    count = len(h5[name].keys())
                    if self.min_parts <= count <= self.max_parts:
                        self.min_part_count = min(self.min_part_count, count)
                        self.max_part_count = max(self.max_part_count, count)
                        fragments.append(name)
                except KeyError:
                    continue

        else:
            raise ValueError(
                f"Invalid data path: {self.data_path}. Please provide a folder path or a .hdf5 file."
            )

        # limit or upsample
        if self.limit_val_samples > 0 and len(fragments) > self.limit_val_samples:
            step = len(fragments) // self.limit_val_samples
            fragments = fragments[::step]

        if self.min_dataset_size > 0 and len(fragments) < self.min_dataset_size:
            reps = -(-self.min_dataset_size // len(fragments))
            fragments = fragments * reps

        return fragments

    def _load_from_h5(self, name: str, index: int) -> dict:
        group = self._get_h5_file()[name]
        parts = sorted(list(group.keys()))
        meshes = list(self.pool.map(lambda p: _load_mesh_from_h5(group, p), parts))
        pcs, pns, thr = self._sample_points(meshes)
        return {
            "index": index,
            "name": name,
            "num_parts": len(parts),
            "pointclouds_gt": pcs,
            "pointclouds_normals_gt": pns,
            "overlap_threshold": thr,
        }

    def _load_from_folder(self, frag: str, index: int) -> dict:
        folder = os.path.join(self.data_path, frag)
        ply_files = sorted(glob.glob(os.path.join(folder, "*.ply")))
        meshes = [_load_mesh_from_ply(p) for p in ply_files]
        pcs, pns, overlap_thr = self._sample_points(meshes)
        return {
            "index": index,
            "name": frag,
            "pointclouds_gt": pcs,
            "pointclouds_normals_gt": pns,
            "overlap_threshold": overlap_thr,
            "num_parts": len(meshes),
        }

    def _sample_points(self, meshes: list[trimesh.Trimesh]) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        """Sample points (and normals) from meshes."""

        # Handle non-mesh dataset (e.g., ModelNet)
        if not hasattr(meshes[0], "faces") or any(m.faces.shape[0] == 0 for m in meshes):
            pcs, pns = [], []
            n_parts = len(meshes)
            for m in meshes:
                cnt = self.num_points_to_sample // n_parts
                idx = np.random.choice(len(m.vertices), cnt, replace=True)
                pcs.append(m.vertices[idx])
                pns.append(
                    m.vertex_normals[idx] if m.vertex_normals is not None else np.zeros((cnt, 3))
                )
            overlap_thr = 0.05  # TODO: move to config
            return pcs, pns, overlap_thr

        # Allocate sampling counts by area
        areas = np.array([m.area for m in meshes])
        total_area = areas.sum()
        base = self.min_points_per_part

        remaining_points = self.num_points_to_sample - base * len(meshes)
        counts = (base + (remaining_points * (areas / total_area)).astype(int)).tolist()
        diff = self.num_points_to_sample - sum(counts)
        counts[np.argmax(counts)] += diff

        def _proc_mesh(args):
            mesh, cnt = args
            pts, fidx = sample_points_poisson(mesh, cnt)
            if len(pts) < cnt:
                extra, eidx = trimesh.sample.sample_surface(mesh, cnt - len(pts))
                pts = np.vstack((pts, extra))
                fidx = np.concatenate((fidx, eidx))
            return pts[:cnt], mesh.face_normals[fidx[:cnt]]

        sampled = list(self.pool.map(_proc_mesh, zip(meshes, counts)))
        pcs, pns = zip(*sampled)
        overlap_thr = np.sqrt(2 * total_area / self.num_points_to_sample + 1e-4)
        return list(pcs), list(pns), overlap_thr
    
    def _make_y_up(self, pts_gt: np.ndarray, pns_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A simple transform to change the up axis of the point cloud. 

        Args:
            pts_gt: Point cloud coordinates of shape (N, 3).
            pns_gt: Point cloud normals of shape (N, 3).

        Returns:
            pts_gt: Point cloud coordinates of shape (N, 3).
            pns_gt: Point cloud normals of shape (N, 3).
        """
        if self.up_axis == "y":
            return pts_gt, pns_gt
        elif self.up_axis == "z":
            return pts_gt[:, [0, 2, 1]], pns_gt[:, [0, 2, 1]]
        elif self.up_axis == "x":
            return pts_gt[:, [1, 0, 2]], pns_gt[:, [1, 0, 2]]
        else:
            raise ValueError(f"Invalid up axis: {self.up_axis}")

    def _transform(self, data: dict) -> dict:
        """Apply scaling, rotation, centering, shuffling, and padding."""
        pcs_gt = data["pointclouds_gt"]
        pns_gt = data["pointclouds_normals_gt"]
        n_parts = data["num_parts"]

        counts = np.array([len(pc) for pc in pcs_gt])
        offsets = np.concatenate([[0], np.cumsum(counts)])
        pts_gt = np.concatenate(pcs_gt)
        normals_gt = np.concatenate(pns_gt)

        # Center point clouds
        pts_gt, _ = center_pcd(pts_gt)

        # Make point clouds y-up
        pts_gt, normals_gt = self._make_y_up(pts_gt, normals_gt)

        # Scale point clouds to [-1, 1] and apply random scaling
        scale = np.max(np.abs(pts_gt))
        if self.random_scale_range is not None:
            scale *= np.random.uniform(*self.random_scale_range)
        pts_gt /= scale

        # Initial rotation (quat in wxyz format)
        if self.split == "train":
            pts_gt, normals_gt, init_quat = rotate_pcd(pts_gt, normals_gt)
        else:
            init_quat = np.array([1, 0, 0, 0])

        pts, normals = pts_gt.copy(), normals_gt.copy()

        def _proc_part(i):
            """Process one part: center, rotate, and shuffle."""
            st, ed = offsets[i], offsets[i+1]
            # center the point cloud
            part, trans = center_pcd(pts_gt[st:ed])
            # random rotate the point cloud
            part, norms, quant = rotate_pcd(part, normals_gt[st:ed])
            # random shuffle point order
            _order = np.random.permutation(len(part))
            pts[st: ed] = part[_order]
            normals[st: ed] = norms[_order]
            pts_gt[st:ed] = pts_gt[st:ed][_order]
            normals_gt[st:ed] = normals_gt[st:ed][_order]
            return quant, trans

        results = list(self.pool.map(_proc_part, range(n_parts)))
        quats, trans = zip(*results)

        # Padding to max_parts
        pts_per_part = pad_data(counts, self.max_parts)
        quats = pad_data(np.stack(quats), self.max_parts)
        trans = pad_data(np.stack(trans), self.max_parts)
        scale = pad_data(np.array([scale] * n_parts), self.max_parts)

        # Anchor selection (primary part and extra parts)
        anchor = np.zeros(self.max_parts, bool)
        primary = np.argmax(counts)
        anchor[primary] = True

        # Select extra parts if multi_anchor is enabled
        if self.multi_anchor and n_parts > 2 and np.random.rand() > 1 / n_parts:
            candidates = counts[:n_parts] > self.num_points_to_sample * 0.05
            candidates[primary] = False
            if candidates.any():
                extra_n = np.random.randint(
                    1, min(candidates.sum() + 1, n_parts - 1)
                )
                extra_idx = np.random.choice(
                    np.where(candidates)[0], extra_n, replace=False
                )
                anchor[extra_idx] = True

        # Return the transformed data
        results = {}
        for key in ["index", "name", "overlap_threshold"]:
            results[key] = data[key]

        results["dataset_name"] = self.dataset_name
        results["num_parts"] = n_parts
        results["pointclouds"] = pts.astype(np.float32)
        results["pointclouds_gt"] = pts_gt.astype(np.float32)
        results["pointclouds_normals"] = normals.astype(np.float32)
        results["pointclouds_normals_gt"] = normals_gt.astype(np.float32)
        results["quaternions"] = quats.astype(np.float32)
        results["translations"] = trans.astype(np.float32)
        results["points_per_part"] = pts_per_part.astype(np.int64)
        results["scale"] = scale.astype(np.float32)
        results["anchor_part"] = anchor.astype(bool)
        results["init_rotation"] = init_quat.astype(np.float32)

        return results

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()
        self.pool.shutdown()


if __name__ == "__main__":
    ds = PointCloudDataset(
        split="train",
        data_path="../dataset/ikea.hdf5",
        dataset_name="ikea",
    )
    sample = ds[0]
    for key, val in sample.items():
        if isinstance(val, np.ndarray):
            print(f"{key:<20} {val.shape}, {val.dtype}")
        else:
            print(f"{key:<20} {val}")
