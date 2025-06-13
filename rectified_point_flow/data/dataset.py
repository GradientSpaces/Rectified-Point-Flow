from typing import List, Dict

import h5py
import numpy as np
import trimesh
import logging
from torch.utils.data import Dataset

from .utils import sample_points_poisson, pad_data
from .transform import center_pc, rotate_pc, shuffle_pc


logger = logging.getLogger("Data")

class PointCloudDataset(Dataset):
    """Dataset for point cloud data with weighted sampling."""
    
    def __init__(
        self,
        split: str = "train",
        data_path: str = "../dataset",
        dataset_name: str | list[str] = "everyday",
        min_parts: int = 2,
        max_parts: int = 64,
        num_points_to_sample: int = 5000,
        min_points_per_part: int = 20,
        multi_ref: bool = False,
        limit_samples: int = 0,
    ):
        super().__init__()
        self.split = split
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.multi_ref = multi_ref
        self.limit_samples = limit_samples
        
        self._h5_file = None
        self.min_part_count = float("inf")
        self.max_part_count = 0
        self.data_list = self.get_data_list()

        logger.info(f"| {self.dataset_name:16s} | {self.split:8s} | {len(self.data_list):8d} | [{self.min_part_count:2d}, {self.max_part_count:2d}] |")
        trimesh.util.log.setLevel(logging.ERROR)

    def _get_h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.data_path, "r")
        return self._h5_file

    def get_data_list(self) -> List[str]:
        """Return the list of data samples."""
        h5_file = self._get_h5_file()
        if isinstance(self.dataset_name, str):
            data_list = list(h5_file["data_split"][self.dataset_name][self.split])
        else:
            data_list = []
            for cat in self.dataset_name:
                data_list += list(h5_file["data_split"][cat][self.split])
        data_list = [d.decode("utf-8") for d in data_list]
        
        filtered_data_list = []
        for item in data_list:
            try:
                num_parts = len(h5_file[item].keys())
                if self.min_parts <= num_parts <= self.max_parts:
                    self.min_part_count = min(self.min_part_count, num_parts)
                    self.max_part_count = max(self.max_part_count, num_parts)
                    filtered_data_list.append(item)
            except:
                logger.warning(f"Error getting data list for {item}")
                continue

        # limit the number of samples
        if self.limit_samples > 0 and len(filtered_data_list) > self.limit_samples:
            every_n = len(filtered_data_list) // self.limit_samples
            filtered_data_list = filtered_data_list[::every_n]
        return filtered_data_list

    def get_data(self, index: int) -> Dict:
        """Get data for a given index."""
        name = self.data_list[index]
        h5_file = self._get_h5_file()
        group = h5_file[name]
        pieces = group.keys()
        pieces_names = list(pieces)
        num_parts = len(pieces)
        has_faces = "faces" in group[pieces_names[0]]
        has_normals = "normals" in group[pieces_names[0]]
        
        meshes = [
            trimesh.Trimesh(
                vertices=np.array(group[piece]["vertices"][:]),
                faces=np.array(group[piece]["faces"][:]) if has_faces else np.array([]),
                vertex_normals=np.array(group[piece]["normals"][:]) if has_normals else None,
                process=False,
            )
            for piece in pieces
        ]

        # Sample points
        pointclouds_gt, pointclouds_normals_gt = self.sample_points(meshes)

        data = {
            "index": index,
            "name": name,
            "dataset_name": self.dataset_name,
            "num_parts": num_parts,
            "pointclouds_gt": pointclouds_gt,
            "pointclouds_normals_gt": pointclouds_normals_gt,
        }
        return data

    def sample_points(self, meshes: List[trimesh.Trimesh]) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Sample points from meshes based on area."""
        
        # Handle point only dataset
        if any(mesh.faces.shape[0] == 0 for mesh in meshes):
            num_parts = len(meshes)
            pointclouds_gt = []
            pointclouds_normals_gt = []
            for mesh in meshes:
                sample_idx = np.random.choice(
                    mesh.vertices.shape[0], 
                    self.num_points_to_sample // num_parts, 
                    replace=True
                )
                pts = mesh.vertices[sample_idx]
                pointclouds_gt.append(pts)
                pointclouds_normals_gt.append(mesh.vertex_normals[sample_idx])
            return pointclouds_gt, pointclouds_normals_gt


        # Sample points based on area per part
        areas = [mesh.area for mesh in meshes]
        total_area = sum(areas)
        num_parts = len(meshes)
        base = self.min_points_per_part
        remaining = self.num_points_to_sample - base * num_parts
        points_per_part = [
            base + int(remaining * (area / total_area))
            for area in areas
        ]
        diff = self.num_points_to_sample - sum(points_per_part)
        if diff != 0:
            points_per_part[np.argmax(points_per_part)] += diff

        # Sample points
        pointclouds_gt = []
        pointclouds_normals_gt = []
        for mesh, count in zip(meshes, points_per_part):
            pts, idx = sample_points_poisson(mesh, count)
            if len(pts) > count:
                pts = pts[:count]
                idx = idx[:count]
            elif len(pts) < count:
                extra, extra_idx = trimesh.sample.sample_surface(mesh, count-len(pts))
                pts = np.vstack((pts, extra))
                idx = np.concatenate((idx, extra_idx))
            pointclouds_gt.append(pts)
            pointclouds_normals_gt.append(mesh.face_normals[idx])

        return pointclouds_gt, pointclouds_normals_gt

    def transform(self, data: dict) -> Dict:
        """Transform data for training."""
        num_parts = data["num_parts"]
        pointclouds_gt = data["pointclouds_gt"]
        pointclouds_normals_gt = data["pointclouds_normals_gt"]

        points_per_part = np.array([len(pc) for pc in pointclouds_gt])
        offset = np.concatenate([[0], np.cumsum(points_per_part)])
        pointclouds_gt = np.concatenate(pointclouds_gt)
        pointclouds_normals_gt = np.concatenate(pointclouds_normals_gt)

        # Initial rotation
        pointclouds_gt, pointclouds_normals_gt, init_rot = rotate_pc(
            pointclouds_gt,
            pointclouds_normals_gt,
        )

        # Scale the point cloud
        scale = np.max(np.abs(pointclouds_gt))
        pointclouds_gt = pointclouds_gt / scale

        pointclouds, pointclouds_normals, quaternions, translations = [], [], [], []
        for part_idx in range(num_parts):
            start = offset[part_idx]
            end = offset[part_idx + 1]

            # Center the point cloud
            pointcloud, translation = center_pc(pointclouds_gt[start:end])

            # Random rotate the point cloud
            pointcloud, pointcloud_normals, quaternion = rotate_pc(
                pointcloud, pointclouds_normals_gt[start:end]
            )
            
            # Shuffle point order
            random_order = np.random.permutation(len(pointcloud))
            pointcloud, pointcloud_normals = shuffle_pc(
                pointcloud, pointcloud_normals, random_order
            )
            pointclouds_gt[start:end] = pointclouds_gt[start:end][random_order]
            pointclouds_normals_gt[start:end] = pointclouds_normals_gt[start:end][random_order]

            pointclouds.append(pointcloud)
            pointclouds_normals.append(pointcloud_normals)
            quaternions.append(quaternion)
            translations.append(translation)

        # Concatenate
        pointclouds = np.concatenate(pointclouds).astype(np.float32)
        pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)
        quaternions = np.stack(quaternions).astype(np.float32)
        translations = np.stack(translations).astype(np.float32)

        # Pad data
        points_per_part = pad_data(points_per_part, self.max_parts).astype(np.int64)
        quaternions = pad_data(quaternions, self.max_parts)
        translations = pad_data(translations, self.max_parts)
        scale = pad_data(np.array([scale] * num_parts), self.max_parts)

        # Reference part selection
        ref_part = np.zeros((self.max_parts), dtype=np.float32)
        ref_part_idx = np.argmax(points_per_part[:num_parts])
        ref_part[ref_part_idx] = 1
        ref_part = ref_part.astype(bool)

        if self.multi_ref and num_parts > 2 and np.random.rand() > 1 / num_parts:
            can_be_ref = points_per_part[:num_parts] > self.num_points_to_sample * 0.05
            can_be_ref[ref_part_idx] = False
            can_be_ref_num = np.sum(can_be_ref)
            if can_be_ref_num > 0:
                num_more_ref = np.random.randint(1, min(can_be_ref_num + 1, num_parts - 1))
                more_ref_part_idx = np.random.choice(np.where(can_be_ref)[0], num_more_ref, replace=False)
                ref_part[more_ref_part_idx] = True

        return {
            "index": data["index"],
            "name": data["name"],
            "dataset_name": data["dataset_name"],
            "num_parts": num_parts,
            "pointclouds": pointclouds.astype(np.float32),
            "pointclouds_gt": pointclouds_gt.astype(np.float32),
            "pointclouds_normals": pointclouds_normals.astype(np.float32),
            "pointclouds_normals_gt": pointclouds_normals_gt.astype(np.float32),
            "quaternions": quaternions,
            "translations": translations,
            "points_per_part": points_per_part,
            "scale": scale,
            "ref_part": ref_part,
            "init_rotation": init_rot,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        return self.transform(data)

    def __len__(self):
        return len(self.data_list)
    

if __name__ == "__main__":
    # Test the dataset
    dataset = PointCloudDataset(
        split="train",
        data_path="../dataset/ikea.hdf5",
        dataset_name="ikea",
    )

    sample = dataset[0]
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"{key:<30} {value.shape}, {value.dtype}")
        else:
            print(f"{key:<30} {value}")