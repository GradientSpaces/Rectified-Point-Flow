import logging
from typing import List, Optional
import os

import h5py
import lightning as L
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .dataset import PointCloudDataset

logger = logging.getLogger("Data")


def worker_init_fn(worker_id):
    """Worker function for initializing the h5 file."""
    worker_info = torch.utils.data.get_worker_info()
    concat_dataset: ConcatDataset = worker_info.dataset
    for dataset in concat_dataset.datasets:
        if dataset._h5_file is None and not dataset.use_folder:
            dataset._h5_file = h5py.File(
                dataset.data_path, "r", libver='latest', swmr=True
            )


class PointCloudDataModule(L.LightningDataModule):
    """Lightning data module for point cloud data."""
    
    def __init__(
        self,
        data_root: str = "",
        dataset_names: List[str] = [],
        up_axis: dict[str, str] = {},
        min_parts: int = 2,
        max_parts: int = 64,
        anchor_free: bool = True,
        num_points_to_sample: int = 5000,
        min_points_per_part: int = 20,
        min_dataset_size: int = 2000,
        limit_val_samples: int = 0,
        random_scale_range: tuple[float, float] = (0.75, 1.25),
        batch_size: int = 40,
        num_workers: int = 16,
        multi_anchor: bool = False,
    ):
        """Data module for point cloud data.

        Args:
            data_root: Root directory of the dataset.
            dataset_names: List of dataset names to use.
            up_axis: Dictionary of dataset names to up axis, e.g. {"ikea": "y", "everyday": "z"}.
                     If not provided, the up axis is assumed to be 'y'. This only affects the visualization.
            min_parts: Minimum number of parts in a point cloud.
            max_parts: Maximum number of parts in a point cloud.
            anchor_free: Whether to use anchor-free mode.
                     If True, the anchor part is centered and randomly rotated, like the non-anchor parts (default).
                     If False, the anchor part is not centered and thus its pose in the CoM frame of the GT point cloud is given (align with GARF).
            num_points_to_sample: Number of points to sample from each point cloud.
            min_points_per_part: Minimum number of points per part.
            min_dataset_size: Minimum number of point clouds in a dataset.
            limit_val_samples: Number of point clouds to sample from the validation set.
            random_scale_range: Range of random scale to apply to the point cloud.
            batch_size: Batch size.
            num_workers: Number of workers to use for loading the data.
            multi_anchor: Whether to use multiple anchors for the point cloud.
        """
        super().__init__()
        self.data_root = data_root
        self.up_axis = up_axis
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.anchor_free = anchor_free
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_val_samples = limit_val_samples
        self.min_dataset_size = min_dataset_size
        self.random_scale_range = random_scale_range
        self.multi_anchor = multi_anchor

        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None

        # Initialize dataset paths
        self.dataset_paths = {}
        self.dataset_names = []
        self._initialize_dataset_paths(dataset_names)
    
    def _initialize_dataset_paths(self, dataset_names: List[str]):
        use_all_datasets = len(dataset_names) == 0

        for file in os.listdir(self.data_root):
            if file.endswith(".hdf5"):
                dataset_name = file.split(".")[0]
                if use_all_datasets or dataset_name in dataset_names:
                    self.dataset_names.append(dataset_name)
                    self.dataset_paths[dataset_name] = os.path.join(self.data_root, file)
            elif os.path.isdir(os.path.join(self.data_root, file)):
                dataset_name = file
                if use_all_datasets or dataset_name in dataset_names:
                    self.dataset_names.append(dataset_name)
                    self.dataset_paths[dataset_name] = os.path.join(self.data_root, file)
            else:
                logger.warning(f"Unknown file type: {file} in {self.data_root}. Skipping...")
        logger.info(f"Using {len(self.dataset_paths)} datasets: {list(self.dataset_paths.keys())}")

    def setup(self, stage: str):
        """Set up datasets for training/validation/testing."""
        make_line = lambda: f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--"
        logger.info(make_line())
        logger.info(f"| {'Dataset':<16} | {'Split':<8} | {'Length':<8} | {'Parts':<8} |")
        logger.info(make_line())

        if stage == "fit":
            self.train_dataset = ConcatDataset(
                [
                    PointCloudDataset(
                        split="train",
                        data_path=self.dataset_paths[dataset_name],
                        up_axis=self.up_axis.get(dataset_name, "y"),
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        min_dataset_size=self.min_dataset_size,
                        anchor_free=self.anchor_free,
                        random_scale_range=self.random_scale_range,
                        multi_anchor=self.multi_anchor,
                    )
                    for dataset_name in self.dataset_names
                ]
            )
            logger.info(make_line())
            self.val_dataset = ConcatDataset(
                [
                    PointCloudDataset(
                        split="val",
                        data_path=self.dataset_paths[dataset_name],
                        up_axis=self.up_axis.get(dataset_name, "y"),
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        anchor_free=self.anchor_free,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        limit_val_samples=self.limit_val_samples,
                    )
                    for dataset_name in self.dataset_names
                ]
            )
            logger.info(make_line())
            logger.info("Total Train Samples: " + str(self.train_dataset.cumulative_sizes[-1]))
            logger.info("Total Val Samples: " + str(self.val_dataset.cumulative_sizes[-1]))
            logger.info("Anchor-free Mode: " + str(self.anchor_free))

        elif stage == "validate":
            self.val_dataset = ConcatDataset(
                [
                    PointCloudDataset(
                        split="val",
                        data_path=self.dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        up_axis=self.up_axis.get(dataset_name, "y"),
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        anchor_free=self.anchor_free,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        limit_val_samples=self.limit_val_samples,
                    )
                    for dataset_name in self.dataset_names
                ]
            )
            logger.info(make_line())
            logger.info("Total Val Samples: " + str(self.val_dataset.cumulative_sizes[-1]))
            logger.info("Anchor-free Mode: " + str(self.anchor_free))

        elif stage in ["test", "predict"]:
            self.test_dataset = [
                PointCloudDataset(
                    split="val",
                    data_path=self.dataset_paths[dataset_name],
                    dataset_name=dataset_name,
                    up_axis=self.up_axis.get(dataset_name, "y"),
                    min_parts=self.min_parts,
                    max_parts=self.max_parts,
                    anchor_free=self.anchor_free,
                    num_points_to_sample=self.num_points_to_sample,
                    min_points_per_part=self.min_points_per_part,
                    limit_val_samples=self.limit_val_samples,
                )
                for dataset_name in self.dataset_names
            ]
            logger.info(make_line())
            logger.info("Total Test Samples: " + str(sum(len(dataset) for dataset in self.test_dataset)))
            logger.info("Anchor-free Mode: " + str(self.anchor_free))

    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=True,
            persistent_workers=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            persistent_workers=False,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=False,
            )
            for dataset in self.test_dataset
        ]