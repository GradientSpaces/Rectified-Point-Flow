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
        min_parts: int = 2,
        max_parts: int = 64,
        num_points_to_sample: int = 5000,
        min_points_per_part: int = 20,
        min_dataset_size: int = 2000,
        limit_val_samples: int = 0,
        random_scale_range: tuple[float, float] = (0.75, 1.25),
        batch_size: int = 40,
        num_workers: int = 16,
        multi_anchor: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_names = dataset_names
        self.min_parts = min_parts
        self.max_parts = max_parts
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
        self._dataset_paths = {}
        self._dataset_names = []
        self._initialize_dataset_paths()
    
    def _initialize_dataset_paths(self):
        use_all_datasets = len(self.dataset_names) == 0
        for file in os.listdir(self.data_root):
            if file.endswith(".hdf5"):
                dataset_name = file.split(".")[0]
                if use_all_datasets or dataset_name in self.dataset_names:
                    self._dataset_names.append(dataset_name)
                    self._dataset_paths[dataset_name] = os.path.join(self.data_root, file)
            elif os.path.isdir(os.path.join(self.data_root, file)):
                dataset_name = file
                if use_all_datasets or dataset_name in self.dataset_names:
                    self._dataset_names.append(dataset_name)
                    self._dataset_paths[dataset_name] = os.path.join(self.data_root, file)
            else:
                logger.warning(f"Unknown file type: {file} in {self.data_root}. Skipping...")
        
        logger.info(f"Using {len(self._dataset_paths)} datasets: {list(self._dataset_paths.keys())}")

    def setup(self, stage: str):
        """Set up datasets for training/validation/testing."""
        logger.info(f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--")
        logger.info(f"| {'Dataset':<16} | {'Split':<8} | {'Length':<8} | {'Parts':<8} |")
        logger.info(f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--")

        if stage == "fit":
            self.train_dataset = ConcatDataset(
                [
                    PointCloudDataset(
                        split="train",
                        data_path=self._dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        min_dataset_size=self.min_dataset_size,
                        random_scale_range=self.random_scale_range,
                        multi_anchor=self.multi_anchor,
                    )
                    for dataset_name in self._dataset_names
                ]
            )
            
            logger.info(f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--")

            self.val_dataset = ConcatDataset(
                [
                    PointCloudDataset(
                        split="val",
                        data_path=self._dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        limit_val_samples=self.limit_val_samples,
                    )
                    for dataset_name in self._dataset_names
                ]
            )
            logger.info(f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--")
            logger.info("Total Train Samples: " + str(self.train_dataset.cumulative_sizes[-1]))
            logger.info("Total Val Samples: " + str(self.val_dataset.cumulative_sizes[-1]))

        elif stage in ["test", "predict", "validate"]:
            self.val_dataset = ConcatDataset(
                [
                    PointCloudDataset(
                        split="val",
                        data_path=self._dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        limit_val_samples=self.limit_val_samples,
                    )
                    for dataset_name in self._dataset_names
                ]
            )
            logger.info(f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--")

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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=False,
        )