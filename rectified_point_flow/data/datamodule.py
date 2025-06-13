from typing import List, Optional, Dict

import h5py
import torch
import lightning as L
import logging
from torch.utils.data import DataLoader, ConcatDataset

from .dataset import PointCloudDataset


logger = logging.getLogger("Data")


def worker_init_fn(worker_id):
    """Initialize worker for multi-processing."""
    worker_info = torch.utils.data.get_worker_info()
    concat_dataset: ConcatDataset = worker_info.dataset
    for dataset in concat_dataset.datasets:
        if dataset._h5_file is None:
            dataset._h5_file = h5py.File(dataset.data_path, "r")


class PointCloudDataModule(L.LightningDataModule):
    """Lightning data module for point cloud data."""
    
    def __init__(
        self,
        dataset_paths: Dict[str, str] = {},
        dataset_names: List[str] = [],
        min_parts: int = 2,
        max_parts: int = 64,
        num_points_to_sample: int = 5000,
        min_points_per_part: int = 20,
        batch_size: int = 40,
        num_workers: int = 16,
        limit_samples: int = 0,
        multi_ref: bool = False,
    ):
        super().__init__()
        self.dataset_paths = dataset_paths
        self.dataset_names = dataset_names
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_samples = limit_samples
        self.multi_ref = multi_ref

        self.train_dataset: Optional[ConcatDataset] = None
        self.val_dataset: Optional[ConcatDataset] = None

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
                        data_path=self.dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        multi_ref=self.multi_ref,
                    )
                    for dataset_name in self.dataset_names
                ]
            )
            
            logger.info(f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--")

            self.val_dataset = ConcatDataset(
                [
                    PointCloudDataset(
                        split="val",
                        data_path=self.dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        limit_samples=self.limit_samples,
                    )
                    for dataset_name in self.dataset_names
                ]
            )
            logger.info(f"--{'-' * 16}---{'-' * 8}---{'-' * 8}---{'-' * 8}--")
            logger.info("Total Train Samples: " + str(self.train_dataset.cumulative_sizes[-1]))
            logger.info("Total Val Samples: " + str(self.val_dataset.cumulative_sizes[-1]))

        if stage in ["test", "predict", "validate"]:
            self.val_dataset = ConcatDataset(
                [
                    PointCloudDataset(
                        split="val",
                        data_path=self.dataset_paths[dataset_name],
                        dataset_name=dataset_name,
                        min_parts=self.min_parts,
                        max_parts=self.max_parts,
                        num_points_to_sample=self.num_points_to_sample,
                        min_points_per_part=self.min_points_per_part,
                        limit_samples=self.limit_samples,
                    )
                    for dataset_name in self.categories
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