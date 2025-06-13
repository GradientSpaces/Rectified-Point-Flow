"""Training utilities for Rectified Point Flow."""

import logging
import os
import glob
import hydra
from typing import List

import lightning as L
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig

logger = logging.getLogger("Train")


def find_wandb_run_id(ckpt_path: str) -> str | None:
    """Find the latest wandb run ID from the checkpoint path."""
    ckpt_dir = os.path.dirname(ckpt_path)
    wandb_dir = os.path.join(ckpt_dir, "wandb")
    if os.path.exists(os.path.join(wandb_dir, "latest-run")):
        run_log_path = glob.glob(os.path.join(wandb_dir, "latest-run", "run-*.wandb"))[0]
        run_id = os.path.basename(run_log_path).split(".")[0].split("-")[-1]
        if len(run_id) == 8:
            return run_id
    return None


def setup_wandb_resume(cfg: DictConfig) -> None:
    """Setup wandb resume configuration if checkpoint exists."""
    if "wandb" in cfg.get("loggers", dict()).keys():
        run_id = find_wandb_run_id(cfg.get("ckpt_path"))
        if run_id:
            cfg.loggers.wandb.id = run_id
            cfg.loggers.wandb.resume = "allow"
            print(f"Found the latest wandb run ID: {run_id}. Continue logging to this run.")
        else:
            print("No previous wandb run ID found. Logging to a new run.")


def setup_loggers(cfg: DictConfig) -> List[Logger]:
    """Initialize and setup loggers."""
    loggers: List[Logger] = [
        hydra.utils.instantiate(logger)
        for logger in cfg.get("loggers", dict()).values()
    ]
    return loggers


def log_code_to_wandb(loggers: List[Logger], rank: int) -> None:
    """Log code to wandb if available."""
    if rank == 0:
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                original_cwd = hydra.utils.get_original_cwd()
                logger.experiment.log_code(
                    root=original_cwd,
                    include_fn=lambda path: path.endswith(".py")
                )
                print(f"Code logged to wandb from {original_cwd}")
