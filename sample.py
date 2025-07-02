"""Sampling from a trained Rectified Point Flow."""

import logging
from pathlib import Path
import os
import warnings

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from rectified_point_flow.utils import load_checkpoint_for_module
from rectified_point_flow.visualizer import VisualizationCallback

logger = logging.getLogger("Sample")
warnings.filterwarnings("ignore", module="lightning")  # ignore warning from lightning' connectors

# Optimize for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def setup(cfg: DictConfig):
    """Setup evaluation components."""

    # Instantiate model and load checkpoint
    ckpt_path = cfg.ckpt_path
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(model, ckpt_path)
    model.eval()

    # Instantiate data module
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Setup visualization callback
    vis_config = cfg.get("visualizer", {})
    callbacks = []
    if vis_config:
        vis_callback: VisualizationCallback = hydra.utils.instantiate(vis_config)
        callbacks.append(vis_callback)
    
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=False,
    )
    return model, datamodule, trainer


@hydra.main(version_base="1.3", config_path="./config", config_name="RPF_base_demo")
def main(cfg: DictConfig):
    """Entry point for evaluating the model on validation set."""
    
    # Check if checkpoint exists
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}")
        logger.error("Please provide a valid checkpoint path in the config or via ckpt_path='...' argument")
        return
        
    # Setup components
    model, datamodule, trainer = setup(cfg)

    # Run sampling
    trainer.test(
        model=model,
        datamodule=datamodule, 
        verbose=True
    )
    logger.info("Done!")
    
    # Print location
    log_dir = cfg.get('log_dir')
    vis_dir = Path(log_dir) / "visualizations"
    logger.info(f"Visualization saved to: {vis_dir}")


if __name__ == "__main__":
    main()