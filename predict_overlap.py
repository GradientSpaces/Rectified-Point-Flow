"""Predict overlap probabilities and visualize them on point clouds."""

import logging
from pathlib import Path
import os
import warnings

import hydra
import lightning as L
from omegaconf import DictConfig

from rectified_point_flow.utils import load_checkpoint_for_module
from rectified_point_flow.visualizer import OverlapVisualizationCallback

logger = logging.getLogger("PredictOverlap")
warnings.filterwarnings("ignore", module="lightning")
warnings.filterwarnings("ignore", category=FutureWarning)


def setup(cfg: DictConfig):
    """Setup inference components."""
    
    # load model and checkpoint
    ckpt_path = cfg.ckpt_path
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(model, ckpt_path)
    model.eval()
    
    # data module
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # visualization callback
    vis_config = cfg.get("visualizer", {})
    callbacks = []
    if vis_config:
        vis_callback: OverlapVisualizationCallback = hydra.utils.instantiate(vis_config)
        callbacks.append(vis_callback)
    
    # trainer
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=False,
    )
    
    return model, datamodule, trainer


@hydra.main(version_base="1.3", config_path="./config", config_name="RPF_base_predict_overlap")
def main(cfg: DictConfig):
    """Main function for overlap prediction and visualization."""
    
    # check if checkpoint exists
    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path or not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found at {ckpt_path}")
        logger.error("Please provide a valid checkpoint path in the config or via ckpt_path='...' argument")
        return
    
    # setup components
    model, datamodule, trainer = setup(cfg)
    
    # run inference
    trainer.test(model=model, datamodule=datamodule)
    
    # logging
    logger.info("Done!")
    log_dir = cfg.get('log_dir')
    vis_dir = Path(log_dir) / "visualizations"
    logger.info(f"Overlap visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
