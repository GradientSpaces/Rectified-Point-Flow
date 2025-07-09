"""Training script for Rectified Point Flow."""

import logging
import os
import warnings

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from rectified_point_flow.utils.training import (
    setup_loggers,
    setup_wandb_resume,
    log_config_to_wandb,
    log_code_to_wandb,
)

logger = logging.getLogger("Train")
warnings.filterwarnings("ignore", module="lightning")  # ignore warning from lightning' connectors

# Optimize for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def setup_training(cfg: DictConfig):
    """Setup training components."""
    os.makedirs(cfg.log_dir, exist_ok=True)
    loggers = setup_loggers(cfg)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)
    return model, datamodule, trainer, loggers 

@hydra.main(version_base="1.3", config_path="./config", config_name="RPF_base_main")
def main(cfg: DictConfig):
    """Entry point for training the model."""

    ckpt_path = cfg.get("ckpt_path")
    is_fresh_run = not (ckpt_path and os.path.exists(ckpt_path))

    if is_fresh_run:
        seed = cfg.get("seed", 0)
        L.seed_everything(seed, workers=True, verbose=False)
        logger.info(f"Fresh run with random seed {seed}")
    else:
        logger.info("Resume training from checkpoint, no random seed set.")
        setup_wandb_resume(cfg)

    model, datamodule, trainer, loggers = setup_training(cfg)
    log_config_to_wandb(loggers, cfg)
    log_code_to_wandb(loggers)
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path if not is_fresh_run else None
    )


if __name__ == "__main__":
    main()
