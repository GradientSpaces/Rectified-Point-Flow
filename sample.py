"""Sampling from a trained Rectified Point Flow."""

import logging
from pathlib import Path
import os
import warnings

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from rectified_point_flow.utils import load_checkpoint_for_module, download_rfp_checkpoint, print_eval_table
from rectified_point_flow.visualizer import VisualizationCallback

logger = logging.getLogger("Sample")
warnings.filterwarnings("ignore", module="lightning")  # ignore warning from lightning' connectors
warnings.filterwarnings("ignore", category=FutureWarning)

# Optimize for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEFAULT_CKPT_PATH_HF = "RPF_base_full_anchorfree_ep2000.ckpt"


def setup(cfg: DictConfig):
    """Setup evaluation components."""

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is None:
        ckpt_path = download_rfp_checkpoint(DEFAULT_CKPT_PATH_HF, './weights')
    elif not os.path.exists(ckpt_path):
        logger.error(f"Checkpoint not found: {ckpt_path}")
        logger.error("Please provide a valid checkpoint in the config or via ckpt_path='...' argument")
        exit(1)

    seed = cfg.get("seed", None)
    if seed is not None:
        L.seed_everything(seed, workers=True, verbose=False)
        logger.info(f"Seed set to {seed} for sampling")

    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    load_checkpoint_for_module(model, ckpt_path)
    model.eval()

    vis_config = cfg.get("visualizer", {})
    callbacks = []
    if vis_config and cfg["visualizer"]["renderer"] != "none":
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

    model, datamodule, trainer = setup(cfg)
    eval_results = trainer.test(
        model=model,
        datamodule=datamodule, 
        verbose=False,
    )
    print_eval_table(eval_results, datamodule.dataset_names)
    logger.info("Visualizations saved to:" + str(Path(cfg.get('log_dir')) / "visualizations"))
    logger.info("Evaluation results saved to:" + str(Path(cfg.get('log_dir')) / "results"))


if __name__ == "__main__":
    main()