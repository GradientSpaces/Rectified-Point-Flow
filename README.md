# Rectified Point Flow: Generic Point Cloud Pose Estimation

[![ProjectPage](https://img.shields.io/badge/Project_Page-RPF-blue)](https://rectified-pointflow.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2506.05282-blue?logo=arxiv&color=%23B31B1B)](https://arxiv.org/abs/2506.05282) [![Hugging Face (LCM) Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/gradient-spaces/Rectified-Point-Flow) [![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

[Tao Sun](https://taosun.io/) *<sup>,1</sup>,
[Liyuan Zhu](https://www.zhuliyuan.net/) *<sup>,1</sup>,
[Shengyu Huang](https://shengyuh.github.io/)<sup>2</sup>,
[Shuran Song](https://shurans.github.io/)<sup>1</sup>,
[Iro Armeni](https://ir0.github.io/)<sup>1</sup>

<sup>1</sup>Stanford University, <sup>2</sup>NVIDIA Research | * denotes equal contribution


**_TL;DR:_** *Assemble unposed parts into complete objects by learning a point-wise flow model.* 


## Overview

We introduce **Rectified Point Flow** (RPF), a unified parameterization that formulates pairwise point cloud registration and multi-part shape assembly as a single conditional generative problem. Given unposed point clouds, our method learns a continuous point-wise velocity field that transports noisy points toward their target positions, from which part poses are recovered. In contrast to prior work that regresses part-wise poses with ad-hoc symmetry handling, our method intrinsically learns assembly symmetries without symmetry labels.

<p align="center">
  <a href="">
    <img src="https://rectified-pointflow.github.io/images/overview_flow_asm.png" width="100%">
  </a>
</p>

## 🛠️ Setup

First, please clone the repo:

```bash
git clone https://github.com/GradientSpaces/Rectified-Point-Flow.git
cd Rectified-Point-Flow
```

We use a Python 3.10 environment for compatibility with the dependencies:

```bash
conda create -n py310-rpf python=3.10 -y
conda activate py310-rpf
```

Then, use [`poetry`](https://python-poetry.org/) or [`uv`](https://docs.astral.sh/uv/) to install the dependencies:

```bash
poetry install  # or `uv sync`
```

Alternatively, we provide an [`install.sh`](install.sh) script to bootstrap the environment via pip only:

```bash
bash install.sh
```

This evironment includes `PyTorch 2.5.1`, `PyTorch3D 0.7.8`, and `flash-attn 2.7.4`. We've tested it on NVIDIA RTX4090/A100/H100 GPUs with CUDA 12.4.

## ✨  Demo

![RPF Demo](assets/merged_trajectory_grid.gif)

**Assembly Generation:** To sample the trained RPF model on demo data, please run:

```bash
python sample.py data_root=./demo/data
```
This saves images of the input (unposed) parts and multiple generations for possible assemblies.
We use [Mitsuba](https://mitsuba.readthedocs.io/en/latest/) for high quality ray-traced rendering, as shown above. For a faster rendering, please switch to [PyTorch3D PointsRasterizer](https://pytorch3d.readthedocs.io/en/latest/modules/renderer/points/rasterizer.html#pytorch3d.renderer.points.rasterizer.PointsRasterizer) by adding `visualizer.renderer=pytorch3d`. 
To save the flow trajectory as a GIF animation, use `visualizer.save_trajectory=true`.

More rendering options are available in [config/visualizer](config/visualizer/default.yaml).

**Overlap Prediction:** To visualize the overlap probabilities predicted by the encoder, please run:

```bash
python predict_overlap.py data_root=./demo/data
```

**Checkpoints:** The scripts will automatically download trained checkpoints from our [HuggingFace repo](https://huggingface.co/gradient-spaces/Rectified-Point-Flow/tree/main):
- `RPF_base_full_*.ckpt`: Full model checkpoint for assembly generation.
- `RPF_base_pretrain_*.ckpt`: Only the encoder checkpoint for overlap prediction.

To use custom checkpoints, please set `ckpt_path` in the config file or pass the argument `ckpt_path=...` to the command.


## 🚀 Training

The RPF training process consists of two stages:
1. **Encoder Pretraining**: Train the point cloud encoder on the overlap point detection task.
2. **Flow Model Training**: Train the full flow model with the pretrained encoder frozen.

### Encoder Pretraining

First, pretrain the point cloud encoder (Point Transformer) on the overlap point detection task:

```bash
python train.py --config-name "RPF_base_pretrain" \
    trainer.num_nodes=1 \
    trainer.devices=2 \
    data_root="../dataset" \
    data.batch_size=200 \
    data.num_workers=32 \
    data.limit_val_samples=1000 \
```

- `data.batch_size`: Batch size per GPU. Defaults to 200 for 80GB GPU.
- `data.num_workers`: Number of data workers per GPU. Defaults to 32.
- `data.limit_val_samples`: Limit validation samples per dataset for faster evaluation. Defaults to 1000.

### Flow Model Training

Train the full RPF model with the pretrained encoder:

```bash
python train.py --config-name "RPF_base_main" \
    trainer.num_nodes=1 \
    trainer.devices=8 \
    data_root="../dataset" \
    data.batch_size=40 \
    data.num_workers=16 \
    model.encoder_ckpt="./weights/RPF_base_pretrain.ckpt"
```

- `model.encoder_ckpt`: Path to pretrained encoder checkpoint.
- `data.batch_size`: Batch size per GPU. Defaults to 40 for 80GB GPU.


## 📚 More Details

### Training Data

The model is trained on a combination of following datasets. Please be aware that datasets have different licenses. We will release the dataset processing scripts and processed files soon.

<details>
<summary>Click to expand the list of training datasets.</summary>

| Dataset | Task | Part segmentation | #Parts | License | 
|:---|:---|:---|:---|:---|
| [IKEA-Manual](https://yunongliu1.github.io/ikea-video-manual/) | Shape assembly | Defined by IKEA manuals | [2, 19] | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [PartNet](https://partnet.cs.stanford.edu/) | Shape assembly | Annotated by human | [2, 64] | [MIT License](https://mit-license.org/) |
| [BreakingBad-Everyday](https://breaking-bad-dataset.github.io/) | Shape assembly | Simulated fractures via [fracture-modes](https://github.com/sgsellan/fracture-modes#dataset) | [2, 49] | [MIT License](https://mit-license.org/) |
| [Two-by-Two](https://tea-lab.github.io/TwoByTwo/) | Shape assembly | Annotated by human | 2 | [MIT License](https://mit-license.org/) | 
| [ModelNet-40](https://modelnet.cs.princeton.edu/#) | Pairwise registration | Following [Predator](https://github.com/prs-eth/OverlapPredator) split | 2 | [Custom](https://modelnet.cs.princeton.edu/#) |
| [TUD-L](https://bop.felk.cvut.cz/datasets/) | Pairwise registration | Real scans with partial observations | 2 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Objverse](https://objaverse.allenai.org/) | Overlap prediction | Segmented by [SAMPart3D](https://github.com/GradientSpaces/SAMPart3D) | [3, 12] | [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) |
</details>


### Custom Datasets

RPF supports two data formats: PLY files and HDF5, but we strongly recommend using HDF5 for faster I/O.
We provide scripts to help convert between these two formats. See [dataset_process/](dataset_process/) for more details.


### Training and Finetuning

<details>
<summary>Click to expand more usage examples.</summary>

#### Override Parameters

Override any configuration parameter from the command line:

```bash
# Adjust learning rate and batch size
python train.py --config-name "RPF_base_main" \
    model.optimizer.lr=5e-5 \
    data.batch_size=32 \
    trainer.max_epochs=1500

# Use different dataset combination
python train.py --config-name "RPF_base_main" \
    data=ikea \
    data.dataset_paths.ikea="../dataset/ikea.hdf5"
```

#### Finetuning Flow Model from Checkpoint

Finetuning the flow model from a checkpoint:

```bash
python train.py --config-name "RPF_base_main" \
    model.flow_model_ckpt="./weights/RPF_base.ckpt"
```

#### Resume Training from Checkpoint

Resume interrupted training from a Lightning checkpoint:

```bash
python train.py --config-name "RPF_base_main" \
    ckpt_path="./output/RPF_base_joint/last.ckpt"
```

#### Multi-GPU Training

Our code supports distributed training across multiple GPUs:

```bash
# Default: DDP with automatic multi-GPU detection, which uses all available GPUs.
python train.py --config-name "RPF_base_main" \
    trainer.devices="auto" \
    trainer.strategy="ddp"

# You can specify number of GPUs and nodes.
python train.py --config-name "RPF_base_main" \
    trainer.num_nodes=2 \
    trainer.devices=8 \
    trainer.strategy="ddp"
```
</details>

### Configurations

RPF uses [Hydra](https://hydra.cc/) for configuration management.

<details>
<summary> The configuration is organized into following groups. Click to expand.</summary>

#### Root Configurations (`config/`)
- `RPF_base_pretrain.yaml`: Root config for encoder pretraining.
- `RPF_base_main.yaml`: Root config for flow model training.

Relevant parameters:
- `data_root`: Path to the directory containing HDF5 files.
- `experiment_name`: The name used for WandB run.
- `log_dir`: Directory for checkpoints and logs (default: `./output/${experiment_name}`).
- `seed`: Random seed for reproducibility (default: 42).

#### Model Configurations (`config/model/`)
- `rectified_point_flow.yaml`: Main RPF model configuration
  - `optimizer`: AdamW optimizer settings (lr: 1e-4, weight_decay: 1e-6)
  - `lr_scheduler`: MultiStepLR with milestones at [1000, 1300, 1600, 1900]
  - `timestep_sampling`: Timestep sampling strategy ("u_shaped")
  - `inference_sampling_steps`: Number of inference steps (default: 20)
- `encoder/ptv3_object.yaml`: Point Transformer V3 encoder configuration
- `flow_model/point_cloud_dit.yaml`: Diffusion Transformer (DiT) configuration

#### Data Configurations (`config/data/`)
- `ikea.yaml`: Single dataset configuration example
- `ikea_partnet_everyday_twobytwo_modelnet_tudl.yaml`: Multi-dataset config for flow model training.
- `ikea_partnet_everyday_twobytwo_modelnet_tudl_objverse.yaml`: Multi-dataset config for encoder pretraining.

Relevant parameters:
- `num_points_to_sample`: Points to sample per part (default: 5000)
- `min_parts`/`max_parts`: Range of parts per scene (2-64)
- `min_points_per_part`: Minimum points required per part (default: 20)
- `multi_anchor`: Enable multi-anchor training (default: true)

#### Training Configurations (`config/trainer/`)
Define parameters for Lightning's [Trainer](https://lightning.ai/docs/pytorch/latest/common/trainer.html#). You can add/adjust all the settings supported by Trainer.
- `main.yaml`: Flow model training settings.
- `pretrain.yaml`: Pretraining settings.

#### Logging Configurations (`config/loggers/`)
- `wandb.yaml`: Weights & Biases logging configuration

</details>


## 🐛 Troubleshooting

**Slow I/O**: We find that the flow model training can be bound by the I/O. This typically leads to a low GPU utilization (e.g., < 80%). We've optimized the setting based on our systems (one node of 8xH100 with 112 CPU cores) and you may need to adjust your own settings. Here are some suggestions:

- More threads per worker: Increase `num_threads=2` in [rectified_point_flow/data/dataset.py](rectified_point_flow/data/dataset.py).
- More workers per GPU: Increase `data.num_workers=32` based on your CPU cores.
- Use [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils) for faster point sampling: Enable with `USE_PCU=1 python train.py ...`.
- Use HDF5 format and store the files on faster storage (e.g., SSD or NVMe).

**Loss overflow**: We do find numerical instabilities during training, especially loss overflowing to NaN. If you encounter this when training, you may try to use `bf16` precision by adding `trainer.precision=bf16`.

## ☑️ Todo List
- [x] Release model & demo code
- [ ] Release full training code & checkpoints
- [ ] Release processed dataset files
- [ ] Support running without flash-attn
- [ ] Online demo

## 📝 Citation

If you find the code or data useful for your research, please cite our paper:

```bibtex
@inproceedings{sun2025_rpf,
      author = {Sun, Tao and Zhu, Liyuan and Huang, Shengyu and Song, Shuran and Armeni, Iro},
      title = {Rectified Point Flow: Generic Point Cloud Pose Estimation},
      booktitle = {arxiv preprint arXiv:2506.05282},
      year = {2025},
    }
```

## Acknowledgments

Some codes in this repo are borrowed from open-source projects, including [DiT](https://github.com/facebookresearch/DiT), [PointTransformer](https://github.com/Pointcept/Pointcept), and [GARF](https://github.com/ai4ce/GARF). We appreciate their valuable contributions!


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
