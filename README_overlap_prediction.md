# Overlap Prediction and Visualization

This script (`predict_overlap.py`) provides functionality to predict overlap probabilities per point using a trained PointCloudEncoder and visualize the results with color-coded point clouds using a Lightning callback architecture.

## Overview

The script loads a pre-trained PointCloudEncoder model, runs inference on test data to predict overlap probabilities for each point, and generates visualizations where points are colored according to their overlap probability using various matplotlib colormaps. The visualization is now handled by the `OverlapVisualizationCallback` class.

## Architecture

### Callback-Based Design

The visualization system has been refactored into a modular callback-based architecture:

- **`VisualizationCallback`** (Base class): Common functionality for all point cloud visualizations
- **`FlowVisualizationCallback`**: Specialized for rectified point flow models (trajectory visualization)  
- **`OverlapVisualizationCallback`**: Specialized for overlap prediction models (probability visualization)

### Utility Functions

New utility functions in `rectified_point_flow/utils/render.py`:

- **`probs_to_colors(probs, colormap)`**: Converts [0,1] probabilities to RGB colors
- **`part_ids_to_colors(part_ids, colormap)`**: Converts part IDs to colors (renamed from `generate_part_colors`)

## Features

- **Overlap Prediction**: Uses a trained PointCloudEncoder to predict overlap probabilities (0-1) for each point
- **Color Mapping**: Maps overlap probabilities to colors using customizable matplotlib colormaps
- **Multiple Colormaps**: Generates visualizations with different color schemes (viridis, plasma, inferno, hot, coolwarm)
- **Flexible Rendering**: Supports both PyTorch3D and Mitsuba renderers
- **Configurable**: Fully configurable through Hydra YAML configs
- **Callback Architecture**: Integrates seamlessly with PyTorch Lightning training/testing loops

## Usage

### Basic Usage

```bash
python predict_overlap.py ckpt_path=/path/to/your/encoder_checkpoint.ckpt
```

### Advanced Usage

```bash
python predict_overlap.py \
    ckpt_path=/path/to/your/checkpoint.ckpt \
    data_root=/path/to/your/dataset \
    visualizer.renderer=pytorch3d \
    visualizer.max_samples_per_batch=20 \
    data.batch_size=2
```

## Configuration

The script uses `config/RPF_base_predict_overlap.yaml` with the overlap visualizer:

```yaml
# Required: Path to trained PointCloudEncoder checkpoint
ckpt_path: /path/to/your/checkpoint.ckpt

# Use overlap visualization callback
defaults:
  - visualizer: overlap

# Data settings
data_root: "./demo/data"
data:
  batch_size: 1
  num_workers: 1

# Output
log_dir: ./demo/
```

### Visualizer Configuration

The overlap visualizer is configured in `config/visualizer/overlap.yaml`:

```yaml
_target_: rectified_point_flow.visualizer.OverlapVisualizationCallback

# Rendering settings
renderer: "pytorch3d"
image_size: 512
point_radius: 0.015
camera_dist: 4.0
camera_elev: 20.0
camera_azim: 30.0

# Overlap-specific settings
colormaps:
  - "viridis"
  - "plasma" 
  - "inferno"
  - "hot"
  - "coolwarm"
```

## Output

The script generates:

1. **Visualizations**: Point cloud images with color-coded overlap probabilities
   - Multiple colormap variants per sample
   - Saved as PNG files in `{log_dir}/visualizations/`

2. **Statistics**: Console output showing overlap probability statistics for each sample:
   - Mean, standard deviation, min/max values

## File Structure

```
demo/
└── visualizations/
    ├── 0000-sample_name_overlap_viridis.png
    ├── 0000-sample_name_overlap_plasma.png
    ├── 0000-sample_name_overlap_inferno.png
    ├── 0000-sample_name_overlap_hot.png
    └── 0000-sample_name_overlap_coolwarm.png
```

## Color Interpretation

- **Low Overlap (Blue/Purple)**: Points with low probability of overlap with other parts
- **High Overlap (Yellow/Red)**: Points with high probability of overlap with other parts
- **Colormap Variations**: 
  - `viridis`: Blue → Green → Yellow
  - `plasma`: Purple → Pink → Yellow
  - `inferno`: Black → Red → Yellow
  - `hot`: Black → Red → Yellow → White
  - `coolwarm`: Blue → White → Red

## Key Classes and Functions

### `OverlapVisualizationCallback`

PyTorch Lightning callback for overlap prediction visualization.

**Key Parameters:**
- `colormaps`: List of matplotlib colormap names
- `renderer`: "pytorch3d" or "mitsuba"
- `max_samples_per_batch`: Limit number of samples to visualize

**Usage:**
```python
from rectified_point_flow.visualizer import OverlapVisualizationCallback

callback = OverlapVisualizationCallback(
    colormaps=["viridis", "plasma"],
    renderer="pytorch3d",
    max_samples_per_batch=5
)
```

### `probs_to_colors(probs, colormap="viridis")`

Converts overlap probabilities [0, 1] to RGB colors using matplotlib colormaps.

**Args:**
- `probs`: Tensor of shape (N,) with values in [0, 1]
- `colormap`: Matplotlib colormap name

**Returns:**
- RGB colors tensor of shape (N, 3)

**Usage:**
```python
from rectified_point_flow.utils.render import probs_to_colors
import torch

overlap_probs = torch.rand(1000)  # Random probabilities
colors = probs_to_colors(overlap_probs, "viridis")
# colors.shape -> torch.Size([1000, 3])
```

### `part_ids_to_colors(part_ids, colormap="default")`

Converts part IDs to colors for visualization (renamed from `generate_part_colors`).

**Args:**
- `part_ids`: Tensor of shape (N,) containing part IDs
- `colormap`: Colormap name or type

**Returns:**
- RGB colors tensor of shape (N, 3)

## Integration with PyTorch Lightning

The callback integrates seamlessly with Lightning's testing loop:

```python
import lightning as L
from rectified_point_flow.visualizer import OverlapVisualizationCallback

# Create callback
vis_callback = OverlapVisualizationCallback()

# Setup trainer with callback
trainer = L.Trainer(callbacks=[vis_callback])

# Run testing - visualizations are automatically generated
trainer.test(model, datamodule)
```

## Requirements

- PyTorch
- PyTorch Lightning
- Hydra
- Matplotlib
- NumPy
- PIL/Pillow
- PyTorch3D (for pytorch3d renderer)
- Mitsuba 3 (optional, for mitsuba renderer)

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Ensure `ckpt_path` points to a valid PointCloudEncoder checkpoint
2. **CUDA out of memory**: Reduce `data.batch_size` or `visualizer.max_samples_per_batch`
3. **Renderer errors**: Switch between "pytorch3d" and "mitsuba" renderers
4. **Data loading errors**: Verify `data_root` path and dataset format

### Example Error Resolution

```bash
# If PyTorch3D rendering fails, use Mitsuba
python predict_overlap.py ckpt_path=your_checkpoint.ckpt visualizer.renderer=mitsuba

# If memory issues occur, reduce batch size
python predict_overlap.py ckpt_path=your_checkpoint.ckpt data.batch_size=1 visualizer.max_samples_per_batch=5

# Custom colormap selection
python predict_overlap.py ckpt_path=your_checkpoint.ckpt visualizer.colormaps=[viridis,plasma]
```

## Model Requirements

The script expects a trained PointCloudEncoder model that:
- Has been trained for overlap prediction
- Returns predictions with `overlap_prob` key containing probabilities in [0, 1]
- Is compatible with the data format used in your dataset

Ensure your checkpoint was saved from a model trained with the overlap prediction objective.

## Migration from Function-Based Approach

If upgrading from the previous function-based approach:

1. **Old**: Used standalone functions for visualization
2. **New**: Uses `OverlapVisualizationCallback` integrated with Lightning
3. **Benefits**: Better integration, cleaner architecture, easier configuration
4. **Migration**: Update configs to use the new visualizer callback 