# Dataset Processing

This directory contains scripts for converting datasets between different formats, useful for processing custom datasets.

## Supported Formats

Our data loader supports two formats: PLY files in a filesystem and HDF5 archives.

### 1. PLY Files

Store PLY files per part under a directory hierarchy as follows:

```xml
data_root/
└── <dataset_name>/
    ├── data_split/
    │   ├── train.txt
    │   └── val.txt
    ├── <object_name>/
    │   ├── <fragment_name>/
    │   │   ├── part_000.ply
    │   │   ├── part_001.ply
    │   │   └── ...
    │   └── ...
    └── <object_name>/
        └── ...
```

- The `data_root` directory can contain multiple datasets.
- The `data_split/{train,val}.txt` files list fragment paths (one per line) for each split:
  ```xml
  <dataset_name>/<object_name>/<fragment_name>
  ```
- See the [demo/data](../demo/data) directory for a complete example.

### 2. HDF5 (Recommended)

Pack data into a single [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file per dataset, organized as follows:

```xml
data_root/
└── <dataset_name>.hdf5
    ├── data_split/
    │   └── <dataset_name>/
    │       ├── train                 : list[str]
    │       └── val                   : list[str]
    └── <dataset_name>/
        ├── <object_name>/
        │   └── <fragment_name>/
        │       └── <part_idx>/
        │           ├── vertices      : float32[n, 3]
        │           ├── normals       : float32[n, 3]
        │           └── faces         : int64[m, 3], optional
        └── ...
```

- `<dataset_name>`, `<object_name>`, and `<fragment_name>` can be any string.
- `<part_idx>` is a 0-based index indicating the part number.
- The `faces` field is optional. If empty, the part is treated as a pure point cloud.
- The `data_split/<dataset_name>/{train,val}` groups contain lists of fragment keys:
  ```xml
  <dataset_name>/<object_name>/<fragment_name>
  ```
- We **stronly recommend** using HDF5 for training due to efficiency in multi-process reading and reduced file count in the storage.

## Loading Datasets

The `PointCloudDataset` class automatically detects the format and handles both, as follows:

```python
from rectified_point_flow.data.dataset import PointCloudDataset

# Load PLY files format
dataset = PointCloudDataset(
    split="train",
    data_path="path/to/your_dataset",
    dataset_name="your_dataset",
    # ... other parameters
)

# Load HDF5 format
dataset = PointCloudDataset(
    split="train", 
    data_path="path/to/your_dataset.hdf5",
    dataset_name="your_dataset",
    # ... other parameters
)
```


You can also configure datasets from different formats in your config file (`config/data.yaml`), like:

```yaml
dataset_paths:
  ikea: "${data_root}/ikea"                # PLY files format
  partnet_v0: "${data_root}/partnet.hdf5"  # HDF5 format
dataset_names: ["ikea", "partnet_v0"]
```

## Format Conversion

### 1. Convert PLY Files to HDF5

We provide a lightweight script to convert PLY files to the HDF5 format, as follows:

```bash
python convert_ply_to_h5.py \
    --data_root     "data_root/" \
    --dataset_name  "dataset_name" \
    --output_path   "data_root_h5/dataset_name.hdf5"
```

**For large-scale datasets:** Please refer to `convert_objverse_to_h5.py`, which we use to convert the [Objaverse](https://objaverse.allenai.org/) dataset efficiently by parallel computing. You may reuse its functions for your own dataset.

### 2. Export HDF5 to PLY Files

We also provide a script to export HDF5 datasets back to PLY format for inspection, visualization, or editing, as follows:

```bash
python export_ply_from_h5.py \
    --data_root          "data_root_h5/" \
    --output_dir         "./demo/data/" \
    --samples_per_split  10 \
    --datasets           "ikea" "partnet_v0"
```

This example exports the 10 samples from `ikea` and `partnet_v0` datasets to the `demo/data` directory.

