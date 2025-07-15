# Guide for Using Your Own Datasets

This directory contains scripts for converting datasets between different formats, useful for processing your own datasets. 

## Data Formats

Our data loader supports two data formats: 
- PLY files in a filesystem, and
- HDF5 archives.

### 1. PLY Files

Store PLY files per part under a directory hierarchy as follows:

```xml
data_root/
└── <dataset_name>/
    ├── data_split/
    │   ├── train.txt
    │   └── val.txt
    ├── <dataset_name>/
    │   ├── <object_name>/
    │   │   ├── <fragmentation_name>/
    │   │   │   ├── part_000.ply
    │   │   │   ├── part_001.ply
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
```

- The `data_root` directory can contain multiple datasets.
- The `<fragmentation_name>` represents a way to fragment an `<object_name>` into parts, under which `part_<idx>.ply` is a 0-based index indicating the part PLY file.
- The `data_split/{train,val}.txt` files list fragment paths (one per line) for each split:
  ```xml
  <dataset_name>/<object_name>/<fragmentation_name>
  ```
- All PLY files should have the `vertices` field. The `vertex_normals` and `faces` fields are optional. If `faces` is empty, the part is treated as a pure point cloud.
- Both binary and ascii PLY files are supported. 
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
        │   └── <fragmentation_name>/
        │       └── <part_idx>/
        │           ├── vertices      : float32[n, 3]
        │           ├── normals       : float32[n, 3], optional
        │           └── faces         : int64[m, 3], optional
        └── ...
```

- `<dataset_name>`, `<object_name>`, and `<fragmentation_name>` can be any string.
- `<part_idx>` is a 0-based index indicating the part number.
- The `vertices` field is required. The `normals` and `faces` fields are optional. If `faces` is empty, the part is treated as a pure point cloud.
- The `data_split/<dataset_name>/{train,val}` groups contain lists of fragment keys:
  ```xml
  <dataset_name>/<object_name>/<fragmentation_name>
  ```
- We **strongly recommend** using HDF5 for training due to efficiency in multi-process reading and reduced file count in the storage.

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

### Config File

In the config file (`config/data.yaml`), you can specify the dataset to use in the `dataset_names` field. It will automatically detect the format as well.

```yaml
# For example, if you have the following datasets:
# ./dataset/ikea/         # PLY files format
# ./dataset/partnet.hdf5  # HDF5 format
# ./dataset/custom.hdf5   # HDF5 format

# You can specify to use only ikea and custom datasets:
data_root: "./dataset"
data:
  dataset_names: ["ikea", "custom"]
```