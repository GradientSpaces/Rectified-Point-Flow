#!/usr/bin/env python3
"""
Export data from HDF5 files to PLY format for easy inspection.

This script reads the HDF5 datasets and exports a subset of fragments
as PLY files in a structure compatible with the dataset.py.

Usage:
    python export_ply_from_h5.py --data_root /path/to/h5/files --output_dir demo/data --samples_per_dataset 10
"""

import argparse
import logging
from pathlib import Path
import os

import h5py
import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Export")

# Dataset names and corresponding HDF5 files
DATASET_NAMES = ["ikea", "partnet_v0", "everyday", "twobytwo", "modelnet", "tudl"]
DATASET_CONFIG = {
    "ikea": "ikea.hdf5",
    "partnet_v0": "partnet.hdf5", 
    "everyday": "breaking_bad.hdf5",
    "twobytwo": "2by2.hdf5",
    "modelnet": "modelnet.hdf5",
    "tudl": "tudl.hdf5"
}

def load_mesh_from_h5(group, part_name):
    """Load one mesh part from an HDF5 group."""
    sub_grp = group[part_name]
    verts = np.array(sub_grp["vertices"][:])
    faces = np.array(sub_grp["faces"][:]) if "faces" in sub_grp else np.array([])
    norms = np.array(sub_grp["normals"][:]) if "normals" in sub_grp else None
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, process=False)

def export_mesh_to_ply(mesh, output_path):
    """Export a trimesh object to PLY file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if len(mesh.faces) == 0:
        # Handle point cloud data (no mesh)
        vertices = mesh.vertices
        normals = mesh.vertex_normals if mesh.vertex_normals is not None else np.zeros_like(vertices)
        
        # Write PLY header for point cloud
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("end_header\n")
            
            # Write vertex data
            for i in range(len(vertices)):
                v = vertices[i]
                n = normals[i]
                f.write(f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]}\n")
    else:
        mesh.export(str(output_path))

def export_dataset(h5_path, dataset_name, output_dir, samples_per_split=5):
    """Export demo data for one dataset."""
    logger.info(f"Processing dataset: {dataset_name} from {h5_path}")
    if not os.path.exists(h5_path):
        logger.warning(f"Dataset file not found: {h5_path}")
        return
    
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = dataset_output_dir / "data_split"
    split_dir.mkdir(exist_ok=True)
    
    try:
        with h5py.File(h5_path, "r", libver='latest', swmr=True) as h5_file:
            if "data_split" not in h5_file or dataset_name not in h5_file["data_split"]:
                return
            
            data_split = h5_file["data_split"][dataset_name]
            for split_name in data_split.keys():
                logger.info(f"Processing split: {split_name}")
                split_data = data_split[split_name]
                fragment_names = [name.decode() if isinstance(name, bytes) else name for name in split_data[:]]
                
                # Select a subset for demo
                if len(fragment_names) > samples_per_split:
                    # Select evenly spaced samples
                    step = len(fragment_names) // samples_per_split
                    selected_fragments = fragment_names[::step][:samples_per_split]
                else:
                    selected_fragments = fragment_names
                
                logger.info(f"Selected {len(selected_fragments)} fragments from {len(fragment_names)} total")
                
                # Write split file
                split_file = split_dir / f"{split_name}.txt"
                with open(split_file, 'w') as f:
                    for frag_name in selected_fragments:
                        f.write(f"{frag_name}\n")
                
                # Export each selected fragment
                for frag_name in selected_fragments:
                    try:
                        if frag_name not in h5_file:
                            logger.warning(f"Fragment {frag_name} not found in H5 file")
                            continue
                            
                        frag_group = h5_file[frag_name]
                        part_names = list(frag_group.keys())
                        
                        # Filter parts with valid data
                        valid_parts = []
                        for part_name in part_names:
                            part_group = frag_group[part_name]
                            if "vertices" in part_group and len(part_group["vertices"]) > 0:
                                valid_parts.append(part_name)
                            
                        logger.info(f"Exporting {frag_name} with {len(valid_parts)} parts")
                        frag_dir = dataset_output_dir / frag_name
                        frag_dir.mkdir(parents=True, exist_ok=True)
                        for i, part_name in enumerate(valid_parts):
                            try:
                                mesh = load_mesh_from_h5(frag_group, part_name)
                                ply_path = frag_dir / f"part_{i:03d}.ply"
                                export_mesh_to_ply(mesh, ply_path)
                                
                            except Exception as e:
                                logger.error(f"Error exporting part {part_name} from {frag_name}: {e}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"Error processing fragment {frag_name}: {e}")
                        continue
                        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export demo data from HDF5 to PLY format")
    parser.add_argument("--data_root", required=True, help="Root directory containing HDF5 files")
    parser.add_argument("--output_dir", default="demo/data", help="Output directory for demo data")
    parser.add_argument("--samples_per_split", type=int, default=5, help="Number of samples per split to export")
    parser.add_argument("--datasets", nargs="+", default=DATASET_NAMES, 
                        choices=DATASET_NAMES, help="Datasets to export")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting demo data export to: {output_dir}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Samples per split: {args.samples_per_split}")
    
    # Export each dataset
    for dataset_name in args.datasets:
        h5_filename = DATASET_CONFIG[dataset_name]
        h5_path = os.path.join(args.data_root, h5_filename)
        export_dataset(h5_path, dataset_name, args.output_dir, args.samples_per_split)
    
    logger.info("Demo data export completed!")


if __name__ == "__main__":
    main() 