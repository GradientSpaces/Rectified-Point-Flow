#!/usr/bin/env python3
"""
Convert PLY files back to HDF5 format.

This script reads PLY files organized in the demo data structure and converts them
back to HDF5 format compatible with dataset.py.

Usage:
    python convert_ply_to_h5.py --data_root ../dataset --dataset_name ikea --output_path ikea_converted.hdf5
"""

import argparse
import logging
from pathlib import Path
import glob

import h5py
import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PLYtoH5")


def load_ply_file(ply_path):
    """Load a PLY file and return vertices, faces, and normals."""
    try:
        mesh = trimesh.load(str(ply_path), process=False)
        
        vertices = np.array(mesh.vertices, dtype=np.float32)
        
        # Handle faces - empty array if no faces (point cloud)
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            faces = np.array(mesh.faces, dtype=np.int32)
        else:
            faces = np.array([], dtype=np.int32).reshape(0, 3)
        
        # Handle normals
        normals = None
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            normals = np.array(mesh.vertex_normals, dtype=np.float32)
        elif len(faces) > 0 and hasattr(mesh, 'face_normals'):
            # Compute vertex normals from face normals for mesh data
            normals = np.zeros_like(vertices, dtype=np.float32)
            for i, face in enumerate(faces):
                face_normal = mesh.face_normals[i]
                for vertex_idx in face:
                    normals[vertex_idx] += face_normal
            # Normalize
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.where(norms > 0, normals / norms, normals)
        else:
            # Default normals pointing up for point clouds
            normals = np.zeros_like(vertices, dtype=np.float32)
            normals[:, 2] = 1.0
            
        return vertices, faces, normals
        
    except Exception as e:
        logger.error(f"Error loading PLY file {ply_path}: {e}")
        return None, None, None


def convert_ply_to_h5(data_root, dataset_name, output_path):
    """Convert PLY files for one dataset to HDF5 format."""
    logger.info(f"Converting dataset: {dataset_name}")
    
    dataset_dir = Path(data_root) / dataset_name
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return False
    
    # Read split information
    split_dir = dataset_dir / "data_split"
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        return False
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as h5_file:
        # Create data_split structure
        data_split_group = h5_file.create_group("data_split")
        dataset_split_group = data_split_group.create_group(dataset_name)
        
        all_fragments = set()
        
        # Process each split
        for split_file in split_dir.glob("*.txt"):
            split_name = split_file.stem
            logger.info(f"Processing split: {split_name}")
            
            # Read fragment names from split file
            with open(split_file, 'r') as f:
                fragment_names = [line.strip() for line in f if line.strip()]
            
            # Store split information (encode as bytes for HDF5 compatibility)
            fragment_names_encoded = [name.encode('utf-8') for name in fragment_names]
            dataset_split_group.create_dataset(split_name, data=fragment_names_encoded)
            
            all_fragments.update(fragment_names)
            logger.info(f"Found {len(fragment_names)} fragments in {split_name} split")
        
        # Process each fragment
        for fragment_name in sorted(all_fragments):
            logger.info(f"Processing fragment: {fragment_name}")
            fragment_dir = dataset_dir / fragment_name
            
            if not fragment_dir.exists():
                logger.warning(f"Fragment directory not found: {fragment_dir}")
                continue
            
            # Get all PLY files in fragment directory, sorted by name
            ply_files = sorted(glob.glob(str(fragment_dir / "*.ply")))
            
            if not ply_files:
                logger.warning(f"No PLY files found in {fragment_dir}")
                continue
            
            # Create fragment group in HDF5
            fragment_group = h5_file.create_group(fragment_name)
            
            # Process each part
            valid_parts = 0
            for i, ply_path in enumerate(ply_files):
                # Extract part name from filename (e.g., part_00.ply -> part_00)
                part_filename = Path(ply_path).stem
                part_name = part_filename  # Keep original naming
                
                logger.debug(f"Processing part: {part_name} from {ply_path}")
                
                vertices, faces, normals = load_ply_file(ply_path)
                
                if vertices is None or len(vertices) == 0:
                    logger.warning(f"Failed to load or empty PLY file: {ply_path}")
                    continue
                
                # Create part group
                part_group = fragment_group.create_group(part_name)
                
                # Store vertices (required)
                part_group.create_dataset("vertices", data=vertices, compression='gzip')
                
                # Store faces (if available and non-empty)
                if faces is not None and len(faces) > 0:
                    part_group.create_dataset("faces", data=faces, compression='gzip')
                
                # Store normals (if available)
                if normals is not None and len(normals) > 0:
                    part_group.create_dataset("normals", data=normals, compression='gzip')
                
                valid_parts += 1
                logger.debug(f"Stored part {part_name}: {len(vertices)} vertices, {len(faces)} faces")
            
            if valid_parts > 0:
                logger.info(f"Converted fragment {fragment_name} with {valid_parts} parts")
            else:
                logger.warning(f"No valid parts found for fragment {fragment_name}")
                # Remove empty fragment group
                del h5_file[fragment_name]
    
    logger.info(f"Successfully created HDF5 file: {output_path}")
    return True


def verify_h5_structure(h5_path, dataset_name):
    """Verify that the created HDF5 file has the expected structure."""
    logger.info("Verifying HDF5 structure...")
    
    try:
        with h5py.File(h5_path, 'r') as h5_file:
            if "data_split" not in h5_file:
                logger.error("Missing 'data_split' group")
                return False
            if dataset_name not in h5_file["data_split"]:
                logger.error(f"Missing dataset '{dataset_name}' in data_split")
                return False
            
            splits = list(h5_file["data_split"][dataset_name].keys())
            logger.info(f"Found splits: {splits}")
            
            # Check fragment structure
            fragment_count = 0
            for key in h5_file.keys():
                if key != "data_split":
                    fragment_group = h5_file[key]
                    fragment_count += 1
                    
                    has_vertices = False
                    for part_name in fragment_group.keys():
                        part_group = fragment_group[part_name]
                        if "vertices" in part_group:
                            has_vertices = True
                            break
                    
                    if not has_vertices:
                        logger.warning(f"Fragment {key} has no parts with vertices")
            
            logger.info(f"Found {fragment_count} fragments total")
            logger.info("HDF5 structure verification passed!")
            return True
            
    except Exception as e:
        logger.error(f"Error verifying HDF5 structure: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PLY files back to HDF5 format")
    parser.add_argument("--data_root", required=True, help="Input directory containing PLY data structure")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset to convert")
    parser.add_argument("--output_path", help="Output HDF5 file path (default: {dataset_name}_converted.hdf5)")
    parser.add_argument("--verify", action="store_true", help="Verify HDF5 structure after conversion")
    
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = f"{args.dataset_name}_converted.hdf5"
    
    logger.info(f"Converting PLY data to HDF5")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Dataset name: {args.dataset_name}")
    logger.info(f"Output file: {args.output_path}")
    
    success = convert_ply_to_h5(args.data_root, args.dataset_name, args.output_path)
    
    if success:
        logger.info("Conversion completed successfully!")
        
        if args.verify:
            verify_success = verify_h5_structure(args.output_path, args.dataset_name)
            if not verify_success:
                logger.error("HDF5 verification failed!")
                exit(1)
    else:
        logger.error("Conversion failed!")
        exit(1)


if __name__ == "__main__":
    main()
