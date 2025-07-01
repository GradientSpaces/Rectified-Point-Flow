import glob
import hashlib
import os
from collections import defaultdict
from multiprocessing import Pool, Process, Queue, cpu_count

import h5py
import numpy as np
import trimesh


def is_val(name, val_ratio=0.25):
    stable_hash = int(hashlib.sha256(str(name).encode('utf-8')).hexdigest(), 16) % 10**8
    return stable_hash < int(val_ratio * 10**8)


def hdf5_writer(queue: Queue, output_file, total_items):
    """
    Single-writer process that reads processed scene results from a queue
    and writes them into an HDF5 file.
    """
    with h5py.File(output_file, 'w') as h5file:
        objaverse_group = h5file.create_group('objaverse_v1')
        train_keys, val_keys = [], []
        items_written = 0
        while items_written < total_items:
            result = queue.get()  # Blocks if queue is empty
            if result is None:
                break
            split, scene = result
            if split == 'skip':
                items_written += 1
                continue
            scene_id = scene["scene_id"]
            obj_type = scene["obj_type"]
            if obj_type not in objaverse_group:
                objaverse_group.create_group(obj_type)
            obj_group = objaverse_group[obj_type]
            scene_group = obj_group.create_group(scene_id)
            for fractured in scene["fractured"]:
                fractured_group = scene_group.create_group(fractured["fractured_id"])
                for idx, part in enumerate(fractured["parts"]):
                    part_group = fractured_group.create_group(str(idx))
                    part_group.create_dataset('vertices', data=part['vertices'])
                    part_group.create_dataset('faces', data=part['faces'])
                    # part_group.create_dataset('shared_faces', data=part['shared_faces'])
                key = f'objaverse_v1/{obj_type}/{scene_id}/{fractured["fractured_id"]}'
                if split == 'train':
                    train_keys.append(key)
                else:
                    val_keys.append(key)
            items_written += 1
            print(f"Write {items_written}/{total_items}")
            
        # Write data-split info.
        data_split_group = h5file.create_group('data_split')
        data_split_partnet = data_split_group.create_group('objaverse_v1')
        string_dt = h5py.string_dtype(encoding='utf-8')
        data_split_partnet.create_dataset('train', data=np.array(train_keys, dtype=object), dtype=string_dt)
        data_split_partnet.create_dataset('val', data=np.array(val_keys, dtype=object), dtype=string_dt)
    print(f"HDF5 dataset saved to '{output_file}'")


def process_level(sample_dir, scene_id, level=0):
    mesh_path = os.path.join(sample_dir, 'ply', f'{scene_id}_0_{level:02d}.ply')
    face_labels_path = os.path.join(sample_dir, 'cluster_out', f'{scene_id}_0_{level:02d}.npy')
    mesh = trimesh.load(mesh_path, process=False)
    face_labels = np.load(face_labels_path)

    # Map original labels to 0-based part IDs
    unique_labels = np.unique(face_labels)
    label_remap = {old: new for new, old in enumerate(unique_labels)}

    # Group face indices by new part ID
    part_faces_dict = defaultdict(list)
    for face_idx, original_label in enumerate(face_labels):
        part_id = label_remap[original_label]
        part_faces_dict[part_id].append(face_idx)

    part_list = []
    for part_id, face_indices in part_faces_dict.items():
        faces = mesh.faces[face_indices]
        unique_verts = np.unique(faces)
        remapped_faces = np.searchsorted(unique_verts, faces)
        part_vertices = mesh.vertices[unique_verts]
        # part_mesh = trimesh.Trimesh(vertices=part_vertices, faces=remapped_faces)
        part_list.append({
            'vertices': part_vertices,
            'faces': remapped_faces,
        })
    return part_list


def process_and_queue(sample_dir):
    """
    Return:
        - split: 'train' or 'val'
        - results: a dictionary containing:
            - scene_id: the ID of the scene
            - obj_type: the type of object (e.g., 'chair', 'table')
            - fractured: a list of fractured parts, each containing:
                - fractured_id: ID of the fractured part
                - parts : a list of parts, each containing:
                    - vertices: vertices of the part
                    - faces: faces of the part
    """
    scene_id = os.path.basename(sample_dir)
    obj_type = os.path.basename(os.path.dirname(sample_dir))
    result = {
        'scene_id': scene_id,
        'obj_type': obj_type,
    }
    try:
        fractured_list = []
        levels = list(range(3, 16))
        for level in np.random.choice(levels, size=3, replace=False):
            fractured_id = f'level_{level}'
            parts = process_level(sample_dir, scene_id, level)
            fractured_list.append({
                'fractured_id': fractured_id,
                'parts': parts,
            })
        result['fractured'] = fractured_list
        split = 'train' if not is_val(f"{obj_type}/{scene_id}") else 'val'
        print(f"Processed {obj_type}/{scene_id}")
    except Exception as e:
        split = 'skip'
        result = None
    return (split, result)

def parallel_process_and_write(data_root, output_file, queue_max_size=50):
    """
    Processes all scenes in parallel by feeding them into a bounded multiprocessing queue.
    A dedicated writer process consumes the results and writes them to an HDF5 file.
    The queue limits in-memory data to the specified max size.
    """
    obj_dirs = glob.glob(os.path.join(data_root, '*', '*'))
    total_jobs = len(obj_dirs)
    result_queue = Queue(maxsize=queue_max_size)
    num_workers = cpu_count()

    # Callback function that places results into the queue.
    def worker_callback(result):
        result_queue.put(result, block=True)

    pool = Pool(processes=num_workers)
    for sample_dir in obj_dirs:
        pool.apply_async(process_and_queue, args=(sample_dir,), callback=worker_callback)
    pool.close()

    # Start the writer process
    writer_proc = Process(target=hdf5_writer, args=(result_queue, output_file, total_jobs))
    writer_proc.start()

    pool.join()
    writer_proc.join()

if __name__ == "__main__":
    data_root = './object'
    output_hdf5 = '/home/users/taosun/Data/objaverse.hdf5'
    parallel_process_and_write(data_root, output_hdf5, queue_max_size=50)
