## Bimanual Affordance: Based on Affogato Dataset

### 📂 Project Structure

```text
affogato / 
    - code / 
        - check.py                      # Validation tool: Sync check between image paths and 3D point clouds
        - gobjaverse_to_affogato.py     # Mapping script: Establishes correspondence between datasets
        - read_npy.py                   # Visualization core: Handles .npy format and Open3D rendering
        - read_glb.py                   # Utility for reading raw GLB 3D files
        - download_*.py                 # Model weight download scripts (Gemma/Molmo)
        - infer_*.py                    # Inference scripts (Work in Progress)
    - dataset / 
        - affogato /                    # Point cloud data (split into part000 ~ 015)
        - gobjaverse /                  # Multi-view images (currently focused on Electronics)
        - gobjaverse_280k_index_to_objaverse.json # Original indexing file
        - gobjaverse_to_affogato.json   # Generated final mapping index
```

### 📊 Dataset Overview

1. Affogato Dataset (Provide 3D Point Cloud)
    - Path Format: `dataset/affogato/affogato_all_part***/{object_id}/xyzc.npy`
2. G-Objaverse Dataset (Provide Multi-view Images)
    - Path Format: `dataset/gobjaverse/electronics/{subdir}/{index}/{object_index}/00000/*****.png`

### ⚙️ Data Pipeline & Validation

**Step 1: Generate Mapping Index**

Since the directory structures and naming conventions differ between datasets, run the mapping script to pair `gobjaverse` image paths with `affogato` point cloud paths:

```Bash
python code/gobjaverse_to_affogato.py
```

This produces `dataset/gobjaverse_to_affogato.json `.

**Step 2: Verify Alignment**

To ensure the mapping is correct (i.e., the image and the point cloud belong to the same object), use `check.py` for manual spot-checking:

```Bash
# Replace with the specific object_id you wish to inspect
python code/check.py --object_id 02886b7e7b2649c8b5ffaf53c0783c2a
```