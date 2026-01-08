# Virtual Golf Coach

A computer vision pipeline for analyzing golf swings using 3D human pose estimation. Extracts SMPL body model parameters from images to enable biomechanical analysis of golf swing mechanics.

## Overview

This project uses [HMR 2.0](https://github.com/shubham-goel/4D-Humans) (Human Mesh Recovery) to reconstruct a 3D human body mesh from a single image. The pipeline:

1. Detects humans in the image using YOLOv8
2. Estimates 3D body pose and shape using HMR 2.0
3. Renders mesh overlays for visualization
4. Saves SMPL parameters for downstream analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/AllenGrahamHart/golf.git
cd golf

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install 4D-Humans
pip install -e 4D-Humans/

# Download SMPL model from https://smplify.is.tue.mpg.de/
# Place at: data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

## Usage

```bash
# Basic usage
python -m src.inference --image input/swing.jpg --output output/

# With mesh export
python -m src.inference --image input/swing.jpg --output output/ --save_mesh

# Skip side view rendering
python -m src.inference --image input/swing.jpg --output output/ --no_side_view

# Use GPU (if available)
python -m src.inference --image input/swing.jpg --output output/ --device cuda
```

## Output Files

| File | Description |
|------|-------------|
| `{name}_render.png` | Front view with 3D mesh overlay |
| `{name}_side.png` | Side view of the mesh |
| `{name}_params.npz` | SMPL parameters (pose, shape, camera) |
| `{name}_person{i}.obj` | 3D mesh file (with `--save_mesh`) |

## SMPL Parameters

The `_params.npz` file contains:

### body_pose `(N, 23, 3, 3)`
Rotation matrices for 23 body joints. Index mapping:

| Idx | Joint | Idx | Joint | Idx | Joint |
|-----|-------|-----|-------|-----|-------|
| 0 | L_Hip | 8 | Spine3 | 16 | L_Shoulder |
| 1 | R_Hip | 9 | L_Foot | 17 | R_Shoulder |
| 2 | Spine1 | 10 | R_Foot | 18 | L_Elbow |
| 3 | L_Knee | 11 | Neck | 19 | R_Elbow |
| 4 | R_Knee | 12 | L_Collar | 20 | L_Wrist |
| 5 | Spine2 | 13 | R_Collar | 21 | R_Wrist |
| 6 | L_Ankle | 14 | Head | 22 | L_Hand |
| 7 | R_Ankle | 15 | Chest | — | R_Hand* |

*R_Hand is SMPL joint 23, stored at body_pose index 22.

### global_orient `(N, 1, 3, 3)`
Root pelvis orientation as a 3x3 rotation matrix.

### betas `(N, 10)`
Body shape coefficients (PCA basis). Controls height, weight, and body proportions.

### cam_t `(N, 3)`
Camera translation `[tx, ty, tz]` in meters.

### boxes `(N, 4)`
Detection bounding boxes `[x1, y1, x2, y2]`.

## Loading Parameters in Python

```python
import numpy as np

params = np.load('output/swing_params.npz')

body_pose = params['body_pose']      # (N, 23, 3, 3)
betas = params['betas']              # (N, 10)
global_orient = params['global_orient']  # (N, 1, 3, 3)
cam_t = params['cam_t']              # (N, 3)

# Example: Get left elbow rotation matrix
left_elbow = body_pose[0, 18]  # 3x3 rotation matrix
```

## Project Structure

```
golf/
├── src/
│   └── inference.py      # Main inference pipeline
├── 4D-Humans/            # HMR 2.0 submodule
├── data/                 # SMPL model files
├── input/                # Input images
├── output/               # Results
└── requirements.txt
```

## Memory Optimization

The pipeline uses YOLOv8-nano (~6MB) instead of ViTDet-H (~700MB) for detection, and implements sequential model loading with garbage collection between stages. This prevents memory exhaustion on systems with limited RAM.

## References

- [4D-Humans / HMR 2.0](https://github.com/shubham-goel/4D-Humans)
- [SMPL Body Model](https://smpl.is.tue.mpg.de/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
