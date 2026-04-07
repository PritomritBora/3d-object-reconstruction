# Real-Time 3D Mesh Generation Pipeline

Converts a set of RGB images (< 200) into a clean, editable triangle mesh (OBJ/PLY) in under 5 minutes on a consumer GPU.

---

## Architecture

```
Images
  │
  ▼
[1] Feature Extraction & SfM  (COLMAP)
    • Blur filter — removes motion-blurred frames using center-crop Laplacian variance
    • SIFT feature extraction + exhaustive/sequential matching
    • Incremental SfM → camera poses + sparse 3D point cloud
  │
  ▼
[2] Reconstruction  (COLMAP sparse cloud)
    • Track-length filter — keeps points seen in ≥ 20th percentile of views
      (object points appear in many views; background points appear in few)
    • Statistical outlier removal
  │
  ▼
[3] Meshing  (Open3D + trimesh)
    • Poisson surface reconstruction (adaptive depth 7–9)
    • Largest connected component filter
    • Hole filling (trimesh)
    • Laplacian smoothing (3 iterations)
    • Quadric decimation → ≤ 50K faces
    • Vertex normal computation
  │
  ▼
OBJ / PLY
```

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. CUDA recommended but not required.

### 2. COLMAP

```bash
# Ubuntu/Debian (CPU build — sufficient for SfM)
sudo apt install colmap

# Or via conda (includes CUDA support for optional dense MVS)
conda install -c conda-forge colmap
```

> The pipeline uses CPU SIFT by default (`FeatureExtraction.use_gpu 0`), so a CPU-only COLMAP build works fine.

---

## Usage

```bash
python run.py --input ./images --output mesh.obj
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Directory of RGB images (.jpg / .JPG / .png) |
| `--output` | required | Output mesh path (.obj or .ply) |
| `--max-faces` | 50000 | Max triangles after decimation |
| `--work-dir` | `<output>_workspace` | Intermediate file directory |
| `--keep-workspace` | off | Keep intermediate files after completion |

---

## Capture Guidelines

Mesh quality depends heavily on input quality.

| Factor | Recommendation |
|--------|---------------|
| Distance | 30–60 cm from object |
| Coverage | Full 360° orbit at 2–3 height levels |
| Speed | Move slowly — 30–40s per full orbit |
| Lighting | Bright, diffuse (avoid harsh shadows or direct sunlight) |
| Background | Plain or dark background preferred |
| Frame count | 50–120 frames |
| Sharpness | Center-crop Laplacian variance > 50 (checked automatically) |

Check your capture sharpness before running:

```bash
python3 -c "
import cv2, numpy as np, os, sys
folder = sys.argv[1]
scores = []
for f in os.listdir(folder):
    img = cv2.imread(f'{folder}/{f}', 0)
    if img is not None:
        h, w = img.shape
        scores.append(cv2.Laplacian(img[h//4:3*h//4, w//4:3*w//4], cv2.CV_64F).var())
print(f'Frames: {len(scores)}  Median sharpness: {np.median(scores):.0f}  (target: >50)')
" ./images
```

---

## Timing Benchmark

Measured on RTX 3060, Buddha head dataset (67 images, 2736×1080):

| Stage | Time |
|-------|------|
| Feature extraction / SfM | ~44s |
| Reconstruction (sparse cloud) | ~3s |
| Meshing (Poisson + cleanup) | ~2s |
| **Total** | **~49s** |

---

## Output

- Triangle mesh with vertex normals
- ≤ 50K faces after quadric decimation
- Degenerate triangles removed
- Largest connected component kept
- Small holes filled (requires `trimesh`)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `open3d` | Normal estimation, Poisson reconstruction, decimation, export |
| `opencv-python` | Image loading, blur detection, resizing |
| `trimesh` | Hole filling |
| `torch` + `torchvision` | Available for optional depth model extensions |
| `colmap` (binary) | SfM — must be installed separately (see Setup) |

---

## Troubleshooting

**Few images registered (< 50% of input)**
Check sharpness scores. If median < 20, the video has too much motion blur — move the camera more slowly.

**Empty or very small mesh**
COLMAP needs texture to find keypoints. Dark matte objects (black plastic, metal) are challenging. Use brighter lighting or place the object on a textured surface.

**COLMAP not found**
Install via `sudo apt install colmap` (Ubuntu) or `brew install colmap` (macOS).

**OOM during meshing**
Reduce `--max-faces` or the point cloud will be downsampled more aggressively before Poisson.
