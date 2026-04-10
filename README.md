# 3D Face Mesh Generation Pipeline

Converts a set of RGB images (< 200) into a clean, editable triangle mesh (OBJ/PLY) in under 5 minutes.

---

## Architecture

```
Images
  │
  ▼
[1] Feature Extraction & SfM  (COLMAP)
    • Blur filter — removes motion-blurred frames using center-crop Laplacian variance
    • SIFT feature extraction + adaptive matching strategy:
      - Exhaustive matching (all pairs) for ≤ 130 images — best quality
      - Sequential matching (adjacent frames) for > 130 images — stays within 5-minute budget
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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


Requires Python 3.9+.

### 2. COLMAP

```bash
# Ubuntu/Debian
sudo apt install colmap

# macOS
brew install colmap
```

> The pipeline uses CPU SIFT by default, so a CPU-only COLMAP build works fine.
> Tested with COLMAP 3.9.1 (apt install on Ubuntu 24.04).

---

## Run the pipeline

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

## Development environment

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| OS | Ubuntu 24.04, kernel 6.17 |
| Python | 3.10.20 |
| COLMAP | 3.9.1 (apt install, CPU SIFT) |
| open3d | 0.19.0 |
| opencv | 4.13.0 |
| trimesh | 4.11.5 |

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
