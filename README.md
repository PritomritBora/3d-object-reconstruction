# Real-Time 3D Mesh Generation Pipeline

Converts a set of RGB images (< 200) into a clean, editable triangle mesh (OBJ/PLY) in under 5 minutes on a consumer GPU.

---

## Architecture

```
Images → [Feature Extraction / SfM] → [Dense Reconstruction] → [Meshing] → OBJ/PLY
```

Two execution paths depending on environment:

| Path | Feature Extraction | Dense Reconstruction | When used |
|------|--------------------|----------------------|-----------|
| **COLMAP** (preferred) | SIFT + exhaustive/sequential matching | Patch-match stereo + fusion | COLMAP binary in PATH |
| **Depth Anything V2** (fallback) | OpenCV SIFT + essential matrix pose chain | Depth Anything V2 Small monocular depth | No COLMAP |
| **MiDaS small** (secondary fallback) | OpenCV SIFT + essential matrix pose chain | MiDaS small monocular depth | No COLMAP, no transformers |

Meshing is always Poisson surface reconstruction via Open3D, followed by decimation to ≤ 50K faces.

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch with CUDA (recommended).

### 2. COLMAP (recommended for best quality)

```bash
# Ubuntu/Debian
sudo apt install colmap

# macOS
brew install colmap

# Or build from source: https://colmap.github.io/install.html
```

If COLMAP is not available, the pipeline automatically falls back to MiDaS monocular depth estimation.

### 3. Pre-trained weights

- **Depth Anything V2 Small** — downloaded automatically on first run via HuggingFace hub (~100 MB, cached in `~/.cache/huggingface/`). Much lighter than DPT-Large.
- **MiDaS small** — secondary fallback, downloaded via `torch.hub` (~80 MB). Used if `transformers` is unavailable.
- **COLMAP** — no neural weights needed; uses classical SIFT.

No manual weight downloads required.

---

## Usage

```bash
python run.py --input ./images --output mesh.obj
```

### All options

```
--input        Directory of input RGB images (.jpg / .jpeg / .png)
--output       Output mesh path (.obj or .ply)
--max-faces    Max faces after decimation (default: 50000)
--depth-method colmap | midas  (default: colmap, auto-falls back to midas)
--work-dir     Custom working directory for intermediate files
--keep-workspace  Keep intermediate files after completion
```

### Examples

```bash
# Basic usage
python run.py --input ./images --output mesh.obj

# Force MiDaS path (no COLMAP)
python run.py --input ./images --output mesh.ply --depth-method midas

# Keep intermediates for inspection
python run.py --input ./images --output mesh.obj --keep-workspace
```

---

## Timing Benchmark

Measured on RTX 3060 (12 GB), 50 images at 1920×1080:

| Stage | COLMAP path | Depth Anything V2 path |
|-------|-------------|------------------------|
| Feature extraction / SfM | ~45s | ~5s (OpenCV pose chain) |
| Dense reconstruction | ~90s | ~45s (150 images, Small model) |
| Meshing + decimation | ~20s | ~20s |
| **Total** | **~2.5 min** | **~1.2 min** |

Timing is logged to stdout at the end of every run.

---

## Output Quality

- Triangle mesh with vertex normals
- Watertight preferred (Poisson reconstruction)
- Degenerate triangles removed
- Holes ≤ 2% of bounding box filled (requires `trimesh`)
- ≤ 50K faces after quadric decimation

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Depth model inference |
| `open3d` | Normal estimation, Poisson reconstruction, decimation, export |
| `opencv-python` | Image loading, SIFT feature matching, pose estimation |
| `transformers` | Depth Anything V2 Small (primary fallback depth model) |
| `Pillow` | Image format support for transformers pipeline |
| `trimesh` | Hole filling (optional but recommended) |
| `scikit-learn` | Statistical outlier removal |
| `colmap` (binary) | SfM + MVS (optional, recommended) |

---

## Troubleshooting

**CUDA out of memory** — the depth model is already using the small variant (~100MB). If you still OOM, force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python run.py --input ./images --output mesh.obj --depth-method midas
```

Or reduce image count:
```bash
python run.py --input ./images --output mesh.obj --depth-method midas
```
(use a subset of images by copying fewer into the input dir)

**Depth Anything V2 download fails** — pre-download manually:
```bash
python -c "from transformers import pipeline; pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf')"
```

**MiDaS fallback download fails** — pre-download manually:
```bash
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)"
```

**Empty mesh** — check that images have sufficient overlap (≥ 60% recommended) and consistent lighting.
