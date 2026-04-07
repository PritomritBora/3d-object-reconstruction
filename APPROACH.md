# Approach & Design Decisions

A breakdown of every major technical choice in the pipeline, why it was picked, and what the alternatives are.

---

## Stage 1 — Feature Extraction & Structure-from-Motion (SfM)

### What this stage does
Finds matching keypoints across images and solves for camera poses (where each photo was taken from). Output is a sparse 3D point cloud + calibrated cameras.

---

### Chosen: COLMAP

COLMAP is the de-facto standard for offline multi-view SfM. It uses SIFT keypoints, exhaustive or sequential matching, and a robust incremental reconstruction solver.

Why it was picked:
- Battle-tested on thousands of real-world datasets
- Handles wide baselines, textureless surfaces, and varying lighting better than most alternatives
- Produces metric-scale reconstructions (real-world units)
- GPU-accelerated SIFT via CUDA
- Fully offline, no weights to download

Tradeoffs:
- Requires a system binary install (not pip-installable)
- Slow on large image sets (100+ images can take 2–4 min for matching alone)
- Fails on low-texture objects (white walls, shiny surfaces)

---

### Alternatives considered

| Option | Pros | Cons | When to prefer |
|--------|------|------|----------------|
| **OpenMVG** | Lightweight, modular C++ | Less maintained, harder to install | Research/custom pipelines |
| **Meshroom (AliceVision)** | Full GUI + pipeline | Heavy dependency, not scriptable easily | Manual artist workflows |
| **GLOMAP** | Faster global SfM than COLMAP | Newer, less tested | Large unordered collections |
| **SuperPoint + SuperGlue** | Neural keypoints, better on low-texture | Requires GPU, slower per-pair | Challenging scenes, indoor |
| **LoFTR** | Dense matching, no keypoints needed | Very slow on many pairs | Textureless objects |
| **DUSt3R / MASt3R** | End-to-end pose + depth, no SfM solver | Large model, newer/less stable | When COLMAP fails entirely |

---

## Stage 2 — Dense Reconstruction

### What this stage does
Uses the camera poses from Stage 1 to compute a dense depth map for every image, then fuses them into a single 3D point cloud.

---

### Chosen: COLMAP Patch-Match Stereo + Stereo Fusion

COLMAP's MVS (Multi-View Stereo) runs patch-match on GPU to produce per-pixel depth maps, then fuses them into a consistent point cloud.

Why it was picked:
- Tightly integrated with COLMAP SfM output (no format conversion)
- GPU-accelerated, fast on consumer hardware
- Produces dense, accurate geometry with geometric consistency filtering
- Fully offline

Tradeoffs:
- Requires COLMAP binary
- Needs accurate camera poses from Stage 1 — if SfM fails, this fails too
- Memory-intensive on high-res images

---

### Fallback: MiDaS DPT-Large (monocular depth)

When COLMAP is unavailable or SfM fails, MiDaS estimates depth from each image independently using a transformer-based neural network.

Why it was picked as fallback:
- Works with zero camera pose information
- Runs entirely in PyTorch (pip-installable)
- DPT-Large is the best accuracy/speed tradeoff in the MiDaS family
- Weights auto-download via torch.hub

Tradeoffs:
- Depth is relative (no metric scale) — back-projection is approximate
- Per-image clouds don't align perfectly without poses
- Less accurate than multi-view stereo on well-captured datasets

---

### Alternatives considered

| Option | Pros | Cons | When to prefer |
|--------|------|------|----------------|
| **OpenMVS** | High quality MVS, open source | Separate install, extra format conversion | When COLMAP MVS is too slow |
| **MVSNet / CasMVSNet** | Neural MVS, good on low-texture | Needs camera poses, slower inference | Challenging textureless scenes |
| **UniDepth** | Metric monocular depth (no scale ambiguity) | Newer, less tested | When metric scale matters without multi-view |
| **Depth Anything v2** | Fast, strong generalization | Relative depth only | Speed-critical fallback |
| **ZoeDepth** | Metric monocular depth | Heavier model | Indoor scenes with known scale |
| **NeRF-based (Instant-NGP)** | Photorealistic, implicit geometry | Slow to train, hard to mesh | High-quality single-object capture |
| **3D Gaussian Splatting** | Very fast rendering | Meshing from splats is lossy | Visualization, not editing |

---

## Stage 3 — Surface Reconstruction (Point Cloud → Mesh)

### What this stage does
Converts the unstructured 3D point cloud into a clean triangle mesh with normals.

---

### Chosen: Poisson Surface Reconstruction (Open3D)

Poisson reconstruction fits a smooth implicit surface to the oriented point cloud by solving a Poisson equation. Open3D's implementation is fast and well-maintained.

Why it was picked:
- Produces watertight meshes by design — no holes from the algorithm itself
- Handles noisy, non-uniform point clouds well
- Smooth output, good for downstream editing
- Fast (seconds, not minutes)
- No neural weights, fully deterministic

Tradeoffs:
- Requires well-oriented normals — bad normals = bad mesh
- Can over-smooth fine details
- Adds geometry in areas with no point coverage (trimmed by density filter)

---

### Alternatives considered

| Option | Pros | Cons | When to prefer |
|--------|------|------|----------------|
| **Ball Pivoting (BPA)** | Preserves sharp features | Leaves holes, sensitive to radius param | Dense, clean point clouds |
| **Alpha Shapes** | Simple, exact on clean data | Holes on sparse data, slow on large clouds | Small, clean scans |
| **Marching Cubes** | Fast, parallelizable | Requires volumetric grid, staircase artifacts | Voxel-based pipelines |
| **TSDF Fusion (Open3D)** | Real-time capable, good for RGBD | Needs depth + poses aligned | RGBD camera input |
| **Neural (NeuS / VolSDF)** | Smooth, high-quality implicit surface | Slow to optimize per-scene | Offline, high-quality single object |
| **FlexiCubes** | Differentiable, editable | Research-stage, complex setup | Differentiable rendering pipelines |

---

## Stage 3b — Mesh Decimation

### Chosen: Quadric Error Metrics (Open3D `simplify_quadric_decimation`)

QEM decimation collapses edges while minimizing geometric error, preserving shape better than uniform subdivision removal.

Why it was picked:
- Industry standard for mesh simplification
- Fast, deterministic
- Built into Open3D — no extra dependency

Tradeoffs:
- Can collapse thin features (ears, fingers) aggressively
- No semantic awareness

### Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **Vertex clustering** | Faster | Lower quality |
| **Instant Meshes** | Produces quad-dominant remesh | External binary, slower |
| **PyMeshLab** | Many algorithms, scriptable | Extra install |

---

## Stage 3c — Hole Filling

### Chosen: trimesh `repair.fill_holes`

Simple boundary loop detection and ear-clipping fill. Handles the ≤ 2% bounding box hole constraint.

### Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **PyMeshFix** | Robust, handles complex topology | Extra C++ dependency |
| **MeshLab filters** | Many options | Not scriptable from Python easily |
| **Manual Poisson re-run at higher depth** | No extra dep | Slower, may over-smooth |

---

## Normal Estimation

### Chosen: PCA-based normals (Open3D) + consistent tangent plane orientation

Open3D estimates normals via PCA on local neighborhoods, then propagates consistent orientation using a minimum spanning tree on the tangent plane graph.

Why: fast, robust, no neural weights, works on any point cloud density.

### Alternatives

| Option | Pros | Cons |
|--------|------|------|
| **Jet fitting** | More accurate on curved surfaces | Slower |
| **Neural normals (PCPNet)** | Better on noisy data | Requires model weights |
| **From depth maps directly** | Exact normals | Only works if depth maps are available |

---

## Framework Choice: PyTorch

Required by the spec. MiDaS inference runs in PyTorch. COLMAP and Open3D are framework-agnostic C++ tools called via subprocess / Python bindings — they don't conflict with PyTorch and don't require it.

If the constraint were lifted, ONNX Runtime would be faster for MiDaS inference on CPU.

---

## Summary: Recommended Stack by Scenario

| Scenario | Stage 1 | Stage 2 | Stage 3 |
|----------|---------|---------|---------|
| Best quality, COLMAP available | COLMAP SIFT | COLMAP patch-match | Poisson + QEM |
| No COLMAP, GPU available | — | MiDaS DPT-Large | Poisson + QEM |
| Low-texture object | SuperPoint+SuperGlue | CasMVSNet | Poisson + QEM |
| RGBD camera input | — | TSDF Fusion | Marching Cubes |
| Maximum quality, no time limit | COLMAP / LoFTR | OpenMVS | NeuS |
