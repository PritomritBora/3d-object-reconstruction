# Pipeline Iteration Log

A chronological record of every setup we tried, what broke, and why we changed it.

---

## Iteration 1 — Initial Setup (MiDaS DPT_Large + Poisson)

### Architecture
```
Images → [COLMAP SfM or MiDaS fallback] → [Point cloud] → [Poisson] → OBJ/PLY
```

### Stage breakdown
- Stage 1: COLMAP SfM (exhaustive matcher) or MiDaS fallback (no poses)
- Stage 2: COLMAP dense MVS, or MiDaS DPT_Large monocular depth → back-projected point cloud
- Stage 3: Poisson surface reconstruction (Open3D, depth=9) → QEM decimation → hole fill

### MiDaS fallback approach
When COLMAP was unavailable, MiDaS DPT_Large estimated depth per image. Each depth map was back-projected into 3D and stacked with a `z += i * 0.01` offset to separate frames.

### Issues
- **System crash (OOM)**: `DPT_Large` is a ViT-L model requiring ~4–6 GB VRAM just to load. On 150 images it reliably ran out of memory.
- **Geometrically broken**: The `z += i * 0.01` offset produced a layered pancake of flat depth maps, not a real 3D object. No pose estimation was done, so all clouds were in camera space with no registration.
- **`dpt_transform`** resizes inputs to at least 384px minimum — on large images this amplified the OOM.

---

## Iteration 2 — MiDaS Small + OpenCV Pose Chain

### Changes from Iteration 1
- Replaced `DPT_Large` with `MiDaS_small` (~80 MB, fits in <2 GB VRAM)
- Added OpenCV SIFT + essential matrix chaining to estimate relative camera poses
- Back-projected each depth map into world space using the chained poses
- Added `torch.cuda.empty_cache()` every 20 frames
- Capped input resolution at 518px before depth network

### Issues
- **Still shattered geometry**: Essential matrix from `recoverPose` gives up-to-scale translation — the translation magnitude is arbitrary (unit vector). Each frame's depth was in a different metric scale, so the registered point cloud was still incoherent.
- **Drift**: Chaining relative poses accumulates error — by frame 50+ the poses were significantly drifted from ground truth.
- No COLMAP sparse points were used for scale reference.

---

## Iteration 3 — TSDF Fusion (first attempt)

### Architecture change
```
Images → COLMAP SfM (tuned) → Depth Anything V2 Small → Scale Align → TSDF → Mesh
```

### Key ideas
- Use COLMAP for accurate metric poses (solves the scale/drift problem)
- Use depth model only for dense depth per frame (not for poses)
- Scale-align each depth map to COLMAP sparse points before TSDF integration
- TSDF fusion (`ScalableTSDFVolume`) naturally produces watertight mesh

### COLMAP tuning
- Images resized to 1024px before COLMAP
- `--SiftExtraction.max_num_features 4096`
- `--SiftMatching.max_num_matches 32768`
- Sequential matcher for video frames (>30 images)
- CPU SIFT (`use_gpu 0`) to avoid COLMAP CUDA build issues

### Issues
- **COLMAP GPU build crash**: `SiftExtraction.use_gpu 1` failed because the installed COLMAP binary was not built with CUDA support. Fixed by setting `use_gpu 0`.
- **TSDF OOM crash at image 10/150**: `TSDF_VOXEL_SIZE=0.004` created ~125x too many voxels. Processing all 150 full-res images filled RAM.
  - Fix: voxel size `0.004` → `0.02`, subsample to 40 frames, resize depth to 320px before integration.
- **`create_from_depth_image` AttributeError**: The CUDA build of open3d (`open3d.cuda.pybind`) doesn't expose `create_from_depth_image`. Fixed by using `create_from_color_and_depth` with a zero-filled dummy color image, which exists on both CPU and CUDA builds.

---

## Iteration 4 — TSDF Fusion (scale alignment fix)

### Problem identified
The mesh output looked like shattered flat planes — completely unrecognisable.

### Root cause
MiDaS outputs **disparity** (inverse depth): high value = close, low value = far.
The scale alignment was doing:
```
s = dot(colmap_z, midas_value) / dot(midas_value, midas_value)
depth_metric = raw * s
```
This fits `s * midas_value ≈ colmap_depth`, but `midas_value` is disparity, not depth. The resulting `s` was meaningless — each frame got a different wrong scale, warping depth maps into random orientations.

### Fix attempted
- Invert MiDaS output first: `pseudo_depth = 1 / disparity`
- Fit affine `s * (1/d) + t ≈ colmap_depth` via least squares
- Adaptive voxel size from COLMAP bounding box diagonal / 200

### Issues
- Mesh still looked like shattered planes (Size X: 31, Size Y: 27, Size Z: 29 — room-scale, not object-scale)
- The affine fit was still unreliable because MiDaS small on a cluttered scene (sofa, table, decorative items in background) produces depth values dominated by the background, not the foreground object
- TSDF was integrating the entire room, not just the target object

---

## Iteration 5 — Drop TSDF, Use COLMAP Directly (current)

### Architecture
```
Images → COLMAP SfM → [Dense MVS if available] → [Sparse cloud fallback] → Poisson → OBJ/PLY
```

### Key insight
COLMAP already ran successfully: **150 images registered, 68,207 point observations, 31.7s**. The sparse `points3D.txt` contains metric-scale 3D points with known positions. There is no reason to run a depth model at all when COLMAP has already solved the geometry.

### Stage 2 — two-tier approach
1. **Try COLMAP dense MVS** (`patch_match_stereo` + `stereo_fusion`) — best quality dense point cloud
2. **Fallback: read `points3D.txt`** directly — already metric, already registered, no neural network needed

### Stage 3 — Poisson on point cloud
- Statistical outlier removal
- PCA normal estimation + consistent orientation
- Poisson reconstruction (depth=9)
- Density trim (bottom 5% removed)
- Largest component filter (removes debris < 5% of main component)
- Hole filling (trimesh)
- Light Laplacian smoothing (3 iterations)
- QEM decimation → 50K faces
- Vertex normal recompute + export

### Depth fallback (no COLMAP)
Kept for environments without COLMAP binary:
- MiDaS small disparity → inverted to pseudo-depth → median-normalised to 1m
- OpenCV SIFT pose chain for registration
- Back-projected point cloud → Poisson

---

## Summary Table

| Iteration | Depth Model | Poses | Fusion | Result |
|-----------|-------------|-------|--------|--------|
| 1 | MiDaS DPT_Large | None (z-offset hack) | Poisson | OOM crash |
| 2 | MiDaS small | OpenCV essential matrix | Poisson | Incoherent cloud (scale drift) |
| 3 | Depth Anything V2 / MiDaS small | COLMAP | TSDF | OOM → API error (fixed) |
| 4 | MiDaS small | COLMAP | TSDF | Shattered planes (disparity inversion bug) |
| 5 | None (COLMAP sparse/dense) | COLMAP | Poisson | Current — correct metric geometry |

---

## Key Lessons

- **MiDaS outputs disparity, not depth** — always invert before any metric operation
- **TSDF voxel size must match scene scale** — hardcoding 4mm for a room-scale scene = instant OOM
- **Don't use a depth model when you already have COLMAP** — COLMAP sparse points are metric, registered, and reliable; monocular depth adds noise and complexity for no gain
- **COLMAP CUDA build issues** — always default `use_gpu 0` unless you've verified the binary was built with CUDA
- **open3d CUDA build has a different API** — `create_from_depth_image` missing; use `create_from_color_and_depth` for cross-build compatibility
- **Essential matrix translation is up-to-scale** — chaining relative poses without metric scale reference produces drifted, incoherent point clouds
