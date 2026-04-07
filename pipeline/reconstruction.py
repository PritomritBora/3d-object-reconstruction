"""
Stage 2 — Dense reconstruction → point cloud.

Primary path:  COLMAP dense MVS (patch_match_stereo + stereo_fusion)
               Requires CUDA-enabled COLMAP binary.
               Produces a dense PLY point cloud (~100K–1M points).

Fallback path: COLMAP sparse point cloud (points3D.txt)
               Always available when COLMAP SfM succeeds.
               Produces a sparser PLY (~1K–10K points) but is fast and reliable.
"""
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class Reconstructor:
    def __init__(self, work_dir: Path, method: str = "colmap"):
        self.work_dir = work_dir
        self.method = method
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, sfm_result) -> Optional[Path]:
        """
        Run reconstruction. Returns path to a PLY point cloud or None on failure.
        Uses COLMAP sparse point cloud from SfM — fast, reliable, works on any hardware.
        """
        return self._sparse_to_ply(sfm_result)

    # ── COLMAP dense MVS (optional, requires CUDA COLMAP) ─────────
    def _colmap_dense_mvs(self, sfm_result) -> Optional[Path]:
        """
        EXPERIMENTAL — not used by default.

        Dense reconstruction via COLMAP patch-match stereo + stereo fusion.
        Requires a CUDA-enabled COLMAP binary (conda install -c conda-forge colmap).

        Results depend heavily on the point cloud bbox scale — if the scene
        includes background, the voxel downsample in meshing destroys detail.
        To enable: replace `_sparse_to_ply` with this method in `run()`.
        """
        dense_dir = self.work_dir / "dense"
        if dense_dir.exists():
            shutil.rmtree(dense_dir)
        dense_dir.mkdir(parents=True, exist_ok=True)
        fused_ply = self.work_dir / "fused.ply"

        # Use the binary sparse model (not TXT) for undistortion
        binary_model = sfm_result.binary_sparse_dir or sfm_result.sparse_dir

        log.info("COLMAP: undistorting images...")
        if self._cmd([
            "image_undistorter",
            "--image_path", str(sfm_result.images_dir),
            "--input_path", str(binary_model),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
            "--max_image_size", "800",
        ]) != 0:
            log.warning("image_undistorter failed.")
            return None

        log.info("COLMAP: patch_match_stereo (GPU)...")
        if self._cmd([
            "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
            "--PatchMatchStereo.gpu_index", "0",
            "--PatchMatchStereo.num_samples", "5",
            "--PatchMatchStereo.num_iterations", "2",
            "--PatchMatchStereo.max_image_size", "640",
        ]) != 0:
            log.warning("patch_match_stereo failed.")
            return None

        log.info("COLMAP: stereo_fusion...")
        if self._cmd([
            "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(fused_ply),
            "--StereoFusion.min_num_pixels", "3",
            "--StereoFusion.max_reproj_error", "2",
        ]) != 0:
            log.warning("stereo_fusion failed.")
            return None

        log.info(f"Dense MVS complete: {fused_ply}")
        return fused_ply

    # ── Sparse cloud fallback ──────────────────────────────────────
    def _sparse_to_ply(self, sfm_result) -> Optional[Path]:
        """
        Convert COLMAP sparse points3D.txt → a cleaned PLY point cloud.

        Applies two filters:
          1. Track-length filter: keeps only points seen in >= 20th percentile
             of views. Object points appear in many views; background points
             appear in few — this naturally separates them.
          2. Statistical outlier removal: removes isolated noise points.
        """
        points_file = sfm_result.sparse_dir / "points3D.txt"
        if not points_file.exists():
            log.error(f"points3D.txt not found: {points_file}")
            return None

        pts, colors, track_lengths = [], [], []
        with open(points_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
                # Track length = number of images that observe this point
                track_lengths.append((len(parts) - 8) // 2)

        if not pts:
            log.error("No points in points3D.txt")
            return None

        pts = np.array(pts, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)
        track_lengths = np.array(track_lengths, dtype=np.int32)

        log.info(f"Sparse cloud: {len(pts):,} points "
                 f"(track_len: min={track_lengths.min()} "
                 f"median={int(np.median(track_lengths))} "
                 f"max={track_lengths.max()})")

        # Track-length filter — adaptive percentile based on track length spread.
        # High spread (max/median > 4) means object points have much longer tracks
        # than background — use a tighter filter. Low spread = uniform scene, be lenient.
        spread = track_lengths.max() / max(np.median(track_lengths), 1)
        if spread > 8:
            percentile = 40  # tight: object clearly dominates track lengths
        else:
            percentile = 20  # lenient: uniform scene, don't over-filter
        min_track = max(2, int(np.percentile(track_lengths, percentile)))
        log.info(f"Track filter: spread={spread:.1f}× → percentile={percentile} min_track={min_track}")
        mask = track_lengths >= min_track
        if mask.sum() >= 50:
            pts, colors = pts[mask], colors[mask]
        log.info(f"After track filter (min_track={min_track}): {len(pts):,} points")

        # Foreground depth filter — keep the nearest cluster of points.
        # Uses camera-space Z from all registered poses to find the closest
        # dense cluster, then discards points beyond 2× that distance.
        # This removes background (fence, wall) when object is in foreground.
        pts, colors = self._foreground_depth_filter(pts, colors, sfm_result.poses)

        # Statistical outlier removal
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        log.info(f"After outlier removal: {len(pcd.points):,} points")

        out_ply = self.work_dir / "sparse.ply"
        o3d.io.write_point_cloud(str(out_ply), pcd, write_ascii=False)
        log.info(f"Saved → {out_ply}")
        return out_ply

    def _foreground_depth_filter(self, pts, colors, poses):
        """
        Keep only points in the nearest depth cluster.

        Projects all sparse points into camera space across all registered views,
        computes the median camera-space Z, then keeps only points within
        [0, median_z * 1.5]. This discards background points (fence, wall)
        when the object is closer to the camera than the background.

        Falls back to returning all points if poses are unavailable.
        """
        if not poses:
            return pts, colors

        all_z = []
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])

        for img_name, (T_wc, K) in list(poses.items())[:20]:  # sample 20 views
            T_cw = np.linalg.inv(T_wc)
            pts_cam = (T_cw @ pts_h.T).T
            zs = pts_cam[:, 2]
            all_z.append(zs[zs > 0.01])

        if not all_z:
            return pts, colors

        all_z = np.concatenate(all_z)
        # 20th percentile = nearest significant cluster (foreground object)
        near_z = float(np.percentile(all_z, 20))
        far_z = float(np.percentile(all_z, 80))

        # Only apply depth filter if there's a clear foreground/background separation
        # (background is >2× further than foreground)
        log.info(f"Depth analysis: near_z={near_z:.2f}m far_z={far_z:.2f}m ratio={far_z/near_z:.1f}×")
        if far_z < near_z * 2.0:
            log.info("Depth range small — skipping depth filter (object fills frame)")
            return pts, colors

        # Keep points within 2× the near depth — removes distant background
        depth_limit = near_z * 2.0

        # Compute per-point median depth across all sampled views
        per_point_z = np.zeros(len(pts), dtype=np.float64)
        count = np.zeros(len(pts), dtype=np.int32)
        for img_name, (T_wc, K) in list(poses.items())[:20]:
            T_cw = np.linalg.inv(T_wc)
            pts_cam = (T_cw @ pts_h.T).T
            zs = pts_cam[:, 2]
            valid = zs > 0.01
            per_point_z[valid] += zs[valid]
            count[valid] += 1

        count = np.maximum(count, 1)
        mean_z = per_point_z / count
        mask = (mean_z > 0.01) & (mean_z <= depth_limit)

        if mask.sum() < 50:
            log.warning("Foreground depth filter too aggressive, skipping.")
            return pts, colors

        log.info(f"Foreground depth filter (limit={depth_limit:.2f}m): "
                 f"{mask.sum():,}/{len(pts):,} points kept")
        return pts[mask], colors[mask]

    # ── Helpers ────────────────────────────────────────────────────
    def _cmd(self, args: list) -> int:
        """Run a COLMAP command, return exit code. Logs stderr on failure."""
        result = subprocess.run(["colmap"] + args, capture_output=True, text=True)
        if result.returncode != 0:
            log.debug(result.stderr[-2000:])
        return result.returncode
