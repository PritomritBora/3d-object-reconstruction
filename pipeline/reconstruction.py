"""
Stage 2 — Dense reconstruction → point cloud.

Primary:  COLMAP patch_match_stereo + stereo_fusion (CPU mode, no GPU required)
Fallback: Multi-view depth fusion — run depth model on every frame, back-project
          to world space, then keep only points that are consistent across
          multiple views (geometric consistency filter). No manual depth windows.
"""
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

DEPTH_MODEL_HF    = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_MIDAS = "MiDaS_small"
MAX_DEPTH_SIDE    = 518
MAX_FRAMES        = 30
PROJ_STEP         = 4     # pixel subsampling for back-projection
MAX_PTS_PER_FRAME = 5000
CONSISTENCY_K     = 8     # object surface needs more neighbours than background noise


class Reconstructor:
    def __init__(self, work_dir: Path, method: str = "colmap"):
        self.work_dir = work_dir
        self.method = method
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, sfm_result) -> Optional[Path]:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Reconstruction device: {device}")
        return self._sparse_to_ply(sfm_result)

    def _sparse_to_ply(self, sfm_result) -> Optional[Path]:
        """COLMAP sparse points → cleaned PLY. Simple and reliable."""
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
                # Track length = number of images that see this point
                # Stored as pairs after index 8: image_id feature_id ...
                track_len = (len(parts) - 8) // 2
                track_lengths.append(track_len)

        if not pts:
            log.error("No points in points3D.txt")
            return None

        pts = np.array(pts, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)
        track_lengths = np.array(track_lengths, dtype=np.int32)

        log.info(f"Sparse cloud: {len(pts):,} points "
                 f"(track_len: min={track_lengths.min()} median={int(np.median(track_lengths))} max={track_lengths.max()})")

        # Filter by track length — points seen in more views are more reliable
        # Background points typically have short tracks (seen in 2-3 views)
        # Object points have longer tracks (seen in many views as camera orbits)
        min_track = max(3, int(np.percentile(track_lengths, 25)))
        pts = pts[track_lengths >= min_track]
        colors = colors[track_lengths >= min_track]
        log.info(f"After track filter (min_track={min_track}): {len(pts):,} points")

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

    # ── TSDF fusion → mesh ─────────────────────────────────────────
    def _tsdf_fusion(self, sfm_result, depth_fn, device) -> Optional[Path]:
        import cv2, torch
        import open3d as o3d

        poses = sfm_result.poses
        sparse_pts_per_img = sfm_result.sparse_points

        # Adaptive voxel size from sparse cloud
        all_sparse = []
        for pts in sparse_pts_per_img.values():
            all_sparse.append(pts)
        if all_sparse:
            all_pts = np.concatenate(all_sparse, axis=0)
            diag = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0)))
            voxel = float(np.clip(diag / 150.0, 0.003, 0.03))
        else:
            voxel = 0.01
        sdf_trunc = voxel * 4.0
        log.info(f"TSDF voxel={voxel:.4f}m sdf_trunc={sdf_trunc:.4f}m")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

        image_paths = sfm_result.image_paths
        if len(image_paths) > MAX_FRAMES:
            step = len(image_paths) // MAX_FRAMES
            image_paths = image_paths[::step][:MAX_FRAMES]
            log.info(f"Subsampled to {len(image_paths)} frames")

        integrated = 0
        log.info(f"TSDF fusion: {len(image_paths)} frames...")

        with torch.no_grad():
            for i, img_path in enumerate(image_paths):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]

                img_name = img_path.name
                if img_name not in poses:
                    continue
                T_wc, K = poses[img_name]

                scale = min(1.0, MAX_DEPTH_SIDE / max(h, w))
                small = cv2.resize(img_rgb, (int(w*scale), int(h*scale))) if scale < 1.0 else img_rgb
                raw = depth_fn(small)
                if raw.shape[:2] != (h, w):
                    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)

                depth = self._align_depth(raw, sparse_pts_per_img.get(img_name),
                                          T_wc, K, h, w)
                if depth is None:
                    continue

                # Resize depth for TSDF (saves RAM)
                tsdf_scale = min(1.0, 320 / max(h, w))
                th, tw = int(h * tsdf_scale), int(w * tsdf_scale)
                depth_small = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_LINEAR)
                K_small = K.copy()
                K_small[0] *= tsdf_scale
                K_small[1] *= tsdf_scale

                depth_f32 = np.clip(depth_small, 0.0, 20.0).astype(np.float32)
                o3d_depth = o3d.geometry.Image(depth_f32)
                o3d_color = o3d.geometry.Image(np.zeros((th, tw, 3), dtype=np.uint8))

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color, o3d_depth,
                    depth_scale=1.0, depth_trunc=20.0,
                    convert_rgb_to_intensity=False,
                )

                fx, fy = K_small[0,0], K_small[1,1]
                cx, cy = K_small[0,2], K_small[1,2]
                intrinsic = o3d.camera.PinholeCameraIntrinsic(tw, th, fx, fy, cx, cy)
                T_cw = np.linalg.inv(T_wc)
                volume.integrate(rgbd, intrinsic, T_cw)
                integrated += 1

                if (i+1) % 5 == 0:
                    log.info(f"  Integrated {i+1}/{len(image_paths)}")
                if device.type == "cuda" and (i+1) % 10 == 0:
                    torch.cuda.empty_cache()

        if integrated == 0:
            log.error("No frames integrated.")
            return None

        log.info(f"Extracting mesh from TSDF ({integrated} frames)...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        n = len(mesh.triangles)
        log.info(f"TSDF mesh: {n:,} faces")

        if n == 0:
            return None

        out_ply = self.work_dir / "tsdf_mesh.ply"
        o3d.io.write_triangle_mesh(str(out_ply), mesh, write_vertex_normals=True)
        return out_ply
        """Load COLMAP sparse points, return (pts, colors) or (None, None)."""
        points_file = sfm_result.sparse_dir / "points3D.txt"
        if not points_file.exists():
            return None, None
        pts, colors = [], []
        with open(points_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
        if not pts:
            return None, None
        pts = np.array(pts, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)
        pts, colors = self._remove_outliers_colored(pts, colors)
        return pts, colors

    def _depth_fusion_pts(self, sfm_result, depth_fn, seg_fn, device,
                          sparse_pts: np.ndarray) -> Optional[np.ndarray]:
        """
        Run depth fusion but crop strictly to sparse point cloud bbox + small margin.
        This gives dense coverage of the object without room-scale noise.
        """
        import cv2, torch

        # Tight crop box from sparse points
        margin = 0.15  # 15cm margin around sparse bbox
        lo = sparse_pts.min(axis=0) - margin
        hi = sparse_pts.max(axis=0) + margin

        image_paths = sfm_result.image_paths
        poses = sfm_result.poses
        sparse_pts_per_img = sfm_result.sparse_points

        if len(image_paths) > MAX_FRAMES:
            step = len(image_paths) // MAX_FRAMES
            image_paths = image_paths[::step][:MAX_FRAMES]

        all_pts = []
        log.info(f"Depth fusion densification on {len(image_paths)} frames "
                 f"(crop box: {(hi-lo).round(3)}m)...")

        with torch.no_grad():
            for i, img_path in enumerate(image_paths):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]

                img_name = img_path.name
                if img_name not in poses:
                    continue
                T_wc, K = poses[img_name]

                # Foreground mask
                fg_mask = None
                if seg_fn is not None:
                    try:
                        fg_mask = seg_fn(img_rgb)
                        if fg_mask.shape != (h, w):
                            fg_mask = cv2.resize(fg_mask.astype(np.uint8), (w, h),
                                                 interpolation=cv2.INTER_NEAREST).astype(bool)
                    except Exception:
                        fg_mask = None

                scale = min(1.0, MAX_DEPTH_SIDE / max(h, w))
                small = cv2.resize(img_rgb, (int(w*scale), int(h*scale))) if scale < 1.0 else img_rgb
                raw = depth_fn(small)
                if raw.shape[:2] != (h, w):
                    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)

                depth = self._align_depth(raw, sparse_pts_per_img.get(img_name),
                                          T_wc, K, h, w, fg_mask)
                if depth is None:
                    continue

                if fg_mask is not None:
                    depth = np.where(fg_mask, depth, 0.0)

                pts = self._backproject(depth, T_wc, K, h, w)
                if pts is None or len(pts) == 0:
                    continue

                # Crop to sparse bbox
                in_box = np.all((pts >= lo) & (pts <= hi), axis=1)
                pts = pts[in_box]
                if len(pts) == 0:
                    continue

                if len(pts) > MAX_PTS_PER_FRAME:
                    idx = np.random.choice(len(pts), MAX_PTS_PER_FRAME, replace=False)
                    pts = pts[idx]

                all_pts.append(pts)

                if device.type == "cuda" and (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()

        if not all_pts:
            return None

        merged = np.concatenate(all_pts, axis=0)
        log.info(f"Dense pts before filter: {len(merged):,}")
        merged = self._consistency_filter(merged)
        log.info(f"Dense pts after filter: {len(merged):,}")
        return merged
        """
        Convert COLMAP sparse points3D.txt → cleaned PLY.
        Works best when object fills the frame (close-up capture).
        """
        points_file = sfm_result.sparse_dir / "points3D.txt"
        if not points_file.exists():
            log.warning("points3D.txt not found.")
            return None

        pts, colors = [], []
        with open(points_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([int(parts[4]), int(parts[5]), int(parts[6])])

        if not pts:
            return None

        pts = np.array(pts, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)
        log.info(f"Sparse cloud: {len(pts):,} points")

        # Statistical outlier removal
        pts, colors = self._remove_outliers_colored(pts, colors)
        log.info(f"After outlier removal: {len(pts):,} points")

        out_ply = self.work_dir / "sparse.ply"
        self._save_ply_colored(pts, colors, out_ply)
        return out_ply

    def _remove_outliers_colored(self, pts, colors, k=20, std_ratio=2.0):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd, idx = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
        return np.asarray(pcd.points).astype(np.float32), colors[idx]

    def _save_ply_colored(self, pts, colors, path):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
        log.info(f"Saved {len(pts):,} points → {path}")
    def _get_fg_sparse_pts(self, sfm_result, seg_fn) -> Optional[np.ndarray]:
        """
        Returns foreground sparse points in world space, or None.
        Used to crop the depth fusion cloud to the object region.
        """
        points_file = sfm_result.sparse_dir / "points3D.txt"
        if not points_file.exists():
            return None

        import cv2
        poses = sfm_result.poses

        pts3d = {}
        with open(points_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                pid = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                pts3d[pid] = np.array([x, y, z], dtype=np.float32)

        if not pts3d or seg_fn is None:
            return None

        image_paths = sfm_result.image_paths
        step = max(1, len(image_paths) // 20)
        sample_imgs = image_paths[::step][:20]

        masks = {}
        for img_path in sample_imgs:
            img_name = img_path.name
            if img_name not in poses:
                continue
            T_wc, K = poses[img_name]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            try:
                fg = seg_fn(img_rgb)
                if fg.shape != (h, w):
                    fg = cv2.resize(fg.astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                masks[img_name] = (fg, K, np.linalg.inv(T_wc), h, w)
            except Exception:
                continue

        if not masks:
            return None

        pts_array = np.array(list(pts3d.values()), dtype=np.float64)
        pts_h = np.hstack([pts_array, np.ones((len(pts_array), 1))])
        fg_votes = np.zeros(len(pts_array), dtype=np.int32)
        total_votes = np.zeros(len(pts_array), dtype=np.int32)

        for img_name, (fg_mask, K, T_cw, h, w) in masks.items():
            pts_cam = (T_cw @ pts_h.T).T
            zs = pts_cam[:, 2]
            visible = zs > 0.01
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            us = (pts_cam[:, 0] * fx / (zs + 1e-8) + cx).astype(int)
            vs = (pts_cam[:, 1] * fy / (zs + 1e-8) + cy).astype(int)
            in_frame = visible & (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
            total_votes[in_frame] += 1
            idx = np.where(in_frame)[0]
            if len(idx) > 0:
                vs_c = vs[idx].clip(0, h-1)
                us_c = us[idx].clip(0, w-1)
                in_fg = fg_mask[vs_c, us_c]
                fg_votes[idx[in_fg]] += 1

        keep = (total_votes > 0) & (fg_votes / (total_votes + 1e-8) >= 0.3)
        fg_pts = pts_array[keep].astype(np.float32)

        # Cluster around centroid to remove stray background points
        if len(fg_pts) > 10:
            centroid = fg_pts.mean(axis=0)
            dists = np.linalg.norm(fg_pts - centroid, axis=1)
            mean_d, std_d = dists.mean(), dists.std()
            tight = fg_pts[dists < mean_d + 1.5 * std_d]
            if len(tight) >= 10:
                fg_pts = tight

        log.info(f"Foreground sparse anchor: {len(fg_pts):,} points "
                 f"bbox={np.linalg.norm(fg_pts.max(axis=0)-fg_pts.min(axis=0)):.3f}m")
        return fg_pts if len(fg_pts) >= 10 else None
    def _sparse_foreground(self, sfm_result, seg_fn) -> Optional[Path]:
        """
        Use COLMAP sparse 3D points directly, filtered to foreground.
        For each sparse point, project it into every image it's visible in,
        check if it lands in the rembg foreground mask, and keep it only if
        it's foreground in the majority of views.
        This gives a clean, metric-scale point cloud of just the object.
        """
        import cv2

        points_file = sfm_result.sparse_dir / "points3D.txt"
        if not points_file.exists():
            log.warning("points3D.txt not found, skipping sparse foreground.")
            return None

        poses = sfm_result.poses  # name -> (T_wc, K)

        # Parse all 3D points
        pts3d = {}  # point_id -> (xyz, [image_names])
        with open(points_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                pid = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                # track: pairs of (image_id, feature_id) — we need image names
                # stored in the TRACK field starting at index 8
                pts3d[pid] = np.array([x, y, z], dtype=np.float32)

        if not pts3d:
            log.warning("No 3D points found.")
            return None

        log.info(f"Sparse cloud: {len(pts3d):,} points — filtering to foreground...")

        if seg_fn is None:
            # No segmenter — just use all sparse points
            pts = np.array(list(pts3d.values()), dtype=np.float32)
            pts = self._remove_outliers(pts)
            out_ply = self.work_dir / "sparse_fg.ply"
            self._save_ply(pts, out_ply)
            log.info(f"Sparse cloud (no mask): {len(pts):,} points")
            return out_ply

        # Build foreground masks for a subset of images
        # Sample up to 20 images evenly
        image_paths = sfm_result.image_paths
        step = max(1, len(image_paths) // 20)
        sample_imgs = image_paths[::step][:20]

        masks = {}  # image_name -> (fg_mask, K, T_cw, h, w)
        for img_path in sample_imgs:
            img_name = img_path.name
            if img_name not in poses:
                continue
            T_wc, K = poses[img_name]
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            try:
                fg = seg_fn(img_rgb)
                if fg.shape != (h, w):
                    fg = cv2.resize(fg.astype(np.uint8), (w, h),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                T_cw = np.linalg.inv(T_wc)
                masks[img_name] = (fg, K, T_cw, h, w)
            except Exception:
                continue

        if not masks:
            log.warning("No masks computed, using full sparse cloud.")
            pts = np.array(list(pts3d.values()), dtype=np.float32)
            pts = self._remove_outliers(pts)
            out_ply = self.work_dir / "sparse_fg.ply"
            self._save_ply(pts, out_ply)
            return out_ply

        # For each 3D point, check how many views see it as foreground
        pts_array = np.array(list(pts3d.values()), dtype=np.float64)  # (N, 3)
        pts_h = np.hstack([pts_array, np.ones((len(pts_array), 1))])   # (N, 4)

        fg_votes = np.zeros(len(pts_array), dtype=np.int32)
        total_votes = np.zeros(len(pts_array), dtype=np.int32)

        for img_name, (fg_mask, K, T_cw, h, w) in masks.items():
            # Project all points into this image
            pts_cam = (T_cw @ pts_h.T).T  # (N, 4)
            zs = pts_cam[:, 2]
            visible = zs > 0.01

            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            us = (pts_cam[:, 0] * fx / (zs + 1e-8) + cx).astype(int)
            vs = (pts_cam[:, 1] * fy / (zs + 1e-8) + cy).astype(int)

            in_frame = visible & (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
            total_votes[in_frame] += 1

            # Vectorised foreground check
            idx = np.where(in_frame)[0]
            if len(idx) > 0:
                vs_c = vs[idx].clip(0, h-1)
                us_c = us[idx].clip(0, w-1)
                in_fg = fg_mask[vs_c, us_c]
                fg_votes[idx[in_fg]] += 1

        # Keep points that are foreground in >50% of views they appear in
        keep = (total_votes > 0) & (fg_votes / (total_votes + 1e-8) >= 0.3)
        fg_pts = pts_array[keep].astype(np.float32)

        log.info(f"Foreground sparse points: {keep.sum():,} / {len(pts_array):,} "
                 f"(total_votes>0: {(total_votes>0).sum():,}, "
                 f"max_fg_ratio: {(fg_votes/(total_votes+1e-8)).max():.3f})")

        if len(fg_pts) < 50:
            log.warning(f"Too few foreground sparse points ({len(fg_pts)}), using full cloud.")
            fg_pts = pts_array.astype(np.float32)

        fg_pts = self._remove_outliers(fg_pts)
        log.info(f"After outlier removal: {len(fg_pts):,} points")

        out_ply = self.work_dir / "sparse_fg.ply"
        self._save_ply(fg_pts, out_ply)
        return out_ply

    # ── COLMAP dense MVS (kept for reference, requires CUDA COLMAP) ─
    def _colmap_dense_mvs(self, sfm_result) -> Optional[Path]:
        dense_dir = self.work_dir / "dense"
        if dense_dir.exists():
            shutil.rmtree(dense_dir)
        dense_dir.mkdir(parents=True, exist_ok=True)
        fused_ply = self.work_dir / "fused.ply"

        log.info("COLMAP: undistorting images...")
        ret = self._cmd([
            "image_undistorter",
            "--image_path", str(sfm_result.images_dir),
            "--input_path", str(sfm_result.sparse_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
            "--max_image_size", "800",
        ])
        if ret != 0:
            log.warning("image_undistorter failed.")
            return None

        log.info("COLMAP: patch_match_stereo (CPU)...")
        ret = self._cmd([
            "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
            "--PatchMatchStereo.gpu_index", "-1",
            "--PatchMatchStereo.num_samples", "5",
            "--PatchMatchStereo.num_iterations", "3",
            "--PatchMatchStereo.max_image_size", "800",
        ])
        if ret != 0:
            log.warning("patch_match_stereo failed.")
            return None

        log.info("COLMAP: stereo_fusion...")
        ret = self._cmd([
            "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(fused_ply),
            "--StereoFusion.min_num_pixels", "3",
            "--StereoFusion.max_reproj_error", "2",
        ])
        if ret != 0:
            log.warning("stereo_fusion failed.")
            return None

        log.info(f"COLMAP dense MVS complete: {fused_ply}")
        return fused_ply

    # ── Background segmenter ───────────────────────────────────────
    def _load_segmenter(self):
        """
        Load rembg for background removal.
        Returns a callable: np.ndarray (H,W,3 uint8) -> np.ndarray (H,W bool mask)
        where True = foreground.
        Falls back to None if rembg not installed.
        """
        try:
            from rembg import new_session, remove
            from PIL import Image
            session = new_session("u2net_human_seg" if False else "u2net")
            log.info("rembg background removal ready.")

            def _segment(img_rgb: np.ndarray) -> np.ndarray:
                pil = Image.fromarray(img_rgb)
                out = remove(pil, session=session, only_mask=True)
                mask = np.array(out) > 128
                return mask

            return _segment
        except ImportError:
            log.warning("rembg not installed — no background removal. "
                        "Install with: pip install rembg")
            return None
        except Exception as e:
            log.warning(f"rembg failed to load ({e}) — no background removal.")
            return None

    # ── Depth fusion ───────────────────────────────────────────────
    def _depth_fusion(self, sfm_result, depth_fn, seg_fn, device, fg_sparse=None) -> Optional[Path]:
        """
        1. Run depth model on each frame → disparity map.
        2. Align disparity to metric depth using COLMAP sparse points (affine fit).
        3. Back-project ALL pixels (no manual depth window).
        4. Apply multi-view geometric consistency: keep only points that have
           >= CONSISTENCY_K neighbours within CONSISTENCY_R metres in the
           merged cloud. This naturally removes background noise and floating
           points without any object-specific assumptions.
        """
        import cv2
        import torch

        image_paths = sfm_result.image_paths
        poses = sfm_result.poses
        sparse_pts = sfm_result.sparse_points

        if len(image_paths) > MAX_FRAMES:
            step = len(image_paths) // MAX_FRAMES
            image_paths = image_paths[::step][:MAX_FRAMES]
            log.info(f"Subsampled to {len(image_paths)} frames for depth fusion")

        all_pts = []
        log.info(f"Running depth fusion on {len(image_paths)} frames...")

        with torch.no_grad():
            for i, img_path in enumerate(image_paths):
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]

                img_name = img_path.name
                if img_name not in poses:
                    log.debug(f"No pose for {img_name}, skipping.")
                    continue
                T_wc, K = poses[img_name]

                # Depth model inference
                scale = min(1.0, MAX_DEPTH_SIDE / max(h, w))
                small = cv2.resize(img_rgb, (int(w*scale), int(h*scale))) if scale < 1.0 else img_rgb
                raw = depth_fn(small)
                if raw.shape[:2] != (h, w):
                    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)

                # Apply foreground mask if available
                fg_mask = None
                if seg_fn is not None:
                    try:
                        import cv2 as _cv2
                        fg_mask = seg_fn(img_rgb)
                        if fg_mask.shape != (h, w):
                            fg_mask = _cv2.resize(
                                fg_mask.astype(np.uint8), (w, h),
                                interpolation=_cv2.INTER_NEAREST
                            ).astype(bool)
                        fg_ratio = fg_mask.mean()
                        if i == 0:
                            log.info(f"Foreground mask: {fg_ratio*100:.1f}% of pixels")
                    except Exception as e:
                        log.debug(f"Segmentation failed for frame {i}: {e}")
                        fg_mask = None

                # Align depth using only sparse points that land in the foreground mask
                depth = self._align_depth(
                    raw, sparse_pts.get(img_name), T_wc, K, h, w, fg_mask
                )
                if depth is None:
                    continue

                # Zero out background pixels so _backproject skips them
                if fg_mask is not None:
                    depth = np.where(fg_mask, depth, 0.0)

                if i == 0:
                    valid = depth > 0.01
                    if valid.any():
                        log.info(f"Depth stats frame 0: min={depth[valid].min():.3f} "
                                 f"max={depth[valid].max():.3f} "
                                 f"median={np.median(depth[valid]):.3f}m "
                                 f"fg_pixels={valid.sum():,}")
                pts = self._backproject(depth, T_wc, K, h, w)
                if pts is None or len(pts) == 0:
                    continue

                # Per-frame cap
                if len(pts) > MAX_PTS_PER_FRAME:
                    idx = np.random.choice(len(pts), MAX_PTS_PER_FRAME, replace=False)
                    pts = pts[idx]

                all_pts.append(pts)

                if (i + 1) % 10 == 0:
                    log.info(f"  Processed {i+1}/{len(image_paths)}")

                if device.type == "cuda" and (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()

        if not all_pts:
            log.error("No depth maps produced.")
            return None

        merged = np.concatenate(all_pts, axis=0)
        log.info(f"Raw point cloud: {len(merged):,} points")
        bbox_diag = float(np.linalg.norm(merged.max(axis=0) - merged.min(axis=0)))
        log.info(f"Point cloud bbox diagonal: {bbox_diag:.3f}m  "
                 f"center: {merged.mean(axis=0).round(3)}")

        # Multi-view consistency filter
        merged = self._consistency_filter(merged)
        log.info(f"After consistency filter: {len(merged):,} points")

        # Crop to foreground sparse bounding box + margin
        if fg_sparse is not None and len(fg_sparse) >= 10:
            obj_size = float(np.linalg.norm(fg_sparse.max(axis=0) - fg_sparse.min(axis=0)))
            margin = obj_size * 0.6 + 0.3
            lo = fg_sparse.min(axis=0) - margin
            hi = fg_sparse.max(axis=0) + margin
            crop_mask = np.all((merged >= lo) & (merged <= hi), axis=1)
            merged = merged[crop_mask]
            log.info(f"After spatial crop (obj_size={obj_size:.2f}m margin={margin:.2f}m): "
                     f"{len(merged):,} points")

        if len(merged) < 100:
            log.error("Too few points after filtering.")
            return None

        out_ply = self.work_dir / "cloud.ply"
        self._save_ply(merged, out_ply)
        return out_ply

    def _consistency_filter(self, pts: np.ndarray) -> np.ndarray:
        """
        Keep only points that have >= CONSISTENCY_K neighbours within
        a radius based on median nearest-neighbour distance * 3.
        This adapts to actual point density regardless of scene scale.
        """
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        # Estimate radius from median nearest-neighbour distance
        # Use a random subsample for speed
        sample_n = min(2000, len(pts))
        idx = np.random.choice(len(pts), sample_n, replace=False)
        pcd_sample = o3d.geometry.PointCloud()
        pcd_sample.points = o3d.utility.Vector3dVector(pts[idx].astype(np.float64))
        tree = o3d.geometry.KDTreeFlann(pcd_sample)
        nn_dists = []
        for j in range(min(500, sample_n)):
            k, _, dist2 = tree.search_knn_vector_3d(pcd_sample.points[j], 2)
            if k >= 2:
                nn_dists.append(np.sqrt(dist2[1]))
        if not nn_dists:
            return pts
        median_nn = float(np.median(nn_dists))
        radius = median_nn * 2.0  # tighter — background points are less dense than object surface
        log.info(f"Consistency filter: median_nn={median_nn:.4f}m radius={radius:.4f}m k={CONSISTENCY_K}")

        pcd_clean, _ = pcd.remove_radius_outlier(nb_points=CONSISTENCY_K, radius=radius)
        result = np.asarray(pcd_clean.points).astype(np.float32)

        if len(result) < 200:
            log.warning(f"Consistency filter too aggressive ({len(result)} pts), using statistical filter.")
            pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            result = np.asarray(pcd_clean.points).astype(np.float32)

        return result

    # ── Depth alignment ────────────────────────────────────────────
    def _align_depth(
        self,
        raw: np.ndarray,
        sparse_pts_world: Optional[np.ndarray],
        T_wc: np.ndarray,
        K: np.ndarray,
        h: int, w: int,
        fg_mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Align depth model output to metric scale using COLMAP sparse points.
        If fg_mask is provided, prefer sparse points that fall inside the
        foreground region for fitting — avoids background contamination.
        """
        raw = np.clip(raw.astype(np.float64), 1e-4, None)

        if sparse_pts_world is not None and len(sparse_pts_world) >= 6:
            T_cw = np.linalg.inv(T_wc)
            pts_h = np.hstack([sparse_pts_world, np.ones((len(sparse_pts_world), 1))])
            pts_cam = (T_cw @ pts_h.T).T
            zs_gt = pts_cam[:, 2]

            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            us = (pts_cam[:, 0] * fx / (zs_gt + 1e-8) + cx).astype(int)
            vs = (pts_cam[:, 1] * fy / (zs_gt + 1e-8) + cy).astype(int)
            valid = (zs_gt > 0.01) & (us >= 0) & (us < w) & (vs >= 0) & (vs < h)

            # Prefer sparse points inside the foreground mask
            if fg_mask is not None and valid.sum() > 0:
                vc = vs.clip(0, h-1)
                uc = us.clip(0, w-1)
                in_fg = valid & fg_mask[vc, uc]
                if in_fg.sum() >= 4:
                    valid = in_fg

            if valid.sum() >= 4:
                zs_c = zs_gt[valid]
                raw_at_pts = raw[vs[valid], us[valid]]
                inv_at_pts = 1.0 / np.clip(raw_at_pts, 1e-4, None)

                # Fit A: direct  zs_c = s*raw + t
                A_dir = np.stack([raw_at_pts, np.ones_like(raw_at_pts)], axis=1)
                s_dir, t_dir = np.linalg.lstsq(A_dir, zs_c, rcond=None)[0]
                res_dir = np.mean((s_dir * raw_at_pts + t_dir - zs_c)**2) if s_dir > 0 else 1e18

                # Fit B: inverse  zs_c = s*(1/raw) + t
                A_inv = np.stack([inv_at_pts, np.ones_like(inv_at_pts)], axis=1)
                s_inv, t_inv = np.linalg.lstsq(A_inv, zs_c, rcond=None)[0]
                res_inv = np.mean((s_inv * inv_at_pts + t_inv - zs_c)**2) if s_inv > 0 else 1e18

                log.debug(f"Fit A (direct): s={s_dir:.4f} t={t_dir:.4f} res={res_dir:.4f}")
                log.debug(f"Fit B (inverse): s={s_inv:.4f} t={t_inv:.4f} res={res_inv:.4f}")

                if res_dir < res_inv and s_dir > 0:
                    best_res = res_dir
                    depth = s_dir * raw + t_dir
                elif s_inv > 0:
                    best_res = res_inv
                    depth = s_inv * (1.0 / raw) + t_inv
                else:
                    depth = None

                # Reject frames with poor fit — prevents catastrophic misalignment
                MAX_RESIDUAL = 2.0  # metres squared
                if depth is not None and best_res > MAX_RESIDUAL:
                    log.debug(f"Rejecting frame: fit residual {best_res:.3f} > {MAX_RESIDUAL}")
                    depth = None

                if depth is not None:
                    return np.clip(depth, 0.01, 100.0).astype(np.float32)

        # Fallback: scale raw so median matches sparse median depth
        if sparse_pts_world is not None and len(sparse_pts_world) >= 1:
            T_cw = np.linalg.inv(T_wc)
            pts_h = np.hstack([sparse_pts_world, np.ones((len(sparse_pts_world), 1))])
            zs = (T_cw @ pts_h.T).T[:, 2]
            target = float(np.median(zs[zs > 0.01])) if (zs > 0.01).any() else 1.0
        else:
            target = 1.0

        # Try both direct and inverse, pick whichever gives positive depths near target
        med_raw = float(np.median(raw))
        med_inv = float(np.median(1.0 / raw))
        if abs(med_raw - target) < abs(med_inv - target):
            depth = raw * (target / (med_raw + 1e-8))
        else:
            depth = (1.0 / raw) * (target / (med_inv + 1e-8))

        return np.clip(depth, 0.01, 100.0).astype(np.float32)

    def _backproject(self, depth: np.ndarray, T_wc: np.ndarray,
                     K: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        ys, xs = np.mgrid[0:h:PROJ_STEP, 0:w:PROJ_STEP]
        zs = depth[ys, xs].astype(np.float64)
        mask = zs > 0.01   # 0.0 means masked out by depth window
        xs, ys, zs = xs[mask].ravel(), ys[mask].ravel(), zs[mask].ravel()
        if len(xs) == 0:
            return None
        X = (xs - cx) * zs / fx
        Y = (ys - cy) * zs / fy
        pts_cam = np.stack([X, Y, zs, np.ones_like(zs)], axis=-1)
        return (T_wc @ pts_cam.T).T[:, :3].astype(np.float32)

    # ── Depth model loading ────────────────────────────────────────
    def _load_depth_model(self, device) -> Optional[Callable]:
        import torch
        try:
            from transformers import pipeline as hf_pipeline
            log.info("Loading Depth Anything V2 Small...")
            pipe = hf_pipeline(
                task="depth-estimation", model=DEPTH_MODEL_HF,
                device=0 if device.type == "cuda" else -1,
            )
            def _da2(img_rgb):
                from PIL import Image
                return np.array(pipe(Image.fromarray(img_rgb))["depth"], dtype=np.float32)
            log.info("Depth Anything V2 Small ready.")
            return _da2
        except Exception as e:
            log.warning(f"Depth Anything V2 unavailable ({e}), trying MiDaS_small...")

        try:
            midas = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL_MIDAS,
                                   trust_repo=True, verbose=False)
            midas.to(device).eval()
            transform = torch.hub.load("intel-isl/MiDaS", "transforms",
                                       trust_repo=True, verbose=False).small_transform
            def _midas(img_rgb):
                inp = transform(img_rgb).to(device)
                with torch.no_grad():
                    pred = midas(inp)
                return torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=img_rgb.shape[:2],
                    mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy().astype(np.float32)
            log.info("MiDaS small ready.")
            return _midas
        except Exception as e:
            log.error(f"Failed to load depth model: {e}")
            return None

    # ── Helpers ────────────────────────────────────────────────────
    def _remove_outliers(self, pts: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std_ratio)
        return np.asarray(pcd.points).astype(np.float32)

    def _save_ply(self, pts: np.ndarray, path: Path):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)
        log.info(f"Saved {len(pts):,} points → {path}")

    @staticmethod
    def _default_K(w: int, h: int) -> np.ndarray:
        fx = w / (2 * np.tan(np.radians(30)))
        return np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float64)

    def _cmd(self, args: list) -> int:
        result = subprocess.run(["colmap"] + args, capture_output=True, text=True)
        if result.returncode != 0:
            log.debug(result.stderr[-2000:])
        return result.returncode
