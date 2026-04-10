"""
Stage 1 — Feature extraction and Structure-from-Motion (SfM).

Uses COLMAP to:
  1. Extract SIFT keypoints from each image
  2. Match keypoints across image pairs (exhaustive for <exhaustive_limit, sequential otherwise)
  3. Run incremental SfM to recover camera poses and a sparse 3D point cloud

Output: SfMResult containing camera poses, intrinsics, and per-image sparse 3D points.
        These are used downstream for depth alignment and dense reconstruction.
"""
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.config import get as cfg

log = logging.getLogger(__name__)


@dataclass
class SfMResult:
    """Carries everything downstream stages need from SfM."""
    sparse_dir: Path            # TXT sparse model (cameras/images/points3D.txt)
    binary_sparse_dir: Path     # Binary sparse model (for dense MVS undistortion)
    images_dir: Path            # Directory of (possibly resized) images used by COLMAP
    image_paths: List[Path]     # Original input image paths
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # image_name -> (4x4 world_T_cam, 3x3 K)
    sparse_points: Dict[str, np.ndarray] = field(default_factory=dict)
    # image_name -> (N, 3) array of 3D world points visible in that image
    method: str = "colmap"


class FeatureExtractor:
    def __init__(self, work_dir: Path, method: str = "colmap"):
        self.work_dir = work_dir
        self.method = method
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, image_paths: List[Path]) -> Optional[SfMResult]:
        """Run feature extraction and SfM. Returns SfMResult or None on failure."""
        if shutil.which("colmap"):
            return self._run_colmap(image_paths)
        log.error("COLMAP binary not found in PATH. Please install COLMAP.")
        return None

    # ── COLMAP SfM ─────────────────────────────────────────────────
    def _run_colmap(self, image_paths: List[Path]) -> Optional[SfMResult]:
        db_path = self.work_dir / "database.db"
        sparse_dir = self.work_dir / "sparse"
        sparse_txt_dir = self.work_dir / "sparse_txt"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        sparse_txt_dir.mkdir(parents=True, exist_ok=True)

        # Resize images — scale down more aggressively for large sets
        sharp_paths = self._filter_blurry(image_paths)
        if len(sharp_paths) > cfg("colmap", "exhaustive_limit"):
            resize_px = cfg("colmap", "resize_sequential")  # smallest for sequential
        elif len(sharp_paths) > 100:
            resize_px = cfg("colmap", "resize_large")
        else:
            resize_px = cfg("colmap", "resize_small")
        resized_dir = self._resize_images(sharp_paths, max_size=resize_px)
        use_dir = resized_dir if resized_dir else sharp_paths[0].parent

        n = len(sharp_paths)
        log.info(f"COLMAP: extracting features ({n} images, resize={resize_px}px)...")

        # Reduce features for large sets — faster extraction and matching
        max_features = cfg("colmap", "max_features") if n <= cfg("colmap", "exhaustive_limit") \
            else cfg("colmap", "max_features_sequential")

        if self._cmd([
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(use_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "0",
            f"--SiftExtraction.max_num_features", str(max_features),
            f"--SiftExtraction.peak_threshold", str(cfg("colmap", "peak_threshold")),
        ]) != 0:
            return None

        exhaustive_max = cfg("colmap", "exhaustive_limit")
        matcher = "sequential_matcher" if n > exhaustive_max else "exhaustive_matcher"
        max_matches = "16384" if matcher == "sequential_matcher" else "32768"
        log.info(f"COLMAP: matching ({matcher}, n={n})...")
        if self._cmd([
            matcher,
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", "0",
            "--SiftMatching.max_num_matches", max_matches,
            *(["--SequentialMatching.overlap", "10"] if matcher == "sequential_matcher" else []),
        ]) != 0:
            return None

        log.info("COLMAP: mapping (SfM)...")
        if self._cmd([
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(use_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.num_threads", "8",
            "--Mapper.init_min_tri_angle", "2",
            "--Mapper.min_num_matches", "10",
            "--Mapper.abs_pose_min_num_inliers", "10",
        ]) != 0:
            return None

        model_dirs = list(sparse_dir.iterdir())
        if not model_dirs:
            log.error("No COLMAP sparse model produced.")
            return None

        # Pick the sub-model with the most registered images
        model_dir = max(model_dirs, key=self._count_registered_images)
        log.info(f"Using sparse model: {model_dir}")

        # Export to TXT for easy parsing
        if self._cmd([
            "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(sparse_txt_dir),
            "--output_type", "TXT",
        ]) != 0:
            log.warning("model_converter failed, using binary model directly.")
            sparse_txt_dir = model_dir

        poses, sparse_points = self._parse_colmap_txt(sparse_txt_dir)
        log.info(f"SfM done: {len(poses)} registered images, "
                 f"{sum(len(v) for v in sparse_points.values())} point observations")

        return SfMResult(
            sparse_dir=sparse_txt_dir,
            binary_sparse_dir=model_dir,
            images_dir=use_dir,
            image_paths=image_paths,
            poses=poses,
            sparse_points=sparse_points,
            method="colmap",
        )

    def _count_registered_images(self, model_dir: Path) -> int:
        """Query COLMAP model_analyzer to get registered image count."""
        try:
            r = subprocess.run(
                ["colmap", "model_analyzer", "--path", str(model_dir)],
                capture_output=True, text=True
            )
            for line in (r.stdout + r.stderr).splitlines():
                if "Registered images:" in line:
                    return int(line.split(":")[-1].strip())
        except Exception:
            pass
        return 0

    def _filter_blurry(self, image_paths: List[Path]) -> List[Path]:
        """
        Remove motion-blurred frames using Laplacian variance on the center crop.
        Center crop avoids bokeh background giving false low sharpness scores.
        Keeps the sharpest 85% of frames, always keeping at least 20.
        """
        try:
            import cv2
            scores = []
            for p in image_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape
                crop = img[h//4:3*h//4, w//4:3*w//4]
                scores.append((p, cv2.Laplacian(crop, cv2.CV_64F).var()))

            if not scores:
                return image_paths

            scores.sort(key=lambda x: x[1], reverse=True)
            keep_n = max(20, int(len(scores) * cfg("input", "blur_threshold_pct") / 100))
            sharp = [(p, s) for p, s in scores[:keep_n] if s >= 5.0]
            if len(sharp) < 10:
                sharp = scores[:max(10, keep_n)]

            log.info(f"Blur filter (center crop): {len(sharp)}/{len(image_paths)} frames kept "
                     f"(min={min(s for _,s in sharp):.1f} max={max(s for _,s in sharp):.1f})")
            return [p for p, _ in sharp]
        except Exception as e:
            log.warning(f"Blur filter failed ({e}), using all frames.")
            return image_paths

    def _resize_images(self, image_paths: List[Path], max_size: int = 1024) -> Optional[Path]:
        """Resize images to max_size px longest side for faster SIFT extraction."""
        try:
            import cv2
            out_dir = self.work_dir / "resized"
            out_dir.mkdir(exist_ok=True)
            for p in image_paths:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h, w = img.shape[:2]
                scale = min(1.0, max_size / max(h, w))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_dir / p.name), img)
            return out_dir
        except Exception as e:
            log.warning(f"Image resize failed ({e}), using originals.")
            return None

    def _parse_colmap_txt(self, txt_dir: Path):
        """
        Parse COLMAP TXT model into poses and sparse points.

        Returns:
            poses: dict of image_name -> (T_wc 4x4, K 3x3)
            sparse_points: dict of image_name -> (N, 3) world points
        """
        poses, sparse_points = {}, {}
        cameras_file = txt_dir / "cameras.txt"
        images_file = txt_dir / "images.txt"
        points_file = txt_dir / "points3D.txt"

        if not cameras_file.exists() or not images_file.exists():
            return poses, sparse_points

        # Parse camera intrinsics
        cam_params = {}
        with open(cameras_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                w, h = int(parts[2]), int(parts[3])
                params = list(map(float, parts[4:]))
                fx = params[0]
                cx = params[1] if len(params) > 1 else w / 2
                cy = params[2] if len(params) > 2 else h / 2
                cam_params[cam_id] = np.array(
                    [[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64
                )

        # Parse image poses and point track IDs
        image_point_ids = {}
        with open(images_file) as f:
            lines = [l for l in f if not l.startswith("#") and l.strip()]
        i = 0
        while i < len(lines):
            parts = lines[i].split()
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            img_name = parts[9]

            R = self._quat_to_rot(qw, qx, qy, qz)
            T_cw = np.eye(4)
            T_cw[:3, :3] = R
            T_cw[:3, 3] = [tx, ty, tz]
            T_wc = np.linalg.inv(T_cw)

            K = cam_params.get(cam_id, np.eye(3))
            poses[img_name] = (T_wc, K)

            if i + 1 < len(lines):
                kp_parts = lines[i + 1].split()
                pt_ids = [int(kp_parts[j]) for j in range(2, len(kp_parts), 3)
                          if int(kp_parts[j]) != -1]
                image_point_ids[img_name] = pt_ids
            i += 2

        # Parse 3D point positions
        point3d_xyz = {}
        if points_file.exists():
            with open(points_file) as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    point3d_xyz[int(parts[0])] = np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )

        # Build per-image sparse point arrays
        for img_name, pt_ids in image_point_ids.items():
            pts = [point3d_xyz[pid] for pid in pt_ids if pid in point3d_xyz]
            if pts:
                sparse_points[img_name] = np.array(pts, dtype=np.float64)

        return poses, sparse_points

    @staticmethod
    def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        n = qw*qw + qx*qx + qy*qy + qz*qz
        if n < 1e-10:
            return np.eye(3)
        s = 2.0 / n
        return np.array([
            [1 - s*(qy*qy+qz*qz),   s*(qx*qy-qz*qw),   s*(qx*qz+qy*qw)],
            [  s*(qx*qy+qz*qw), 1 - s*(qx*qx+qz*qz),   s*(qy*qz-qx*qw)],
            [  s*(qx*qz-qy*qw),   s*(qy*qz+qx*qw), 1 - s*(qx*qx+qy*qy)],
        ])

    def _cmd(self, args: list) -> int:
        """Run a COLMAP command, return exit code."""
        result = subprocess.run(["colmap"] + args, capture_output=True, text=True)
        if result.returncode != 0:
            log.warning(result.stderr[-1000:])
        return result.returncode
