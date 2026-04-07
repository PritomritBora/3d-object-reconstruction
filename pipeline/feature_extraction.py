"""
Stage 1 — Feature extraction and Structure-from-Motion (SfM).

Primary path : COLMAP (tuned for speed) via subprocess
               Exports sparse model as TXT for downstream scale alignment.
Fallback path: OpenCV SIFT + essential matrix chaining (no COLMAP binary needed)
"""
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SfMResult:
    sparse_dir: Path            # COLMAP sparse model dir (TXT format) or work_dir
    images_dir: Path
    image_paths: List[Path]
    # image name -> (4x4 world_T_cam, 3x3 K)
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # image name -> list of (x,y,z) sparse 3D points visible in that image
    sparse_points: Dict[str, np.ndarray] = field(default_factory=dict)
    method: str = "colmap"


class FeatureExtractor:
    def __init__(self, work_dir: Path, method: str = "colmap"):
        self.work_dir = work_dir
        self.method = method
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, image_paths: List[Path]) -> Optional[SfMResult]:
        if self.method == "colmap" and shutil.which("colmap"):
            result = self._run_colmap(image_paths)
            if result is not None:
                return result
            log.warning("COLMAP failed, falling back to OpenCV pose estimation.")
        return self._run_opencv_poses(image_paths)

    # ── COLMAP ─────────────────────────────────────────────────────
    def _run_colmap(self, image_paths: List[Path]) -> Optional[SfMResult]:
        images_dir = image_paths[0].parent

        # Filter out blurry frames — they cause COLMAP registration failures
        image_paths = self._filter_blurry(image_paths)
        log.info(f"After blur filter: {len(image_paths)} sharp frames")
        db_path = self.work_dir / "database.db"
        sparse_dir = self.work_dir / "sparse"
        sparse_txt_dir = self.work_dir / "sparse_txt"

        # Always start fresh — stale database causes partial re-registration
        if db_path.exists():
            db_path.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        if sparse_txt_dir.exists():
            shutil.rmtree(sparse_txt_dir)
        resized_dir = self.work_dir / "resized"
        if resized_dir.exists():
            shutil.rmtree(resized_dir)

        sparse_dir.mkdir(parents=True, exist_ok=True)
        sparse_txt_dir.mkdir(parents=True, exist_ok=True)

        n = len(image_paths)
        log.info(f"COLMAP: extracting features ({n} images)...")

        # Resize images to ~1024px longest side for speed
        resized_dir = self._resize_images(image_paths)
        use_dir = resized_dir if resized_dir else images_dir

        ret = self._cmd([
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(use_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "0",
            "--SiftExtraction.max_num_features", "8192",
        ])
        if ret != 0:
            return None

        matcher = "sequential_matcher" if n > 120 else "exhaustive_matcher"
        log.info(f"COLMAP: matching ({matcher})...")
        ret = self._cmd([
            matcher,
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", "0",
            "--SiftMatching.max_num_matches", "32768",
        ])
        if ret != 0:
            return None

        log.info("COLMAP: mapping (SfM)...")
        ret = self._cmd([
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(use_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.num_threads", "8",
            "--Mapper.init_min_tri_angle", "2",
            "--Mapper.multiple_models", "0",
            "--Mapper.min_num_matches", "10",
            "--Mapper.abs_pose_min_num_inliers", "10",
        ])
        if ret != 0:
            return None

        model_dirs = list(sparse_dir.iterdir())
        if not model_dirs:
            log.error("No COLMAP sparse model produced.")
            return None

        # Pick the model with the most registered images
        def _count_images(d):
            try:
                r = subprocess.run(
                    ["colmap", "model_analyzer", "--path", str(d)],
                    capture_output=True, text=True
                )
                # model_analyzer writes to stderr
                output = r.stdout + r.stderr
                for line in output.splitlines():
                    if "Registered images:" in line:
                        return int(line.split(":")[-1].strip())
            except Exception:
                pass
            return 0

        model_dir = max(model_dirs, key=_count_images)
        log.info(f"Using sparse model: {model_dir}")

        # Export to TXT for easy parsing
        ret = self._cmd([
            "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(sparse_txt_dir),
            "--output_type", "TXT",
        ])
        if ret != 0:
            log.warning("model_converter failed, trying to read binary directly.")
            sparse_txt_dir = model_dir

        poses, intrinsics, sparse_points = self._parse_colmap_txt(sparse_txt_dir)
        if not poses:
            log.warning("Could not parse COLMAP model, poses will be empty.")

        log.info(f"SfM done: {len(poses)} registered images, "
                 f"{sum(len(v) for v in sparse_points.values())} point observations")

        return SfMResult(
            sparse_dir=sparse_txt_dir,
            images_dir=use_dir,
            image_paths=image_paths,
            poses=poses,
            sparse_points=sparse_points,
            method="colmap",
        )

    def _filter_blurry(self, image_paths: List[Path], threshold: float = 8.0) -> List[Path]:
        """Remove motion-blurred frames using Laplacian variance.
        Uses adaptive threshold: removes bottom 15% of frames by sharpness."""
        try:
            import cv2
            scores = []
            for p in image_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                scores.append((p, cv2.Laplacian(img, cv2.CV_64F).var()))

            if not scores:
                return image_paths

            # Adaptive: keep top 85% sharpest frames, but always keep at least 20
            scores.sort(key=lambda x: x[1], reverse=True)
            keep_n = max(20, int(len(scores) * 0.85))
            sharp = [p for p, _ in scores[:keep_n]]
            # Also apply absolute minimum threshold to remove truly blurry frames
            sharp = [p for p, s in scores[:keep_n] if s >= 5.0]
            if len(sharp) < 10:
                sharp = [p for p, _ in scores[:max(10, keep_n)]]

            log.info(f"Blur filter: {len(sharp)}/{len(image_paths)} frames kept "
                     f"(min_score={min(s for _,s in scores[:len(sharp)]):.1f}, "
                     f"max_score={max(s for _,s in scores[:len(sharp)]):.1f})")
            return sharp
        except Exception as e:
            log.warning(f"Blur filter failed ({e}), using all frames.")
            return image_paths

    def _resize_images(self, image_paths: List[Path]) -> Optional[Path]:
        """Resize images to max 1024px longest side into a temp dir."""
        try:
            import cv2
            out_dir = self.work_dir / "resized"
            out_dir.mkdir(exist_ok=True)
            for p in image_paths:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h, w = img.shape[:2]
                scale = min(1.0, 1024 / max(h, w))
                if scale < 1.0:
                    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_dir / p.name), img)
            return out_dir
        except Exception as e:
            log.warning(f"Image resize failed ({e}), using originals.")
            return None

    def _parse_colmap_txt(self, txt_dir: Path):
        """Parse COLMAP TXT model → poses, intrinsics, sparse points per image."""
        poses = {}
        intrinsics = {}
        sparse_points = {}  # image_name -> (N,3) array of 3D points

        cameras_file = txt_dir / "cameras.txt"
        images_file = txt_dir / "images.txt"
        points_file = txt_dir / "points3D.txt"

        if not cameras_file.exists() or not images_file.exists():
            return poses, intrinsics, sparse_points

        # Parse cameras
        cam_params = {}
        with open(cameras_file) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model = parts[1]
                w, h = int(parts[2]), int(parts[3])
                params = list(map(float, parts[4:]))
                # SIMPLE_RADIAL or PINHOLE: fx, cx, cy
                fx = params[0]
                cx = params[1] if len(params) > 1 else w / 2
                cy = params[2] if len(params) > 2 else h / 2
                K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
                cam_params[cam_id] = K

        # Parse images (poses)
        # Also collect which 3D point IDs are visible per image
        image_point_ids = {}  # image_name -> list of point3D_ids
        with open(images_file) as f:
            lines = [l for l in f if not l.startswith("#") and l.strip()]
        i = 0
        while i < len(lines):
            parts = lines[i].split()
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            cam_id = int(parts[8])
            img_name = parts[9]

            R = self._quat_to_rot(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])
            # COLMAP stores cam_T_world; invert to world_T_cam
            T_cw = np.eye(4)
            T_cw[:3, :3] = R
            T_cw[:3, 3] = t
            T_wc = np.linalg.inv(T_cw)

            K = cam_params.get(cam_id, np.eye(3))
            poses[img_name] = (T_wc, K)
            intrinsics[img_name] = K

            # Next line: 2D keypoints + point3D ids
            if i + 1 < len(lines):
                kp_parts = lines[i + 1].split()
                pt_ids = []
                for j in range(2, len(kp_parts), 3):
                    pid = int(kp_parts[j])
                    if pid != -1:
                        pt_ids.append(pid)
                image_point_ids[img_name] = pt_ids
            i += 2

        # Parse 3D points
        point3d_xyz = {}
        if points_file.exists():
            with open(points_file) as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    pid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    point3d_xyz[pid] = np.array([x, y, z])

        # Build per-image sparse point arrays
        for img_name, pt_ids in image_point_ids.items():
            pts = [point3d_xyz[pid] for pid in pt_ids if pid in point3d_xyz]
            if pts:
                sparse_points[img_name] = np.array(pts, dtype=np.float64)

        return poses, intrinsics, sparse_points

    @staticmethod
    def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
        n = qw*qw + qx*qx + qy*qy + qz*qz
        if n < 1e-10:
            return np.eye(3)
        s = 2.0 / n
        R = np.array([
            [1 - s*(qy*qy+qz*qz),   s*(qx*qy-qz*qw),   s*(qx*qz+qy*qw)],
            [  s*(qx*qy+qz*qw), 1 - s*(qx*qx+qz*qz),   s*(qy*qz-qx*qw)],
            [  s*(qx*qz-qy*qw),   s*(qy*qz+qx*qw), 1 - s*(qx*qx+qy*qy)],
        ])
        return R

    def _cmd(self, args: list) -> int:
        result = subprocess.run(["colmap"] + args, capture_output=True, text=True)
        if result.returncode != 0:
            log.debug(result.stderr[-2000:])
        return result.returncode

    # ── OpenCV fallback ────────────────────────────────────────────
    def _run_opencv_poses(self, image_paths: List[Path]) -> SfMResult:
        """
        Estimate poses via SIFT + essential matrix chaining.
        No sparse 3D points available, so scale alignment will be skipped.
        """
        import cv2
        log.info("Estimating poses via OpenCV SIFT (no COLMAP)...")

        img0 = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
        h, w = (img0.shape[:2] if img0 is not None else (720, 1280))
        fx = w / (2 * np.tan(np.radians(30)))
        K = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float64)

        sift = cv2.SIFT_create(nfeatures=2000)
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        poses = {}
        cumulative = np.eye(4, dtype=np.float64)
        poses[image_paths[0].name] = (cumulative.copy(), K)

        prev_gray = img0
        prev_kp, prev_desc = sift.detectAndCompute(prev_gray, None) if img0 is not None else ([], None)

        for i in range(1, len(image_paths)):
            img = cv2.imread(str(image_paths[i]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                poses[image_paths[i].name] = (cumulative.copy(), K)
                continue

            kp, desc = sift.detectAndCompute(img, None)
            moved = False
            if desc is not None and prev_desc is not None and len(kp) >= 8:
                matches = matcher.knnMatch(prev_desc, desc, k=2)
                good = [m for m, n_ in matches if m.distance < 0.75 * n_.distance]
                if len(good) >= 8:
                    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                    pts2 = np.float32([kp[m.trainIdx].pt for m in good])
                    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    if E is not None:
                        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
                        T_rel = np.eye(4)
                        T_rel[:3, :3] = R
                        T_rel[:3, 3] = t.ravel()
                        cumulative = cumulative @ np.linalg.inv(T_rel)
                        moved = True

            poses[image_paths[i].name] = (cumulative.copy(), K)
            prev_gray, prev_kp, prev_desc = img, kp, desc

        log.info(f"OpenCV pose estimation done: {len(poses)} images")
        return SfMResult(
            sparse_dir=self.work_dir,
            images_dir=image_paths[0].parent,
            image_paths=image_paths,
            poses=poses,
            sparse_points={},
            method="opencv",
        )
