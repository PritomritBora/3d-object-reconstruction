"""
Pipeline orchestrator — wires the three stages together and logs timing.

Stage 1: Feature extraction / SfM  (COLMAP)
Stage 2: Reconstruction             (sparse point cloud from SfM)
Stage 3: Meshing                    (Poisson + cleanup + decimation)
"""
import logging
import time
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class MeshPipeline:
    def __init__(self, work_dir: Path, max_faces: int = 50000, depth_method: str = "colmap"):
        self.work_dir = work_dir
        self.max_faces = max_faces
        self.depth_method = depth_method
        self.timings: dict = {}

    def run(self, image_paths: List[Path], output_path: Path) -> bool:
        """
        Run the full pipeline. Returns True on success, False on failure.
        Timing for each stage is logged and stored in self.timings.
        """
        from pipeline.feature_extraction import FeatureExtractor
        from pipeline.reconstruction import Reconstructor
        from pipeline.meshing import Mesher

        log.info("=" * 60)
        log.info("  3D Mesh Generation Pipeline")
        log.info("=" * 60)

        # Stage 1 — Feature extraction and SfM
        with _Timer("feature_extraction", self.timings):
            sfm_result = FeatureExtractor(
                work_dir=self.work_dir / "sfm",
                method=self.depth_method,
            ).run(image_paths)

        if sfm_result is None:
            log.error("Stage 1 (feature extraction) failed.")
            return False
        log.info(f"[TIMING] Feature extraction: {self.timings['feature_extraction']:.1f}s")

        # Stage 2 — Reconstruction (sparse point cloud)
        with _Timer("reconstruction", self.timings):
            recon_path = Reconstructor(
                work_dir=self.work_dir / "dense",
                method=self.depth_method,
            ).run(sfm_result)

        if recon_path is None:
            log.error("Stage 2 (reconstruction) failed.")
            return False
        log.info(f"[TIMING] Reconstruction: {self.timings['reconstruction']:.1f}s")

        # Stage 3 — Meshing, cleanup, decimation, export
        with _Timer("meshing", self.timings):
            success = Mesher(max_faces=self.max_faces).run(recon_path, output_path)

        if not success:
            log.error("Stage 3 (meshing) failed.")
            return False
        log.info(f"[TIMING] Meshing: {self.timings['meshing']:.1f}s")

        # Summary
        log.info("─" * 40)
        log.info("Stage timings:")
        for stage, t in self.timings.items():
            log.info(f"  {stage:<25} {t:.1f}s")

        return True


class _Timer:
    """Context manager that measures wall-clock time for a named stage."""

    def __init__(self, name: str, store: dict):
        self.name = name
        self.store = store

    def __enter__(self):
        self._start = time.time()
        log.info(f"▶ Stage: {self.name}")
        return self

    def __exit__(self, *_):
        self.store[self.name] = time.time() - self._start
