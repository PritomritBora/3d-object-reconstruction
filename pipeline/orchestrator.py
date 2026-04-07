"""
Pipeline orchestrator — wires all stages together and logs timing.

Stage 1: Feature extraction / SfM  (COLMAP tuned, or OpenCV fallback)
Stage 2: Depth estimation + TSDF fusion → raw mesh
Stage 3: Mesh cleanup + decimation + export
"""
import logging
import time
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Enable DEBUG for reconstruction to diagnose depth alignment
logging.getLogger("pipeline.reconstruction").setLevel(logging.DEBUG)


class MeshPipeline:
    def __init__(self, work_dir: Path, max_faces: int = 50000, depth_method: str = "colmap"):
        self.work_dir = work_dir
        self.max_faces = max_faces
        self.depth_method = depth_method
        self.timings: dict = {}

    def run(self, image_paths: List[Path], output_path: Path) -> bool:
        from pipeline.feature_extraction import FeatureExtractor
        from pipeline.reconstruction import Reconstructor
        from pipeline.meshing import Mesher

        log.info("=" * 60)
        log.info("  3D Mesh Generation Pipeline")
        log.info("=" * 60)

        # Stage 1 — SfM / pose estimation
        with _Timer("feature_extraction", self.timings):
            extractor = FeatureExtractor(
                work_dir=self.work_dir / "sfm",
                method=self.depth_method,
            )
            sfm_result = extractor.run(image_paths)

        if sfm_result is None:
            log.error("Stage 1 failed.")
            return False
        log.info(f"[TIMING] Feature extraction: {self.timings['feature_extraction']:.1f}s")

        # Stage 2 — Dense reconstruction → point cloud or mesh
        with _Timer("reconstruction", self.timings):
            reconstructor = Reconstructor(
                work_dir=self.work_dir / "dense",
                method=self.depth_method,
            )
            recon_path = reconstructor.run(sfm_result)

        if recon_path is None:
            log.error("Stage 2 failed.")
            return False
        log.info(f"[TIMING] Reconstruction: {self.timings['reconstruction']:.1f}s")

        # Stage 3 — Meshing / cleanup + decimation + export
        with _Timer("meshing", self.timings):
            mesher = Mesher(max_faces=self.max_faces)
            success = mesher.run(recon_path, output_path)

        if not success:
            log.error("Stage 3 failed.")
            return False
        log.info(f"[TIMING] Meshing: {self.timings['meshing']:.1f}s")

        log.info("─" * 40)
        log.info("Stage timings:")
        for stage, t in self.timings.items():
            log.info(f"  {stage:<25} {t:.1f}s")

        return True


class _Timer:
    def __init__(self, name: str, store: dict):
        self.name = name
        self.store = store

    def __enter__(self):
        self._t = time.time()
        log.info(f"▶ Stage: {self.name}")
        return self

    def __exit__(self, *_):
        self.store[self.name] = time.time() - self._t
