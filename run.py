"""
Real-Time 3D Mesh Generation Pipeline
Entry point: python run.py --input ./images --output mesh.obj
"""
import argparse
import sys
import time
from pathlib import Path

from pipeline.orchestrator import MeshPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time 3D Mesh Generation Pipeline"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Directory containing input RGB images (< 200)"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output mesh file path (.obj or .ply)"
    )
    parser.add_argument(
        "--max-faces", type=int, default=50000,
        help="Max faces after decimation (default: 50000)"
    )
    parser.add_argument(
        "--depth-method", choices=["colmap", "midas"], default="colmap",
        help="Depth estimation method (default: colmap)"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Working directory for intermediate files (default: <output>_workspace)"
    )
    parser.add_argument(
        "--keep-workspace", action="store_true",
        help="Keep intermediate files after completion"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if not args.input.exists() or not args.input.is_dir():
        print(f"[ERROR] Input directory not found: {args.input}")
        sys.exit(1)

    images = sorted(list(args.input.glob("*.jpg")) +
                    list(args.input.glob("*.jpeg")) +
                    list(args.input.glob("*.png")))

    if len(images) == 0:
        print(f"[ERROR] No images found in {args.input}")
        sys.exit(1)

    if len(images) > 200:
        print(f"[WARN] Found {len(images)} images, pipeline requires < 200. Using first 200.")
        images = images[:200]

    print(f"[INFO] Found {len(images)} images")

    # Set working directory
    work_dir = args.work_dir or args.output.parent / f"{args.output.stem}_workspace"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    total_start = time.time()
    pipeline = MeshPipeline(
        work_dir=work_dir,
        max_faces=args.max_faces,
        depth_method=args.depth_method,
    )

    success = pipeline.run(
        image_paths=images,
        output_path=args.output,
    )

    total_elapsed = time.time() - total_start
    print(f"\n[TIMING] Total wall-clock time: {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")

    if not success:
        print("[ERROR] Pipeline failed. Check logs above.")
        sys.exit(1)

    print(f"[SUCCESS] Mesh saved to: {args.output}")

    if not args.keep_workspace:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
