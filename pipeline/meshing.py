"""
Stage 3 — Surface reconstruction, cleanup, and export.

Input : PLY file — either a point cloud (from SfM sparse cloud) or a mesh.
        Point clouds go through Poisson surface reconstruction first.
        Meshes go straight to cleanup.

Output: Clean OBJ or PLY with vertex normals, ≤ max_faces triangles.

Pipeline:
  1. Poisson reconstruction (if input is a point cloud)
  2. Remove degenerate / duplicate geometry
  3. Keep only the largest connected component
  4. Fill small holes (via trimesh)
  5. Light Laplacian smoothing
  6. Quadric decimation to ≤ max_faces
  7. Recompute vertex normals
  8. Export
"""
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Mesher:
    def __init__(self, max_faces: int = 50000):
        self.max_faces = max_faces

    def run(self, input_path: Path, output_path: Path) -> bool:
        """
        Process a PLY point cloud or mesh into a clean triangle mesh.
        Returns True on success, False on failure.
        """
        try:
            import open3d as o3d
        except ImportError:
            log.error("open3d not found. Install with: pip install open3d")
            return False

        log.info(f"Loading: {input_path}")

        # Detect whether input is a point cloud or mesh
        # (both can be stored as PLY — check for triangle data)
        test_mesh = o3d.io.read_triangle_mesh(str(input_path))
        if len(test_mesh.triangles) > 0:
            log.info(f"Input is a mesh: {len(test_mesh.triangles):,} faces")
            mesh = test_mesh
        else:
            pcd = o3d.io.read_point_cloud(str(input_path))
            if len(pcd.points) < 10:
                log.error(f"Point cloud too small: {len(pcd.points)} points.")
                return False
            log.info(f"Input is a point cloud: {len(pcd.points):,} points — running Poisson")
            mesh = self._poisson_from_pcd(pcd)
            if mesh is None:
                return False

        # Geometry cleanup
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Keep only the main object — discard floating debris
        mesh = self._keep_largest_component(mesh)
        log.info(f"After component filter: {len(mesh.triangles):,} faces")

        # Fill holes before decimation for better topology
        mesh = self._fill_holes(mesh)

        # Light smoothing to reduce Poisson staircase artifacts
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        # Decimate to target face count
        n = len(mesh.triangles)
        if n > self.max_faces:
            log.info(f"Decimating {n:,} → {self.max_faces:,} faces...")
            mesh = mesh.simplify_quadric_decimation(self.max_faces)
            log.info(f"After decimation: {len(mesh.triangles):,} faces")

        # Final cleanup and normals
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        self._quality_report(mesh)
        return self._export(mesh, output_path)

    # ── Poisson surface reconstruction ─────────────────────────────
    def _poisson_from_pcd(self, pcd):
        """
        Reconstruct a watertight surface from a point cloud using
        Poisson surface reconstruction (Kazhdan & Hoppe 2013).

        Voxel size is estimated from the median nearest-neighbour distance
        so it adapts to both sparse (~6K points) and dense (~500K points) clouds.
        """
        import open3d as o3d

        n = len(pcd.points)
        pts_arr = np.asarray(pcd.points)

        # Estimate voxel size from median nearest-neighbour distance
        sample_n = min(2000, n)
        idx = np.random.choice(n, sample_n, replace=False)
        sample_pcd = o3d.geometry.PointCloud()
        sample_pcd.points = o3d.utility.Vector3dVector(pts_arr[idx].astype(np.float64))
        tree = o3d.geometry.KDTreeFlann(sample_pcd)
        nn_dists = []
        for j in range(min(500, sample_n)):
            k, _, dist2 = tree.search_knn_vector_3d(sample_pcd.points[j], 2)
            if k >= 2:
                nn_dists.append(np.sqrt(dist2[1]))
        median_nn = float(np.median(nn_dists)) if nn_dists else 0.01
        voxel = max(0.002, median_nn * 1.5)
        log.info(f"Voxel size: {voxel:.4f}m (median_nn={median_nn:.4f}m, {n:,} points)")

        # Downsample if very dense to keep Poisson memory manageable
        if n > 200_000:
            pcd = pcd.voxel_down_sample(voxel_size=voxel)
            while len(pcd.points) > 200_000:
                voxel *= 1.5
                pcd = pcd.voxel_down_sample(voxel_size=voxel)
            log.info(f"Downsampled to {len(pcd.points):,} points")

        # Normal estimation — PCA-based with consistent orientation
        log.info("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 5, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # Higher Poisson depth = more detail but more RAM
        # depth=9 for dense clouds, depth=7 for sparse
        depth = 9 if len(pcd.points) > 50_000 else 7
        log.info(f"Poisson reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )

        # Trim the lowest-density outer shell (Poisson artifacts at boundaries)
        densities_np = np.asarray(densities)
        mesh.remove_vertices_by_mask(densities_np < np.quantile(densities_np, 0.005))
        log.info(f"Poisson mesh: {len(mesh.triangles):,} faces")
        return mesh

    # ── Mesh cleanup helpers ───────────────────────────────────────
    def _keep_largest_component(self, mesh):
        """Remove all connected components except the largest one."""
        import open3d as o3d
        tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        cluster_n_tris = np.asarray(cluster_n_tris)
        if len(cluster_n_tris) == 0:
            return mesh
        largest = int(np.argmax(cluster_n_tris))
        mesh.remove_triangles_by_mask(tri_clusters != largest)
        mesh.remove_unreferenced_vertices()
        return mesh

    def _fill_holes(self, mesh):
        """
        Fill small open boundaries using trimesh ear-clipping.
        Gracefully skips if trimesh is not installed.
        """
        try:
            import trimesh
            import open3d as o3d
            tm = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                process=False,
            )
            before = len(tm.faces)
            trimesh.repair.fill_holes(tm)
            filled = o3d.geometry.TriangleMesh()
            filled.vertices = o3d.utility.Vector3dVector(tm.vertices)
            filled.triangles = o3d.utility.Vector3iVector(tm.faces)
            log.info(f"Hole filling: {before:,} → {len(tm.faces):,} faces")
            return filled
        except ImportError:
            log.warning("trimesh not found, skipping hole filling.")
            return mesh
        except Exception as e:
            log.warning(f"Hole filling failed ({e}), skipping.")
            return mesh

    def _quality_report(self, mesh):
        """Log mesh quality metrics."""
        log.info("─" * 40)
        log.info(f"  Vertices  : {len(mesh.vertices):,}")
        log.info(f"  Faces     : {len(mesh.triangles):,}")
        log.info(f"  Watertight: {mesh.is_watertight()}")
        log.info(f"  Manifold  : {mesh.is_edge_manifold()}")
        if not mesh.is_watertight():
            log.warning("  Mesh is not watertight — open boundaries present")
        log.info("─" * 40)

    def _export(self, mesh, output_path: Path) -> bool:
        """Write mesh to OBJ or PLY. Returns True on success."""
        import open3d as o3d
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ext = output_path.suffix.lower()
        if ext not in (".obj", ".ply"):
            log.warning(f"Unknown extension '{ext}', writing as .ply")
            output_path = output_path.with_suffix(".ply")

        ok = o3d.io.write_triangle_mesh(
            str(output_path), mesh,
            write_ascii=(ext == ".obj"),
            write_vertex_normals=True,
            write_triangle_uvs=False,
        )
        if ok:
            log.info(f"Mesh saved: {output_path}  ({len(mesh.triangles):,} faces)")
        else:
            log.error("Failed to write mesh.")
        return bool(ok)
