"""
Stage 3 — Mesh cleanup, hole filling, decimation, and export.

Input : triangle mesh PLY from TSDF fusion
Output: clean OBJ or PLY with vertex normals, ≤ max_faces triangles
"""
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class Mesher:
    def __init__(self, max_faces: int = 50000):
        self.max_faces = max_faces

    def run(self, mesh_path: Path, output_path: Path) -> bool:
        try:
            import open3d as o3d
        except ImportError:
            log.error("open3d not found. Install with: pip install open3d")
            return False

        log.info(f"Loading: {mesh_path}")
        ext = mesh_path.suffix.lower()

        # Detect input type — point cloud goes through Poisson, mesh goes straight to cleanup
        pcd = o3d.io.read_point_cloud(str(mesh_path))
        is_point_cloud = len(pcd.points) > 0 and len(pcd.points) != 0

        # A PLY can be either — check if it has faces
        test_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        has_faces = len(test_mesh.triangles) > 0

        if has_faces:
            log.info(f"Input is a mesh: {len(test_mesh.triangles):,} faces")
            mesh = test_mesh
        else:
            log.info(f"Input is a point cloud: {len(pcd.points):,} points — running Poisson reconstruction")
            mesh = self._poisson_from_pcd(pcd)
            if mesh is None:
                return False

        # ── 1. Basic geometry cleanup ──────────────────────────────
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # ── 2. Remove small floating components ───────────────────
        mesh = self._remove_small_components(mesh)
        log.info(f"After component filter: {len(mesh.triangles):,} faces")

        # ── 3. Fill holes BEFORE smoothing/decimation ─────────────
        mesh = self._fill_holes(mesh)

        # ── 4. Light Laplacian smoothing ───────────────────────────
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        # ── 5. Decimate ────────────────────────────────────────────
        n = len(mesh.triangles)
        if n > self.max_faces:
            log.info(f"Decimating {n:,} → {self.max_faces:,} faces...")
            mesh = mesh.simplify_quadric_decimation(self.max_faces)
            log.info(f"After decimation: {len(mesh.triangles):,} faces")

        # ── 6. Final cleanup + normals ─────────────────────────────
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        # ── 7. Quality report ──────────────────────────────────────
        self._quality_report(mesh)

        # ── 8. Export ──────────────────────────────────────────────
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

    def _poisson_from_pcd(self, pcd):
        import open3d as o3d

        n = len(pcd.points)
        log.info(f"Point cloud input: {n:,} points")

        pts_arr = np.asarray(pcd.points)

        # Estimate a sensible voxel size from median nearest-neighbour distance
        # Sample 2000 points to estimate nn distance efficiently
        sample_n = min(2000, n)
        idx = np.random.choice(n, sample_n, replace=False)
        pcd_sample = o3d.geometry.PointCloud()
        pcd_sample.points = o3d.utility.Vector3dVector(pts_arr[idx].astype(np.float64))
        tree = o3d.geometry.KDTreeFlann(pcd_sample)
        nn_dists = []
        for j in range(min(500, sample_n)):
            k, _, dist2 = tree.search_knn_vector_3d(pcd_sample.points[j], 2)
            if k >= 2:
                nn_dists.append(np.sqrt(dist2[1]))
        median_nn = float(np.median(nn_dists)) if nn_dists else 0.01
        # Voxel = 2× median nn, but cap between 0.005m and 0.1m
        voxel = float(np.clip(median_nn * 2.0, 0.005, 0.1))
        log.info(f"Estimated voxel size: {voxel:.4f}m (median_nn={median_nn:.4f}m)")

        if n > 80_000:
            pcd = pcd.voxel_down_sample(voxel_size=voxel)
            log.info(f"Downsampled to {len(pcd.points):,} points")
            # If still too many, double voxel until under 80K
            while len(pcd.points) > 80_000:
                voxel *= 1.5
                pcd = pcd.voxel_down_sample(voxel_size=voxel)
                log.info(f"Re-downsampled to {len(pcd.points):,} points (voxel={voxel:.4f}m)")

        pts_arr = np.asarray(pcd.points)
        normal_radius = voxel * 5

        log.info("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # depth=7: ~1GB RAM, fast, sufficient for 50K face output
        log.info("Running Poisson reconstruction (depth=7)...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False
        )
        densities_np = np.asarray(densities)
        mesh.remove_vertices_by_mask(densities_np < np.quantile(densities_np, 0.01))
        log.info(f"Poisson mesh: {len(mesh.triangles):,} faces")
        return mesh

    def _remove_small_components(self, mesh):
        """Keep only the single largest connected component."""
        import open3d as o3d
        tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        tri_clusters = np.asarray(tri_clusters)
        cluster_n_tris = np.asarray(cluster_n_tris)
        if len(cluster_n_tris) == 0:
            return mesh
        largest = int(np.argmax(cluster_n_tris))
        remove_mask = tri_clusters != largest
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
        return mesh

    def _fill_holes(self, mesh):
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
            after = len(tm.faces)
            filled = o3d.geometry.TriangleMesh()
            filled.vertices = o3d.utility.Vector3dVector(tm.vertices)
            filled.triangles = o3d.utility.Vector3iVector(tm.faces)
            log.info(f"Hole filling: {before:,} → {after:,} faces")
            return filled
        except ImportError:
            log.warning("trimesh not found, skipping hole filling.")
            return mesh
        except Exception as e:
            log.warning(f"Hole filling failed ({e}), skipping.")
            return mesh

    def _quality_report(self, mesh):
        n_v = len(mesh.vertices)
        n_f = len(mesh.triangles)
        watertight = mesh.is_watertight()
        manifold = mesh.is_edge_manifold()
        log.info("─" * 40)
        log.info(f"  Vertices  : {n_v:,}")
        log.info(f"  Faces     : {n_f:,}")
        log.info(f"  Watertight: {watertight}")
        log.info(f"  Manifold  : {manifold}")
        if not watertight:
            log.warning("  Mesh is not watertight — may have open boundaries")
        if n_f > self.max_faces:
            log.warning(f"  Face count {n_f:,} exceeds limit {self.max_faces:,}")
        log.info("─" * 40)
