import unittest
import numpy as np
import pandas as pd

try:
    import open3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import pyvista
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from geo3dmodel.model3d import (
    Trimesh3d,
    build_surface_from_disks,
    build_vertical_surface,
    compute_surface_boundary_polygon,
    generate_masked_surface_grid,
    fit_surface_z_weighted_avg,
    fit_surface_z_quadric,
    fit_surface_z_rbf,
    fit_surface_z_bspline,
    interpolate_surface_z_weighted_avg,
    interpolate_surface_z_quadric,
    interpolate_surface_z_rbf,
    interpolate_surface_z_bspline,
    build_transformed_trimesh3d,
)
from geokitpy import geotensors as gkp
import shapely


class TestModel3D(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(True)
        
    def test_trimesh3d_dataclass(self):
        verts = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        faces = [[0, 1, 2]]
        mesh = Trimesh3d(vertices=verts, faces=faces)
        self.assertIsInstance(mesh.vertices, np.ndarray)
        self.assertIsInstance(mesh.faces, np.ndarray)
        self.assertEqual(mesh.vertices.shape, (3, 3))
        self.assertEqual(mesh.faces.shape, (1, 3))
        # Test .triangles alias compatibility
        self.assertTrue(np.array_equal(mesh.triangles, mesh.faces))

    def test_build_surface_from_disks(self):
        disk1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        disk2 = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]])
        mesh = build_surface_from_disks([disk1, disk2])
        self.assertIsInstance(mesh, Trimesh3d)
        self.assertEqual(mesh.vertices.shape[0], 6)

    def test_build_vertical_surface(self):
        line = shapely.LineString([(0, 0), (10, 0)])
        mesh = build_vertical_surface(line)
        self.assertIsInstance(mesh, Trimesh3d)
        self.assertTrue(len(mesh.vertices) > 0)
        self.assertTrue(len(mesh.faces) > 0)

    def test_fit_with_weighted_avg_basic(self):
        # Create a synthetic planar patch of 3D points
        np.random.seed(42)
        x = np.random.uniform(0, 10, 80)
        y = np.random.uniform(0, 10, 80)
        z = 0.5 * x + 0.2 * y + np.random.normal(0, 0.05, 80)
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        mesh = df.pcloud.fit_with_weighted_avg(n_elements=15)
        self.assertIsInstance(mesh, Trimesh3d)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertEqual(mesh.faces.shape[1], 3)
        self.assertTrue(np.array_equal(mesh.triangles, mesh.faces))

    def test_fit_with_weighted_avg_concave_boundary(self):
        # L-shaped point distribution to test concave boundary
        np.random.seed(42)
        n = 50
        pts1 = np.c_[np.random.uniform(0, 10, n), np.random.uniform(0, 2, n), np.zeros(n)]
        pts2 = np.c_[np.random.uniform(0, 2, n), np.random.uniform(0, 10, n), np.zeros(n)]
        pts = np.vstack([pts1, pts2])
        df = pd.DataFrame(pts, columns=["x", "y", "z"])

        mesh_concave = df.pcloud.fit_with_weighted_avg(boundary="concave", concave_ratio=0.3, n_elements=15)
        self.assertIsInstance(mesh_concave, Trimesh3d)
        self.assertGreater(len(mesh_concave.vertices), 0)
        self.assertGreater(len(mesh_concave.faces), 0)

    def test_fit_with_weighted_avg_weightings(self):
        np.random.seed(42)
        x = np.random.uniform(-5, 5, 60)
        y = np.random.uniform(-5, 5, 60)
        z = np.sin(x) + np.cos(y)
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        for w in ["idw", "gaussian", "exponential", "uniform"]:
            mesh = df.pcloud.fit_with_weighted_avg(n_elements=10, weighting=w)
            self.assertIsInstance(mesh, Trimesh3d)
            self.assertGreater(len(mesh.vertices), 0)

    def test_fit_with_weighted_avg_spacing(self):
        np.random.seed(42)
        x = np.random.uniform(0, 20, 50)
        y = np.random.uniform(0, 20, 50)
        z = np.zeros(50)
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        mesh = df.pcloud.fit_with_weighted_avg(spacing=2.5)
        self.assertIsInstance(mesh, Trimesh3d)
        self.assertGreater(len(mesh.vertices), 0)

    def test_pcloud_pca(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = 2.0 * x - 1.5 * y + 3.0
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        pca_obj = df.pcloud.pca(n_components=3)
        self.assertEqual(pca_obj.components_.shape, (3, 3))
        self.assertEqual(len(pca_obj.explained_variance_), 3)

        # Less than 3 points raises ValueError
        df_small = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        with self.assertRaises(ValueError):
            df_small.pcloud.pca()

    def test_pcloud_svd(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = 2.0 * x - 1.5 * y + 3.0
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        u, s, vt = df.pcloud.svd()
        self.assertEqual(u.shape[0], 50)
        self.assertEqual(len(s), 3)
        self.assertEqual(vt.shape, (3, 3))

    def test_pcloud_ransac_standalone(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = 0.5 * x + 1.0
        x_out = np.random.uniform(0, 10, 10)
        y_out = np.random.uniform(0, 10, 10)
        z_out = np.random.uniform(50, 100, 10)

        df = pd.DataFrame({
            "x": np.concatenate([x, x_out]),
            "y": np.concatenate([y, y_out]),
            "z": np.concatenate([z, z_out]),
        })

        plane, residual = df.pcloud.ransac(distance_threshold=0.1, random_state=42)
        self.assertIsInstance(plane, gkp.Plane)
        self.assertIsInstance(residual, float)

        plane2, residual2, inliers = df.pcloud.ransac(
            distance_threshold=0.1, random_state=42, return_inliers=True
        )
        self.assertIsInstance(inliers, pd.Series)
        self.assertEqual(len(inliers), 60)
        self.assertEqual(list(inliers.index), list(df.index))
        self.assertGreaterEqual(inliers.iloc[:50].sum(), 45)
        self.assertLessEqual(inliers.iloc[50:].sum(), 2)

    def test_fit_plane_to_points_methods(self):
        # Dipping plane: z = 0.5*x + 0.0*y + 2.0
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = 0.5 * x + 2.0
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        for method in ["pca", "least_squares", "svd", "ransac"]:
            plane, residual = df.pcloud.fit_plane_to_points(method=method)
            self.assertIsInstance(plane, gkp.Plane)
            self.assertIsInstance(residual, float)
            self.assertGreaterEqual(residual, 0.0)
            self.assertGreaterEqual(plane.dip, 0.0)
            self.assertLessEqual(plane.dip, 90.0)

    def test_fit_plane_to_points_exact_3points(self):
        pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        df = pd.DataFrame(pts, columns=["x", "y", "z"])

        for method in ["pca", "least_squares", "svd"]:
            plane, residual = df.pcloud.fit_plane_to_points(method=method)
            self.assertIsInstance(plane, gkp.Plane)
            self.assertIsInstance(residual, float)
            self.assertAlmostEqual(residual, 0.0, places=5)
            self.assertAlmostEqual(plane.dip, 0.0, delta=1.0)

    def test_fit_plane_to_points_vertical(self):
        # Vertical plane: x = 5.0
        y = np.linspace(0, 10, 20)
        z = np.linspace(0, 10, 20)
        x = np.full_like(y, 5.0)
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        for method in ["pca", "svd"]:
            plane, residual = df.pcloud.fit_plane_to_points(method=method)
            self.assertIsInstance(plane, gkp.Plane)
            self.assertAlmostEqual(residual, 0.0, places=4)
            self.assertAlmostEqual(plane.dip, 90.0, delta=1.0)

    def test_distance_to_plane_immutable(self):
        df = pd.DataFrame({
            "x": [0.0, 1.0, 0.0, 0.0],
            "y": [0.0, 0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 0.0, 2.0],
        })
        orig_cols = list(df.columns)
        plane, _ = df.iloc[:3].pcloud.fit_plane_to_points(method="pca")

        dists_signed = df.pcloud.distance_to_plane(plane, signed=True)
        self.assertIsInstance(dists_signed, pd.Series)
        self.assertEqual(len(dists_signed), 4)
        self.assertEqual(list(df.columns), orig_cols)

        dists_abs = df.pcloud.distance_to_plane(plane, signed=False)
        self.assertTrue(np.all(dists_abs >= 0))
        self.assertAlmostEqual(dists_abs.iloc[3], 2.0, places=4)

    def test_evaluate_fit_plane(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        z = np.random.normal(0, 0.1, 40)
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        plane, _ = df.pcloud.fit_plane_to_points(method="pca")
        metrics = df.pcloud.evaluate_fit_plane(plane)
        self.assertIsInstance(metrics, dict)
        for k in ["rmse", "mae", "mad", "max", "sum"]:
            self.assertIn(k, metrics)
            self.assertIsInstance(metrics[k], float)
            self.assertGreaterEqual(metrics[k], 0.0)

    def test_fit_surface_to_points_methods(self):
        np.random.seed(42)
        x = np.random.uniform(-5, 5, 80)
        y = np.random.uniform(-5, 5, 80)
        z = 0.05 * (x**2 + y**2)
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        for method in ["weighted_avg", "quadric", "rbf", "bspline"]:
            mesh = df.pcloud.fit_surface_to_points(method=method, n_elements=10)
            self.assertIsInstance(mesh, Trimesh3d)
            self.assertGreater(len(mesh.vertices), 0)
            self.assertGreater(len(mesh.faces), 0)

        # Also test cubic polynomial (degree 3)
        mesh_cubic = df.pcloud.fit_surface_to_points(method="quadric", degree=3, n_elements=10)
        self.assertIsInstance(mesh_cubic, Trimesh3d)

    def test_distance_to_surface(self):
        df = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 1.0, 2.0],
            "z": [0.0, 0.0, 0.0],
        })
        orig_cols = list(df.columns)
        mesh = df.pcloud.fit_surface_to_points(method="weighted_avg", n_elements=5)
        dists = df.pcloud.distance_to_surface(mesh)

        self.assertIsInstance(dists, pd.Series)
        self.assertEqual(len(dists), 3)
        self.assertEqual(list(df.columns), orig_cols)

    def test_pcloud_fit_surface_z_individual_methods(self):
        np.random.seed(42)
        x = np.random.uniform(-5, 5, 60)
        y = np.random.uniform(-5, 5, 60)
        z = 0.1 * x**2 + 0.1 * y**2
        df = pd.DataFrame({"x": x, "y": y, "z": z})

        mesh_quad = df.pcloud.fit_surface_z_quadric(n_elements=10)
        self.assertIsInstance(mesh_quad, Trimesh3d)

        mesh_rbf = df.pcloud.fit_surface_z_rbf(n_elements=10, smooth=0.1)
        self.assertIsInstance(mesh_rbf, Trimesh3d)

        mesh_bspline = df.pcloud.fit_surface_z_bspline(n_elements=10)
        self.assertIsInstance(mesh_bspline, Trimesh3d)

        mesh_wavg = df.pcloud.fit_surface_z_weighted_avg(n_elements=10)
        self.assertIsInstance(mesh_wavg, Trimesh3d)

    def test_pcloud_interpolate_surface_z_aliases(self):
        df = pd.DataFrame({
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "z": [0.0, 0.1, 0.2, 0.1, 0.2, 0.3],
        })

        mesh_interp_quad = df.pcloud.interpolate_surface_z_quadric(n_elements=5)
        self.assertIsInstance(mesh_interp_quad, Trimesh3d)

        mesh_interp_rbf = df.pcloud.interpolate_surface_z_rbf(n_elements=5)
        self.assertIsInstance(mesh_interp_rbf, Trimesh3d)

        mesh_interp_bspline = df.pcloud.interpolate_surface_z_bspline(n_elements=5)
        self.assertIsInstance(mesh_interp_bspline, Trimesh3d)

        mesh_interp_wavg = df.pcloud.interpolate_surface_z_weighted_avg(n_elements=5)
        self.assertIsInstance(mesh_interp_wavg, Trimesh3d)

        mesh_with_quad = df.pcloud.fit_with_quadric(n_elements=5)
        self.assertIsInstance(mesh_with_quad, Trimesh3d)

        mesh_with_rbf = df.pcloud.fit_with_rbf(n_elements=5)
        self.assertIsInstance(mesh_with_rbf, Trimesh3d)

        mesh_with_bspline = df.pcloud.fit_with_bspline(n_elements=5)
        self.assertIsInstance(mesh_with_bspline, Trimesh3d)

    def test_module_level_surface_z_aliases(self):
        self.assertIs(interpolate_surface_z_quadric, fit_surface_z_quadric)
        self.assertIs(interpolate_surface_z_rbf, fit_surface_z_rbf)
        self.assertIs(interpolate_surface_z_bspline, fit_surface_z_bspline)
        self.assertIs(interpolate_surface_z_weighted_avg, fit_surface_z_weighted_avg)

    @unittest.skipIf(not HAS_OPEN3D, "open3d not installed")
    def test_open3d_feature(self):
        pass

    @unittest.skipIf(not HAS_PYVISTA, "pyvista not installed")
    def test_pyvista_feature(self):
        pass


if __name__ == "__main__":
    unittest.main()
