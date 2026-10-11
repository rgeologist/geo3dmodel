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
    build_surface_from_polylines,
    build_vertical_surface_from_one_polyline_extrusion,
    build_triangulated_surface_from_lengths,
    build_vertical_surface_polyline_and_zlevels,
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
    map_rgb_image_to_vertical_surface,
    map_datarray_to_vertical_surface,
    convert_path_to_orientation_vectors,
    project_structures_to_xsection,
    intersect_structures_and_xsection,
    generate_structural_vectors,
)
from geokitpy import geotensors as gkp
import shapely
import xarray as xr

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None


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

        # Test .unpack() method
        x_flat, y_flat, z_flat, i, j, k = mesh.unpack()
        self.assertTrue(np.array_equal(x_flat, [0, 1, 0]))
        self.assertTrue(np.array_equal(y_flat, [0, 0, 1]))
        self.assertTrue(np.array_equal(z_flat, [0, 0, 0]))
        self.assertTrue(np.array_equal(i, [0]))
        self.assertTrue(np.array_equal(j, [1]))
        self.assertTrue(np.array_equal(k, [2]))

        # Test aliases
        for a, b in zip(mesh.to_xyz_ijk(), (x_flat, y_flat, z_flat, i, j, k)):
            self.assertTrue(np.array_equal(a, b))
        for a, b in zip(mesh.unpack_xyz_ijk(), (x_flat, y_flat, z_flat, i, j, k)):
            self.assertTrue(np.array_equal(a, b))

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
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertEqual(mesh.faces.shape[1], 3)

    def test_build_surface_from_polylines(self):
        line1 = shapely.LineString([(0, 0, 10), (10, 0, 10)])
        line2 = shapely.LineString([(0, 0, 0), (10, 0, 0)])
        mesh = build_surface_from_polylines(line1, line2)
        self.assertIsInstance(mesh, Trimesh3d)
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertEqual(mesh.faces.shape[1], 3)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)
        self.assertTrue(np.array_equal(mesh.triangles, mesh.faces))

    def test_build_vertical_surface_from_one_polyline_extrusion(self):
        line = shapely.LineString([(0, 0, 10), (10, 5, 10)])
        mesh = build_vertical_surface_from_one_polyline_extrusion(line, extrusion_distance=-20)
        self.assertIsInstance(mesh, Trimesh3d)
        self.assertEqual(mesh.vertices.shape[1], 3)
        self.assertEqual(mesh.faces.shape[1], 3)
        self.assertGreater(len(mesh.vertices), 0)
        self.assertGreater(len(mesh.faces), 0)
        self.assertTrue(np.array_equal(mesh.triangles, mesh.faces))

    def test_build_triangulated_surface_from_lengths(self):
        mesh = build_triangulated_surface_from_lengths(n_horizontal=5, n_vertical=4)
        self.assertIsInstance(mesh, Trimesh3d)
        self.assertEqual(mesh.vertices.shape, (20, 3))
        self.assertEqual(mesh.faces.shape[1], 3)
        self.assertTrue(np.array_equal(mesh.triangles, mesh.faces))

    def test_build_vertical_surface_polyline_and_zlevels(self):
        x = np.array([0.0, 5.0, 10.0])
        y = np.array([0.0, 2.0, 4.0])
        z = np.array([0.0, -10.0, -20.0, -30.0])
        mesh = build_vertical_surface_polyline_and_zlevels(x=x, y=y, z=z)
        self.assertIsInstance(mesh, Trimesh3d)
        self.assertEqual(mesh.vertices.shape, (12, 3))
        self.assertEqual(mesh.faces.shape[1], 3)
        self.assertTrue(np.array_equal(mesh.triangles, mesh.faces))

    def test_map_rgb_image_to_vertical_surface(self):
        x = np.array([0.0, 5.0, 10.0])
        y = np.array([0.0, 2.0, 4.0])
        z = np.array([0.0, -10.0, -20.0, -30.0])
        rgb = np.ones((len(z), len(x), 4), dtype=float)
        x_flat, y_flat, z_flat, i_idx, j_idx, k_idx, rgb_flat = map_rgb_image_to_vertical_surface(
            rgb=rgb, x=x, y=y, z=z
        )
        self.assertEqual(len(x_flat), 12)
        self.assertEqual(len(i_idx), len(j_idx))
        self.assertEqual(len(j_idx), len(k_idx))
        self.assertEqual(rgb_flat.shape, (12, 4))

    def test_map_datarray_to_vertical_surface(self):
        x = np.array([0.0, 5.0, 10.0])
        y = np.array([0.0, 2.0, 4.0])
        z = np.array([0.0, -10.0, -20.0, -30.0])
        data = np.ones((len(z), len(x)))
        da = xr.DataArray(data, coords={"z": z, "x": x, "y": ("x", y)}, dims=["z", "x"])
        x_flat, y_flat, z_flat, i_idx, j_idx, k_idx, colors = map_datarray_to_vertical_surface(
            datarray=da
        )
        self.assertEqual(len(x_flat), 12)
        self.assertEqual(len(i_idx), len(j_idx))
        self.assertEqual(colors.shape, (12, 4))

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

    @unittest.skipIf(not HAS_GEOPANDAS, "geopandas not installed")
    def test_convert_path_to_orientation_vectors(self):
        pts = [shapely.Point(0, 0, 0), shapely.Point(10, 0, 0), shapely.Point(10, 10, 0)]
        gdf = gpd.GeoDataFrame({"geometry": pts})
        res = convert_path_to_orientation_vectors(gdf)
        self.assertIn("vx_xsection", res.columns)
        self.assertIn("vy_xsection", res.columns)
        self.assertIn("vz_xsection", res.columns)
        self.assertEqual(res["vx_xsection"].iloc[1], 10.0)
        self.assertEqual(res["vy_xsection"].iloc[2], 10.0)

    @unittest.skipIf(not HAS_GEOPANDAS, "geopandas not installed")
    def test_intersect_structures_and_xsection(self):
        pts = [shapely.Point(0, 0, 0), shapely.Point(10, 0, 0)]
        gdf = gpd.GeoDataFrame({
            "geometry": pts,
            "vx_xsection": [1.0, 1.0],
            "vy_xsection": [0.0, 0.0],
            "vz_xsection": [0.0, 0.0],
            "structure_strike": [90.0, 90.0],
            "structure_dip": [45.0, 45.0],
            "point_type": ["structure_measurement", "structure_contact"],
        })
        res = intersect_structures_and_xsection(gdf)
        self.assertIn("x_intersection", res.columns)
        self.assertIn("y_intersection", res.columns)
        self.assertIn("z_intersection", res.columns)
        self.assertIn("stru_apparent_dip", res.columns)

    @unittest.skipIf(not HAS_GEOPANDAS, "geopandas not installed")
    def test_generate_structural_vectors(self):
        pts = [shapely.Point(0, 0, 0), shapely.Point(10, 0, 0)]
        gdf = gpd.GeoDataFrame({
            "geometry": pts,
            "vx_xsection": [1.0, 1.0],
            "vy_xsection": [0.0, 0.0],
            "vz_xsection": [0.0, 0.0],
            "structure_strike": [90.0, 90.0],
            "structure_dip": [45.0, 45.0],
            "point_type": ["structure_measurement", "structure_contact"],
        })
        gdf = intersect_structures_and_xsection(gdf)
        gdf["hor_distance_geogis"] = [0.0, 10.0]
        res = generate_structural_vectors(gdf, symbol_size=50)
        self.assertIn("x0_stru_symbol", res.columns)
        self.assertIn("x1_stru_symbol", res.columns)
        self.assertIn("z0_stru_symbol", res.columns)
        self.assertIn("z1_stru_symbol", res.columns)

    @unittest.skipIf(not (HAS_GEOPANDAS and HAS_TRIMESH), "geopandas or trimesh not installed")
    def test_project_structures_to_xsection(self):
        path_pts = [shapely.Point(0, 0, 0), shapely.Point(100, 0, 0)]
        path_gdf = gpd.GeoDataFrame({"geometry": path_pts})
        stru_pts = [shapely.Point(50, 10, 0)]
        stru_gdf = gpd.GeoDataFrame({
            "geometry": stru_pts,
            "structure_strike": [90.0],
            "structure_dip": [45.0],
        })
        res = project_structures_to_xsection(path_gdf, stru_gdf)
        self.assertEqual(len(res), 1)
        self.assertIsInstance(res.iloc[0], shapely.Point)


if __name__ == "__main__":
    unittest.main()
