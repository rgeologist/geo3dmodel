import unittest
import numpy as np
import pandas as pd
import geo3dmodel

class TestBootstrapFitPlane(unittest.TestCase):
    
    def setUp(self):
        # Create a known plane: z = 0.5x + 0.3y + noise
        rng = np.random.default_rng(42)
        n = 200
        x = rng.uniform(-10, 10, n)
        y = rng.uniform(-10, 10, n)
        z = 0.5 * x + 0.3 * y + rng.normal(0, 0.1, n)
        self.df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    def test_output_shape(self):
        result = self.df.pcloud.bootstrap_fit_plane(n_realizations=50, random_state=42)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result.columns), 11)  # 11 columns
        self.assertLessEqual(len(result), 50)
        self.assertGreater(len(result), 0)
    
    def test_columns(self):
        result = self.df.pcloud.bootstrap_fit_plane(n_realizations=10, random_state=42)
        expected = ['strike', 'dip', 'dip_direction', 'residual_rmse', 'residual_sum',
                    'centroid_x', 'centroid_y', 'centroid_z',
                    'normal_x', 'normal_y', 'normal_z']
        self.assertEqual(list(result.columns), expected)
    
    def test_reproducibility(self):
        r1 = self.df.pcloud.bootstrap_fit_plane(n_realizations=20, random_state=123)
        r2 = self.df.pcloud.bootstrap_fit_plane(n_realizations=20, random_state=123)
        pd.testing.assert_frame_equal(r1, r2)
    
    def test_sample_fraction(self):
        result = self.df.pcloud.bootstrap_fit_plane(
            n_realizations=30, sample_fraction=0.5, random_state=42
        )
        self.assertGreater(len(result), 0)
    
    def test_valid_ranges(self):
        result = self.df.pcloud.bootstrap_fit_plane(n_realizations=50, random_state=42)
        self.assertTrue((result['strike'] >= 0).all() and (result['strike'] < 360).all())
        self.assertTrue((result['dip'] >= 0).all() and (result['dip'] <= 90).all())
    
    def test_too_few_points(self):
        df_small = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6]})
        with self.assertRaises(ValueError):
            df_small.pcloud.bootstrap_fit_plane(n_realizations=10)
    
    def test_methods(self):
        for method in ['pca', 'svd', 'least_squares']:
            result = self.df.pcloud.bootstrap_fit_plane(
                method=method, n_realizations=10, random_state=42
            )
            self.assertGreater(len(result), 0)
    
    def test_ransac(self):
        result = self.df.pcloud.bootstrap_fit_plane(
            method='ransac', n_realizations=10, random_state=42
        )
        self.assertGreater(len(result), 0)


class TestOrientationConfidencePolygon(unittest.TestCase):
    
    def setUp(self):
        # Simulate a tight cluster of orientations
        rng = np.random.default_rng(42)
        self.strikes = 45.0 + rng.normal(0, 5, 100)
        self.dips = 30.0 + rng.normal(0, 3, 100)
    
    def test_output_shape(self):
        from geo3dmodel.model3d import orientation_confidence_polygon
        polygon = orientation_confidence_polygon(self.strikes, self.dips)
        self.assertEqual(polygon.ndim, 2)
        self.assertEqual(polygon.shape[1], 2)
    
    def test_confidence_values(self):
        from geo3dmodel.model3d import orientation_confidence_polygon
        p90 = orientation_confidence_polygon(self.strikes, self.dips, confidence=0.90)
        p50 = orientation_confidence_polygon(self.strikes, self.dips, confidence=0.50)
        # P50 polygon should generally be smaller (contained within P90)
        self.assertIsNotNone(p90)
        self.assertIsNotNone(p50)
    
    def test_too_few_orientations(self):
        from geo3dmodel.model3d import orientation_confidence_polygon
        with self.assertRaises(ValueError):
            orientation_confidence_polygon([10, 20], [30, 40])
    
    def test_invalid_confidence(self):
        from geo3dmodel.model3d import orientation_confidence_polygon
        with self.assertRaises(ValueError):
            orientation_confidence_polygon(self.strikes, self.dips, confidence=1.5)

if __name__ == '__main__':
    unittest.main()
