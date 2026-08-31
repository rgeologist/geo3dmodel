import unittest
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from geo3dmodel.model3d_matplotlib import Model3D_matplotlib
    HAS_MODEL3D = True
except ImportError:
    HAS_MODEL3D = False

class TestModel3DMatplotlib(unittest.TestCase):
    @unittest.skipIf(not HAS_MATPLOTLIB or not HAS_MODEL3D, "Missing matplotlib or model3d_matplotlib")
    def setUp(self):
        self.model = Model3D_matplotlib()

    @unittest.skipIf(not HAS_MATPLOTLIB or not HAS_MODEL3D, "Missing matplotlib or model3d_matplotlib")
    def test_initialization(self):
        self.assertIsNotNone(self.model.fig)
        self.assertIsNotNone(self.model.ax)

    @unittest.skipIf(not HAS_MATPLOTLIB or not HAS_MODEL3D, "Missing matplotlib or model3d_matplotlib")
    def test_format_fig_white(self):
        self.model.format_fig(template='white', legend_separate=False)
        self.assertEqual(self.model.fig.patch.get_facecolor(), (1.0, 1.0, 1.0, 1.0)) # White

    @unittest.skipIf(not HAS_MATPLOTLIB or not HAS_MODEL3D, "Missing matplotlib or model3d_matplotlib")
    def test_format_fig_paraview(self):
        self.model.format_fig(template='paraview', legend_separate=False)
        # Paraview background is '#333333'
        self.assertTrue(self.model.fig.patch.get_facecolor() != (1.0, 1.0, 1.0, 1.0))

    @unittest.skipIf(not HAS_MATPLOTLIB or not HAS_MODEL3D, "Missing matplotlib or model3d_matplotlib")
    def test_methods_exist(self):
        # Just test that calling them doesn't raise NotImplementedError
        self.model.plot_borehole(None)
        self.model.add_borehole_interval()
        self.model.plot_point_cloud(None)

if __name__ == '__main__':
    unittest.main()
