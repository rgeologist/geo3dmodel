# -*- coding: utf-8 -*-
"""Unit tests for model3d_pyvista.py."""

import unittest
from unittest.mock import MagicMock, patch

try:
    import pyvista
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from geo3dmodel.model3d_pyvista import Model3D_pyvista

class TestModel3DPyvista(unittest.TestCase):
    """Test suite for Model3D_pyvista class."""
    
    @unittest.skipIf(not HAS_PYVISTA, "PyVista is not installed.")
    def test_init_success(self):
        """Test successful initialization when pyvista is available."""
        model = Model3D_pyvista()
        self.assertIsNotNone(model.multi_block)
        self.assertIsNotNone(model.plotter)
        
    @patch('geo3dmodel.model3d_pyvista.PYVISTA', False)
    def test_init_fails_without_pyvista(self):
        """Test initialization raises ImportError when pyvista is missing."""
        with self.assertRaises(ImportError):
            Model3D_pyvista()

    @unittest.skipIf(not HAS_PYVISTA, "PyVista is not installed.")
    def test_methods_exist(self):
        """Test that all expected methods exist and can be called."""
        model = Model3D_pyvista()
        
        # We can mock parameters to just test they exist and don't raise NotImplementedError unexpectedly
        model.plot_borehole(borehole=None)
        model.plot_rectangular_mesh()
        model.plot_plane(points_3d_df=None)
        model.plot_point_cloud(df=None)
        model.add_borehole_path(borehole=None)
        model.add_boreholes_paths(borehole_dict={})
        model.add_borehole_interval()
        model.plot_log_along_borehole(borehole=None, log=None)
        model.plot_one_structure_as_disk(strike=0, dip=0, center=(0,0,0))
        
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
