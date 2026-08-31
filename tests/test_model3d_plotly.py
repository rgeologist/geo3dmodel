import unittest
import sys
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from geo3dmodel.model3d_plotly import Model3D_plotly
    HAS_MODEL3D = True
except ImportError:
    HAS_MODEL3D = False

class TestModel3DPlotly(unittest.TestCase):
    @unittest.skipIf(not HAS_PLOTLY or not HAS_MODEL3D, "Missing plotly or model3d_plotly")
    def setUp(self):
        self.model = Model3D_plotly(local_zero=(0, 0, 0))

    @unittest.skipIf(not HAS_PLOTLY or not HAS_MODEL3D, "Missing plotly or model3d_plotly")
    def test_initialization(self):
        self.assertEqual(self.model.local_zero, (0, 0, 0))
        self.assertIsInstance(self.model.fig, go.Figure)
        self.assertEqual(self.model.traces, [])

    @unittest.skipIf(not HAS_PLOTLY or not HAS_MODEL3D, "Missing plotly or model3d_plotly")
    def test_add_borehole_xyz(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        z = np.array([1, 2, 3])
        self.model.add_borehole_xyz(x=x, y=y, z=z)
        self.assertEqual(len(self.model.traces), 1)
        self.assertIsInstance(self.model.traces[0], go.Scatter3d)

    @unittest.skipIf(not HAS_PLOTLY or not HAS_MODEL3D, "Missing plotly or model3d_plotly")
    def test_generate_default_color_dict(self):
        bh_dict = {'BH1': MagicMock(), 'BH2': MagicMock()}
        color_dict = self.model.generate_default_color_dict(bh_dict)
        self.assertEqual(len(color_dict), 2)
        self.assertIn('BH1', color_dict)
        self.assertIn('BH2', color_dict)

    @unittest.skipIf(not HAS_PLOTLY or not HAS_MODEL3D, "Missing plotly or model3d_plotly")
    def test_merge_traces_and_remove(self):
        self.model.add_borehole_xyz(x=np.array([1]), y=np.array([1]), z=np.array([1]))
        self.model.merge_traces()
        self.assertEqual(len(self.model.fig.data), 1)
        self.model.remove_all_traces()
        self.assertEqual(len(self.model.fig.data), 0)

    @unittest.skipIf(not HAS_PLOTLY or not HAS_MODEL3D, "Missing plotly or model3d_plotly")
    def test_templates(self):
        pv_template = self.model.paraview_template()
        self.assertIsNotNone(pv_template)
        paper_template = self.model.paper_template()
        self.assertIsNotNone(paper_template)

if __name__ == '__main__':
    unittest.main()
