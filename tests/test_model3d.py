import unittest
import numpy as np

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

from geo3dmodel.model3d import Trimesh3d, build_surface_from_disks, build_vertical_surface
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

    @unittest.skipIf(not HAS_OPEN3D, "open3d not installed")
    def test_open3d_feature(self):
        pass

    @unittest.skipIf(not HAS_PYVISTA, "pyvista not installed")
    def test_pyvista_feature(self):
        pass

if __name__ == "__main__":
    unittest.main()
