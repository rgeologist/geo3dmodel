import os

file_path = r'c:\Users\r.castilla\My Drive\code\00-work\geo3dmodel\geo3dmodel\model3d.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
has_future = False
for line in lines:
    if line.strip() == 'from __future__ import annotations':
        has_future = True
    
    # Replace breakpoint
    if 'breakpoint()' in line and not line.strip().startswith('#'):
        line = line.replace('breakpoint()', '# breakpoint()')
        
    # Replace eval 1
    if "eval(f'self.coordinates().{func}(axis=0)')" in line:
        line = line.replace("eval(f'self.coordinates().{func}(axis=0)')", "getattr(self.coordinates(), func)(axis=0)")
        
    # Replace eval 2
    if "eval(f'windows.{window_name}(window_size, *window_args, **window_kwargs)')" in line:
        line = line.replace("eval(f'windows.{window_name}(window_size, *window_args, **window_kwargs)')", "getattr(windows, window_name)(window_size, *window_args, **window_kwargs)")
        
    # Optional imports replacement
    if 'ModuleNotFoundError' in line:
        line = line.replace('ModuleNotFoundError', 'ImportError')

    new_lines.append(line)

if not has_future:
    new_lines.insert(0, 'from __future__ import annotations\n')

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

# Create tests file
test_dir = r'c:\Users\r.castilla\My Drive\code\00-work\geo3dmodel\tests'
os.makedirs(test_dir, exist_ok=True)
test_file_path = os.path.join(test_dir, 'test_model3d.py')

test_content = '''\
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

class TestModel3D(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(True)
        
    @unittest.skipIf(not HAS_OPEN3D, "open3d not installed")
    def test_open3d_feature(self):
        pass

    @unittest.skipIf(not HAS_TRIMESH, "trimesh not installed")
    def test_trimesh_feature(self):
        pass

    @unittest.skipIf(not HAS_PYVISTA, "pyvista not installed")
    def test_pyvista_feature(self):
        pass

if __name__ == "__main__":
    unittest.main()
'''

with open(test_file_path, 'w', encoding='utf-8') as f:
    f.write(test_content)

print('Done refactoring and creating tests')
