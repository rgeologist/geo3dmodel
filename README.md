# Geo3DModel

**Geo3DModel** is a robust and flexible 3D modeling package designed for geosciences. It provides a core module to construct geological 3D objects, alongside concrete implementations using powerful visualization libraries like **PyVista** and **Plotly** to seamlessly plot them.

---

## 🌟 Features

- **Core Modeling (`model3d.py`):** Allows you to build and manipulate 3D objects commonly used in geosciences.
- **Abstract Base Engine:** A clean `Model3D_abstract` interface defining standard methods for 3D modeling.
- **Multiple Visualization Backends:** The Plotly and PyVista implementations allow you to easily plot and visualize the 3D objects you've built.
  - **PyVista (`model3d_pyvista`)**: High-performance, VTK-based 3D rendering for large and complex geological datasets.
  - **Plotly (`model3d_plotly`)**: Interactive, browser-based 3D visualizations ideal for web applications and notebooks.
- **Extensive Tooling:** Built on top of a powerful scientific stack including `numpy`, `pandas`, `xarray`, `scipy`, `shapely`, and `scikit-image`.
- **Optional Dependency Handling:** Smooth degradation if heavy dependencies (like `trimesh`, `pyvista`, or `open3d`) are not installed.

## 📦 Installation

You can install the package via `pip`. From the root of the repository, run:

```bash
pip install .
```

To install the development dependencies (which includes `pytest` for running the test suite):

```bash
pip install .[dev]
```

## 🚀 Quick Start

Here is a quick overview of how you can use the package to build and visualize your data:

### Building 3D Objects
```python
from geo3dmodel import model3d

# Build a generic rectangular mesh
vertices, faces = model3d.build_rectangular_mesh(...)
```

### Using the PyVista Backend
```python
from geo3dmodel import Model3D_pyvista

# Initialize the PyVista model
model = Model3D_pyvista()

# (Add your geological meshes, structured grids, or point clouds here)
# model.plot_structured_grid(...)
```

### Using the Plotly Backend
```python
from geo3dmodel import Model3D_plotly

# Initialize the Plotly model for interactive browser plots
model = Model3D_plotly()

# (Add your traces and render)
```

## 🧪 Running Tests

This project comes with a comprehensive unit test suite leveraging `pytest`. To run the tests, simply execute:

```bash
pytest
```
The test suite gracefully skips backend-specific tests if the respective optional dependencies are missing from your environment.

## 🛠️ Architecture

- `geo3dmodel/model3d.py`: Core logic allowing you to build common 3D geoscience objects.
- `geo3dmodel/model3d_abstract.py`: The abstract base class dictating the API contract.
- `geo3dmodel/model3d_pyvista.py`: The concrete PyVista implementation for plotting.
- `geo3dmodel/model3d_plotly.py`: The concrete Plotly implementation for plotting.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is open-source and available under standard licenses.
