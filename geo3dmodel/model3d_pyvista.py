# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:39:08 2021

@author: Raymi Castilla
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    import pyvista
    PYVISTA = True
except ImportError:
    PYVISTA = False

try:
    import xarray
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from .model3d_abstract import Model3D_abstract

class Model3D_pyvista(Model3D_abstract):
    """3D Model representation using PyVista.

    Attributes:
        multi_block (Any): PyVista MultiBlock dataset.
        plotter (Any): PyVista Plotter instance.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Model3D_pyvista object.

        Args:
            **kwargs: Arbitrary keyword arguments. Can include 'multi_block' and 'plotter'.

        Raises:
            ImportError: If the pyvista module is not installed.
        """
        if not PYVISTA:            
            raise ImportError('Module pyvista is needed to instantiate this class.')
        
        self.multi_block = kwargs.get('multi_block', pyvista.MultiBlock())
        self.plotter = kwargs.get('plotter', pyvista.Plotter())
    
    def plot_borehole(self, borehole: Any, **kwargs: Any) -> None:
        """Plot a single borehole.

        Args:
            borehole (Any): The borehole object to plot.
            **kwargs: Additional plotting arguments.
        """
        pass
    
    def plot_rectangular_mesh(
        self,
        strike: Optional[float] = None,
        dip: Optional[float] = None,
        nodes: Optional[Any] = None,
        triangles: Optional[Any] = None,
        center: Tuple[float, float, float] = (0, 0, 0),
        i_elements: Optional[int] = None,
        i_length: Optional[float] = None,
        j_elements: Optional[int] = None,
        j_length: Optional[float] = None,
        row: int = 1,
        col: int = 1,
        **kwargs: Any
    ) -> None:
        """Plot a rectangular mesh.

        Args:
            strike (float, optional): Strike angle. Defaults to None.
            dip (float, optional): Dip angle. Defaults to None.
            nodes (Any, optional): Nodes of the mesh. Defaults to None.
            triangles (Any, optional): Triangles of the mesh. Defaults to None.
            center (Tuple[float, float, float], optional): Center coordinate. Defaults to (0,0,0).
            i_elements (int, optional): Number of i elements. Defaults to None.
            i_length (float, optional): Length of i elements. Defaults to None.
            j_elements (int, optional): Number of j elements. Defaults to None.
            j_length (float, optional): Length of j elements. Defaults to None.
            row (int, optional): Subplot row index. Defaults to 1.
            col (int, optional): Subplot column index. Defaults to 1.
            **kwargs: Additional plotting arguments.
        """
        pass
    
    def plot_plane(
        self,
        points_3d_df: Any,
        c_hull_kwargs: Optional[Dict[str, Any]] = None,
        trace_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Plot a plane fitted to 3D points.

        Args:
            points_3d_df (Any): DataFrame containing 3D points.
            c_hull_kwargs (dict, optional): Convex hull arguments. Defaults to None.
            trace_kwargs (dict, optional): Trace arguments. Defaults to None.
            **kwargs: Additional plotting arguments.
        """
        if c_hull_kwargs is None:
            c_hull_kwargs = {}
        if trace_kwargs is None:
            trace_kwargs = {}
        pass
    
    def plot_point_cloud(self, df: Any, **kwargs: Any) -> None:
        """Plot a point cloud.

        Args:
            df (Any): DataFrame containing point cloud data.
            **kwargs: Additional plotting arguments.
        """
        pass
    
    def add_borehole_path(self, borehole: Any, **kwargs: Any) -> None:
        """Add the path of a borehole to the model.

        Args:
            borehole (Any): The borehole object.
            **kwargs: Additional arguments.
        """
        pass
    
    def add_boreholes_paths(
        self,
        borehole_dict: Dict[str, Any],
        names_to_skip: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Add paths for multiple boreholes to the model.

        Args:
            borehole_dict (dict): Dictionary mapping names to borehole objects.
            names_to_skip (list, optional): List of borehole names to skip. Defaults to None.
            **kwargs: Additional arguments.
        """
        if names_to_skip is None:
            names_to_skip = []
        pass
    
    def add_borehole_interval(self) -> None:
        """Add a borehole interval to the model."""
        pass
    
    def plot_log_along_borehole(self, borehole: Any, log: Any) -> None:
        """Plot log data along a borehole path.

        Args:
            borehole (Any): The borehole object.
            log (Any): The log data to plot.
        """
        pass
    
    def plot_one_structure_as_disk(
        self,
        strike: float,
        dip: float,
        center: Tuple[float, float, float],
        disk_kwargs: Optional[Dict[str, Any]] = None,
        mesh_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Plot a single geological structure as a disk.

        Args:
            strike (float): Strike angle of the structure.
            dip (float): Dip angle of the structure.
            center (Tuple[float, float, float]): Center coordinate (x, y, z).
            disk_kwargs (dict, optional): Arguments for disk creation. Defaults to None.
            mesh_kwargs (dict, optional): Arguments for mesh representation. Defaults to None.
        """
        if disk_kwargs is None:
            disk_kwargs = {}
        if mesh_kwargs is None:
            mesh_kwargs = {}
        pass
    
    def add_structured_grid(self, data_array: Any, grid_name: str) -> None:
        """Add a structured grid to the multi-block dataset.

        Args:
            data_array (Any): An xarray DataArray containing grid data.
            grid_name (str): Name to assign to the grid in the multi-block dataset.

        Raises:
            ImportError: If xarray is not installed.
        """
        if not HAS_XARRAY:
            raise ImportError("Module xarray is needed to use this method.")
            
        grid = data_array.pgrid.to_pyvista_structuredgrid()
        data = grid.point_data['values']
        nan_mask = np.isnan(data)
        grid.hide_points(nan_mask)
        self.multi_block[grid_name] = grid
    
    def plot_structured_grid(self, data_array: Any) -> None:
        """Plot a structured grid.

        Args:
            data_array (Any): An xarray DataArray or similar object.

        Raises:
            ImportError: If the pyvista module is not installed.
        """
        if not PYVISTA:            
            raise ImportError('Module pyvista is needed to run this function')
            
        grid = pyvista.UniformGrid()
        
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, cmap="viridis", opacity=0.5, show_edges=True)
        plotter.show()


if __name__ == '__main__':
    pass