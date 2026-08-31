# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:39:08 2021

@author: Raymi Castilla
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from .model3d_abstract import Model3D_abstract

class Model3D_matplotlib(Model3D_abstract):
    """3D Model representation using Matplotlib.

    Attributes:
        fig (matplotlib.figure.Figure): Matplotlib figure instance.
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Matplotlib 3D axes instance.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Model3D_matplotlib object.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
    
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
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
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
            center (Tuple[float, float, float], optional): Center coordinate. Defaults to (0.0,0.0,0.0).
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
    
    def add_one_structure_as_disk(
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
        pass

    def _set_axes_equal(self) -> None:
        """Make axes of 3D plot have equal scale."""
        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        self.ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def format_fig(self, template: str = 'white', legend_separate: bool = False) -> None:
        """Format the figure, setting equal aspect ratio and applying templates.

        Args:
            template (str, optional): The template to use ('white' or 'paraview'). Defaults to 'white'.
            legend_separate (bool, optional): If True, places legend in a separate figure. Defaults to False.
        """
        # Ensure equal aspect ratio
        try:
            self.ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass
        self._set_axes_equal()

        if template == 'paraview':
            self.fig.patch.set_facecolor('#333333')
            self.ax.set_facecolor('#333333')
            self.ax.xaxis.label.set_color('white')
            self.ax.yaxis.label.set_color('white')
            self.ax.zaxis.label.set_color('white')
            self.ax.tick_params(axis='x', colors='white')
            self.ax.tick_params(axis='y', colors='white')
            self.ax.tick_params(axis='z', colors='white')
            text_color = 'white'
        else:
            self.fig.patch.set_facecolor('white')
            self.ax.set_facecolor('white')
            text_color = 'black'
            
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            if legend_separate:
                fig_leg = plt.figure()
                ax_leg = fig_leg.add_subplot(111)
                ax_leg.axis('off')
                if template == 'paraview':
                    fig_leg.patch.set_facecolor('#333333')
                ax_leg.legend(handles, labels, loc='center', framealpha=1.0, facecolor=self.fig.patch.get_facecolor(), edgecolor=text_color, labelcolor=text_color)
            else:
                self.ax.legend(facecolor=self.fig.patch.get_facecolor(), edgecolor=text_color, labelcolor=text_color)
