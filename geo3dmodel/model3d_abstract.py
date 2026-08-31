# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:39:08 2021

@author: Raymi Castilla
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional


class Model3D_abstract(ABC):
    """Abstract base class for 3D modeling operations."""

    @abstractmethod
    def plot_borehole(self, borehole: Any, **kwargs: Any) -> None:
        """Plots a single borehole in 3D space.

        Args:
            borehole (Any): The borehole object to plot.
            **kwargs (Any): Additional keyword arguments for plotting.

        Returns:
            None: This method does not return a value.
        """
        ...

    @abstractmethod
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
        """Plots a rectangular mesh in 3D space.

        Args:
            strike (float, optional): The strike angle of the mesh. Defaults to None.
            dip (float, optional): The dip angle of the mesh. Defaults to None.
            nodes (Any, optional): The nodes of the mesh. Defaults to None.
            triangles (Any, optional): The triangles defining the mesh. Defaults to None.
            center (Tuple[float, float, float], optional): The center coordinates (x, y, z). Defaults to (0.0, 0.0, 0.0).
            i_elements (int, optional): Number of elements along the i-axis. Defaults to None.
            i_length (float, optional): Length along the i-axis. Defaults to None.
            j_elements (int, optional): Number of elements along the j-axis. Defaults to None.
            j_length (float, optional): Length along the j-axis. Defaults to None.
            row (int, optional): Row index for subplot placement. Defaults to 1.
            col (int, optional): Column index for subplot placement. Defaults to 1.
            **kwargs (Any): Additional keyword arguments for plotting.

        Returns:
            None: This method does not return a value.
        """
        ...

    @abstractmethod
    def plot_plane(
        self,
        points_3d_df: Any,
        c_hull_kwargs: Optional[Dict[str, Any]] = None,
        trace_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Plots a plane based on a set of 3D points.

        Args:
            points_3d_df (Any): DataFrame containing the 3D points.
            c_hull_kwargs (Dict[str, Any], optional): Keyword arguments for convex hull. Defaults to None.
            trace_kwargs (Dict[str, Any], optional): Keyword arguments for trace formatting. Defaults to None.
            **kwargs (Any): Additional keyword arguments for plotting.

        Returns:
            None: This method does not return a value.
        """
        pass

    @abstractmethod
    def plot_point_cloud(self, df: Any, **kwargs: Any) -> None:
        """Plots a 3D point cloud from a DataFrame.

        Args:
            df (Any): DataFrame containing the point cloud data.
            **kwargs (Any): Additional keyword arguments for plotting.

        Returns:
            None: This method does not return a value.
        """
        ...

    @abstractmethod
    def add_borehole_path(self, borehole: Any, **kwargs: Any) -> None:
        """Adds a path for a single borehole to the current 3D plot.

        Args:
            borehole (Any): The borehole object whose path is added.
            **kwargs (Any): Additional keyword arguments for path formatting.

        Returns:
            None: This method does not return a value.
        """
        pass

    @abstractmethod
    def add_boreholes_paths(
        self,
        borehole_dict: Dict[str, Any],
        names_to_skip: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """Adds paths for multiple boreholes to the current 3D plot.

        Args:
            borehole_dict (Dict[str, Any]): Dictionary of borehole objects, keyed by name.
            names_to_skip (List[str], optional): List of borehole names to skip. Defaults to None.
            **kwargs (Any): Additional keyword arguments for path formatting.

        Returns:
            None: This method does not return a value.
        """
        pass

    @abstractmethod
    def add_borehole_interval(self) -> None:
        """Adds an interval representation for a borehole to the plot.

        Returns:
            None: This method does not return a value.
        """
        pass

    @abstractmethod
    def plot_log_along_borehole(self, borehole: Any, log: Any) -> None:
        """Plots a specific log along the trajectory of a borehole.

        Args:
            borehole (Any): The borehole object.
            log (Any): The log data to plot along the borehole.

        Returns:
            None: This method does not return a value.
        """
        pass

    @abstractmethod
    def add_one_structure_as_disk(
        self,
        strike: float,
        dip: float,
        center: Tuple[float, float, float],
        disk_kwargs: Optional[Dict[str, Any]] = None,
        mesh_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Adds a single planar structure represented as a 3D disk.

        Args:
            strike (float): The strike angle of the structure.
            dip (float): The dip angle of the structure.
            center (Tuple[float, float, float]): The center coordinates (x, y, z) of the disk.
            disk_kwargs (Dict[str, Any], optional): Keyword arguments for disk generation. Defaults to None.
            mesh_kwargs (Dict[str, Any], optional): Keyword arguments for mesh formatting. Defaults to None.

        Returns:
            None: This method does not return a value.
        """
        pass

if __name__ == '__main__':
    ...