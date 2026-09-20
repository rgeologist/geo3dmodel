# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:39:08 2021

@author: Raymi Castilla
"""
from __future__ import annotations

# import os
from os import PathLike
from pathlib import Path
import copy
import time
import logging
from itertools import islice, cycle, repeat
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict, Literal,Sequence

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

import plotly.express as px
from plotly.offline import plot as offline_plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shapely import LineString
import xarray as xr

import geokitpy as gkp
from .model3d_abstract import Model3D_abstract
from . import model3d as m3d


class Model3D_plotly(Model3D_abstract):
    """A 3D model representation using Plotly for rendering."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Model3D_plotly instance.

        Args:
            **kwargs: Arbitrary keyword arguments.
                specs (list): Subplot specifications. Defaults to [[{"type": "scene"}]].
                local_zero (tuple): The local zero coordinates (x, y, z). Defaults to (0,0,0).
        """
        # specs = kwargs.pop('specs', [[{"type": "scene"}]])
        local_zero = kwargs.pop('local_zero', (0, 0, 0))
        fig = go.Figure()
        # fig = make_subplots(rows=rows, cols=cols, 
        #                     specs=specs, **kwargs)
        self.local_zero = local_zero
        self.x0, self.y0, self.z0 = self.local_zero
        self.fig = fig
        self.traces: list[go.BaseTraceType] = []

    def add_borehole_xyz(
        self,
        *,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        **kwargs: Any
    ) -> None:
        """
        Add a borehole trace given its x, y, and z coordinates.

        Args:
            x (ArrayLike): X coordinates.
            y (ArrayLike): Y coordinates.
            z (ArrayLike): Z coordinates.
            **kwargs: Additional keyword arguments for go.Scatter3d.
        """
        trace = go.Scatter3d(
            x=x - self.x0,
            y=y - self.y0,
            z=z - self.z0,
            **kwargs
        )
        self.traces.append(trace)

    def add_borehole(
        self,
        borehole: Any,
        survey_name: str = 'preferred',
        **kwargs: Any
    ) -> None:
        """
        Add a borehole trace using data from a specific survey.

        Args:
            borehole (Any): The borehole object containing surveys.
            survey_name (str): The survey name to use. Defaults to 'preferred'.
            **kwargs: Additional keyword arguments.
        """
        df = borehole.surveys[survey_name].data        
        x, y, z = df.loc[:, ['x', 'y', 'z']].to_numpy().T        
        self.add_borehole_xyz(x=x, y=y, z=z, **kwargs)

    def plot_borehole(self, borehole: Any, **kwargs: Any) -> None:
        """
        Plot a borehole trace. (Alias for add_borehole)

        Args:
            borehole (Any): The borehole object.
            **kwargs: Additional keyword arguments.
        """
        self.add_borehole(borehole, **kwargs)
    
    
    
    def add_cylinder_around_path(
        self,*,
        cylinder = m3d.CylinderAlongPath,
        **formatting_kwargs)->None:
        """
        Adds a cylinder around a path. The cylinder can have variable radius 
        
        Args:
            cylinder (m3d.CylinderAlongPath): CylinderAlongPath instance
            cap_ends (bool): Wheather the ends of the cylinder are caped or not 
            **formatting_kwargs
        """        
        
        cylinder_trimesh = m3d.build_cylinder_around_path(
            cylinder)
        self.add_triangulated_surface(
            cylinder_trimesh,
            **formatting_kwargs
        )
        
    def add_cylinder_around_borehole(
            self,*
            borehole:gkp.Borehole,
            survey_name: str = 'preferred',
            xyz_columns: Sequence|str='xyz',
            cylinder = m3d.CylinderAlongPath,
            formatting_kwargs:dict|None=None,
            )->None:
        
        
        survey = borehole.surveys[survey_name]
        col_names = [col for col in xyz_columns]
        path_df = survey.loc[:,col_names]
        self.add_cylinder_around_path(cylinder=cylinder)
        
    # def plot_borehole_as_tube(self, borehole, **kwargs):
    #     #it doesn't work currently
    #     row = kwargs.pop('row', 1)
    #     col = kwargs.pop('col', 1)
    #     
    #     df = borehole.survey
    #     points = df.loc[:,['x','y','z']].values
    #     
    #     poly = pv.PolyData()
    #     poly.points = points
    #     cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    #     cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    #     cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    #     poly.lines = cells
    #     line = poly
    # 
    #     line["scalars"] = np.arange(line.n_points)
    #     tube = line.tube(radius=0.1)
    #     
    #     triang=tube.triangulate()
    #     points = triang.points
    #     vertex = triang.faces.reshape(-1,4)
    #     
    #     x, y, z = points[0:500,:].T
    #     
    #     trace = go.Mesh3d(x=x, y=y, z=z, **kwargs)
    #     
    #     self.traces.append(trace)
    

    
    
    @staticmethod
    def generate_default_color_dict(borehole_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a default color dictionary for a set of boreholes.

        Args:
            borehole_dict (Dict[str, Any]): Dictionary mapping borehole names to objects.

        Returns:
            Dict[str, str]: Dictionary mapping borehole names to color strings.
        """
        color_dict = {}
        color_lst = [color for color in px.colors.qualitative.Dark24]
        if len(color_lst) > 5:
            color_lst.pop(5) # drop black in the 5th position
        colors = cycle(color_lst)
        for bh_name, color in zip(borehole_dict.keys(), colors):
            color_dict[bh_name] = color
            
        return color_dict

    def add_wellheads_markers(
        self,
        borehole_dict: Dict[str, Any],
        names_to_skip: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Add wellhead markers for a dictionary of boreholes.
        Note: The original class had two add_wellheads methods. This is the first one, renamed to avoid conflict.

        Args:
            borehole_dict (Dict[str, Any]): Dictionary of boreholes.
            names_to_skip (Optional[List[str]]): List of borehole names to skip. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if names_to_skip is None:
            names_to_skip = []
            
        color_dict = kwargs.pop('color_dict', None)
        wellhead_coords_system = kwargs.pop('wellhead_coords_system', 'lv95')
        
        x0, y0, z0 = self.local_zero
        
        if color_dict is None:     
            color_dict = self.generate_default_color_dict(borehole_dict)
        
        for bh_name, bh in borehole_dict.items():
            check_names = [name in bh_name for name in names_to_skip]
            if any(check_names):
                continue
            
            try:
                color = color_dict[bh_name]
                x = bh.wellhead[wellhead_coords_system][0] - x0
                y = bh.wellhead[wellhead_coords_system][1] - y0
                z = bh.wellhead[wellhead_coords_system][2] - z0
                trace = go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers', name=bh_name,
                    marker=dict(color=color, size=12),
                    **kwargs
                )
                kwargs.pop('legendgrouptitle', None)
                self.traces.append(trace) 
            except AttributeError as e:
                logging.warning(f"Failed to add wellhead marker for {bh_name}: {e}")

    def add_borehole_path(
        self,
        borehole: gkp.borehole.Borehole,
        coord_system: str,
        **kwargs: Any
    ) -> None:
        """
        Add a 3D line trace for a borehole path.

        Args:
            borehole (gkp.borehole.Borehole): The borehole object.
            coord_system (str): The coordinate system to use.
            **kwargs: Additional keyword arguments.

        Raises:
            AttributeError: If the borehole has no preferred survey.
        """
        try:
            bh_survey = borehole.surveys.preferred.data
        except AttributeError:
            raise AttributeError(f'borehole {borehole.name} has no survey associated to it')

        color = kwargs.pop('color', None)
        
        hover_template  = 'x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}'
        hover_template += '<br>md: %{text:.f}'
        kwargs.update(dict(hovertemplate=hover_template))
            
        try:
            wh_coords = (
                borehole.header.easting[coord_system],
                borehole.header.northing[coord_system],
                borehole.header.elevation_gl.iloc[0],
            )
            x, y, z = (bh_survey.loc[:, ['x', 'y', 'z']] + wh_coords).values.T
            
        except AttributeError:
            logging.warning(f'Borehole {borehole.name} will not be plotted')
            return
            
        text = bh_survey.md
        kwargs.update(dict(text=text))
        
        try:
            trace = go.Scatter3d(
                x=x - self.x0,
                y=y - self.y0,
                z=z - self.z0,
                mode='lines', name=borehole.name,
                line=dict(color=color, width=5),
                **kwargs
            )
            kwargs.pop('legendgrouptitle', None)
            self.traces.append(trace) 
        except (ValueError, AttributeError) as e:
            logging.warning(f"Failed to add borehole path for {borehole.name}: {e}")

    def add_wellhead(
        self,
        borehole: gkp.borehole.Borehole,
        coord_system: str,
        **kwargs: Any
    ) -> None:
        """
        Add a 3D cone trace representing a wellhead.

        Args:
            borehole (gkp.borehole.Borehole): The borehole object.
            coord_system (str): The coordinate system.
            **kwargs: Additional keyword arguments.
        """
        color = kwargs.pop('color', None)
        
        x0, y0, z0 = self.local_zero
        
        hover_template = 'x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}'
        kwargs.update(dict(hovertemplate=hover_template))
            
        try:
            x, y, z = (
                borehole.header.easting[coord_system],
                borehole.header.northing[coord_system],
                borehole.header.elevation_gl.iloc[0],
            )
        except AttributeError:
            logging.warning(f'Wellhead of {borehole.name} will not be plotted')
            return
            
        try:
            trace = go.Cone(
                x=np.asarray([x]) - x0,
                y=np.asarray([y]) - y0,
                z=np.asarray([z]) - z0,
                u=[0], v=[0], w=[1],
                name=borehole.name,                            
                **kwargs
            )
            self.traces.append(trace) 
        except (ValueError, AttributeError) as e:
            logging.warning(f"Failed to add wellhead for {borehole.name}: {e}")
            
    def add_boreholes_paths(
        self,
        borehole_dict: Dict[str, gkp.borehole.Borehole],
        coord_system: str,
        names_to_skip: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Add paths for multiple boreholes.

        Args:
            borehole_dict (Dict[str, gkp.borehole.Borehole]): Dictionary of boreholes.
            coord_system (str): Coordinate system to use.
            names_to_skip (Optional[List[str]]): List of borehole names to skip. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if names_to_skip is None:
            names_to_skip = []
             
        color_dict = kwargs.pop('color_dict', None)
        
        if color_dict is None:     
            color_dict = self.generate_default_color_dict(borehole_dict)
        
        for bh_name, bh in borehole_dict.items():
            check_names = [name in bh_name for name in names_to_skip]
            if any(check_names):
                continue
            
            self.add_borehole_path(bh, coord_system, color=color_dict.get(bh_name), **kwargs)
    
    def add_wellheads(
        self,
        borehole_dict: Dict[str, gkp.borehole.Borehole],
        coord_system: str,
        names_to_skip: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Add wellheads for multiple boreholes using 3D cones.

        Args:
            borehole_dict (Dict[str, gkp.borehole.Borehole]): Dictionary of boreholes.
            coord_system (str): Coordinate system to use.
            names_to_skip (Optional[List[str]]): List of borehole names to skip. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if names_to_skip is None:
            names_to_skip = []
             
        color_dict = kwargs.pop('color_dict', None)
        
        if color_dict is None:     
            color_dict = self.generate_default_color_dict(borehole_dict)
        
        for bh_name, bh in borehole_dict.items():
            check_names = [name in bh_name for name in names_to_skip]
            if any(check_names):
                continue
                
            kwargs.update(dict(color=color_dict.get(bh_name)))
            self.add_wellhead(bh, coord_system, **kwargs)
            
    def plot_boreholes_3d_lines(
        self,
        borehole_dict: Dict[str, Any],
        names_to_skip: Optional[List[str]] = None,
        **kwargs: Any
    ) -> None:
        """
        Plot multiple boreholes as 3D lines using Plotly Express.

        Args:
            borehole_dict (Dict[str, Any]): Dictionary of boreholes.
            names_to_skip (Optional[List[str]]): List of names to skip. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if names_to_skip is None:
            names_to_skip = []
            
        survey_lst = []
        for bh_name, bh in borehole_dict.items():
            check_names = [name in bh_name for name in names_to_skip]
            if any(check_names):
                continue
            
            df = bh.survey.reset_index()
            df['bh_name'] = bh_name
            survey_lst.append(df)
        
        # unify all borehole surveys in a single dataframe
        if survey_lst:
            bh_df = pd.concat(survey_lst, ignore_index=True)
            self.fig = px.line_3d(bh_df, x="x", y="y", z="z", color='bh_name')
    
    def add_borehole_interval(
        self,
        *,
        survey: pd.DataFrame,
        md_start: float,
        md_end: float,
        sampling: float = 0.05,
        x_column: str = 'x',
        y_column: str = 'y',
        z_column: str = 'z',
        interval_fmt_kwargs: Optional[Dict[str, Any]] = None,
        trace_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a specific interval of a borehole trace.

        Args:
            survey (pd.DataFrame): The survey dataframe.
            md_start (float): Measured depth start.
            md_end (float): Measured depth end.
            sampling (float): Interpolation sampling rate. Defaults to 0.05.
            x_column (str): Column name for x. Defaults to 'x'.
            y_column (str): Column name for y. Defaults to 'y'.
            z_column (str): Column name for z. Defaults to 'z'.
            interval_fmt_kwargs (Optional[Dict[str, Any]]): Line format kwargs. Defaults to None.
            trace_kwargs (Optional[Dict[str, Any]]): Trace kwargs. Defaults to None.
        """
        if interval_fmt_kwargs is None:
            interval_fmt_kwargs = {}
            
        if trace_kwargs is None:
            trace_kwargs = {}
            
        customdata = trace_kwargs.pop('customdata', None)
        x0, y0, z0 = self.local_zero
        bh_survey = survey
        
        md_array = np.arange(md_start, md_end, sampling)
        
        x = np.interp(md_array, bh_survey.md, bh_survey[x_column]) - x0
        y = np.interp(md_array, bh_survey.md, bh_survey[y_column]) - y0
        z = np.interp(md_array, bh_survey.md, bh_survey[z_column]) - z0
        
        if customdata is not None:
            h_template = 'x:%{x:.1f}<br>y:%{y:.1f}<br>z:%{z:.1f}'
            if isinstance(customdata, str):
                customdata_list = list(repeat(customdata, len(x)))            
                h_template += '<br>%{customdata}<extra></extra>'
            elif isinstance(customdata, list):
                for ind, d in enumerate(customdata):
                    h_template += '<br>%{customdata' + '[{}]'.format(ind) + '}'
                customdata_list = [[cdata] * len(x) for cdata in customdata]
            else:
                customdata_list = customdata
                
            trace_kwargs.update(dict(customdata=customdata_list, hovertemplate=h_template))
        
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines', line=interval_fmt_kwargs,
            **trace_kwargs
        )
        self.traces.append(trace)
        
    def add_borehole_with_stratigraphy(
        self,
        borehole_survey: pd.DataFrame,
        *,
        stratigraphy: pd.DataFrame,
        colors: pd.DataFrame,
        **kwargs: Any
    ) -> None:
        """
        Add a borehole trace colored by stratigraphy.

        Args:
            borehole_survey (pd.DataFrame): The borehole survey dataframe.
            stratigraphy (pd.DataFrame): Stratigraphy dataframe with top_md and bottom_md.
            colors (pd.DataFrame): Colors dataframe.
            **kwargs: Additional keyword arguments.
        """
        interval_fmt_kwargs = kwargs.pop('interval_fmt_kwargs', {})        
        trace_kwargs = kwargs.pop('trace_kwargs', {})
        
        depth_top = stratigraphy['top_md']
        depth_bottom = stratigraphy['bottom_md']       
                
        names = stratigraphy.index.to_list()
        colors_array = colors.set_index('name').loc[names, ['R', 'G', 'B']].to_numpy()
        color_strings = [f"rgb({row[0]},{row[1]},{row[2]})" for row in colors_array]
        iterator = zip(depth_top, depth_bottom, color_strings, names)
        
        trace_kwargs.update(dict(showlegend=True))
        for top, bottom, color, name in iterator:
            interval_fmt_kwargs.update(dict(color=color))
            trace_kwargs.update(dict(customdata=name))
            self.add_borehole_interval(
                survey=borehole_survey,
                md_start=top,
                md_end=bottom,
                sampling=1,
                interval_fmt_kwargs=interval_fmt_kwargs,
                trace_kwargs=trace_kwargs,
                **kwargs
            )            
            trace_kwargs.update(dict(showlegend=False))
    
    def plot_log_along_borehole(
        self,
        borehole: Any,
        log: Any,
        colorscale: str = 'Viridis',
        width: int = 10,
        **kwargs: Any
    ) -> None:
        """
        Plot a 3D line representing a log along a borehole trajectory.

        Args:
            borehole (Any): The borehole object.
            log (Any): The log object containing values and depth index.
            colorscale (str): The Plotly colorscale. Defaults to 'Viridis'.
            width (int): Line width. Defaults to 10.
            **kwargs: Additional keyword arguments.
        """
        trace_kwargs = kwargs.pop('trace_kwargs', {})
        every_x_meters = kwargs.pop('every_x_meters', None)
        
        survey = borehole.survey
        x0, y0, z0 = self.local_zero
        
        if every_x_meters:
            new_log = gkp.wlog.resample(log, func='median', resolution=every_x_meters)
            log = new_log
        
        # adapt borehole survey index to wlog index
        survey = borehole.survey.reindex(method='nearest', index=log.index)

        hovertemplate = f'{log.name}'
        hovertemplate += ': %{customdata[0]:.0f}'
        hovertemplate += '<br>MD: %{customdata[1]:.1f}<extra></extra>'        
        customdata = [[val, depth] for val, depth in zip(log.values, log.index.values)]
        
        line_dict = dict(color=log.values, width=width, colorscale=colorscale)
        line_dict.update(trace_kwargs)
        
        trace = go.Scatter3d(
            x=survey.x - x0, y=survey.y - y0, z=survey.z - z0,
            mode='lines', line=line_dict,
            name=f'{log.name}:{borehole.name}',
            hovertemplate=hovertemplate,
            customdata=customdata,
            **kwargs
        )
        self.traces.append(trace)
    
    def plot_all_logs_along_all_boreholes(
        self,
        borehole_dict: Dict[str, Any],
        log_dict: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Plot multiple logs along multiple boreholes.

        Args:
            borehole_dict (Dict[str, Any]): Dictionary of boreholes.
            log_dict (Dict[str, Any]): Dictionary of logs.
            **kwargs: Additional keyword arguments.
        """
        for bh_name, borehole in borehole_dict.items():
            for log_name, log_data in log_dict.items():
                self.plot_log_along_borehole(borehole, log_data, **kwargs)
        
    def plot_rectangular_mesh(
        self,
        *,
        strike: Optional[float] = None,
        dip: Optional[float] = None,
        nodes: Optional[np.ndarray] = None,
        triangles: Optional[np.ndarray] = None,
        center: Tuple[float, float, float] = (0, 0, 0),
        i_elements: Optional[int] = None,
        i_length: Optional[float] = None,
        j_elements: Optional[int] = None,
        j_length: Optional[float] = None,
        row: int = 1,
        col: int = 1,
        **kwargs: Any
    ) -> None:
        """
        Plot a rectangular 3D mesh.

        Args:
            strike (Optional[float]): Strike angle in degrees.
            dip (Optional[float]): Dip angle in degrees.
            nodes (Optional[np.ndarray]): Precomputed nodes array.
            triangles (Optional[np.ndarray]): Precomputed triangles array.
            center (Tuple[float, float, float]): Center point coordinates. Defaults to (0,0,0).
            i_elements (Optional[int]): Number of elements in i direction.
            i_length (Optional[float]): Length in i direction.
            j_elements (Optional[int]): Number of elements in j direction.
            j_length (Optional[float]): Length in j direction.
            row (int): Subplot row. Defaults to 1.
            col (int): Subplot col. Defaults to 1.
            **kwargs: Additional keyword arguments for go.Mesh3d.
        """
        if None not in (strike, dip):
            nodes_out, triangles_out = m3d.build_rectangular_mesh(
                strike=strike, dip=dip, center=center,
                i_elements=i_elements, i_length=i_length,
                j_elements=j_elements, j_length=j_length
            )
            nodes = nodes_out
            triangles = triangles_out
        
        if nodes is not None and triangles is not None:
            x, y, z = nodes.T
            i, j, k = triangles.T
            
            trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)
            self.traces.append(trace)
        
    def add_triangulated_surface(
        self,
        triangulated_surface: m3d.Trimesh3d,
        **kwargs: Any
    ) -> None:
        """
        Add a triangulated surface trace.

        Args:
            triangulated_surface (m3d.Trimesh3d): The Trimesh3d surface object.
            **kwargs: Additional keyword arguments for go.Mesh3d.
        """
        mesh = triangulated_surface
        trace = go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            **kwargs
        )
        self.traces.append(trace)
    
    def create_trace_for_one_structure_as_disk(
        self,
        strike: float,
        dip: float,
        center: ArrayLike,
        draw_outline: bool = False,
        outline_kwargs: Optional[Dict[str, Any]] = None,
        disk_kwargs: Optional[Dict[str, Any]] = None,
        mesh_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[go.Mesh3d, Optional[go.Scatter3d]]:
        """
        Create Plotly traces (mesh and optional outline) for a structural disk.

        Args:
            strike (float): Strike angle in degrees.
            dip (float): Dip angle in degrees.
            center (ArrayLike): Center point of the disk.
            draw_outline (bool): Whether to draw an outline. Defaults to False.
            outline_kwargs (Optional[Dict[str, Any]]): Kwargs for outline trace. Defaults to None.
            disk_kwargs (Optional[Dict[str, Any]]): Kwargs for disk generation. Defaults to None.
            mesh_kwargs (Optional[Dict[str, Any]]): Kwargs for mesh trace. Defaults to None.

        Returns:
            Tuple[go.Mesh3d, Optional[go.Scatter3d]]: The mesh trace and outline trace.
        """
        if disk_kwargs is None:
            disk_kwargs = {}
        if mesh_kwargs is None:
            mesh_kwargs = {}
        if outline_kwargs is None:
            outline_kwargs = {}
        
        try:
            disk, triangles = m3d.build_disk(strike, dip, center=center, **disk_kwargs)
            i, j, k = triangles.T
        except TypeError as te:
            logging.error(f"Error building disk: {te}")
            raise te

        hovertemplate = mesh_kwargs.get('hovertemplate', None)
        if hovertemplate is None:            
            hovertemplate = 'strike:%{customdata[0]:.0f}'
            hovertemplate += '<br>dip:%{customdata[1]:.0f}<extra></extra>'
            
        customdata = mesh_kwargs.get('customdata', None)
        if customdata is None:
            customdata = [strike, dip]        
        customdata_list = [customdata] * disk.shape[0]
        
        name = mesh_kwargs.get('name', None)
        if name is None:
            name = 'structure'
        
        mesh_kwargs.update(dict(
            hovertemplate=hovertemplate,
            customdata=customdata_list,
            name=name
        ))
        
        mesh = go.Mesh3d(
            x=disk[:, 0], y=disk[:, 1], z=disk[:, 2],
            i=i, j=j, k=k, **mesh_kwargs
        )
        outline = None
        if draw_outline:
            outline = go.Scatter3d(
                x=disk[:, 0], y=disk[:, 1], z=disk[:, 2],
                mode='lines', **outline_kwargs
            )
        
        return (mesh, outline)

    def add_one_structure_as_disk(
        self,
        strike: float,
        dip: float,
        center: ArrayLike,
        disk_kwargs: Optional[Dict[str, Any]] = None,
        mesh_kwargs: Optional[Dict[str, Any]] = None,
        row: int = 1,
        col: int = 1
    ) -> None:
        """
        Add a single structural element as a 3D disk trace.

        Args:
            strike (float): Strike angle in degrees.
            dip (float): Dip angle in degrees.
            center (ArrayLike): Center coordinates.
            disk_kwargs (Optional[Dict[str, Any]]): Kwargs for disk builder. Defaults to None.
            mesh_kwargs (Optional[Dict[str, Any]]): Kwargs for mesh trace. Defaults to None.
            row (int): Subplot row. Defaults to 1.
            col (int): Subplot col. Defaults to 1.
        """
        if disk_kwargs is None:
            disk_kwargs = {}
        if mesh_kwargs is None:
            mesh_kwargs = {}
            
        mesh, outline = self.create_trace_for_one_structure_as_disk(
            strike, dip, center,
            disk_kwargs=disk_kwargs,
            mesh_kwargs=mesh_kwargs
        )
        
        self.traces.append(mesh)
        if outline is not None:
            self.traces.append(outline)
            
    def add_multiple_structures_as_disks(
        self,
        df: pd.DataFrame,
        draw_outline: bool = False,
        outline_kwargs: Optional[Dict[str, Any]] = None,
        disk_kwargs: Optional[Dict[str, Any]] = None,
        mesh_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add multiple structural elements as 3D disk traces from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing strike, dip, x, y, z.
            draw_outline (bool): Whether to draw outlines. Defaults to False.
            outline_kwargs (Optional[Dict[str, Any]]): Kwargs for outline trace. Defaults to None.
            disk_kwargs (Optional[Dict[str, Any]]): Kwargs for disk generation. Defaults to None.
            mesh_kwargs (Optional[Dict[str, Any]]): Kwargs for mesh trace. Defaults to None.
        """
        if mesh_kwargs is None:
            mesh_kwargs = {}
            
        showlegend = True
        customdata_columns = mesh_kwargs.pop('customdata_columns', None)
        
        for pl in df.itertuples():            
            mesh_kwargs_1 = copy.deepcopy(mesh_kwargs)
            center = np.array([
                [pl.x - self.x0],
                [pl.y - self.y0],
                [pl.z - self.z0]
            ])
            
            if customdata_columns is None:
                customdata = None
            else:             
                customdata = [getattr(pl, col) for col in customdata_columns]
            
            mesh_kwargs_1.update({'showlegend': showlegend, 'customdata': customdata})
            
            mesh, outline = self.create_trace_for_one_structure_as_disk(
                pl.strike, pl.dip,
                center=center,
                draw_outline=draw_outline,
                outline_kwargs=outline_kwargs,
                disk_kwargs=disk_kwargs,
                mesh_kwargs=mesh_kwargs_1
            )
            
            self.traces.append(mesh)  
            if outline is not None:
                self.traces.append(outline)
            showlegend = False
        
    def plot_fit_to_pointcloud(self, df: pd.DataFrame, **kwargs: Any) -> None:
        """
        Plot a fitted surface or mesh to a point cloud.

        Args:
            df (pd.DataFrame): DataFrame with point cloud data and pcloud accessor.
            **kwargs: Additional keyword arguments.
        """
        mode = kwargs.pop('mode', 'plane')
        row = kwargs.pop('row', 1)
        col = kwargs.pop('col', 1)
        trace_kwargs = kwargs.pop('trace_kwargs', {})
        
        if mode == 'plane':
            c_hull_kwargs = kwargs.pop('c_hull_kwargs', {})
            self.plot_plane(
                df, row=row, col=col,
                c_hull_kwargs=c_hull_kwargs,
                trace_kwargs=trace_kwargs
            )
            
        elif mode == 'interpolate':
            x_grd, y_grd, z_grd = df.pcloud.interpolated_grid(num_pts=100)
            self.plot_gridded_surface(
                x_grd, y_grd, z_grd,
                row=1, col=1, **trace_kwargs
            )
            
        elif mode == 'mesh':
            triangulation = df.pcloud.mesh_points()
            i, j, k = triangulation.triangles.T
            trace = go.Mesh3d(
                x=df.x, y=df.y, z=df.z,
                i=i, j=j, k=k, 
                opacity=0.7, **trace_kwargs
            )
            self.traces.append(trace)
            
        elif mode == 'open3d':
            mesh = df.pcloud.fit_with_open3d(**kwargs)
            triangles = np.asarray(mesh.triangles)
            vertices = np.asarray(mesh.vertices)
            trace = go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                **trace_kwargs
            )
            self.traces.append(trace)
            
        elif mode == 'weighted_avg':
            mesh = df.pcloud.fit_with_weighted_avg(**kwargs)
            triangles = np.asarray(mesh.faces if hasattr(mesh, 'faces') else mesh.triangles)
            vertices = np.asarray(mesh.vertices)
            trace = go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                **trace_kwargs
            )
            self.traces.append(trace)
         
    def add_surface_from_datastructure(
        self,
        data_structure: Union[pd.DataFrame, xr.DataArray],
        **kwargs: Any
    ) -> None:
        """
        Add a 3D surface trace from a pandas DataFrame or xarray DataArray.

        Args:
            data_structure (Union[pd.DataFrame, xr.DataArray]): Data source.
            **kwargs: Additional keyword arguments.
        """
        sample_every = kwargs.pop('sample_every', 1)
        if isinstance(data_structure, pd.DataFrame):
            plot_func = go.Mesh3d
            x, y, z = data_structure.loc[::sample_every, ['x', 'y', 'z']].values.T
        elif isinstance(data_structure, xr.DataArray): 
            plot_func = go.Surface
            darray = data_structure[::sample_every, ::sample_every]
            x, y, z = darray.x, darray.y, darray.data
        self.add_surface(x, y, z, plot_func=plot_func, **kwargs)
        
    def add_surface(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        plot_func: Any = None,
        **kwargs: Any
    ) -> None:
        """
        Add a 3D surface (Mesh3d or Surface) given x, y, and z coordinates.

        Args:
            x (ArrayLike): X coordinates.
            y (ArrayLike): Y coordinates.
            z (ArrayLike): Z coordinates.
            plot_func (Any): Plotting function to use (go.Mesh3d or go.Surface).
            **kwargs: Additional keyword arguments.
            
        Raises:
            ValueError: If plot_func is not recognized.
        """
        if plot_func == go.Mesh3d:
            self.add_irregular_surface(x, y, z, **kwargs)
        elif plot_func == go.Surface:
            self.add_gridded_surface(x, y, z, **kwargs)
        else:
            raise ValueError('plot_func is not valid')
            
    def add_irregular_surface(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        **kwargs: Any
    ) -> None:
        """
        Add an irregular 3D surface using Delaunay triangulation (go.Mesh3d).

        Args:
            x (ArrayLike): 1D array of X coordinates.
            y (ArrayLike): 1D array of Y coordinates.
            z (ArrayLike): 1D array of Z coordinates.
            **kwargs: Additional keyword arguments for go.Mesh3d.
        """
        trace = go.Mesh3d(x=x - self.x0, y=y - self.y0, z=z - self.z0, **kwargs)
        self.traces.append(trace)
    
    def add_gridded_surface(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        **kwargs: Any
    ) -> None:
        """
        Add a gridded 3D surface (go.Surface).

        Args:
            x (ArrayLike): X coordinates.
            y (ArrayLike): Y coordinates.
            z (ArrayLike): 2D array of Z coordinates.
            **kwargs: Additional keyword arguments for go.Surface.

        Raises:
            ValueError: If z is not a 2D array-like structure.
        """
        if len(np.array(z).shape) != 2:
            raise ValueError('z must be a 2D list or array')
        trace = go.Surface(x=x - self.x0, y=y - self.y0, z=z - self.z0, **kwargs)
        self.traces.append(trace)
        
    def add_vertical_surface_from_trace(
        self,
        trace: LineString,
        **kwargs: Any
    ) -> None:
        """
        Add a vertical surface trace based on a Shapely LineString.

        Args:
            trace (LineString): The LineString representing the trace.
            **kwargs: Additional keyword arguments.
        """
        if trace.has_z:
            coords = [(c[0] - self.x0, c[1] - self.y0, c[2] - self.z0) for c in trace.coords]
        else:
            coords = [(c[0] - self.x0, c[1] - self.y0) for c in trace.coords]
        
        new_ls = LineString(coords)
        
        max_segment_length = kwargs.pop('max_segment_length', 100)
        xsection_top = kwargs.pop('xsection_top', 900)
        xsection_bottom = kwargs.pop('xsection_bottom', -4000)
        
        trimesh_obj = m3d.build_vertical_surface(
            new_ls,
            max_segment_length=max_segment_length,
            xsection_top=xsection_top,
            xsection_bottom=xsection_bottom
        )
        self.add_triangulated_surface(trimesh_obj, **kwargs)
        
    def plot_plane(
        self,
        points_3d_df: pd.DataFrame,
        row: int = 1,
        col: int = 1,
        c_hull_kwargs: Optional[Dict[str, Any]] = None,
        trace_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Plot a 3D plane using a convex hull from 3D points.

        Args:
            points_3d_df (pd.DataFrame): DataFrame with 3D points.
            row (int): Subplot row. Defaults to 1.
            col (int): Subplot col. Defaults to 1.
            c_hull_kwargs (Optional[Dict[str, Any]]): Convex hull kwargs. Defaults to None.
            trace_kwargs (Optional[Dict[str, Any]]): Trace kwargs. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if c_hull_kwargs is None:
            c_hull_kwargs = {}
        if trace_kwargs is None:
            trace_kwargs = {}
            
        text = trace_kwargs.pop('text', None)
        hull3D = points_3d_df.pcloud.convex_hull_3D(**c_hull_kwargs)
        
        if text is not None:
            text = list(text) * len(hull3D)
            trace_kwargs.update({'text': text})
            
        trace = go.Mesh3d(
            x=hull3D[:, 0], y=hull3D[:, 1], z=hull3D[:, 2],
            **trace_kwargs
        )
        self.traces.append(trace)
        
    def plot_point_cloud(
        self,
        df: pd.DataFrame,
        marker_dict: Optional[Dict[str, Any]] = None,
        other_trace_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """
        Plot a 3D point cloud using a scatter trace.

        Args:
            df (pd.DataFrame): DataFrame with x, y, z columns.
            marker_dict (Optional[Dict[str, Any]]): Marker format dict. Defaults to None.
            other_trace_kwargs (Optional[Dict[str, Any]]): Other trace kwargs. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if marker_dict is None:
            marker_dict = {}
        if other_trace_kwargs is None:
            other_trace_kwargs = {}
            
        trace = go.Scatter3d(
            x=df.x - self.x0, y=df.y - self.y0, z=df.z - self.z0,
            mode='markers', marker=marker_dict,
            **other_trace_kwargs
        )
        self.traces.append(trace)
    
    def merge_all_traces(
        self,
        ) -> None:
        """
        Merge registered traces into the Plotly figure.

        Args:
            
        """
        self.fig.add_traces(self.traces)

    def add_trace(self, trace:go.BaseTraceType)->None:
        """
        add one trace object into the Plotly figure.
    
        Args:
            
        """
        self.fig.add_traces([trace])
        
    
    merge_traces = merge_all_traces
    
    
    def remove_all_traces(self) -> None:
        """Clear all traces from the Plotly figure."""
        self.fig.data = ()
    
    @staticmethod
    def paraview_template(layout_kwargs: Optional[Dict[str, Any]] = None) -> go.Layout:
        """
        Get a Paraview-style Plotly layout template.

        Args:
            layout_kwargs (Optional[Dict[str, Any]]): Additional layout arguments. Defaults to None.

        Returns:
            go.Layout: The configured Plotly Layout object.
        """
        if layout_kwargs is None:
            layout_kwargs = {}
            
        scene = layout_kwargs.pop('scene', {})        
        scene.update(
            yaxis_title="Northing (m)",
            xaxis_title="Easting (m)",
            zaxis_title="Elevation (m)",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        )
        
        camera = layout_kwargs.pop('camera', {})
        camera.update(eye=dict(x=-1.2, y=-1.2, z=1))  
        
        layout_kwargs.update(
            scene=scene,
            scene_camera=camera,
            scene_aspectmode='data',
            scene_aspectratio=dict(x=1, y=1, z=1),
            font_color='white',
            paper_bgcolor='rgb(85,88,110)',
            legend={'itemsizing': 'constant'}
        )
        
        return go.Layout(layout_kwargs)
    
    @staticmethod
    def paper_template(layout_kwargs: Optional[Dict[str, Any]] = None) -> go.Layout:
        """
        Get a paper-ready Plotly layout template.

        Args:
            layout_kwargs (Optional[Dict[str, Any]]): Additional layout arguments. Defaults to None.

        Returns:
            go.Layout: The configured Plotly Layout object.
        """
        if layout_kwargs is None:
            layout_kwargs = {}
            
        default_scene = dict(
            yaxis_title="Local Northing (m)",
            xaxis_title="Local Easting (m)",
            zaxis_title="Elevation (m)",
            xaxis=dict(showbackground=True),
            yaxis=dict(showbackground=True),
            zaxis=dict(showbackground=True)
        )        
        scene = layout_kwargs.pop('scene', {})
        default_scene.update(scene)
        scene = default_scene
        
        default_camera = dict(eye=dict(x=-1.2, y=-1.2, z=1))        
        camera = layout_kwargs.pop('camera', {})
        default_camera.update(camera)
        camera = default_camera
        
        layout_kwargs.update(
            scene=scene,
            scene_camera=camera, 
            scene_aspectmode='data',
            scene_aspectratio=dict(x=1, y=1, z=1),
            font_color='black',
            paper_bgcolor='rgb(255,255,255)',
            legend={'itemsizing': 'constant'}
        )                            
        
        return go.Layout(layout_kwargs)
        
    def format_fig(self, **kwargs: Any) -> None:
        """
        Apply a template layout to the Plotly figure.

        Args:
            **kwargs: Additional keyword arguments including 'template'.
        """
        template = kwargs.pop('template', self.paraview_template())        
        self.fig.update_layout(template)
    
    def plot_offline(self) -> None:
        """Plot the figure offline."""
        offline_plot(self.fig)
        
    def show_in_browser(self) -> None:
        """Display the figure in the default web browser."""
        self.fig.show(renderer='browser')
        
    def save_file(self,
                  filepath:PathLike,
                  overwrite_question:bool=True,
                  **kwargs: Any) -> None:
        """
        Save the plot to a file (HTML, PNG, JPEG, SVG, PDF).

        Args:
            name (str): Output filename.
            **kwargs: Additional arguments for the write function.
        """
        save = True
        
        filepath = Path(filepath)
        
        if filepath.name.endswith('.html'):
            func = self.fig.write_html
        elif filepath.name.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
            func = self.fig.write_image
        else:
            filepath = filepath.with_name(filepath.name + '.html')
            func = self.fig.write_html            
            
        if filepath.exists() and overwrite_question:
            overwrite = input("File already exists. Overwrite? (y/n): ")
            if overwrite.lower() not in ['y', 'yes']:
                save = False
        
        if save:
            func(filepath, **kwargs)            
            if filepath.exists():
                logging.info('File successfully saved')
        else:
            logging.info('File not saved')  

    # def write_html(self, name):
    #     
    #     if not name.endswith('.html'):
    #         name += '.html'            
    #     
    #     save = True
    #         
    #     if os.path.isfile(name):
    #         overwrite = input("File already exists. Overwrite? (y/n): ")
    #         if overwrite not in ['y','yes','Y','YES']:
    #             # os.remove(name)
    #             save = False
    #     
    #     if save:
    #         self.fig.write_html(name)
    #         if os.path.isfile(name):
    #             print('File succesfuly saved')
    #     else:
    #         print('File not saved')
            
    # def write_svg(self, name)

    def show(self, **kwargs: Any) -> None:
        """
        Show the Plotly figure.

        Args:
            **kwargs: Additional keyword arguments for fig.show().
        """
        self.fig.show(**kwargs)


try:
    # delete the accessor to avoid warning 
    del xr.DataArray().pgrid
except AttributeError:
    pass

# @xr.register_dataarray_accessor('pgrid')
# class Pgrid:
#     
#     def __init__(self, data_array):     
#         self._obj = data_array
#         
#     @classmethod
#     def factory(cls, axes, origin, step, size):
#         # xyz = st.Axes.xyz()
#         
#         xyz_list=[np.arange(0, s7ep*s1ze, step=s7ep) for s7ep, s1ze in zip(step, size)]
#         
#         #translate to origin
#         xyz_list = [coord+or1gin for coord, or1gin in zip(xyz_list, origin)]
#         
#         #rotate to align with axes
#         rot_matrix = axes.dircos(st.Axes.xyz())
#         #Coordinates vector
#         xv, yv, zv = np.meshgrid(*xyz_list)   
#         v=st.Vector(np.stack((xv,yv,zv), axis=-1))      
#         #rotation
#         vr = v@rot_matrix
#         
#         darray = xr.DataArray(np.empty(size),
#                               coords=[('x', vr[0,:,0,0]),
#                                       ('y', vr[:,0,0,1]),
#                                       ('z', vr[0,0,:,2])])
#         
#         return darray
        
#---- MAIN
if __name__ == '__main__':
    ...


# ---------------------------------------------------------------------------
# Camera & Palette Helpers
# ---------------------------------------------------------------------------

DEFAULT_CAMERA = dict(
    eye=dict(x=-1.5, y=-1.5, z=1.2),
    center=dict(x=0.0, y=0.0, z=0.0),
    up=dict(x=0.0, y=0.0, z=1.0)
)


def get_default_camera(
    eye: Optional[Dict[str, float]] = None,
    center: Optional[Dict[str, float]] = None,
    up: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, float]]:
    """Return a default or customized camera layout dict."""
    cam = copy.deepcopy(DEFAULT_CAMERA)
    if eye is not None:
        cam["eye"].update(eye)
    if center is not None:
        cam["center"].update(center)
    if up is not None:
        cam["up"].update(up)
    return cam


def get_cluster_palette() -> List[str]:
    """Return a qualitative hex color palette for clusters (Dark24 without pure black)."""
    color_lst = [c for c in px.colors.qualitative.Dark24]
    if len(color_lst) > 5:
        color_lst.pop(5)  # Drop pure black (#222222)
    return color_lst


def assign_cluster_color(existing_colors: Iterable[str]) -> str:
    """Pick the next qualitative color not currently used by active clusters."""
    palette = get_cluster_palette()
    used = set(existing_colors)
    for color in palette:
        if color not in used:
            return color
    idx = len(list(used)) % len(palette)
    return palette[idx]


# ---------------------------------------------------------------------------
# Stereonet 2D Projection Functions
# ---------------------------------------------------------------------------

def project_lambert(
    lon: ArrayLike,
    lat: ArrayLike,
    center_lon: float = 0.0,
    center_lat: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Lambert azimuthal equal-area projection from spherical coordinates (radians) to 2D (x, y).

    Normalized such that the primitive outer circle has radius R = 1.0.
    """
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    cos_lat = np.cos(lat_arr)
    sin_lat = np.sin(lat_arr)
    diff_lon = lon_arr - center_lon
    cos_diff_lon = np.cos(diff_lon)

    inner_k = (
        1.0
        + np.sin(center_lat) * sin_lat
        + np.cos(center_lat) * cos_lat * cos_diff_lon
    )
    inner_k = np.where(inner_k <= 0.0, 1e-15, inner_k)
    k = 1.0 / np.sqrt(inner_k)
    x = k * cos_lat * np.sin(diff_lon)
    y = k * (np.cos(center_lat) * sin_lat - np.sin(center_lat) * cos_lat * cos_diff_lon)
    return x, y


def project_stereographic(
    lon: ArrayLike,
    lat: ArrayLike,
    center_lon: float = 0.0,
    center_lat: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Stereographic equal-angle projection from spherical coordinates (radians) to 2D (x, y).

    Normalized such that the primitive outer circle has radius R = 1.0.
    """
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    cos_lat = np.cos(lat_arr)
    sin_lat = np.sin(lat_arr)
    diff_lon = lon_arr - center_lon
    cos_diff_lon = np.cos(diff_lon)

    inner_k = (
        1.0
        + np.sin(center_lat) * sin_lat
        + np.cos(center_lat) * cos_lat * cos_diff_lon
    )
    inner_k = np.where(inner_k <= 0.0, 1e-15, inner_k)
    k = 1.0 / inner_k
    x = k * cos_lat * np.sin(diff_lon)
    y = k * (np.cos(center_lat) * sin_lat - np.sin(center_lat) * cos_lat * cos_diff_lon)
    return x, y


# ---------------------------------------------------------------------------
# Color Conversion & RGBA Helpers
# ---------------------------------------------------------------------------

def hex_to_rgba(color_str: str, alpha: float = 1.0) -> str:
    """Convert hex (#RRGGBB or #RGB) or rgb(...) string to rgba(r, g, b, alpha)."""
    if not color_str:
        return f"rgba(255, 255, 255, {alpha})"
    color_str = color_str.strip()
    if color_str.startswith("rgba"):
        return color_str
    if color_str.startswith("rgb("):
        inner = color_str[4:-1]
        return f"rgba({inner}, {alpha})"
    clean = color_str.lstrip("#")
    if len(clean) == 6:
        r = int(clean[0:2], 16)
        g = int(clean[2:4], 16)
        b = int(clean[4:6], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    elif len(clean) == 3:
        r = int(clean[0] * 2, 16)
        g = int(clean[1] * 2, 16)
        b = int(clean[2] * 2, 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(255, 255, 255, {alpha})"


# ---------------------------------------------------------------------------
# Functional Trace & Figure Builders for Dash
# ---------------------------------------------------------------------------

def build_point_cloud_trace(
    df: pd.DataFrame,
    marker_colors: Optional[Union[str, Sequence[str]]] = None,
    marker_opacity: Optional[Union[float, Sequence[float]]] = None,
    marker_size: int = 3,
    customdata: Optional[Any] = None,
    hovertext: Optional[Any] = None,
    name: str = "Point Cloud"
) -> go.Scatter3d:
    """Build a standalone go.Scatter3d trace from a DataFrame."""
    m_dict: Dict[str, Any] = dict(size=marker_size)
    if marker_colors is not None:
        m_dict["color"] = marker_colors
    else:
        m_dict["color"] = "#FFFFFF"
    if marker_opacity is not None:
        m_dict["opacity"] = marker_opacity
    else:
        m_dict["opacity"] = 0.8

    kwargs: Dict[str, Any] = {}
    if customdata is not None:
        kwargs["customdata"] = customdata
    if hovertext is not None:
        kwargs["hovertext"] = hovertext
        kwargs["hoverinfo"] = "text"

    return go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode="markers",
        marker=m_dict,
        name=name,
        **kwargs
    )


def build_point_cloud_figure(
    df: pd.DataFrame,
    marker_colors: Optional[Union[str, Sequence[str]]] = None,
    marker_opacity: Optional[Union[float, Sequence[float]]] = None,
    uirevision: str = "static_view",
    template: Optional[go.Layout] = None,
    dragmode: str = "orbit",
    title: Optional[str] = None
) -> go.Figure:
    """Build a 3D Plotly figure with uirevision preservation and dark Paraview-style theme."""
    trace = build_point_cloud_trace(
        df,
        marker_colors=marker_colors,
        marker_opacity=marker_opacity,
        customdata=np.arange(len(df)),
        hovertext=[f"Point #{i}" for i in range(len(df))]
    )
    fig = go.Figure(data=[trace])
    layout = template if template is not None else Model3D_plotly.paraview_template()
    fig.update_layout(layout)
    fig.update_layout(
        uirevision=uirevision,
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        scene=dict(
            dragmode=dragmode,
            camera=get_default_camera()
        )
    )
    if title:
        fig.update_layout(title=title)
    return fig


def build_selection_plane_figure(
    df: pd.DataFrame,
    plane_mode: str = "XY",
    active_indices: Optional[Sequence[int]] = None,
    cluster_colors: Optional[Dict[int, str]] = None,
    hidden_indices: Optional[Set[int]] = None,
    dragmode: str = "lasso",
    camera_eye: Optional[Dict[str, float]] = None,
    camera: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Build the linked 2D Selection Plane figure (Scattergl) supporting box and lasso."""
    n = len(df)
    active_set = set(active_indices) if active_indices else set()
    clusters = cluster_colors or {}
    hidden_set = set(hidden_indices) if hidden_indices else set()

    colors = []
    for i in range(n):
        if i in hidden_set:
            colors.append("rgba(0,0,0,0.0)")
        elif i in active_set:
            colors.append("rgba(255,215,0,1.0)")  # Gold for active selection
        elif i in clusters:
            colors.append(hex_to_rgba(clusters[i], 0.85))
        else:
            base_alpha = 0.5 if (clusters or active_set) else 0.8
            colors.append(f"rgba(255,255,255,{base_alpha})")

    if plane_mode == "XY":
        x_label, y_label = "Easting (m)", "Northing (m)"
        x_pts, y_pts = df["x"].values, df["y"].values
    elif plane_mode == "XZ":
        x_label, y_label = "Easting (m)", "Elevation (m)"
        x_pts, y_pts = df["x"].values, df["z"].values
    elif plane_mode == "YZ":
        x_label, y_label = "Northing (m)", "Elevation (m)"
        x_pts, y_pts = df["y"].values, df["z"].values
    elif plane_mode == "Camera":
        cam = camera or {}
        eye = cam.get("eye", camera_eye or DEFAULT_CAMERA["eye"])
        center = cam.get("center", DEFAULT_CAMERA["center"])
        up = cam.get("up", DEFAULT_CAMERA["up"])
        proj = m3d.project_points_to_view(df[["x", "y", "z"]].values, eye=eye, center=center, up=up)
        x_pts, y_pts = proj[:, 0], proj[:, 1]
        x_label, y_label = "Camera View X (m)", "Camera View Y (m)"
    else:
        x_pts, y_pts = df["x"].values, df["y"].values
        x_label, y_label = "X", "Y"

    trace = go.Scattergl(
        x=x_pts,
        y=y_pts,
        mode="markers",
        marker=dict(
            size=5,
            color=colors,
            opacity=1.0
        ),
        customdata=np.arange(n),
        hoverinfo="none",
        name="Points"
    )

    # Autoscale when camera orientation changes; preserve zoom for fixed planes
    if plane_mode == "Camera":
        cam_ref = camera or ({"eye": camera_eye} if camera_eye else DEFAULT_CAMERA)
        eye_ref = cam_ref.get("eye", {})
        ui_rev = f"camera_{eye_ref.get('x',0):.2f}_{eye_ref.get('y',0):.2f}_{eye_ref.get('z',0):.2f}"
    else:
        ui_rev = f"plane_{plane_mode}"

    fig = go.Figure(data=[trace])
    fig.update_layout(
        dragmode=dragmode,
        uirevision=ui_rev,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(
            title=x_label,
            zeroline=False,
            showgrid=True,
            gridcolor="#333344"
        ),
        yaxis=dict(
            title=y_label,
            zeroline=False,
            showgrid=True,
            gridcolor="#333344",
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor="#1e1e28",
        paper_bgcolor="#1e1e28",
        font_color="#dddddd"
    )
    return fig


def build_stereonet_figure(
    projection: str = "equal_area",
    step_deg: int = 10,
    title: str = "Stereonet (Schmidt Equal-Area)"
) -> go.Figure:
    """Build the static stereonet grid figure in pure Plotly."""
    grid_lines = m3d.generate_stereonet_grid_lonlat(step_deg=step_deg)
    proj_fn = project_lambert if projection == "equal_area" else project_stereographic

    traces = []
    for lons, lats, kind in grid_lines:
        x, y = proj_fn(lons, lats)
        if kind == "primitive":
            traces.append(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color="#ffffff", width=2),
                hoverinfo="none", showlegend=False
            ))
        elif kind == "crosshair":
            traces.append(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color="#666677", width=1.2, dash="dash"),
                hoverinfo="none", showlegend=False
            ))
        else:
            traces.append(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color="#3a3a4e", width=0.8),
                hoverinfo="none", showlegend=False
            ))

    cardinal_annotations = [
        dict(x=0, y=1.07, text="<b>N</b>", showarrow=False, font=dict(color="#ffffff", size=14)),
        dict(x=1.07, y=0, text="<b>E</b>", showarrow=False, font=dict(color="#ffffff", size=14)),
        dict(x=0, y=-1.07, text="<b>S</b>", showarrow=False, font=dict(color="#ffffff", size=14)),
        dict(x=-1.07, y=0, text="<b>W</b>", showarrow=False, font=dict(color="#ffffff", size=14)),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=13), x=0.5, xanchor="center"),
        annotations=cardinal_annotations,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            visible=False,
            range=[-1.18, 1.18],
            fixedrange=True
        ),
        yaxis=dict(
            visible=False,
            range=[-1.18, 1.18],
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor="#1e1e28",
        paper_bgcolor="#1e1e28",
        showlegend=False
    )
    return fig


def create_plane_disk_traces(
    strike: float,
    dip: float,
    center: Sequence[float],
    radius: float = 1.0,
    color: str = "#636efa",
    opacity: float = 0.45,
    name: str = "Plane",
    draw_outline: bool = True,
    outline_color: Optional[str] = None,
    outline_width: float = 2.5,
    num_sides: int = 36,
    custom_info: Optional[str] = None,
) -> Tuple[go.Mesh3d, Optional[go.Scatter3d]]:
    """Build a 3D Mesh3d disk and optional Scatter3d perimeter outline for a structural plane.

    Args:
        strike: Strike angle in degrees (Right-Hand-Rule).
        dip: Dip angle in degrees (0 to 90).
        center: 3D center coordinates (x, y, z).
        radius: Radius of the disk in dataset units.
        color: Mesh surface color hex or rgba string.
        opacity: Opacity for the mesh surface (0.0 - 1.0).
        name: Name of the plane trace.
        draw_outline: Whether to include a Scatter3d perimeter trace.
        outline_color: Color of the perimeter line (defaults to color).
        outline_width: Line width of perimeter outline.
        num_sides: Number of segments for the circular boundary.
        custom_info: Extra info text for hovertemplate.

    Returns:
        Tuple of (go.Mesh3d, Optional[go.Scatter3d]).
    """
    disk, triangles = m3d.build_disk(
        strike=float(strike),
        dip=float(dip),
        center=tuple(center),
        radius=float(radius),
        num_sides=int(num_sides)
    )
    i, j, k = triangles.T

    hovertemplate = (
        f"<b>{name}</b><br>"
        f"Strike: {strike:.1f}°<br>"
        f"Dip: {dip:.1f}°"
    )
    if custom_info:
        hovertemplate += f"<br>{custom_info}"
    hovertemplate += "<extra></extra>"

    mesh_trace = go.Mesh3d(
        x=disk[:, 0],
        y=disk[:, 1],
        z=disk[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        name=name,
        hovertemplate=hovertemplate,
        showlegend=False,
    )

    outline_trace = None
    if draw_outline:
        outline_c = outline_color or color
        outline_trace = go.Scatter3d(
            x=disk[:, 0],
            y=disk[:, 1],
            z=disk[:, 2],
            mode="lines",
            line=dict(color=outline_c, width=outline_width),
            name=f"{name} outline",
            hoverinfo="none",
            showlegend=False,
        )

    return mesh_trace, outline_trace


def add_pole_to_stereonet(
    fig: go.Figure,
    dip: float,
    dip_azimuth: float,
    color: str = "#FF4136",
    name: str = "Pole",
    projection: str = "equal_area",
    strike: Optional[float] = None,
    residual: Optional[float] = None,
    marker_size: int = 9,
    symbol: str = "circle",
    **kwargs: Any
) -> None:
    """Add a structural plane pole marker to the stereonet.

    Args:
        fig: Target stereonet go.Figure.
        dip: Dip angle in degrees.
        dip_azimuth: Dip direction (azimuth) in degrees.
        color: Marker color.
        name: Name for hover tooltip and legend.
        projection: 'equal_area' (Schmidt) or 'equal_angle' (Wulff).
        strike: Optional strike in degrees for hover display.
        residual: Optional residual error for hover display.
        marker_size: Marker radius in pixels.
        symbol: Plotly scatter marker symbol.
        **kwargs: Extra arguments passed to go.Scatter.
    """
    if strike is None:
        calc_strike = (dip_azimuth - 90) % 360
    else:
        calc_strike = strike

    plane = gkp.Plane(calc_strike, dip)
    lon, lat = plane.to_pole_lonlat()
    proj_fn = project_lambert if projection == "equal_area" else project_stereographic
    x, y = proj_fn(lon, lat)

    hover_parts = [f"<b>{name}</b>", f"Dip: {dip:.1f}° ➔ {dip_azimuth:.1f}°", f"Strike: {calc_strike:.1f}°"]
    if residual is not None:
        hover_parts.append(f"RMSE: {residual:.4f} m")
    hovertext = "<br>".join(hover_parts)

    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode="markers",
        marker=dict(
            size=marker_size,
            color=color,
            symbol=symbol,
            line=dict(color="#ffffff", width=1.2)
        ),
        name=name,
        hovertext=hovertext,
        hoverinfo="text",
        showlegend=False,
        **kwargs
    ))


def add_great_circle_to_stereonet(
    fig: go.Figure,
    strike: float,
    dip: float,
    color: str = "#FF4136",
    name: str = "Great Circle",
    projection: str = "equal_area",
    line_width: float = 1.8,
    dash: str = "solid",
    npoints: int = 100,
    **kwargs: Any
) -> None:
    """Add a structural plane great circle (cyclographic trace) to the stereonet.

    Args:
        fig: Target stereonet go.Figure.
        strike: Strike angle in degrees.
        dip: Dip angle in degrees.
        color: Line color.
        name: Name for hover tooltip.
        projection: 'equal_area' (Schmidt) or 'equal_angle' (Wulff).
        line_width: Width of the great circle line.
        dash: Plotly line dash pattern ('solid', 'dash', 'dot').
        npoints: Number of evaluation points along the great circle.
        **kwargs: Extra arguments passed to go.Scatter.
    """
    plane = gkp.Plane(strike, dip)
    lons, lats = plane.to_greatcircle_lonlat(npoints=npoints)
    proj_fn = project_lambert if projection == "equal_area" else project_stereographic
    x, y = proj_fn(lons, lats)

    hovertext = f"<b>{name}</b><br>Strike: {strike:.1f}° / Dip: {dip:.1f}°"

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color=color, width=line_width, dash=dash),
        name=name,
        hovertext=hovertext,
        hoverinfo="text",
        showlegend=False,
        **kwargs
    ))

