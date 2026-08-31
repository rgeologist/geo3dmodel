# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:39:08 2021

@author: Raymi Castilla
"""

import os
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from itertools import islice, cycle, repeat
import copy
import time
import logging

import plotly.express as px
from plotly.offline import plot as offline_plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shapely import LineString
import xarray as xr
try:
    import trimesh
    TRIMESH=True
except ModuleNotFoundError:
    TRIMESH=False

import geokitpy as gkp
from .model3d_abstract import Model3D_abstract
from . import model3d as m3d

class Model3D_plotly(Model3D_abstract):
    
    def __init__(self, **kwargs):
        specs = kwargs.pop('specs', [[{"type": "scene"}]])
        local_zero=kwargs.pop('local_zero', (0,0,0))
        fig = go.Figure()
        # fig = make_subplots(rows=rows, cols=cols, 
        #                     specs=specs, **kwargs)
        self.local_zero=local_zero
        self.x0, self.y0, self.z0 = self.local_zero
        self.fig    = fig
        self.traces = []
        
    def add_borehole_xyz(self,*,
                         x:ArrayLike,
                         y:ArrayLike,
                         z:ArrayLike, **kwargs):
        
        trace = go.Scatter3d(x=x-self.x0,
                             y=y-self.y0,
                             z=z-self.z0,
                             **kwargs
                             )
        self.traces.append(trace)
        

    
    def add_borehole(self, borehole, survey_name:str='preferred', **kwargs):        
        df = borehole.surveys[survey_name].data        
        x, y, z = df.loc[:,['x','y','z']].to_numpy().T        
        self.add_borehole_xyz(x=x,
                              y=y,
                              z=z, **kwargs)
    
    def plot_borehole(self, borehole, **kwargs):
        self.add_borehole(borehole, **kwargs)
    
    
    # def plot_borehole_as_tube(self, borehole, **kwargs):
    #     #it doesn't work currently
    #     row = kwargs.pop('row', 1)
    #     col = kwargs.pop('col', 1)
        
    #     df = borehole.survey
    #     points = df.loc[:,['x','y','z']].values
        
    #     poly = pv.PolyData()
    #     poly.points = points
    #     cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    #     cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    #     cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    #     poly.lines = cells
    #     line = poly
    
    
    #     line["scalars"] = np.arange(line.n_points)
    #     tube = line.tube(radius=0.1)
        
    #     triang=tube.triangulate()
    #     points = triang.points
    #     vertex = triang.faces.reshape(-1,4)
    #     # pdb.set_trace()
        
    #     x, y, z = points[0:500,:].T
    #     # _, i, j, k = vertex.T
    #     # pdb.set_trace()
        
    #     trace = go.Mesh3d(x=x, y=y, z=z, **kwargs)
        
    #     self.traces.append(trace)
    
    @staticmethod
    def generate_default_color_dict(borehole_dict):
        color_dict = {}
        color_lst = [color for color in px.colors.qualitative.Dark24]
        color_lst.pop(5) #drop black in the 5th position
        colors=cycle(color_lst)
        for bh_name, color in zip(borehole_dict.keys(), colors):
            color_dict.update({bh_name:color})
            
        return color_dict
    
    def add_wellheads(self, borehole_dict, names_to_skip=[], **kwargs):        
        # row = kwargs.pop('row', 1)
        # col = kwargs.pop('col', 1)
        color_dict = kwargs.pop('color_dict', None)
        wellhead_coords_system = kwargs.pop('wellhead_coords_system', 'lv95')
        
        x0, y0, z0 = self.local_zero
        
        if color_dict is None:     
            color_dict=self.generate_default_color_dict(borehole_dict)
    
        
        for bh_name, bh in borehole_dict.items():
            check_names = [name in bh_name for name in names_to_skip]
            if any(check_names):
                continue
            
            
            try:
                color = color_dict[bh_name]
                x = bh.wellhead[wellhead_coords_system][0]-x0
                y = bh.wellhead[wellhead_coords_system][1]-y0
                z = bh.wellhead[wellhead_coords_system][2]-z0
                trace = go.Scatter3d(x=[x], y=[y], z=[z],
                                      mode='markers', name=bh_name,
                                      marker=dict(color=color, size=12),
                                      **kwargs
                                      )
                kwargs.pop('legendgrouptitle', None)
                self.traces.append(trace) 
            except AttributeError:
                breakpoint()
    
    
    
    def add_borehole_path(self, borehole:gkp.borehole.Borehole,
                          coord_system:str, **kwargs): 
        try:
            bh_survey = borehole.surveys.preferred.data
        except AttributeError:
            raise AttributeError(f'borehole {borehole.name} has no survey associated to it')

        
        color = kwargs.pop('color', None)
        
        hover_template  = 'x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}'
        hover_template += '<br>md: %{text:.f}'
        kwargs.update(dict(hovertemplate=hover_template))
            
        try:
            # breakpoint()
            wh_coords = (borehole.header.easting[coord_system],#.item(),
                         borehole.header.northing[coord_system],#.item(),
                         borehole.header.elevation_gl.iloc[0],#.item()
                         )
            x,y,z=(bh_survey.loc[:,['x','y','z']]+wh_coords).values.T
            
        except AttributeError:
            logging.warning('Borehole {borehole.name} will not be plotted')
            return
        text = bh_survey.md
        kwargs.update(dict(text=text))
        # breakpoint()
        try:
            trace = go.Scatter3d(x=x-self.x0,
                                 y=y-self.y0,
                                 z=z-self.z0,
                                 mode='lines', name=borehole.name,
                                 line=dict(color=color, width=5),
                                 **kwargs
                                 )
            kwargs.pop('legendgrouptitle', None)
            self.traces.append(trace) 

        except ValueError:
            breakpoint()

        except AttributeError:
            breakpoint()
    
    def add_wellhead(self, borehole:gkp.borehole.Borehole,
                          coord_system:str,
                          **kwargs):
        
        color = kwargs.pop('color', None)
        
        x0, y0, z0 = self.local_zero
        
        hover_template  = 'x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}'
        kwargs.update(dict(hovertemplate=hover_template))
            
        try:
            x,y,z = (borehole.header.easting[coord_system],#.item(),
                         borehole.header.northing[coord_system],#.item(),
                         borehole.header.elevation_gl.iloc[0],#.item()
                         )

        except AttributeError:
            logging.warning('Wellhead of {borehole.name} will not be plotted')
            return
        # text = bh_survey.md
        # kwargs.update(dict(text=text))
        # breakpoint()
        try:
            trace = go.Cone(x=np.asarray([x])-x0,
                            y=np.asarray([y])-y0,
                            z=np.asarray([z])-z0,
                            u=[0], v=[0], w=[1],
                            name=borehole.name,                            
                            **kwargs)
            self.traces.append(trace) 
        
        except ValueError:
            breakpoint()

        except AttributeError:
            breakpoint()
            
    
    def add_boreholes_paths(self, borehole_dict:dict, coord_system:str, 
                            names_to_skip:list=None, **kwargs):
        
        if names_to_skip is None:
            names_to_skip=[]
             
        color_dict = kwargs.pop('color_dict', None)
        
        if color_dict is None:     
            color_dict=self.generate_default_color_dict(borehole_dict)
        
        for bh_name, bh in borehole_dict.items():
            check_names = [name in bh_name for name in names_to_skip]
            if any(check_names):
                continue
            
            # if bh.survey is None:
            #     continue            
            # breakpoint()
            self.add_borehole_path(bh,coord_system, color=color_dict[bh_name], **kwargs)
    
    def add_wellheads(self, borehole_dict:dict, coord_system:str, 
                            names_to_skip:list=None, **kwargs):
        
        if names_to_skip is None:
            names_to_skip=[]
            
             
        color_dict = kwargs.pop('color_dict', None)
        
        if color_dict is None:     
            color_dict=self.generate_default_color_dict(borehole_dict)
        
        
        for bh_name, bh in borehole_dict.items():
            check_names = [name in bh_name for name in names_to_skip]
            if any(check_names):
                continue
            kwargs.update(dict(color=color_dict[bh_name]))
            
            self.add_wellhead(bh,coord_system, **kwargs)
            
            
    
    def plot_boreholes_3d_lines(self, borehole_dict, names_to_skip=[], **kwargs):
        
        # row = kwargs.pop('row', 1)
        # col = kwargs.pop('col', 1)
            
        survey_lst=[]
        for bh_name, bh in borehole_dict.items():
            check_names = [name in bh_name for name in names_to_skip]
            if any(check_names):
                continue
            
            df = bh.survey.reset_index()
            df['bh_name']=bh_name
            survey_lst.append(df)
        
                
        #unify all borehole surveys in a single dataframe
        bh_df = pd.concat(survey_lst, ignore_index=True)
        
        self.fig = px.line_3d(bh_df, x="x", y="y", z="z",
                          color='bh_name')
        
    
    def add_borehole_interval(self, *,
                              survey:pd.DataFrame,
                              md_start:float,
                              md_end:float,
                              sampling:float=0.05,
                              x_column:str='x',
                              y_column:str='y',
                              z_column:str='z',
                              interval_fmt_kwargs:dict=None,
                              trace_kwargs:dict=None):
        if interval_fmt_kwargs is None:
            interval_fmt_kwargs={}
            
        if trace_kwargs is None:
            trace_kwargs={}
            
        # if customdata == 'Malm':
        #     breakpoint()
        
        customdata = trace_kwargs.pop('customdata', None)
        
        x0, y0, z0 = self.local_zero
        
        bh_survey = survey
        
        md_array = np.arange(md_start, md_end, sampling)
        
        x = np.interp(md_array, bh_survey.md, bh_survey[x_column])-x0
        y = np.interp(md_array, bh_survey.md, bh_survey[y_column])-y0
        z = np.interp(md_array, bh_survey.md, bh_survey[z_column])-z0
        
        if customdata:
            h_template  = 'x:%{x:.1f}<br>y:%{y:.1f}<br>z:%{z:.1f}'
            if isinstance(customdata, str):
                # customdata = [customdata]*len(x)
                customdata = list(repeat(customdata, len(x)))            
                h_template += '<br>%{customdata}<extra></extra>'
                # pdb.set_trace()
            elif isinstance(customdata, list):
                for ind, d in enumerate(customdata):
                    h_template += '<br>%{customdata' + '[{}]'.format(ind) + '}'
                
                customdata = [[cdata]*len(x) for cdata in customdata]
                
            trace_kwargs.update(dict(customdata=customdata,
                                     hovertemplate=h_template))
        
        trace = go.Scatter3d(x=x, y=y, z=z,
                             mode='lines', line=interval_fmt_kwargs,
                             **trace_kwargs)
        
        self.traces.append(trace)
        
    def add_borehole_with_stratigraphy(self, borehole_survey,*,
                                       stratigraphy:pd.DataFrame,
                                       colors:pd.DataFrame,
                                       **kwargs):
        interval_fmt_kwargs = kwargs.pop('interval_fmt_kwargs', {})        
        trace_kwargs = kwargs.pop('trace_kwargs', {})
        
        depth_top    = stratigraphy['top_md']
        depth_bottom = stratigraphy['bottom_md']       
                
        names  = stratigraphy.index.to_list()
        # colors = pych.Suisse().database.colors.set_index('name').loc[names]        
        colors = colors.set_index('name').loc[names, ['R','G','B']].to_numpy()
        colors = [f"rgb({row[0]},{row[1]},{row[2]})" for row in colors]
        iterator = zip(depth_top, depth_bottom, colors, names)
        
        trace_kwargs.update(dict(showlegend=True))
        for top, bottom, color, name in iterator:
            interval_fmt_kwargs.update(dict(color=color))
            trace_kwargs.update(dict(customdata=name))
            self.add_borehole_interval(survey=borehole_survey,
                                       md_start=top,
                                       md_end=bottom,
                                       sampling=1,
                                       interval_fmt_kwargs=interval_fmt_kwargs,
                                       trace_kwargs=trace_kwargs,
                                       **kwargs)            
            trace_kwargs.update(dict(showlegend=False))
    
    
    def plot_log_along_borehole(self, borehole, log, 
                                colorscale='Viridis', width=10, **kwargs):
        # breakpoint()
        trace_kwargs   = kwargs.pop('trace_kwargs', {})
        every_x_meters = kwargs.pop('every_x_meters', None)
        
        survey = borehole.survey
        
        x0, y0, z0 = self.local_zero
        
        adapt_survey=False
        if every_x_meters:
            new_log = gkp.wlog.resample(log, func='median', resolution=every_x_meters)
            log=new_log
            # breakpoint()
            # log.single_log.plot()
        
        #adapt borehole survey index to wlog index
        survey = borehole.survey.reindex(method='nearest', index=log.index)

        hovertemplate = f'{log.name}'
        hovertemplate += ': %{customdata[0]:.0f}'
        hovertemplate += '<br>MD: %{customdata[1]:.1f}<extra></extra>'        
        customdata = [[val, depth] for val, depth in zip(log.values, log.index.values)]
        
        # hovertemplate = 'log value:%{customdata.values:$.0f}'
        # hovertemplate += '<br>MD:%{customdata.index:$.1f}<extra></extra>'
        # customdata = log
        # breakpoint()
        line_dict = dict(dict(color=log.values), width=width, colorscale=colorscale)
        line_dict.update(trace_kwargs)
        trace = go.Scatter3d(x=survey.x-x0, y=survey.y-y0, z=survey.z-z0,
                              mode='lines', line=line_dict,
                             name=f'{log.name}:{borehole.name}',
                             hovertemplate=hovertemplate,
                             customdata=customdata,
                             **kwargs)
        
        self.traces.append(trace)
    
    def plot_all_logs_along_all_boreholes(self, borehole_dict, log_dict, **kwargs):
        
        for bh_name, borehole in borehole_dict.items():
            for log_name, log_data in log_dict.items():
                self.plot_log_along_borehole(borehole, log_name, **kwargs)
        
    
    def plot_rectangular_mesh(self, *, strike=None, dip=None, nodes=None, triangles=None,
                              center=(0,0,0), i_elements=None, i_length=None,
                               j_elements=None, j_length=None, row=1, col=1, **kwargs):
            
        if None not in (strike, dip):
            #strike and dip defined (defines new nodes and triangles even if
            #they are specified in arguments)
            nodes, triangles = m3d.build_rectangular_mesh(strike=strike, dip=dip, center=center,
                                    i_elements=i_elements, i_length=i_length,
                                    j_elements=j_elements, j_length=j_length)
        
        x, y, z = nodes.T
        i, j, k = triangles.T
        
        trace = go.Mesh3d(x=x, y=y, z=z,
                          i=i, j=j, k=k, **kwargs
                          )
        # breakpoint()
        
        self.traces.append(trace)
        
    def add_triangulated_surface(self, triangulated_surface:"trimesh.Trimesh",
                                  **kwargs):
        mesh = triangulated_surface
        # breakpoint()
        trace = go.Mesh3d(x=mesh.vertices[:, 0],
                          y=mesh.vertices[:, 1],
                          z=mesh.vertices[:, 2],
                          i=mesh.faces[:, 0],
                          j=mesh.faces[:, 1],
                          k=mesh.faces[:, 2],
                          **kwargs)
        self.traces.append(trace)
    
    def create_trace_for_one_structure_as_disk(self, strike, dip, center,
                                draw_outline:bool=False, outline_kwargs:dict=None,
                                disk_kwargs:dict=None, mesh_kwargs:dict=None):
        if disk_kwargs is None:
            disk_kwargs={}
        if mesh_kwargs is None:
            mesh_kwargs={}
        if outline_kwargs is None:
            outline_kwargs={}
        
        try:
            disk, triangles = m3d.build_disk(strike, dip, center=center,
                                             **disk_kwargs)
            i, j, k = triangles.T
        except TypeError as te:
            print(te)
            raise te
        # breakpoint()
        hovertemplate = mesh_kwargs.get('hovertemplate', None)
        if hovertemplate is None:            
            hovertemplate = 'strike:%{customdata[0]:.0f}'
            hovertemplate += '<br>dip:%{customdata[1]:.0f}<extra></extra>'
            
        customdata = mesh_kwargs.get('customdata', None)
        if customdata is None:
            customdata = [strike,dip]        
        customdata = [customdata]*disk.shape[0]
        
        
        name = mesh_kwargs.get('name', None)
        if name is None:
            name = 'structure'
        
        mesh_kwargs.update(dict(hovertemplate=hovertemplate,
                                customdata=customdata,
                                name=name))
        
        mesh = go.Mesh3d(x=disk[:,0], y=disk[:,1], z=disk[:,2],
                          i=i, j=j, k=k, **mesh_kwargs)
        outline=None
        if draw_outline:
            outline = go.Scatter3d(x=disk[:,0], y=disk[:,1], z=disk[:,2],
                              mode='lines', **outline_kwargs)
        
        return (mesh, outline)

    def add_one_structure_as_disk(self, strike, dip, center,
                                   disk_kwargs={},
                                   mesh_kwargs={}, row=1, col=1):
        
        mesh, outline = self.create_trace_for_one_structure_as_disk(strike, dip, center,
                                       disk_kwargs=disk_kwargs,
                                       mesh_kwargs=mesh_kwargs)
        
        self.traces.append(mesh)
        if outline is not None:
            self.traces.append(outline)
    
        
    def add_multiple_structures_as_disks(self, df,
                                 draw_outline:bool=False,
                                 outline_kwargs:dict=None,
                                 disk_kwargs:dict=None,
                                 mesh_kwargs:dict=None,):
        
        showlegend = True
        
        customdata_columns = mesh_kwargs.pop('customdata_columns',None)
        # if customdata_columns is None:
        #     customdata=[None]*df.shape[0]
        # else:
        #     df_cdata = df.loc[:,customdata_columns]
        #     customdata=[list(row) for row in df_cdata.values]
        
        # mesh_kwargs.update({'customdata':customdata})
        
        # if any(df.unit_name.isin(['RG-2'])):
        #     breakpoint()
        # breakpoint()
        for pl in df.itertuples():            
            mesh_kwargs_1 = copy.deepcopy(mesh_kwargs)
            center = np.array([[pl.x-self.x0],
                               [pl.y-self.y0],
                               [pl.z-self.z0]])
            
            if customdata_columns is None:
                customdata=None
            else:             
                customdata=[getattr(pl, col) for col in customdata_columns]
            
            mesh_kwargs_1.update({'showlegend':showlegend,
                                  'customdata':customdata})
            
            mesh, outline = self.create_trace_for_one_structure_as_disk(pl.strike, pl.dip,
                                            center=center,
                                            draw_outline=draw_outline,
                                            outline_kwargs=outline_kwargs,
                                            disk_kwargs=disk_kwargs,
                                            mesh_kwargs=mesh_kwargs_1)
            
            self.traces.append(mesh)  
            if outline is not None:
                self.traces.append(outline)
            showlegend=False
        
    
    def plot_fit_to_pointcloud(self, df, **kwargs):
        
        mode   = kwargs.pop('mode', 'plane')
        row    = kwargs.pop('row', 1)
        col    = kwargs.pop('col', 1)
        trace_kwargs = kwargs.pop('trace_kwargs', {})
        
        if mode == 'plane':
            c_hull_kwargs = kwargs.pop('c_hull_kwargs', {})
            self.plot_plane(df, row=row, col=col, c_hull_kwargs=c_hull_kwargs,
                            trace_kwargs=trace_kwargs)
            
        elif mode == 'interpolate':
            x_grd, y_grd, z_grd = df.pcloud.interpolated_grid(num_pts=100)
            self.plot_gridded_surface(x_grd, y_grd, z_grd,
                                      row=1, col=1, **trace_kwargs)
            
        elif mode == 'mesh':
            triangulation = df.pcloud.mesh_points()
            i, j, k = triangulation.triangles.T
            trace = go.Mesh3d(x=df.x, y=df.y, z=df.z,
                              i=i, j=j, k=k, 
                              opacity = 0.7, **trace_kwargs)
            # trace = go.Mesh3d(x=df.x, y=df.y, z=df.z, alphahull=10,
            #                   opacity = 0.7, **trace_kwargs)
            self.traces.append(trace)
            
        elif mode == 'open3d':
            mesh = df.pcloud.fit_with_open3d(**kwargs)
            
            
            triangles = np.asarray(mesh.triangles)
            vertices  = np.asarray(mesh.vertices)
            trace = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], 
                              i=triangles[:,0], j=triangles[:,1], k=triangles[:,2],
                              **trace_kwargs)
            self.traces.append(trace)
            
        elif mode == 'weighted_avg':
            mesh = df.pcloud.fit_with_weighted_avg(**kwargs)
            
            triangles = np.asarray(mesh.triangles)
            vertices  = np.asarray(mesh.vertices)
            # pdb.set_trace()
            trace = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], 
                              i=triangles[:,0], j=triangles[:,1], k=triangles[:,2],
                              **trace_kwargs)
            self.traces.append(trace)
            # self.fig.add_trace(trace, row=1, col=1)
         
    def add_surface_from_datastructure(self,
                            data_structure:(pd.DataFrame, xr.DataArray),
                            **kwargs):
        sample_every=kwargs.pop('sample_every',1)
        if isinstance(data_structure, pd.DataFrame):
            plot_func = go.Mesh3d
            x, y, z = data_structure.loc[::sample_every,['x','y','z']].values.T
        elif isinstance(data_structure, xr.DataArray): 
            plot_func = go.Surface
            darray = data_structure[::sample_every, ::sample_every,]
            x,y,z = darray.x, darray.y, darray.data
        self.add_surface(x, y, z, plot_func=plot_func, **kwargs)
        
    def add_surface(self, x, y, z, plot_func=None, **kwargs):
        if plot_func == go.Mesh3d:
            self.add_irregular_surface(x, y, z, **kwargs)
        elif plot_func == go.Surface:
            self.add_gridded_surface(x, y, z, **kwargs)
        else:
            raise ValueError('plot_func is not valid')
            
        
    def add_irregular_surface(self, x, y, z, **kwargs):
        '''
        'x', 'y' and 'z' are 1D lists. A delaunay triangulation is used to render
        the surface. i, j and k can also be specified if triangulation
        has been done previously
        '''
        trace = go.Mesh3d(x=x-self.x0, y=y-self.y0, z=z-self.z0, **kwargs)
        self.traces.append(trace)
    
    def add_gridded_surface(self, x, y, z, **kwargs):
        '''
        z must be a 2d list
        Coordinates in `x` and `y` can either be 1D lists or {2D arrays}
        '''
        if len(np.array(z).shape)!=2:
            raise ValueError('z must be a 2D list')
        trace = go.Surface(x=x-self.x0, y=y-self.y0, z=z-self.z0, **kwargs)
        self.traces.append(trace)
        
    def add_vertical_surface_from_trace(self, trace:LineString, **kwargs):     
        if trace.has_z:
            coords = [(c[0]-self.x0,c[1]-self.y0,c[2]-self.z0) for c in trace.coords]
        else:
            coords = [(c[0]-self.x0,c[1]-self.y0) for c in trace.coords]
        
        new_ls = LineString(coords)
        
        max_segment_length=kwargs.pop('max_segment_length',100)
        xsection_top=kwargs.pop('xsection_top',900)
        xsection_bottom=kwargs.pop('xsection_bottom',-4000)
        trimesh = m3d.build_vertical_surface(new_ls,
                                             max_segment_length=max_segment_length,
                                             xsection_top=xsection_top,
                                             xsection_bottom=xsection_bottom)
        # breakpoint()
        self.add_triangulated_surface(trimesh, **kwargs)
        
    def plot_plane(self, points_3d_df, row=1, col=1, c_hull_kwargs={},
                   trace_kwargs={}, **kwargs):
        
        text=trace_kwargs.pop('text',None)
        hull3D=points_3d_df.pcloud.convex_hull_3D(**c_hull_kwargs)
        
        if text is not None:
            text = list(text)*len(hull3D)
            trace_kwargs.update({'text':text})
        trace = go.Mesh3d(x=hull3D[:,0], y=hull3D[:,1], z=hull3D[:,2],
                          **trace_kwargs)
        self.traces.append(trace)
        # self.fig.add_trace(trace, row=row, col=col)
        
        
    def plot_point_cloud(self, df:pd.DataFrame, marker_dict:dict=None,
                         other_trace_kwargs:dict=None, **kwargs):
        # breakpoint()
        if marker_dict is None:
            marker_dict={}
            
        if other_trace_kwargs is None:
            other_trace_kwargs={}
            
        trace = go.Scatter3d(x=df.x-self.x0, y=df.y-self.y0, z=df.z-self.z0,
                             mode='markers', marker=marker_dict,
                             **other_trace_kwargs)
        
        self.traces.append(trace)
    
    
    def merge_traces(self, trace=None, rows=1, cols=1):
        #old add_traces
        
        if trace is None:
            self.fig.add_traces(self.traces)
        else:            
            self.fig.add_traces(trace)
    
    def remove_all_traces(self):
        self.fig.data=()
    
    @staticmethod
    def paraview_template(layout_kwargs={}):
        
        scene = layout_kwargs.pop('scene', {})        
        scene.update(yaxis_title="Northing (m)",
                     xaxis_title="Easting (m)",
                     zaxis_title="Elevation (m)",
                     xaxis = dict(showbackground=False),
                     yaxis = dict(showbackground=False),
                     zaxis = dict(showbackground=False)
                     )
        
        camera = layout_kwargs.pop('camera', {})
        camera.update(eye=dict(x=-1.2, y=-1.2, z=1))  
        
        layout_kwargs.update(scene = scene,
                             scene_camera=camera,
                            scene_aspectmode='data',
                            scene_aspectratio=dict(x=1, y=1, z=1),
                            font_color='white',
                            paper_bgcolor='rgb(85,88,110)',
                            legend= {'itemsizing': 'constant'})
        
        paraview_template = go.Layout(layout_kwargs)
                                 
        
        return paraview_template
    
    @staticmethod
    def paper_template(layout_kwargs={}):        
        
        scene = layout_kwargs.pop('scene', {}) 
        
        default_scene=dict(yaxis_title="Local Northing (m)",
                     xaxis_title="Local Easting (m)",
                     zaxis_title="Elevation (m)",
                     xaxis = dict(showbackground=True),
                     yaxis = dict(showbackground=True),
                     zaxis = dict(showbackground=True)
                     )        
        scene = layout_kwargs.pop('scene', {})
        default_scene.update(scene)
        scene=default_scene
        
        default_camera=dict(eye=dict(x=-1.2, y=-1.2, z=1))        
        camera = layout_kwargs.pop('camera', {})
        default_camera.update(camera)
        camera=default_camera
        
        layout_kwargs.update(scene = scene,
                             scene_camera=camera, 
                             scene_aspectmode='data',
                             scene_aspectratio=dict(x=1, y=1, z=1),
                             # autosize=False,
                             # width=2480,
                             # height=3508, #corresponds to a 300ppi A4 size
                             font_color='black',
                             paper_bgcolor='rgb(255,255,255)',
                             legend= {'itemsizing': 'constant'})                            
        
        paper_template = go.Layout(layout_kwargs)                                 
        
        return paper_template
        
        
    
    def format_fig(self, **kwargs):
        template = kwargs.pop('template', self.paraview_template())        
        self.fig.update_layout(template)
    
    def plot_offline(self):
        offline_plot(self.fig)
        
    def show_in_browser(self):
        self.fig.show(renderer='browser')
        
    def save_file(self, name, **kwargs):
        
        save=True
        overwrite_question = kwargs.pop('overwrite_question',True)
        # breakpoint()
        if name.endswith('.html'):
            func = self.fig.write_html
        elif name.endswith(('.png','.jpg','.jpeg','.svg','.pdf')):
            func = self.fig.write_image
        else:
            name += '.html'
            func = self.fig.write_html            
            
        if (os.path.isfile(name) & overwrite_question):
            overwrite = input("File already exists. Overwrite? (y/n): ")
            if overwrite not in ['y','yes','Y','YES']:
                save = False
        
        if save:
            func(name, **kwargs)            
            if os.path.isfile(name):
                logging.info('File succesfuly saved')
        else:
            logging.info('File not saved')  
        
    # def write_html(self, name):
        
    #     if not name.endswith('.html'):
    #         name += '.html'            
        
    #     save = True
            
    #     if os.path.isfile(name):
    #         overwrite = input("File already exists. Overwrite? (y/n): ")
    #         if overwrite not in ['y','yes','Y','YES']:
    #             # os.remove(name)
    #             save = False
        
    #     if save:
    #         self.fig.write_html(name)
    #         if os.path.isfile(name):
    #             print('File succesfuly saved')
    #     else:
    #         print('File not saved')
            
    # def write_svg(self, name)

    def show(self, **kwargs):
        self.fig.show(**kwargs)



try:
    #delete the accesor to avoid warning 
    del xr.DataArray().pgrid
except AttributeError:
    pass


# @xr.register_dataarray_accessor('pgrid')
# class Pgrid:
    
#     def __init__(self, data_array):     
#         self._obj = data_array
        
#     @classmethod
#     def factory(cls, axes, origin, step, size):
#         # xyz = st.Axes.xyz()
        
#         xyz_list=[np.arange(0, s7ep*s1ze, step=s7ep) for s7ep, s1ze in zip(step, size)]
        
#         #translate to origin
#         xyz_list = [coord+or1gin for coord, or1gin in zip(xyz_list, origin)]
        
#         #rotate to align with axes
#         rot_matrix = axes.dircos(st.Axes.xyz())
#         #Coordinates vector
#         xv, yv, zv = np.meshgrid(*xyz_list)   
#         v=st.Vector(np.stack((xv,yv,zv), axis=-1))      
#         #rotation
#         vr = v@rot_matrix
        
#         darray = xr.DataArray(np.empty(size),
#                               coords=[('x', vr[0,:,0,0]),
#                                       ('y', vr[:,0,0,1]),
#                                       ('z', vr[0,0,:,2])])
        
#         return darray
        
        
    

#---- MAIN
if __name__ == '__main__':
    ...
    # pgrid = Pgrid.factory(st.Axes.from_trendplunge(270,0,180,0, 0,90), (0, 0, 0), (1,1,2), (60, 110, 255))
    
    