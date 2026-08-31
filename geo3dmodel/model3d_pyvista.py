# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:39:08 2021

@author: Raymi Castilla
"""

# import os
# import pandas as pd
import numpy as np
# from itertools import islice, cycle, repeat
# import copy
# import time

import xarray

# # from geomdl import BSpline
try:
    import pyvista
    PYVISTA=True
except ModuleNotFoundError:
    PYVISTA=False
    
# import geokitpy as gkp
from .model3d_abstract import Model3D_abstract

class Model3D_pyvista(Model3D_abstract):
    
    def __init__(self, **kwargs):
        if not PYVISTA:            
            raise ValueError('Module pyvista needed to use instantiate this class')
        self.multi_block = kwargs.get('multi_block', pyvista.MultiBlock())
        self.plotter = kwargs.get('plotter', pyvista.Plotter())
    
    
    def plot_borehole(self, borehole, **kwargs):
        ...
    
    
    def plot_rectangular_mesh(self, strike=None, dip=None, nodes=None, triangles=None,
                              center=(0,0,0), i_elements=None, i_length=None,
                               j_elements=None, j_length=None, row=1, col=1, **kwargs):            
        ...
    
    
    def plot_plane(self, points_3d_df, c_hull_kwargs={},
                   trace_kwargs={}, **kwargs):
        pass
    
    
    def plot_point_cloud(self, df, **kwargs):
        ...
    
    
    
    def add_borehole_path(self, borehole, **kwargs):
        pass
    
    
    
    def add_boreholes_paths(self, borehole_dict, names_to_skip=[], **kwargs):   
        pass
            
        
    
    
    def add_borehole_interval(self):
        pass
    
    
    
    def plot_log_along_borehole(self, borehole, log):
        pass
           
    
    
    def plot_one_structure_as_disk(self, strike, dip, center,
                                   disk_kwargs={},
                                   mesh_kwargs={}):
        pass
    
    def add_structured_grid(self, data_array: xarray.DataArray,
                            grid_name:str):
        grid = data_array.pgrid.to_pyvista_structuredgrid()
        data = grid.point_data['values']#.reshape(62, 59, 60)
        nan_mask = np.isnan(data)
        grid.hide_points(nan_mask)
        self.multi_block[grid_name] = grid
    
    def plot_structured_grid(self, xarray):
        if not PYVISTA:            
            raise ValueError('Module pyvista needed to run this function')
    # Create a structured grid
        grid = pyvista.UniformGrid()
        # grid.dimensions = (nx, ny, nz)  # Number of grid points in x, y, z
        # grid.spacing = (1, 1, 1)  # Adjust spacing if needed (e.g., real-world distances)
        # grid.origin = (0, 0, 0)  # Origin of the grid
        
        
        # grid.point_data["values"] = values  # Add values to the grid
        
        # Plot the entire grid
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, cmap="viridis", opacity=0.5, show_edges=True)  # Adjust opacity and edges
        plotter.show()
        
    # def plot_multiple_structures_as_disks(self, df, row=1, col=1,
    #                              disk_kwargs={}, mesh_kwargs={}):
        
    #     showlegend = True
        
    #     customdata_columns = mesh_kwargs.pop('customdata_columns',None)
    #     # if customdata_columns is None:
    #     #     customdata=[None]*df.shape[0]
    #     # else:
    #     #     df_cdata = df.loc[:,customdata_columns]
    #     #     customdata=[list(row) for row in df_cdata.values]
        
    #     # mesh_kwargs.update({'customdata':customdata})
        
    #     # if any(df.unit_name.isin(['RG-2'])):
    #     #     breakpoint()
    #     # breakpoint()
    #     for pl in df.itertuples():            
    #         mesh_kwargs_1 = copy.deepcopy(mesh_kwargs)
    #         center = np.array([[pl.x-self.x0],
    #                            [pl.y-self.y0],
    #                            [pl.z-self.z0]])
            
    #         if customdata_columns is None:
    #             customdata=None
    #         else:             
    #             customdata=[getattr(pl, col) for col in customdata_columns]
            
    #         mesh_kwargs_1.update({'showlegend':showlegend,
    #                               'customdata':customdata})
            
    #         trace=self.make_trace_for_one_structure_as_disk(pl.strike, pl.dip,
    #                                         center=center,
    #                                         disk_kwargs=disk_kwargs,
    #                                         mesh_kwargs=mesh_kwargs_1)
            
    #         self.traces.append(trace)            
    #         showlegend=False
        
    
    # def plot_fit_to_pointcloud(self, df, **kwargs):
        
    #     mode   = kwargs.pop('mode', 'plane')
    #     row    = kwargs.pop('row', 1)
    #     col    = kwargs.pop('col', 1)
    #     trace_kwargs = kwargs.pop('trace_kwargs', {})
        
    #     if mode == 'plane':
    #         c_hull_kwargs = kwargs.pop('c_hull_kwargs', {})
    #         self.plot_plane(df, row=row, col=col, c_hull_kwargs=c_hull_kwargs,
    #                         trace_kwargs=trace_kwargs)
            
    #     elif mode == 'interpolate':
    #         x_grd, y_grd, z_grd = df.pcloud.interpolated_grid(num_pts=100)
    #         self.plot_gridded_surface(x_grd, y_grd, z_grd,
    #                                   row=1, col=1, **trace_kwargs)
            
    #     elif mode == 'mesh':
    #         triangulation = df.pcloud.mesh_points()
    #         i, j, k = triangulation.triangles.T
    #         trace = go.Mesh3d(x=df.x, y=df.y, z=df.z,
    #                           i=i, j=j, k=k, 
    #                           opacity = 0.7, **trace_kwargs)
    #         # trace = go.Mesh3d(x=df.x, y=df.y, z=df.z, alphahull=10,
    #         #                   opacity = 0.7, **trace_kwargs)
    #         self.traces.append(trace)
            
    #     elif mode == 'open3d':
    #         mesh = df.pcloud.fit_with_open3d(**kwargs)
            
            
    #         triangles = np.asarray(mesh.triangles)
    #         vertices  = np.asarray(mesh.vertices)
    #         trace = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], 
    #                           i=triangles[:,0], j=triangles[:,1], k=triangles[:,2],
    #                           **trace_kwargs)
    #         self.traces.append(trace)
            
    #     elif mode == 'weighted_avg':
    #         mesh = df.pcloud.fit_with_weighted_avg(**kwargs)
            
    #         triangles = np.asarray(mesh.triangles)
    #         vertices  = np.asarray(mesh.vertices)
    #         # pdb.set_trace()
    #         trace = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], 
    #                           i=triangles[:,0], j=triangles[:,1], k=triangles[:,2],
    #                           **trace_kwargs)
    #         self.traces.append(trace)
    #         # self.fig.add_trace(trace, row=1, col=1)
            
        
    # def add_gridded_surface(self, x, y, z, row=1, col=1, **kwargs):
    #     '''
    #     z must be a 2d list
    #     Coordinates in `x` and `y` can either be 1D lists or {2D arrays}
    #     '''
    #     if len(np.array(z).shape)!=2:
    #         raise ValueError('z must be a 2D list')
    #     trace = go.Surface(x=x-self.x0, y=y-self.y0, z=z-self.z0, **kwargs)
    #     self.traces.append(trace)
        
        

        
        

    
    

    
    # @staticmethod
    # def paraview_template(layout_kwargs={}):
        
    #     scene = layout_kwargs.pop('scene', {})        
    #     scene.update(yaxis_title="Northing (m)",
    #                  xaxis_title="Easting (m)",
    #                  zaxis_title="Elevation (m)",
    #                  xaxis = dict(showbackground=False),
    #                  yaxis = dict(showbackground=False),
    #                  zaxis = dict(showbackground=False)
    #                  )
        
    #     camera = layout_kwargs.pop('camera', {})
    #     camera.update(eye=dict(x=-1.2, y=-1.2, z=1))  
        
    #     layout_kwargs.update(scene = scene,
    #                          scene_camera=camera,
    #                         scene_aspectmode='data',
    #                         scene_aspectratio=dict(x=1, y=1, z=1),
    #                         font_color='white',
    #                         paper_bgcolor='rgb(85,88,110)',
    #                         legend= {'itemsizing': 'constant'})
        
    #     paraview_template = go.Layout(layout_kwargs)
                                 
        
    #     return paraview_template
    
    # @staticmethod
    # def paper_template(layout_kwargs={}):        
        
    #     scene = layout_kwargs.pop('scene', {}) 
        
    #     default_scene=dict(yaxis_title="Local Northing (m)",
    #                  xaxis_title="Local Easting (m)",
    #                  zaxis_title="Elevation (m)",
    #                  xaxis = dict(showbackground=True),
    #                  yaxis = dict(showbackground=True),
    #                  zaxis = dict(showbackground=True)
    #                  )        
    #     scene = layout_kwargs.pop('scene', {})
    #     default_scene.update(scene)
    #     scene=default_scene
        
    #     default_camera=dict(eye=dict(x=-1.2, y=-1.2, z=1))        
    #     camera = layout_kwargs.pop('camera', {})
    #     default_camera.update(camera)
    #     camera=default_camera
        
    #     layout_kwargs.update(scene = scene,
    #                          scene_camera=camera, 
    #                          scene_aspectmode='data',
    #                          scene_aspectratio=dict(x=1, y=1, z=1),
    #                          # autosize=False,
    #                          # width=2480,
    #                          # height=3508, #corresponds to a 300ppi A4 size
    #                          font_color='black',
    #                          paper_bgcolor='rgb(255,255,255)',
    #                          legend= {'itemsizing': 'constant'})                            
        
    #     paper_template = go.Layout(layout_kwargs)                                 
        
    #     return paper_template
        
        
    
    # def format_fig(self, **kwargs):
    #     template = kwargs.pop('template', self.paraview_template())        
    #     self.fig.update_layout(template)
    
    # def plot_offline(self):
    #     offline_plot(self.fig)
        
    # def show_in_browser(self):
    #     self.fig.show(renderer='browser')
        
    # def save_file(self, name, **kwargs):
        
    #     save=True
    #     overwrite_question = kwargs.pop('overwrite_question',True)
    #     # breakpoint()
    #     if name.endswith('.html'):
    #         func = self.fig.write_html
    #     elif name.endswith(('.png','.jpg','.jpeg','.svg','.pdf')):
    #         func = self.fig.write_image
    #     else:
    #         name += '.html'
    #         func = self.fig.write_html            
            
    #     if (os.path.isfile(name) & overwrite_question):
    #         overwrite = input("File already exists. Overwrite? (y/n): ")
    #         if overwrite not in ['y','yes','Y','YES']:
    #             save = False
        
    #     if save:
    #         func(name, **kwargs)
            
    #         if os.path.isfile(name):
    #             print('File succesfuly saved')
    #     else:
    #         print('File not saved')  
        
    # # def write_html(self, name):
        
    # #     if not name.endswith('.html'):
    # #         name += '.html'            
        
    # #     save = True
            
    # #     if os.path.isfile(name):
    # #         overwrite = input("File already exists. Overwrite? (y/n): ")
    # #         if overwrite not in ['y','yes','Y','YES']:
    # #             # os.remove(name)
    # #             save = False
        
    # #     if save:
    # #         self.fig.write_html(name)
    # #         if os.path.isfile(name):
    # #             print('File succesfuly saved')
    # #     else:
    # #         print('File not saved')
            
    # # def write_svg(self, name)

    # def show(self, **kwargs):
    #     self.fig.show(**kwargs)






        
        
    

#---- MAIN
if __name__ == '__main__':
    ...
    