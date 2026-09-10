from __future__ import annotations
 # -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 07:36:15 2021

@author: Raymi Castilla
"""
from os import PathLike
from copy import copy
import itertools
import numbers
from typing import Callable, Union, List, Tuple, Optional, Literal
# import re
# import logging
import operator as py_operator

import numpy as np
from numpy.typing import NDArray, ArrayLike

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.path as mplpath
from matplotlib import tri
import matplotlib.cm as cm
import matplotlib.colors as colors 

from scipy.spatial import ConvexHull#, convex_hull_plot_2d
from scipy.linalg import LinAlgError
from scipy.interpolate import splprep, splev, griddata, Rbf
# from scipy.signal import windows 
from sklearn.decomposition import PCA

import shapely
from shapely.geometry import LineString, Point, Polygon, box, MultiPoint

import xarray as xr
import skimage


try:
    import open3d
    OPEN3D=True
except ImportError:
    OPEN3D=False

try:
    import trimesh
    TRIMESH=True
except ImportError:
    TRIMESH=False

try:
    import pyvista
    PYVISTA=True
except ImportError:
    PYVISTA=False

import geokitpy as gkp

PointCollection = Union[List[Point], Tuple[Point, ...],
                        MultiPoint, np.ndarray, pd.DataFrame]

def check_strike_dip(strike: float, dip: float) -> None:
    """Checks if the provided strike and dip are valid numbers.

    Args:
        strike (float): The strike angle.
        dip (float): The dip angle.

    Returns:
        None

    Raises:
        TypeError: If strike or dip is not a number or is NaN.
    """
    message  = 'strike and/or dip not a valid type. '
    message += f'strike:{strike}; dip:{dip}'
    type_error = TypeError(message)
    isnumber = lambda x: isinstance(x, numbers.Number)
    if not all([isnumber(val) for val in [strike, dip]]):
        #strike or dip not a number
        raise type_error
    elif any(np.isnan([strike, dip])):           
        #still need to check for np.nan
        #at least one is nan
        raise type_error

def build_flat_disk_perimeter_xy(*, radius: float, num_sides: int) -> Tuple[np.ndarray, np.ndarray]:
    """Builds the x and y coordinates for a flat disk perimeter.

    Args:
        radius (float): The radius of the disk.
        num_sides (int): The number of sides for the disk perimeter.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The x and y coordinates of the disk perimeter.
    """
    angles = np.linspace(0,2*np.pi,num_sides)
    x = radius*np.cos(angles)
    y = radius*np.sin(angles)
    return x,y

def rotate_flat_polygon_strike_dip(*, x: NDArray, y: NDArray, strike: float, dip: float) -> np.ndarray:
    """Rotates a flat polygon based on strike and dip angles.

    Args:
        x (NDArray): The x coordinates of the polygon.
        y (NDArray): The y coordinates of the polygon.
        strike (float): The strike angle.
        dip (float): The dip angle.

    Returns:
        np.ndarray: The rotated coordinates as a 3D array.
    """
    z = np.zeros_like(x)
    coords = np.vstack((x,y,z))    
    rot_mat = np.asarray(gkp.Plane(strike,dip).axes.T)
    new_coords = ((rot_mat @ coords)).T    
    return new_coords

def build_disk_parallel_to_plane(*, plane:gkp.Plane,
                                 center:tuple,
                                 radius:float,
                                 num_sides:int):
    x,y = build_flat_disk_perimeter_xy(radius=radius, num_sides=num_sides)
    x, y = x-center[0], y-center[1]
    new_coords = rotate_flat_polygon_strike_dip(x=x, y=y, strike=plane.strike, dip=plane.dip)
    return new_coords

def generate_disks_around_path(*, path:pd.DataFrame,
                               position_disks:Literal['start', 'middle', 'end'],
                               radius_disks:ArrayLike,
                               num_sides_disks:int)->list:
    
    path_coords = path.loc[:,['x','y','z']]
    path_arr = path_coords.to_numpy()
    planes = gkp.Vector.from_path(path_arr).view(gkp.Plane)
    
    match position_disks:
        case 'start':
            centers=path_arr[:-1,:]
        case 'middle':
            centers=path_arr[:-1,:]+planes/2.
        case 'end':
            centers=path_arr[1:,:]
        case _:
            raise ValueError('position_disks must be either '
                             '"start, "middle" or "end". '
                             f'{position_disks} was given')   
    radius_array = np.asarray(radius_disks)
    if radius_array.ndim == 0:
        radii = itertools.repeat(radius_array.item(), len(centers))
    else:
        if radius_array.size != len(centers):
            raise ValueError(
                'radius_disks must contain one radius per disk center; '
                f'{radius_array.size} radii were provided for '
                f'{len(centers)} centers'
            )
        radii = radius_array.flat

    iterator = zip(planes, centers,
                   radii,
                   itertools.repeat(num_sides_disks))
    disks = [build_disk_parallel_to_plane(plane=pl,
                                     center=center,
                                     radius=rad,
                                     num_sides=num_sides) for
             pl, center, rad, num_sides in iterator]
    return disks
    


def triangulate_perimeter(x: NDArray, y: NDArray) -> Tuple[tri.Triangulation, np.ndarray]:
    """Triangulates a perimeter defined by x and y coordinates.

    Args:
        x (NDArray): The x coordinates of the perimeter.
        y (NDArray): The y coordinates of the perimeter.

    Returns:
        Tuple[tri.Triangulation, np.ndarray]: The triangulation object and the delaunay triangles.
    """
    triang = tri.Triangulation(x,y)
    delaunay_triangles=triang.triangles
    return triang, delaunay_triangles
    
def build_disk(strike: float, dip: float, center: Tuple[float, float, float] = (0, 0, 0), radius: float = 1, num_sides: int = 15, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Builds a 3D disk with the specified strike, dip, center, and radius.

    Args:
        strike (float): The strike angle.
        dip (float): The dip angle.
        center (Tuple[float, float, float], optional): The center coordinates. Defaults to (0,0,0).
        radius (float, optional): The radius of the disk. Defaults to 1.
        num_sides (int, optional): The number of sides for the disk perimeter. Defaults to 15.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The 3D coordinates of the disk and the delaunay triangles.
    """
    check_strike_dip(strike, dip)    
    x,y = build_flat_disk_perimeter_xy(radius=radius, num_sides=num_sides)     
    triangles, delaunay_triangles = triangulate_perimeter(x, y)    
    new_coords = rotate_flat_polygon_strike_dip(x=x, y=y, strike=strike, dip=dip)
    
    center = np.array(center).reshape(-1,3)
    # breakpoint()
    new_coords = new_coords+center
    
    return new_coords, delaunay_triangles


def xy_grid(origin: tuple,
            x_negative_dist: float, x_positive_dist: float,        
            y_negative_dist: float, y_positive_dist: float, 
            resolution: Optional[float] = None, n_elements: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an XY grid based on origin and distances.

    Args:
        origin (tuple): The origin coordinates (x, y).
        x_negative_dist (float): Distance in the negative X direction.
        x_positive_dist (float): Distance in the positive X direction.
        y_negative_dist (float): Distance in the negative Y direction.
        y_positive_dist (float): Distance in the positive Y direction.
        resolution (Optional[float], optional): Grid resolution. Defaults to None.
        n_elements (Optional[int], optional): Number of elements. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The X and Y mesh grids.

    Raises:
        ValueError: If neither resolution nor n_elements is provided.
    """
    xmin = origin[0]+x_negative_dist
    xmax = origin[0]+x_positive_dist    
    ymin = origin[1]+y_negative_dist
    ymax = origin[1]+y_positive_dist
    
    if resolution:
        num_elem_x = (xmax-xmin)//resolution
        num_elem_y = (ymax-ymin)//resolution
    elif n_elements:
        num_elem_x=n_elements
        num_elem_y=n_elements
    else:
        raise ValueError('Either "resolution" or "n_elements" must be indicated')
        
    x_vector = slice(xmin,xmax,(num_elem_x+1)*1j)
    y_vector = slice(ymin,ymax,(num_elem_y+1)*1j)
    Y, X = np.ogrid[y_vector,x_vector]
    # breakpoint()
    return X, np.flipud(Y)


def build_rectangular_mesh(*, strike: float, dip: float, center: Tuple[float, float, float], i_elements: int, i_length: float,
                           j_elements: int, j_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """Builds a 3D rectangular mesh with a specified strike, dip, and dimensions.

    Args:
        strike (float): The strike angle.
        dip (float): The dip angle.
        center (Tuple[float, float, float]): The center coordinates.
        i_elements (int): Number of elements along the I axis.
        i_length (float): Total length along the I axis.
        j_elements (int): Number of elements along the J axis.
        j_length (float): Total length along the J axis.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The 3D coordinates and delaunay triangles.
    """
    import matplotlib.tri as tri
    
    i_points = np.linspace(-i_length/2,i_length/2,i_elements)
    j_points = np.linspace(-j_length/2,j_length/2,j_elements)
    xy = np.array(list(itertools.product(i_points, j_points)))
    
    x,y = xy.T
    triang = tri.Triangulation(x,y)
    delaunay_triangles=triang.triangles
    
    z = np.atleast_2d([0]*xy.shape[0]).T
    coords = np.concatenate((xy, z), axis=1).T
    # breakpoint()
    rot_mat = gkp.Plane(strike,dip).axes.as_matrix.T
    center = np.array(center).reshape(3,-1)
    new_coords = (rot_mat @ coords) + center
    return new_coords.T, delaunay_triangles


def z_for_plane3d(point_on_plane: tuple, plane: Optional[gkp.Plane] = None,
                  strike: Optional[float] = None, dip: Optional[float] = None) -> Callable:
    """Returns a function to compute Z given X and Y for a 3D plane.

    Args:
        point_on_plane (tuple): A point lying on the plane (x, y, z).
        plane (Optional[gkp.Plane], optional): A plane object. Defaults to None.
        strike (Optional[float], optional): The strike angle. Defaults to None.
        dip (Optional[float], optional): The dip angle. Defaults to None.

    Returns:
        Callable: A function taking (x, y) and returning z.
    """
    if plane is None:
        plane = gkp.Plane(strike, dip)
        
    d = (-1*(plane*point_on_plane).sum()).view(np.ndarray)
    
    a, b, c = plane._array
    
    z = lambda x, y: -(d+a*x+b*y)/c
    
    return z
        

def z_plane3d_xygrid(xy_grid: Tuple[np.ndarray, np.ndarray], point_on_plane: tuple, 
                    plane: Optional[gkp.Plane] = None, strike: Optional[float] = None, dip: Optional[float] = None) -> np.ndarray:
    """Calculates Z values for a 2D XY grid based on a 3D plane.

    Args:
        xy_grid (Tuple[np.ndarray, np.ndarray]): The X and Y mesh grids.
        point_on_plane (tuple): A point lying on the plane.
        plane (Optional[gkp.Plane], optional): A plane object. Defaults to None.
        strike (Optional[float], optional): The strike angle. Defaults to None.
        dip (Optional[float], optional): The dip angle. Defaults to None.

    Returns:
        np.ndarray: The computed Z values on the grid.
    """
    z_func = z_for_plane3d(point_on_plane, plane=plane,
                      strike=strike, dip=dip)
    
    z = z_func(*xy_grid)
    # breakpoint()
    
    return z
    
    
def build_vertical_surface(surface_path: shapely.Geometry,
                           max_segment_length: float = 100,
                           xsection_top: float = 900,
                           xsection_bottom: float = -4000,
                           ) -> "trimesh.Trimesh":
    """Builds a vertical 3D surface (mesh) by extruding a 2D path.

    Args:
        surface_path (shapely.Geometry): The 2D path (LineString, etc.).
        max_segment_length (float, optional): Maximum length for segmentizing. Defaults to 100.
        xsection_top (float, optional): The top Z elevation. Defaults to 900.
        xsection_bottom (float, optional): The bottom Z elevation. Defaults to -4000.

    Returns:
        trimesh.Trimesh: The resulting vertical surface mesh.
        
    Raises:
        ValueError: If trimesh is not available or geometry type is unsupported.
    """
    if not TRIMESH:
        raise ValueError("Module trimesh needed to run this function")
    #Read and format data
    if isinstance(surface_path, shapely.MultiPoint):
        path_ls = (shapely.LineString(surface_path.geoms)
                   .segmentize(max_segment_length=max_segment_length))
        
    elif isinstance(surface_path, (shapely.LineString, shapely.Polygon)):
        path_ls = (surface_path
                   .segmentize(max_segment_length=max_segment_length))
        
    elif isinstance(surface_path, shapely.MultiLineString):
        path_ls = (shapely.line_merge(surface_path)
                   .segmentize(max_segment_length=max_segment_length))
    else:
        raise ValueError(f'surface_path is of type {type(surface_path)} and it is not handled yet')
    
    #Adds 3rd dimension if necessary
    polyline_top = np.array([pt for pt in path_ls.coords])
    if polyline_top.shape[1]==2:
        constant_top=np.ones((polyline_top.shape[0],1))*xsection_top
        polyline_top=np.hstack((polyline_top, constant_top))
    polyline_bottom = polyline_top.copy()
    polyline_bottom[:,2] = xsection_bottom
    
    #build vertical surface
    n_points = polyline_top.shape[0]
    faces = []
    for i in range(n_points - 1):
        # Two triangles per quad
        faces.append([i, i + 1, i + n_points])
        faces.append([i + 1, i + 1 + n_points, i + n_points])
    
    vertices = np.vstack((polyline_bottom, polyline_top))
    surface = trimesh.Trimesh(vertices=vertices, faces=faces)
    return surface


def build_surface_from_polylines(line1: shapely.LineString,
                                 line2: shapely.LineString) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int]]:
    """Takes two lines, performs a triangulation, and returns a 3D mesh.

    Args:
        line1 (shapely.LineString): The top line.
        line2 (shapely.LineString): The bottom line.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int]]: The x, y, z coordinates and i, j, k indices for the triangles.

    Raises:
        TypeError: If line1 or line2 is not a shapely.LineString.
    """
    
    if not (isinstance(line1, shapely.LineString) and isinstance(line2, shapely.LineString)):
        raise TypeError("line1 and line2 must be shapely.LineString")
    
    # Let's say you have your two 3D shapely LineStrings: top_line and bottom_line
    # 1. Determine how many resolution points you want (e.g., based on the denser line)
    num_points = max(len(line1.coords), len(line2.coords))
    
    
    # 2. Resample both lines evenly along their lengths
    dists1 = np.linspace(0, line1.length, num_points)
    dists2 = np.linspace(0, line2.length, num_points)
    
    pts1 = np.array([line1.interpolate(d).coords[0] for d in dists1])
    pts2 = np.array([line2.interpolate(d).coords[0] for d in dists2])
    
    # Stack vertices: Top line points come first, followed by Bottom line points
    try:
        all_pts = np.vstack([pts1, pts2])
        x, y, z = all_pts[:, 0], all_pts[:, 1], all_pts[:, 2]
    except IndexError:
        pass
    
    # Initialize index arrays for the mesh triangles
    i_idx, j_idx, k_idx = [], [], []
    
    n = len(pts1)
    for idx in range(n - 1):
        # Vertex indices for the current segment
        t1 = idx          # Current Top
        t2 = idx + 1      # Next Top
        b1 = idx + n      # Current Bottom
        b2 = idx + 1 + n  # Next Bottom
    
        # Triangle 1: Top-Left, Top-Right, Bottom-Left
        i_idx.append(t1)
        j_idx.append(t2)
        k_idx.append(b1)
    
        # Triangle 2: Top-Right, Bottom-Right, Bottom-Left
        i_idx.append(t2)
        j_idx.append(b2)
        k_idx.append(b1)
        
    return x,y,z,i_idx,j_idx,k_idx

def build_vertical_surface_from_one_polyline_extrusion(line: shapely.LineString,
                                             extrusion_distance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int]]:
    """Builds a vertical surface by extruding a single polyline downwards or upwards.

    Args:
        line (shapely.LineString): The input polyline.
        extrusion_distance (float): The distance to extrude in the Z direction.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], List[int]]: The x, y, z coordinates and i, j, k triangle indices.

    Raises:
        TypeError: If line is not a shapely.LineString.
        ValueError: If the polyline does not have a constant Z value.
    """
    if not isinstance(line, shapely.LineString):
        raise TypeError("line must be a shapely.LineString")
        
    if not line.has_z:
        line = shapely.force_3d(line, z=0)
    
    z_line = shapely.get_coordinates(line, include_z=True)[:,2]
    z_line_set = set(z_line)
    if len(z_line_set) != 1:
        raise ValueError('All Z values of line must be equal')
        
    z_line2 = z_line[0]+extrusion_distance
    
    line2 = shapely.force_3d(shapely.force_2d(line), z=z_line2)
        
    tup = build_surface_from_polylines(line, line2)
    return tup

# def build_vertical_surface_from_xy_z_orig(*,x:ArrayLike,
#                                     y:ArrayLike,
#                                     z:ArrayLike)->tuple:    
#     '''
#     takes xy (1D arrays) coordinates that define a horizontal line and
#     applies z values (1D array) to get a vertical surface
    
#     returns x,y,z,i,j,k
    
#     '''
#     X, Z = np.meshgrid(x, z)
#     Y, _ = np.meshgrid(y, z)
#     n_points = len(x)
#     n_layers = len(z)
    
#     # Flatten them into 1D arrays for go.Mesh3d
#     x_flat = X.flatten()
#     y_flat = Y.flatten()
#     z_flat = Z.flatten()
    
#     # 3. Generate Triangles (i, j, k indices) for the vertical faces
#     i_indices = []
#     j_indices = []
#     k_indices = []
    
#     for layer in range(n_layers - 1):
#         for point in range(n_points - 1):
#             # Locate the 4 corners of the current wall segment
#             bottom_left  = layer * n_points + point
#             bottom_right = bottom_left + 1
#             top_left     = (layer + 1) * n_points + point
#             top_right    = top_left + 1
            
#             # Triangle 1 (Bottom-Left, Bottom-Right, Top-Right)
#             i_indices.append(bottom_left)
#             j_indices.append(bottom_right)
#             k_indices.append(top_right)
            
#             # Triangle 2 (Bottom-Left, Top-Right, Top-Left)
#             i_indices.append(bottom_left)
#             j_indices.append(top_right)
#             k_indices.append(top_left)
    
#     tup = x_flat, y_flat, z_flat, i_indices, j_indices, k_indices
#     return tup

# def build_vertical_surface_from_xy_z(*,x:ArrayLike,
#                                     y:ArrayLike,
#                                     z:ArrayLike)->tuple:    
#     '''
#     takes xy (1D arrays) coordinates that define a horizontal line and
#     applies z values (1D array) to get a vertical surface
    
#     returns x,y,z,i,j,k
    
#     '''
    
#     # x and y are in pairs because they define a line on a constant-depth surface
#     xy_idx = range(x.size)
    
#     IDX, Z = np.meshgrid(xy_idx,z)
    
#     idx_flat = IDX.flat
#     z_flat = Z.flat
#     breakpoint()
#     triang_obj = tri.Triangulation(idx_flat, z_flat)
    
#     tup=None
#     # tup = x_flat, y_flat, z_flat, i_indices, j_indices, k_indices
#     return tup

def build_triangulated_surface_from_lengths(*,
                                        n_horizontal: int,
                                        n_vertical: int) -> tri.Triangulation:
    """Builds a triangulated surface from the lengths of 2 dimensions.

    It is used to build any surface with 2 dimensions.
    The final triang_obj can be mapped to any coordinates.
    The resulting triang_obj.triangles contain the flattened indices that can be used
    after to map real coordinates.
    
    It works best if the real coordinates have regular spacing.
    It can be used to generate vertical surfaces like a vertical xsection (seismic or geologic).

    Args:
        n_horizontal (int): The number of horizontal points.
        n_vertical (int): The number of vertical points.

    Returns:
        tri.Triangulation: The created triangulation object.
    """
    hor = range(n_horizontal)
    vert = range(n_vertical)
    H, Z = np.meshgrid(hor,vert)
    triang_obj = tri.Triangulation(H.flat, Z.flat)
    return triang_obj
    
def build_vertical_surface_polyline_and_zlevels(*, x: ArrayLike,
                                                y: ArrayLike,
                                                z: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds a vertical surface from a polyline at fixed depth and a series of depth levels.

    x, y and z are 1D arrays.
    xy define a polyline at a fixed depth = z[0].
    z positions the xy polyline and defines the coordinates at depth.

    Args:
        x (ArrayLike): 1D array of x coordinates for the polyline.
        y (ArrayLike): 1D array of y coordinates for the polyline.
        z (ArrayLike): 1D array of z levels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The x, y, z flats and i, j, k triangle indices.
        
    Raises:
        ValueError: If x and y do not have the same length.
    """    
    if x.size != y.size:
        raise ValueError('x and y must have the same length')
    
    n_horizontal = x.size
    n_vertical   = z.size
    
    triang_obj = build_triangulated_surface_from_lengths(n_horizontal=n_horizontal,
                                                         n_vertical=n_vertical)
    i_indices, j_indices, k_indices = triang_obj.triangles.T
    
    X, Z = np.meshgrid(x, z)
    Y, _ = np.meshgrid(y, z)
        
    # Flatten them into 1D arrays
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()       
            
    tup = x_flat, y_flat, z_flat, i_indices, j_indices, k_indices
    return tup


def map_array_to_cmap(*, values: ArrayLike,
                      mpl_cmap: Optional[colors.Colormap] = None,
                      mpl_cmap_name: Optional[str] = None, **kwargs) -> np.ndarray:
    """Maps an array of values to RGBA colors using a matplotlib colormap.

    Args:
        values (ArrayLike): The array of values to map.
        mpl_cmap (Optional[colors.Colormap], optional): A matplotlib colormap instance. Defaults to None.
        mpl_cmap_name (Optional[str], optional): The name of a registered matplotlib colormap. Defaults to None.
        **kwargs: Additional arguments such as vmin and vmax.

    Returns:
        np.ndarray: The array of mapped vertex colors.
    """
    vmin = kwargs.pop('vmin', values.min())
    vmax = kwargs.pop('vmax', values.max())
    
    values_flat = values.flatten()
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if mpl_cmap:
        cmap = mpl_cmap
    else:
        cmap = cm.get_cmap(mpl_cmap_name)
    vertex_colors = cmap(norm(values_flat))
    return vertex_colors

def import_rgb_image_as_vertical_surface(*, filepath: Optional[PathLike] = None,
                                         im_array: Optional[NDArray] = None,
                                         x: NDArray,
                                         y: NDArray,
                                         z: NDArray) -> xr.Dataset:
    """Imports an RGB image and maps it onto a vertical surface as an xarray Dataset.

    Args:
        filepath (Optional[PathLike], optional): The path to the image file. Defaults to None.
        im_array (Optional[NDArray], optional): The RGB image array. Defaults to None.
        x (NDArray): 1D array of x coordinates for the polyline.
        y (NDArray): 1D array of y coordinates for the polyline.
        z (NDArray): 1D array of z levels.

    Returns:
        xr.Dataset: The xarray Dataset containing the mapped image.

    Raises:
        ValueError: If neither im_array nor filepath is specified, or if shape requirements are not met.
        TypeError: If im_array dimensions are incorrect, or if x and y differ in size.
    """
    if im_array is None:
        if filepath is not None:
            im_array = skimage.io.imread(filepath)
        else:
            raise ValueError('Either im_array or filepath must be specified')
        
    if im_array.shape[-1]<3:
        raise TypeError("im_array expected to be an array with at least 3 in the last dimension")
    
    if im_array.ndim<3:
        raise TypeError("im_array expected to be an array with 3 dimensions")
    
    if x.size != y.size:    
        raise TypeError("x and y must be the same size")
    
    if (x.size != im_array.shape[1]) or (z.size!=im_array.shape[0]):
        raise ValueError("This implementation needs that im_array.shape=(z.size, x.size, 3or4)")
    
    x_diff, y_diff = np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])
    dist = np.sqrt(x_diff**2+y_diff**2)
    
    coords={'x':('distance',x),'y':('distance',y),
            'vertical':z,'distance':dist}
    variables = dict(r=(('vertical', 'distance'), im_array[..., 0]),
                     g=(('vertical', 'distance'), im_array[..., 1]),
                     b=(('vertical', 'distance'), im_array[..., 2]),
                     )
    try:
        variables.update(dict(alpha=(('vertical', 'distance'), im_array[..., 3])))
    except IndexError:
        variables.update(dict(alpha=(('vertical', 'distance'), np.ones(im_array.shape[:2])*255)))
    
    dset = xr.Dataset(data_vars=variables, coords=coords)
    return dset

def map_rgb_image_to_vertical_surface(*, rgb: NDArray,
                                      x: NDArray,
                                      y: NDArray,
                                      z: NDArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maps an RGB image array to a vertical surface grid.

    Args:
        rgb (NDArray): The RGB image array.
        x (NDArray): 1D array of x coordinates for the polyline.
        y (NDArray): 1D array of y coordinates for the polyline.
        z (NDArray): 1D array of z levels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The x, y, z flats, the i, j, k triangle indices, and the flattened RGB array.

    Raises:
        ValueError: If the shapes of rgb, x, and z are not compatible.
    """
    if (x.size != rgb.shape[1]) or (z.size!=rgb.shape[0]):
        raise ValueError("This implementation needs that rgb.shape=(z.size, x.size, 3or4)")
        
    tup1 = build_vertical_surface_polyline_and_zlevels(x=x,
                                                        y=y,
                                                        z=z)
    x_flat, y_flat, z_flat, i_indices, j_indices, k_indices = tup1
    rgb_flat = rgb.reshape(-1,4)
    tup =  x_flat, y_flat, z_flat, i_indices, j_indices, k_indices, rgb_flat
    return tup
    


def map_datarray_to_vertical_surface(*, datarray: xr.DataArray,
                               mpl_cmap_name: str = 'seismic', **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maps an xarray DataArray to a vertical surface grid with a colormap.

    Args:
        datarray (xr.DataArray): The input xarray DataArray containing x, y, and z coordinates.
        mpl_cmap_name (str, optional): The colormap name. Defaults to 'seismic'.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The x, y, z flats, the i, j, k triangle indices, and the vertex colors.
    """
    tup1 = build_vertical_surface_polyline_and_zlevels(x=datarray.x,
                                                        y=datarray.y,
                                                        z=datarray.z)
    x_flat, y_flat, z_flat, i_indices, j_indices, k_indices = tup1
    
    vertex_colors = map_array_to_cmap(values=datarray.data,
                             mpl_cmap_name=mpl_cmap_name, **kwargs)
    tup =  x_flat, y_flat, z_flat, i_indices, j_indices, k_indices, vertex_colors
    return tup
    
    
    

def orientations_to_pcloud(strikes: ArrayLike, dips: ArrayLike, centers: ArrayLike, radius: float = 1, num_sides: int = 15, **kwargs) -> pd.DataFrame:
    """Takes arrays of orientation data and builds disks to create a point cloud.

    Args:
        strikes (ArrayLike): Array of strike angles.
        dips (ArrayLike): Array of dip angles.
        centers (ArrayLike): Array of center coordinates (x, y, z).
        radius (float, optional): The radius of the disks. Defaults to 1.
        num_sides (int, optional): The number of sides for the disks. Defaults to 15.
        **kwargs: Additional keyword arguments.

    Returns:
        pd.DataFrame: A DataFrame representing the point cloud with 'x', 'y', and 'z' columns.
    """
    #takes arrays of orientation data and builds disks
    #and creates a pcloud
    coords_lst=[]
    # breakpoint()
    for strike, dip, center in zip(strikes, dips, centers):
        
        try:
            coords, _ = build_disk(strike, dip, center, radius, num_sides, **kwargs)
            coords_lst.append(coords)
        except TypeError as te:
            print(te)
            continue
    try:
        arr = np.concatenate(coords_lst, axis=0)
    except ValueError:
        pass
        
    df = pd.DataFrame(arr, columns=['x','y','z'])
    return df

def rotation_matrix(new_x: Union[Tuple, List, ArrayLike, gkp.Vector], new_y: Union[Tuple, List, ArrayLike, gkp.Vector], new_z: Union[Tuple, List, ArrayLike, gkp.Vector]) -> np.ndarray:
    """Calculates a rotation matrix.

    Checked with Algorithms in Structural Geology on July 2022.

    Args:
        new_x (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or Vector (vx, vy, vz).
        new_y (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or Vector (vx, vy, vz).
        new_z (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or Vector (vx, vy, vz).

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """    
    
    rotation_matrix = change_base_matrix(new_x, new_y, new_z).T
    
    return rotation_matrix
    
def change_base_matrix(new_x: Union[Tuple, List, ArrayLike, gkp.Vector], new_y: Union[Tuple, List, ArrayLike, gkp.Vector], new_z: Union[Tuple, List, ArrayLike, gkp.Vector]) -> np.ndarray:    
    """Calculates a change of base matrix.

    Checked with Algorithms in Structural Geology on July 2022.

    Args:
        new_x (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or Vector (vx, vy, vz).
        new_y (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or Vector (vx, vy, vz).
        new_z (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or Vector (vx, vy, vz).

    Returns:
        np.ndarray: The 3x3 change of base matrix.
    """    
    if all([len(arg)==2 for arg in locals().values()]):
        #arguments defined as trend & plunge
        v1 = gkp.Vector.from_trendplunge(*new_x)
        v2 = gkp.Vector.from_trendplunge(*new_y)
        v3 = gkp.Vector.from_trendplunge(*new_z)
    elif all([len(arg)==3 for arg in locals().values()]):
        #arguments defined as vectors (arrays) of xyz coordinates
        v1 = new_x
        v2 = new_y
        v3 = new_z
    
    change_base_matrix = np.stack((v1, v2, v3), axis=0)
    
    return change_base_matrix

def change_base(points: ArrayLike, new_x: Union[Tuple, List, ArrayLike, gkp.Vector], new_y: Union[Tuple, List, ArrayLike, gkp.Vector], new_z: Union[Tuple, List, ArrayLike, gkp.Vector]) -> np.ndarray:
    """Applies a change of base to a set of points.

    Args:
        points (ArrayLike): The input points.
        new_x (Union[Tuple, List, ArrayLike, gkp.Vector]): The new x base.
        new_y (Union[Tuple, List, ArrayLike, gkp.Vector]): The new y base.
        new_z (Union[Tuple, List, ArrayLike, gkp.Vector]): The new z base.

    Returns:
        np.ndarray: The points in the new base.
    """
    change_base_mat = change_base_matrix(new_x, new_y, new_z)
    try:
        new_base_pts = change_base_mat @ points  
    except ValueError:
        new_base_pts = change_base_mat @ np.array(points).T
        
    return new_base_pts


def rotate(points: ArrayLike, new_x: Union[Tuple, List, ArrayLike, gkp.Vector], new_y: Union[Tuple, List, ArrayLike, gkp.Vector], new_z: Union[Tuple, List, ArrayLike, gkp.Vector]) -> np.ndarray:
    """Rotates points using a rotation matrix defined by new base vectors.

    multiplication needs to be  rot_matrix @ points
    with rot_matrix = 3x3 and points = 3x1

    Args:
        points (ArrayLike): List or array of points.
        new_x (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or (vx, vy, vz).
        new_y (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or (vx, vy, vz).
        new_z (Union[Tuple, List, ArrayLike, gkp.Vector]): (trend, plunge) or (vx, vy, vz).

    Returns:
        np.ndarray: The rotated points.
    """
    rot_matrix = rotation_matrix(new_x, new_y, new_z)    
    # breakpoint()
    try:
        rotated_pts = rot_matrix @ points
    except ValueError:
        rotated_pts = rot_matrix @ np.array(points).T
        
    return rotated_pts.T

try:
    #delete the accesor to avoid the warning from pandas
    del pd.DataFrame.pcloud
except AttributeError:
    pass

def format_point_collection(points: PointCollection) -> np.ndarray:
    """Formats a collection of points into a numpy array.

    Args:
        points (PointCollection): The collection of points.

    Returns:
        np.ndarray: The formatted numpy array of coordinates.

    Raises:
        ValueError: If the size of the array representing xy(z) is not 2 or 3 columns.
    """
    if isinstance(points, MultiPoint):
        points=[point for point in points.geoms]
    elif isinstance(points, (list, tuple)):
        points = [pt for pt in points if isinstance(pt, Point)]
    elif isinstance(points, (pd.DataFrame, np.ndarray)):
        if isinstance(points, pd.DataFrame):
            array = points.values
        else:
            array = points
        if (array.shape[1]<2) or (array.shape[1]>3):
            raise ValueError('Size of array supposed to represent xy(z) is not right. It should have 2 or 3 columns.')
        else:
            points = np.array(array)
    return points

def mask_points_inside_perimeter(points: PointCollection,
                                 perimeter: shapely.Polygon) -> List[bool]:
    """Creates a boolean mask for points inside a given perimeter polygon.

    Args:
        points (PointCollection): The collection of points to check.
        perimeter (shapely.Polygon): The polygon perimeter.

    Returns:
        List[bool]: A mask indicating which points are inside the perimeter.
    """
    points_arr = format_point_collection(points)
    point_geoms = shapely.points(points_arr)
    mask = perimeter.geometry.values.covers(point_geoms)
    return mask

def mask_with_polygon(datarray: xr.DataArray, polygon: shapely.Polygon,
                      where: Literal['inside', 'outside']) -> np.ndarray:
    """Masks an xarray DataArray with a shapely polygon.

    Args:
        datarray (xr.DataArray): The DataArray to mask.
        polygon (shapely.Polygon): The polygon to use as a mask.
        where (Literal['inside', 'outside']): Whether to mask 'inside' or 'outside' the polygon.

    Returns:
        np.ndarray: The resulting boolean mask array.
    """
    X,Y = np.meshgrid(datarray.x, datarray.y)
    path = mplpath.Path([(x,y) for x,y in zip(*polygon.exterior.coords.xy)])
    flags = path.contains_points(np.hstack((X.flatten()[:,np.newaxis],
                                         Y.flatten()[:,np.newaxis])))
    flags.reshape(*X.shape)
    if where == 'inside':
        #True when inside polygon
        result = flags
    elif where == 'outside':
        #True when outside polygon
        result = np.logical_not(flags)
    return result


@pd.api.extensions.register_dataframe_accessor('pcloud')    
class Pcloud:
    
    def __init__(self, pandas_dataframe: pd.DataFrame, **kwargs):
        """Initializes the Pcloud accessor for a pandas DataFrame.

        Accessor grouping methods to work with a DataFrame containing data on 
        point clouds.   
        It assumes there are columns called x, y, and z.
        x, y, and z correspond to a ENU system.

        Args:
            pandas_dataframe (pd.DataFrame): The DataFrame to attach to.
            **kwargs: Additional keyword arguments.
        """
        
        self.data = pandas_dataframe        
        

    def __repr__(self) -> str:
        """Returns the string representation of the underlying DataFrame.

        Returns:
            str: The string representation.
        """
        return self.data.__repr__()
      
      
    def load_file(self, *args, **kwargs) -> None:
        """Loads a point cloud from a CSV file into the DataFrame.

        Args:
            *args: Positional arguments, expecting the filepath as the first argument.
            **kwargs: Additional keyword arguments for pandas read_csv.
        """
        df = pd.read_csv(args[0])
        self.data = df
          
    
    def x(self) -> pd.Series:
        """Gets the x coordinates.

        Returns:
            pd.Series: The x coordinates.
        """
        x = self.data.x
        return x
    
    def y(self) -> pd.Series:
        """Gets the y coordinates.

        Returns:
            pd.Series: The y coordinates.
        """
        y = self.data.y
        return y
    
    def z(self) -> pd.Series:
        """Gets the z coordinates.

        Returns:
            pd.Series: The z coordinates.
        """
        z = self.data.z
        return z
    
    def coord_columns(self) -> List[str]:
        """Gets the coordinate column names.

        Returns:
            List[str]: A list of coordinate column names.
        """
        coords_columns = [name for name in self.coord_names.values()]
        return coords_columns
        
    def coordinates(self) -> pd.DataFrame:
        """Gets the DataFrame containing only x, y, and z coordinates.

        Returns:
            pd.DataFrame: The coordinates DataFrame.
        """
        return self.data.loc[:,('x','y','z')]
        
    def coord_values(self) -> np.ndarray:
        """Gets the coordinate values as a numpy array.

        Returns:
            np.ndarray: The coordinates array.
        """
        return self.coordinates().values
    
    def centroid(self, func: str = 'mean') -> pd.Series:
        """Calculates the centroid of the point cloud.

        Args:
            func (str, optional): The function to use, e.g., 'mean'. Defaults to 'mean'.

        Returns:
            pd.Series: The centroid coordinates.
        """
        centroid = getattr(self.coordinates(), func)(axis=0)
        return centroid
    
    
    def least_squares(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Calculates the least squares fit of a plane to the point cloud.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[np.ndarray, np.ndarray, int, np.ndarray]: The fit, residual, rank, and singular values.
        """
        points = self.coord_values()
        A = np.array(np.hstack((points[:,0:2], np.ones((points.shape[0],1)))))
        b = np.array(points[:,2].reshape((points.shape[0],1)))
        fit, residual, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        return fit, residual, rank, singular_values
        
        
    def fit_plane_to_points(self, **kwargs) -> Tuple[gkp.Plane, float]:
        """Fits a plane to the points using PCA or least squares.

        points is an array nx3 where each row is a point.

        Args:
            **kwargs: Keyword arguments, including 'method' ('pca' or 'least_squares').

        Returns:
            Tuple[gkp.Plane, float]: The fitted plane and the residual.

        Raises:
            ValueError: If there are not enough points (less than 3) to get a plane.
        """
        method = kwargs.pop('method', 'pca')
        points = self.coord_values()
        
        if points.shape[0] < 3:
            raise ValueError('Not enough points to get plane')
        
        if method == 'least_squares':
            A = np.array(np.hstack((points[:,0:2], np.ones((points.shape[0],1)))))
            b = np.array(points[:,2].reshape((points.shape[0],1)))
            fit, residual, _, _ = np.linalg.lstsq(A, b, rcond=None)  
            # breakpoint()
            # errors = b - A * fit
            
            # A = np.matrix(np.hstack((points[:,0:2], np.ones((points.shape[0],1)))))
            # b = np.array(points[:,2].reshape((points.shape[0],1)))
            # fit = (A.T * A).I * A.T * b
            # errors = b - A * fit
            # residual = np.linalg.norm(errors)
            # pdb.set_trace()
            
            plane = gkp.Vector([-fit[0,0], -fit[1,0], 1]).unit.view(gkp.Plane)
        
        elif method == 'pca':
            pca = PCA(n_components=3)
            pca.fit(points)
            normal_vector = pca.components_[-1] 
            plane = normal_vector.view(gkp.Plane)
            # Compute residuals (perpendicular distances)
            plane_point = pca.mean_
            def point_to_plane_distance(point, plane_point, normal_vector):
                return abs(np.dot(point - plane_point, normal_vector)) / np.linalg.norm(normal_vector)
            residual = np.array([point_to_plane_distance(p, plane_point, normal_vector) for p in points]).sum()
        
        return plane, residual

    def project_points_on_plane(self, plane: gkp.Plane, **kwargs) -> np.ndarray:
        """Projects the point cloud onto a given plane.

        Centers the cloud, projects it on the given plane, and returns the new XYZ coordinates.

        Args:
            plane (gkp.Plane): The plane onto which to project the points.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The projected coordinates.
        """
        #projects the points into a plane with translation
        
        #center the cloud        
        centered_cloud = self.coordinates() - self.centroid()
        
        #project the centered cloud on the given plane 
        new_xyz=plane.project_on_this_plane(centered_cloud.values)
        
        return new_xyz     
            
    def convex_hull_3D(self, plane: Optional[gkp.Plane] = None, **kwargs) -> np.ndarray:
        """Finds a convex hull marking the perimeter of the point cloud.

        Args:
            plane (Optional[gkp.Plane], optional): A plane that fits the point cloud. 
                If not given, it uses PCA. Defaults to None.
            **kwargs: Additional arguments, such as 'spline' (bool) for smoothing.

        Returns:
            np.ndarray: The 3D coordinates of the convex hull.
        """
        spline = kwargs.pop('spline', True)
        #get plane that fits the point cloud.
        #not dependant on location, only the orientation counts
        if plane is None:
            pl, _ = self.fit_plane_to_points(method='pca')
        else:
            pl = plane
        
        new_xyz  = self.project_points_on_plane(pl)
        
        #Calculate a 2D hull defined by the points projected onto the fit plane
        hull = ConvexHull(new_xyz[:,:2])
        
        # plt.figure().subplots(1,1).scatter(new_xyz[:,0],new_xyz[:,1])
        # plt.plot(new_xyz[hull.vertices,0], new_xyz[hull.vertices,1], 'r--', lw=2)
        
        hull_nrows = hull.vertices.shape[0]
        hull_coords = np.column_stack((hull.points[hull.vertices,:],
                                       np.zeros(hull_nrows)))
        
        
        if spline:
            try:
                n_pts = 100
                hull_coords_copy = copy(hull_coords)
                tck, u_spl = splprep(hull_coords_copy[:,:2].T, u=None, s=0.0, per=1) 
                u_new = np.linspace(u_spl.min(), u_spl.max(), n_pts)
                x_new, y_new = splev(u_new, tck, der=0)
                hull_coords = np.column_stack((x_new, y_new,
                                               np.zeros(n_pts)))
            except TypeError:
                pass
        
        
        #rotate back from fit plane reference to xyz
        rot_matrix = pl.axes.as_matrix.T
        # bck_rot=gkp.Axes.xyz().dircos(pl.axes)
        result = np.matmul(rot_matrix, hull_coords.T).T
        
        #back translate to the cloud location        
        centroid = self.centroid().values
        result += centroid
        
        #close hull
        result = np.append(result, result[0,:].reshape((-1,3)), axis=0)
        
        return result
    
    # def move_to_xy(self, plane=None):
    #     #takes a plane and use it to move the pcloud to xy.
    #     #if no plane is given it will get the best fit plane using pca
        
    #     if plane is None:
    #         pl, _ = self.fit_plane_to_points(method='pca')
    #     else:
    #         pl = plane
            
    #     centroid = self.centroid().values
    #     new_xyz  = self.project_points_on_plane(pl) - centroid
        
    #     return new_xyz        
        
    
    def interpolated_grid(self, num_pts: int = 100, plane: Optional[gkp.Plane] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolates points onto a regular grid on a specified plane.

        Args:
            num_pts (int, optional): The number of points for the grid. Defaults to 100.
            plane (Optional[gkp.Plane], optional): The plane to project to. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The X, Y, and Z grids.
        """
        if plane is None:
            pl, _ = self.fit_plane_to_points(method='pca')
        else:
            pl = plane
        
        new_xyz  = self.project_points_on_plane(pl)
        
        xy_points = new_xyz[:,:2]
        values = new_xyz[:,2]
        x      = new_xyz[:,0]
        y      = new_xyz[:,1]
        grid_x, grid_y = np.mgrid[min(x):max(x):num_pts*1j,
                                  min(y):max(y):num_pts*1j]
        
        grid_z = griddata(xy_points, values, (grid_x, grid_y), method='cubic')
        
        flat_grid =  np.stack((grid_x.flatten(),
                               grid_y.flatten(),
                               grid_z.flatten()), axis=1)
        
        #return to original base
        rotation_matrix = pl.axes.__array__
        new_grid = ((flat_grid @ rotation_matrix) + self.centroid().values).reshape((num_pts,num_pts,3))        
       
        return new_grid[:,:,0], new_grid[:,:,1], new_grid[:,:,2]
    
    def mesh_points(self, **kwargs) -> tri.Triangulation:
        """Creates a mesh triangulation of the points projected on a plane.

        Args:
            **kwargs: Additional keyword arguments, such as 'plane'.

        Returns:
            tri.Triangulation: The triangulation object.
        """
        plane = kwargs.pop('plane', None)
        if plane is None:
            pl, _ = self.fit_plane_to_points(method='pca')
        else:
            pl = plane
        
        new_xyz  = self.project_points_on_plane(pl)
        
        triangulation = tri.Triangulation(new_xyz[:,0], new_xyz[:,1])
        
        return triangulation
    
    def fit_with_open3d(self, method: str = 'poisson', **kwargs) -> "open3d.geometry.TriangleMesh":
        """Fits a 3D surface to the point cloud using Open3D algorithms.

        Algorithms available: 'poisson', 'ball_pivoting', 'alpha_shape'.

        Args:
            method (str, optional): The Open3D method to use. Defaults to 'poisson'.
            **kwargs: Additional keyword arguments such as 'plane', 'radius_factor', 'alpha'.

        Returns:
            open3d.geometry.TriangleMesh: The reconstructed surface mesh.

        Raises:
            ValueError: If the open3d module is not installed.
        """
        if not OPEN3D:
            raise ValueError("Module open3d needed to run this function")
        plane = kwargs.pop('plane', None)
        if plane is None:
            pl, _ = self.fit_plane_to_points(method='pca')
        else:
            pl = plane
        
        new_xyz  = self.project_points_on_plane(pl)
        
        pcd = open3d.geometry.PointCloud()
        # pdb.set_trace()
        pcd.points = open3d.utility.Vector3dVector(new_xyz)
        pcd.remove_statistical_outlier(10, 500)        
        pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        centroid = pcd.get_center()
        pcd.translate(np.array([0,0,0]), relative=False)
        # bbox = pcd.get_axis_aligned_bounding_box()
        
        if method == 'poisson':
            defaults = dict(depth=4, width=0, scale=1.1, linear_fit=False)
            defaults.update(kwargs)
            kwargs = defaults
            mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, **kwargs)[0]
        elif method == 'ball_pivoting':
            radius_factor = kwargs.pop('radius_factor', 3)
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = radius_factor * avg_dist            
            mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,open3d.utility.DoubleVector([radius, radius * 2]))
        elif method == 'alpha_shape':
            alpha = kwargs.pop('alpha', 0.5)
            mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
        
        #next lines will 
        #   -build a convex hull around the point cloud
        #   -project it on the surface
        #   -filter out the vertices outside the convex hull
        #   -generate a new vertex array including the convex hull
        #   -re-triangulate and generate a new TriangleMesh
        #   -rotate it and translate it to the original position
        #   -return
        
        #get convex hull around points
        processed_xyz = pd.DataFrame(np.asarray(pcd.points), columns=['x','y','z'])
        hull3D = processed_xyz.pcloud.convex_hull_3D(plane=gkp.Plane(0,0),
                                                  spline=True)
        # pdb.set_trace()
        vertices = np.asarray(mesh.vertices)
        #project the convex hull into the mesh
        #Try to use Rbf. If some error is raised, fall back to griddata
        #griddata cannot extrapolate so zeros are added if needed
        
        try:
            rbf3 = Rbf(vertices[:,0], vertices[:,1], vertices[:,2], degree=-1)
            hull3D[:,2] = rbf3(hull3D[:,0], hull3D[:,1])
        except LinAlgError as e:
            hull3D[:,2] = griddata(vertices[:,:2], vertices[:,2], hull3D[:,:2],
                                   method='cubic', fill_value=0)
            
        #clip z-values of the convex hull so they are less than n*std from mean
        n=1
        z_hull_std = np.std(hull3D[:,2])
        z_hull_mean = np.mean(hull3D[:,2])
        hull3D[:,2] = np.clip(hull3D[:,2],
                              z_hull_mean-n*z_hull_std,
                              z_hull_mean+n*z_hull_std)
        
        #check which vertices of the mesh lie inside the convex hull
        #this is done in 2d using only xy
        path = mplpath.Path(hull3D[:,:2])
        inside_hull=path.contains_points(vertices[:,:2])
        
        #keep only vertices inside hull and concatenate with hull
        vertices_crop = np.vstack((vertices[inside_hull,:], hull3D))
        
        #triangulate the new array
        triangulation = tri.Triangulation(vertices_crop[:,0],vertices_crop[:,1])
        
        trimesh_vertices  = open3d.utility.Vector3dVector(vertices_crop)
        trimesh_triangles = open3d.utility.Vector3iVector(triangulation.triangles)
        mesh_crop = open3d.geometry.TriangleMesh(trimesh_vertices, trimesh_triangles)
        
        
        #plot to test this part of the function. Normally commented
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax=fig.add_subplot(projection='3d')
        # ax.plot(processed_xyz.x, processed_xyz.y, processed_xyz.z, 'k.')
        # orig_tri = tri.Triangulation(vertices[:,0], vertices[:,1],
        #                              triangles=np.asarray(mesh.triangles))
        # ax.plot_trisurf(orig_tri, Z=vertices[:,2])
        # ax.plot(hull3D[:,0], hull3D[:,1], hull3D[:,2], 'r.')
        # pdb.set_trace()
        # ax.plot(hull3D[:,0], hull3D[:,1], 0, 'g.')
        # ax.plot(vertices[:,0], vertices[:,1], vertices[:,2], 'm.')
        # ax.plot(vertices_crop[:,0], vertices_crop[:,1], vertices_crop[:,2], 'b.')
        # ax.plot_trisurf(triangulation, Z=vertices_crop[:,2])
        
        
        #return to original base
        rotation_matrix = pl.axes.__array__.T
        final_mesh = mesh_crop.rotate(rotation_matrix).translate(centroid, relative=False)
        # new_vertices = ((vertices @ rotation_matrix) + centroid)
        
        # triangles = np.asarray(final_mesh.triangles)
        # vertices = np.asarray(final_mesh.vertices)
        
        # return vertices, triangles
        return final_mesh
    
    def fit_with_weighted_avg(self, **kwargs) -> "open3d.geometry.TriangleMesh":
        """Fits a surface to the point cloud using a weighted average technique.

        Projects points on a plane, grids them, and computes Z values by weighted averaging.

        Args:
            **kwargs: Keyword arguments including 'plane', 'spacing', 'n_elements', and 'average_kwargs'.

        Returns:
            open3d.geometry.TriangleMesh: The resulting triangulated mesh.
        """
        
        plane = kwargs.pop('plane', None)
        spacing = kwargs.pop('spacing', None)
        n_elements = kwargs.pop('n_elements', None)
        
        average_kwargs_default = {'closest_nelems':None, 'window_name':None,
                                  'window_args':None, 'window_kwargs':None}
        average_kwargs_default.update(kwargs.pop('average_kwargs',{}))
        average_kwargs=average_kwargs_default
        
        if plane is None:
            pl, _ = self.fit_plane_to_points(method='pca')
        else:
            pl = plane
        
        temp_xyz  = self.project_points_on_plane(pl)
        
        temp_xyz_df = pd.DataFrame(temp_xyz, columns=['x','y','z'])
        hull3D   = temp_xyz_df.pcloud.convex_hull_3D(plane=gkp.Plane(0,0),
                                                  spline=True)
        
        #create grid
        
        xmin=hull3D[:,0].min()
        xmax=hull3D[:,0].max()
        ymin=hull3D[:,1].min()
        ymax=hull3D[:,1].max()
        if spacing:
            n_elem_x = round((xmax-xmin)/spacing)
            n_elem_y = round((ymax-ymin)/spacing)        
        elif n_elements:
            n_elem_x = n_elements
            n_elem_y = n_elements
            
        # xgrid = np.ogrid[xmin:xmax:n_elem_x*1j]
        # ygrid = np.ogrid[ymin:ymax:n_elem_y*1j]
        xgrid, ygrid = np.mgrid[xmin:xmax:n_elem_x*1j, ymin:ymax:n_elem_y*1j]
        grid = np.c_[xgrid.flatten(), ygrid.flatten()]
        
        #check which vertices of the mesh lie inside the convex hull
        #this is done in 2d using only xy
        path = mplpath.Path(hull3D[:,:2])
        inside_hull=path.contains_points(grid)
    
        grid_in_hull = np.c_[grid[inside_hull], [0]*sum(inside_hull)]
        
        def compute_weights(distances: np.ndarray, closest_nelems: Optional[int] = None,
                    window_name: Optional[str] = None, window_args: Optional[Tuple] = None, window_kwargs: Optional[dict] = None) -> np.ndarray:
            """Computes distance-based weights for the averaging algorithm.

            Args:
                distances (np.ndarray): Array of distances to neighboring points.
                closest_nelems (Optional[int], optional): Number of elements to consider. Defaults to None.
                window_name (Optional[str], optional): The name of a scipy window function. Defaults to None.
                window_args (Optional[Tuple], optional): Window arguments. Defaults to None.
                window_kwargs (Optional[dict], optional): Window keyword arguments. Defaults to None.

            Returns:
                np.ndarray: The computed weights.

            Raises:
                ValueError: If the open3d module is not installed.
            """
            
            if not OPEN3D:
                raise ValueError("Module open3d needed to run this function")
            if window_name:
                #All windows are symmetrical, so we need to generate one that's
                #twice the num_elements and take only the half
                if closest_nelems is None:
                    window_size=2*len(distances)
                    closest_nelems = len(distances)
                else:
                    window_size=2*closest_nelems
                window=getattr(windows, window_name)(window_size, *window_args, **window_kwargs)
                #take only half of the window
                weights=window[window_size//2:]
            else:
                #if no window is specified calculate the weights from 1/distances
                argsort = np.argsort(distances)
                weights = 1/distances[argsort[:closest_nelems]]
            return weights
                
        
        closest_nelems = average_kwargs.get('closest_nelems',None)
            
        
        z_lst=[]
        for point in grid_in_hull:
            distances=np.sqrt(np.sum((temp_xyz-point)**2, axis=1))
            argsort = np.argsort(distances)
            z_to_avg=temp_xyz[argsort[:closest_nelems],2]
            weights = compute_weights(distances, **average_kwargs)
            z=np.average(z_to_avg, weights=weights[:len(z_to_avg)])
            z_lst.append(z)
        grid_in_hull[:,2]=z_lst
        
        
        triangulation = tri.Triangulation(grid_in_hull[:,0],grid_in_hull[:,1])
        
        trimesh_vertices  = open3d.utility.Vector3dVector(grid_in_hull)
        trimesh_triangles = open3d.utility.Vector3iVector(triangulation.triangles)
        mesh = open3d.geometry.TriangleMesh(trimesh_vertices, trimesh_triangles)
        
        # # plot to test this part of the function. Normally commented
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax=fig.add_subplot(projection='3d')
        # ax.plot(new_xyz[:,0],new_xyz[:,1],new_xyz[:,2], 'k.')
        ax.plot(hull3D[:,0],hull3D[:,1], 0, 'r-')
        ax.plot(grid_in_hull[:,0], grid_in_hull[:,1], grid_in_hull[:,2], 'g.')
        # # ax.plot_trisurf(triangulation, Z=grid_in_hull[:,2])
        # pdb.set_trace()        
        
        
        #return to original base
        rotation_matrix = pl.axes.as_matrix.T
        final_mesh = mesh.rotate(rotation_matrix).translate(self.centroid().values, relative=False)
        return final_mesh
    
try:
    #delete the accesor to avoid warning 
    del xr.DataArray().pgrid
except AttributeError:
    pass

@xr.register_dataarray_accessor('pgrid')
class Pgrid:
    
    def __init__(self, data_array: xr.DataArray):
        """Accessor to handle structured grids (regular grids with xyz + values).

        Args:
            data_array (xr.DataArray): The xarray DataArray.
        """
        self._obj = data_array
        
    @classmethod
    def factory(cls, axes: gkp.Axes, origin: Tuple[float, float, float], step: Tuple[float, float, float], size: Tuple[int, int, int]) -> xr.DataArray:
        """Creates a structured grid DataArray from basic parameters.

        Args:
            axes (gkp.Axes): The axes definition for orientation.
            origin (Tuple[float, float, float]): The origin (x, y, z).
            step (Tuple[float, float, float]): The spacing step in (x, y, z).
            size (Tuple[int, int, int]): The number of points in (x, y, z).

        Returns:
            xr.DataArray: The generated DataArray.
        """
        # xyz = gkp.Axes.xyz()
        
        coords_list=[np.arange(0, s7ep*s1ze, step=s7ep) for s7ep, s1ze in zip(step, size)]
        
        #translate to origin
        coords_list = [coord+or1gin for coord, or1gin in zip(coords_list, origin)]
        
        #rotate to align with axes
        rot_matrix = axes.dircos(gkp.Axes.xyz())
        #Coordinates vector
        try:
            xv, yv, zv = np.meshgrid(*coords_list)   
            v=gkp.Vector(np.stack((xv,yv,zv), axis=-1))
        except:
            pass
        #rotation
        vr = v@rot_matrix
        
        darray = xr.DataArray(np.empty(size),
                              coords=[('x', vr[0,:,0,0]),
                                      ('y', vr[:,0,0,1]),
                                      ('z', vr[0,0,:,2])])        
        return darray
    
    @classmethod
    def from_arrays(cls, x: ArrayLike, y: ArrayLike, values: ArrayLike, *, z: Optional[ArrayLike] = None, force_structured: bool = False, **kwargs) -> xr.DataArray:
        """Creates a Pgrid DataArray from coordinate arrays and values.

        Args:
            x (ArrayLike): The x coordinates.
            y (ArrayLike): The y coordinates.
            values (ArrayLike): The grid values.
            z (Optional[ArrayLike], optional): The z coordinates. Defaults to None.
            force_structured (bool, optional): Whether to force structured grid generation. Defaults to False.
            **kwargs: Additional keyword arguments like 'nx', 'ny', 'nz'.

        Returns:
            xr.DataArray: The resulting DataArray.
        """
        nx=kwargs.pop('nx',None)
        ny=kwargs.pop('ny',None)
        nz=kwargs.pop('nz',None)
        
        if nx is None:
            nx = len(np.unique(x))
        if ny is None:
            ny = len(np.unique(y))
        if (nz is None) and (z is not None):            
            nz = len(np.unique(z))
        
        if z is None:
            csv_fits_structured_grid = nx * ny == len(x)
        else:
            csv_fits_structured_grid = nx * ny * nz == len(x)
            
        # breakpoint()
        if csv_fits_structured_grid:
            darray = structured_data_to_grid(x,y,values,z=z)
        else:
            if force_structured:
                darray = incomplete_structured_data_to_grid(x,y,values,z=z)
            else:
                kwargs.update({'nx':nx,'ny':ny,'nz':nz})
                darray = non_structured_data_to_grid(x,y,values,z=z,**kwargs)
        
        return darray
        
    @classmethod
    def from_csv(cls, *, filepath: str,
                 skip_header: int,
                 column_mapping: Optional[dict] = None,
                 force_structured: bool = False,
                 read_csv_kwargs: Optional[dict] = None,
                 convert_from_arrays_kwargs: Optional[dict] = None) -> xr.DataArray:
        """Loads a Pgrid DataArray from a CSV file.

        Args:
            filepath (str): The path to the CSV file.
            skip_header (int): Number of header lines to skip.
            column_mapping (Optional[dict], optional): Mapping of column names. Defaults to None.
            force_structured (bool, optional): Whether to force structured grid generation. Defaults to False.
            read_csv_kwargs (Optional[dict], optional): Kwargs for read_csv. Defaults to None.
            convert_from_arrays_kwargs (Optional[dict], optional): Kwargs for from_arrays. Defaults to None.

        Returns:
            xr.DataArray: The generated DataArray.
        """
        if read_csv_kwargs is None: read_csv_kwargs = {}
        if convert_from_arrays_kwargs is None: convert_from_arrays_kwargs = {}
        
        # breakpoint()
        x, y, z, values = read_csv(filepath=filepath, skip_header=skip_header,
                                   column_mapping=column_mapping, **read_csv_kwargs)
        # breakpoint()
        darray=cls.from_arrays(x,y,values,z=z,
                        force_structured=force_structured,
                        **convert_from_arrays_kwargs)

        
        return darray
    
    
    def extract_data_along_path(self, *,                                      
                                      path_geometry: pd.Series,
                                      interp_kwargs: Optional[dict] = None) -> xr.DataArray:
        """Extracts data along a defined 2D path geometry.

        Args:
            path_geometry (pd.Series): The geometry defining the path.
            interp_kwargs (Optional[dict], optional): Arguments for interpolation. Defaults to None.

        Returns:
            xr.DataArray: A DataArray with extracted values along the path.
        """
        if interp_kwargs is None:
            interp_kwargs = {}
        x = [pt.x for pt in path_geometry]
        y = [pt.y for pt in path_geometry]
        
        lstring = shapely.LineString(path_geometry)
        dist = [lstring.project(pt) for pt in path_geometry]
        x_darr = xr.DataArray(x, dims='values')
        y_darr = xr.DataArray(y, dims='values')
        values_2d_darray=self._obj.interp(x=x_darr, y=y_darr, **interp_kwargs)
        
        coords={
                # 'z':values_2d_darray.z.data,
                'distance':dist,
                'x':('distance',x_darr.data),
                'y':('distance',y_darr.data),
                }
        dims=["z",'distance']
        try:
            coords.update({'z':values_2d_darray.z.data,})
            dims=['z','distance']
        except AttributeError:
            dims=['distance']
            
        values = xr.DataArray(values_2d_darray.data,
                                   coords=coords,
                                   dims=dims)
        return values
    
    
    
    def to_pyvista_imagegrid(self) -> Any:
        """Converts the DataArray to a PyVista ImageData grid.

        Returns:
            Any (pyvista.ImageData): The PyVista image grid.

        Raises:
            ValueError: If the pyvista module is not installed.
        """
        if not PYVISTA:
            raise ValueError("Module pyvista needed to run this function")
        darray = self._obj
        x = darray.x
        y = darray.y
        z = darray.z
        # breakpoint()
        grid = pyvista.ImageData()  # Equivalent to UniformGrid
        grid.dimensions = np.array(darray.shape) #+ 1  # PyVista requires dims + 1
        grid.spacing = (np.diff(x).mean(),
                        np.diff(y).mean(),
                        np.diff(z).mean())  # Compute spacing
        grid.origin = (x.min(), y.min(), z.min())  # Set origin
        grid.point_data["values"] = darray.values.flatten(order="F")  # Assign values        
        return grid
    
    def to_pyvista_structuredgrid(self) -> Any:
        """Converts the DataArray to a PyVista StructuredGrid.

        Returns:
            Any (pyvista.StructuredGrid): The PyVista structured grid.
        """
        image_grid=self._obj.pgrid.to_pyvista_imagegrid()
        struct_grid = image_grid.cast_to_structured_grid()
        return struct_grid
        
    
    def plot_pyvista(self, **kwargs) -> Any:
        """Plots the grid using PyVista.

        Args:
            **kwargs: Keyword arguments like 'plotter' and 'threshold'.

        Returns:
            Any (pyvista.Plotter): The active plotter object.

        Raises:
            ValueError: If the pyvista module is not installed.
        """
        if not PYVISTA:
            raise ValueError("Module pyvista needed to run this function")
        plotter = kwargs.get('plotter', pyvista.Plotter())
        threshold = kwargs.get('threshold', None)
        
        grid=self._obj.pgrid.to_pyvista_structuredgrid()
        # breakpoint()
        
        data = grid.point_data['values']#.reshape(62, 59, 60)
        nan_mask = np.isnan(data)
        grid.hide_points(nan_mask)
        
        if threshold:
            grid = grid.threshold(threshold, scalars="values")
        
        grid_actor= plotter.add_mesh(grid, cmap="viridis",
                         show_edges=True, scalars="values")
        return plotter
    
    def plot_matplotlib(self, ax: Any = None) -> Tuple[Any, Any]:
        """Plots the grid using Matplotlib voxels.

        Assumes the cells to fill have a float value different from np.nan
        Assumes regular spacing in all 3 dimensions.

        Args:
            ax (Any, optional): The matplotlib axis. Defaults to None.

        Returns:
            Tuple[Any, Any]: The axis and the faces (voxels) object.
        """
        #Assumes the cells to fill have a float value different from np.nan
        #Assumes regular spacing in all 3 dimensions
        if ax is None:
            ax = plt.figure().add_subplot(projection='3d')        
        
        
        #rearrange the dimensions of self._obj to follow x,y,z order
        darray_T = self._obj.transpose("x", "y", "z")
        
        #Coordinates must have one element less than the values
        # breakpoint()
        x = darray_T.x
        y = darray_T.y
        z = darray_T.z
        X,Y,Z = np.meshgrid(x,y,z)
        
        #which cells to fill
        filled = np.logical_not(darray_T.isnull())[:-1,:-1,:-1]     
        facecolors='gray'        
        faces = ax.voxels(#X,Y,Z,
                          filled, facecolors=facecolors, edgecolors='k')
        return ax, faces
    
    def replace_inside_polygon(self, polygon: shapely.Polygon, fill_value: float = np.nan) -> xr.DataArray:
        """Replaces values inside a polygon with a fill value.

        Args:
            polygon (shapely.Polygon): The masking polygon.
            fill_value (float, optional): The value to use for filling. Defaults to np.nan.

        Returns:
            xr.DataArray: The modified DataArray.
        """
        #True outside the polygon
        mask = mask_with_polygon(self._obj, polygon, where='outside')
        mask = mask.reshape(self._obj.shape[-2:])
        #Keep True values (outside polygon) and replace those inside
        return self._obj.where(mask, fill_value)
        
    def replace_outside_polygon(self, polygon: shapely.Polygon, fill_value: float = np.nan) -> xr.DataArray:
        """Replaces values outside a polygon with a fill value.

        Args:
            polygon (shapely.Polygon): The masking polygon.
            fill_value (float, optional): The value to use for filling. Defaults to np.nan.

        Returns:
            xr.DataArray: The modified DataArray.
        """
        #True inside the polygon
        mask = mask_with_polygon(self._obj, polygon, where='inside')
        mask = mask.reshape(self._obj.shape[-2:])
        #Keep True values (inside polygon) and replace those outside
        return self._obj.where(mask, fill_value)
        
        
    def find_isosurface(self, *, threshold_value: float,                                
                                    mode: Literal['first','last'],
                                    z_mode: Literal['elevation', 'depth']) -> xr.DataArray:
        """Finds the z-values (isosurface) at a specific threshold value.

        Args:
            threshold_value (float): The threshold value.
            mode (Literal['first', 'last']): Whether to find the first or last crossing.
            z_mode (Literal['elevation', 'depth']): The direction of the z-axis.

        Returns:
            xr.DataArray: A DataArray containing the z-values for the isosurface.
        """
        
        idx=self.find_idx_at_threshold(threshold_value=threshold_value,
                                        mode=mode, z_mode=z_mode)    
        z_values = self.interpolate_to_find_values(idx=idx,threshold_value=threshold_value)
        return z_values
    
    
    def find_idx_at_threshold(self, *, threshold_value: float,                                
                                    mode: Literal['first','last'],
                                    z_mode: Literal['elevation', 'depth']) -> np.ndarray:
        """Finds the vertical index where the value crosses a threshold.

        Args:
            threshold_value (float): The threshold value.
            mode (Literal['first', 'last']): Whether to find the first or last crossing.
            z_mode (Literal['elevation', 'depth']): The direction of the z-axis.

        Returns:
            np.ndarray: An array of indices corresponding to the threshold crossing.
        """
        
        #apply threshold to DataArray
        # mask_crossings = ant_darray.data >= threshold_value # Boolean mask of threshold crossings
        #Assumes Z values are sorted in ascending order
        #If Z is elevation then first values are the deepest
        #If Z is depth then first values are the shallowest
        # Z is elevation:
            #If mode is "first":
                # Operator must be "le"
            #If mode is "last":
                # Operator must be "ge"
        # Z is depth:            
            #If mode is "first":
                # Operator must be "ge"
            #If mode is "last":
                # Operator must be "le"
            
        #finding the first occurrence of True (maximum  between True and False)
        #results in finding the deepest case of going under the threshold
                
        #index of the vertical dimension
        z_idx = self._obj.dims.index('z')
        
        datarray = self._obj
        if mode == 'first':
            if z_mode == 'elevation':            
                oper = py_operator.le
            elif z_mode == 'depth':
                oper = py_operator.ge
        elif mode == 'last':            
            datarray = np.flip(datarray,axis=z_idx)            
            if z_mode == 'elevation': 
                oper = py_operator.ge
            elif z_mode == 'depth':
                oper = py_operator.le
                
        mask_crossings = oper(datarray, threshold_value)
        idx = mask_crossings.argmax(axis=z_idx)
        if mode == 'last':
            idx=self._obj.shape[z_idx]-(idx+1)
        
        # Mask rows that don't contain any True values with a placeholder (e.g., -1)
        # has_true = mask_crossings.any(axis=z_idx)
        # clean_indices = np.where(has_true, idx, -1) #-1 or np.nan?

        # # breakpoint()
        # match mode:
        #     case 'first':
        #         mask_crossings = self._obj.data <= threshold_value
        #         idx = mask_crossings.argmax(axis=z_idx)
        #     case 'last':
        #         mask_crossings = np.flip(self._obj,axis=z_idx) >= threshold_value
        #         idx = mask_crossings.argmax(axis=z_idx)
        #         idx=self._obj.shape[z_idx]-(idx)                
        # breakpoint()
        return idx.data
    
    def interpolate_to_find_values(self, *,
                                   idx: xr.DataArray,
                                   threshold_value: float) -> xr.DataArray:
        """Interpolates values to find a more accurate depth for a threshold value.

        Args:
            idx (xr.DataArray): The array of integer indices.
            threshold_value (float): The threshold value.

        Returns:
            xr.DataArray: A DataArray containing the interpolated z-values.
        """
        #index of the vertical dimension
        z_idx = self._obj.dims.index('z')
        #Interpolates to find a more aqurate depth for the threshold value
        vs_values_upper = np.take_along_axis(self._obj.data, np.expand_dims(idx, axis=z_idx), axis=z_idx)
        vs_values_lower = np.take_along_axis(self._obj.data, np.expand_dims(idx-1, axis=z_idx), axis=z_idx)
        z_values_upper = self._obj.z[idx.flat].data.reshape(idx.shape[z_idx],-1)
        z_values_lower = self._obj.z[(idx-1).flat].data.reshape(idx.shape[z_idx],-1)
        proportion = (threshold_value-vs_values_upper)/(vs_values_lower-vs_values_upper)
        z_values = z_values_upper-(proportion*(z_values_upper-z_values_lower)).squeeze()
        result = xr.DataArray(z_values, dims=['y','x'],coords=(self._obj.y, self._obj.x))
        return result



def structured_data_to_grid(x: ArrayLike, y: ArrayLike, values: ArrayLike, z: Optional[ArrayLike] = None) -> xr.DataArray:
    """Converts structured arrays to a regular DataArray grid.

    Args:
        x (ArrayLike): The x coordinates.
        y (ArrayLike): The y coordinates.
        values (ArrayLike): The data values.
        z (Optional[ArrayLike], optional): The z coordinates. Defaults to None.

    Returns:
        xr.DataArray: The resulting regular grid DataArray.
    """
    if z is not None:
        #3D case
        # breakpoint()
        sorted_data = np.lexsort((x, y, z)) # Sort by z, then by y, then by x
        new_values = values[sorted_data]
        xs=x[sorted_data]
        ys=y[sorted_data]
        zs=z[sorted_data]        
        x_unique, y_unique, z_unique = np.unique(x), np.unique(y), np.unique(z)
        nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
        assert np.all(xs.reshape(nz, ny, nx) == x_unique[np.newaxis, np.newaxis, :]), "x not cycling fastest"
        assert np.all(ys.reshape(nz, ny, nx) == y_unique[np.newaxis, :, np.newaxis]), "y not cycling correctly"
        assert np.all(zs.reshape(nz, ny, nx) == z_unique[:, np.newaxis, np.newaxis]), "z not cycling slowest"
        grid = new_values.reshape(nz,ny,nx)    
        darray = xr.DataArray(
                        grid,
                        dims=['z', 'y', 'x'],
                        coords={
                            'z': z_unique,
                            'y': y_unique,
                            'x': x_unique,
                        }
                    )
    else:
        #2D case
        sorted_data = np.lexsort((x, y))
        new_values = values[sorted_data]
        x_unique, y_unique = np.unique(x), np.unique(y)
        nx, ny = len(x_unique), len(y_unique)
        grid = new_values.reshape(ny, nx)
        darray = xr.DataArray(grid,
                              coords=[('y', y_unique),
                                      ('x', x_unique)])
    return darray

def incomplete_structured_data_to_grid(x: ArrayLike, y: ArrayLike, values: ArrayLike, z: Optional[ArrayLike] = None) -> xr.DataArray:
    """Creates a grid from incomplete structured data.
    
    The original grid is structured but the input values do not include the nan
    values that complete the grid.

    Args:
        x (ArrayLike): The x coordinates.
        y (ArrayLike): The y coordinates.
        values (ArrayLike): The data values.
        z (Optional[ArrayLike], optional): The z coordinates. Defaults to None.

    Returns:
        xr.DataArray: The resulting grid DataArray.
    """
    if z is not None:
        #3D case
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        z_unique = np.unique(z)
        nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)
        grid = np.full((nx, ny, nz), np.nan)
        
        # Find indices of x, y, z in the structured grid
        ix = np.searchsorted(x_unique, x)
        iy = np.searchsorted(y_unique, y)
        iz = np.searchsorted(z_unique, z)
        # Use fancy indexing to insert values at correct positions
        grid[ix, iy, iz] = values
        
        darray = xr.DataArray(grid,
                              coords=[('x', x_unique),
                                      ('y', y_unique),
                                      ('z', z_unique)])
    else:
        #2D case
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        nx, ny = len(x_unique), len(y_unique)
        grid = np.full((nx, ny), np.nan)
        
        # Find indices of x, y, z in the structured grid
        ix = np.searchsorted(x_unique, x)
        iy = np.searchsorted(y_unique, y)
        # Use fancy indexing to insert values at correct positions
        grid[ix, iy] = values
        
        darray = xr.DataArray(grid,
                              coords=[('x', x_unique),
                                      ('y', y_unique),])
        
    return darray

def non_structured_data_to_grid(x: ArrayLike, y: ArrayLike, values: ArrayLike, z: Optional[ArrayLike] = None, nx: int = 100, ny: int = 100, nz: int = 100,
                                **kwargs) -> xr.DataArray:
    """Interpolates non-structured data onto a regular grid.

    Args:
        x (ArrayLike): The x coordinates.
        y (ArrayLike): The y coordinates.
        values (ArrayLike): The data values.
        z (Optional[ArrayLike], optional): The z coordinates. Defaults to None.
        nx (int, optional): Grid points in x. Defaults to 100.
        ny (int, optional): Grid points in y. Defaults to 100.
        nz (int, optional): Grid points in z. Defaults to 100.
        **kwargs: Additional interpolation keyword arguments.

    Returns:
        xr.DataArray: The interpolated grid DataArray.
    """
    # breakpoint()
    if z is not None:
        #3D case
        x_lin = np.linspace(x.min(), x.max(), nx)
        y_lin = np.linspace(y.min(), y.max(), ny)
        z_lin = np.linspace(z.min(), z.max(), nz)
        X, Y, Z = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
        grid = griddata((x, y, z), values, (X, Y, Z), **kwargs)
        darray = xr.DataArray(grid,
                              coords=[('x', x_lin),
                                      ('y', y_lin),
                                      ('z', z_lin)])
    else:
        #2D case
        x_lin = np.linspace(x.min(), x.max(), nx)
        y_lin = np.linspace(y.min(), y.max(), ny)
        X, Y, Z = np.meshgrid(x_lin, y_lin, indexing="ij")
        grid = griddata((x, y), values, (X, Y), **kwargs)
        darray = xr.DataArray(grid,
                              coords=[('x', x_lin),
                                      ('y', y_lin)])
            
    return darray   

# def map_columns(filepath: str,
#              **kwargs):
#     '''
#     Find x,y,z columns
#     '''
#     with open(filepath, 'r') as f:
#         headerline = f.readline()
        
#     columns=re.split('\W+', headerline)
#     breakpoint()

def read_csv(*, filepath: str,
             column_mapping: dict,
             **kwargs) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]: 
    """Reads x, y, z, and values from a CSV file.

    column_mapping={'x':0,'y':1,'z':2,'values':3}

    Args:
        filepath (str): Path to the CSV file.
        column_mapping (dict): Dictionary mapping variable names to column indices.
        **kwargs: Additional arguments for np.genfromtxt.

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]: Arrays of x, y, z, and values.

    Raises:
        IndexError: If columns are missing or delimiter is incorrect.
    """       
    # breakpoint()
    arr=np.genfromtxt(filepath, **kwargs)
    try:
        x=arr[:,column_mapping['x']]
        y=arr[:,column_mapping['y']]
        values = arr[:,column_mapping['values']]
    except IndexError:
        print("Make sure you're using the right delimiter")
        raise IndexError("too many indices for array. Make sure you're using the right delimiter to read array from text file")
    try:
        z=arr[:,column_mapping['z']]
    except:
        z=None
    return x, y, z, values

def describe_csv(filepath: str, **kwargs) -> None:
    """Prints a description of the coordinate arrays read from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        **kwargs: Additional arguments passed to read_csv.
    """
    x, y, z, values = read_csv(filepath=filepath, **kwargs)
    nx, ny, nz = len(np.unique(x)), len(np.unique(y)), 1 if z is None else len(np.unique(z))
    print(f'Length of array = {len(x)}')
    print(f'nx*ny*nz = {nx*ny*nz}')
    print(f'sorted x = {np.sort(np.unique(x))}')
    print(f'sorted y = {np.sort(np.unique(y))}')
    if z is not None:
        print(f'sorted z = {np.sort(np.unique(z))}')
    print(f'spacing x = {np.diff(np.sort(np.unique(x)))}')
    print(f'spacing y = {np.diff(np.sort(np.unique(y)))}')
    if z is not None:
        print(f'spacing z = {np.diff(np.sort(np.unique(z)))}')       
#%% MAIN
if __name__ == '__main__':
    
    #TESTS
    
    points = [0,1,0]    
    
    # ----change base
    #rotates coordinate axes 30° clockwise around +z
    new_x  = (110,0)
    new_y  = (30,0)
    new_z  = (0,-90)
    chge_base1 = change_base_matrix(new_x, new_y, new_z)
    pts_in_new_base1 = change_base(points, new_x, new_y, new_z)
    
    # ----rotate points
    #rotates points 30° clockwise around +z
    new_x  = (110,0)
    new_y  = (30,0)
    new_z  = (0,-90)
    rot_matrix1 = rotation_matrix(new_x, new_y, new_z)
    new_points1 = rotate(points, new_x, new_y, new_z)
    
    # ----change base
    #rotates coordinate axes so the new y axis points (30,15)
    new_x  = (110,0)
    new_y  = (30,15)
    new_z  = (0,15-90)
    chge_base2 = change_base_matrix(new_x, new_y, new_z)
    pts_in_new_base2 = change_base(points, new_x, new_y, new_z)
    
    # ----rotate points
    #rotates points so the new vector points (30,15)
    new_x  = (110,0)
    new_y  = (30,15)
    new_z  = (0,15-90)
    rot_matrix2 = rotation_matrix(new_x, new_y, new_z)
    new_points2 = rotate(points, new_x, new_y, new_z)
    
