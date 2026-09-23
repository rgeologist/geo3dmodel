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
from typing import (Callable, Union, List, Tuple,
                    Optional, Literal, Sequence, Any)
# import re
# import logging
import operator as py_operator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray, ArrayLike

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.path as mplpath
from matplotlib import tri
import matplotlib.cm as cm
import matplotlib.colors as colors 

from scipy.spatial import ConvexHull, cKDTree
from scipy.linalg import LinAlgError
from scipy.interpolate import splprep, splev, griddata, Rbf, SmoothBivariateSpline
# from scipy.signal import windows 
from sklearn.decomposition import PCA

import shapely
from shapely.geometry import LineString, Point, Polygon, box, MultiPoint

import xarray as xr
import skimage


class _LazyModuleCheck:
    def __init__(self, mod_name: str):
        self._mod_name = mod_name
        self._cached: Optional[bool] = None

    def __bool__(self) -> bool:
        if self._cached is None:
            import importlib.util
            self._cached = importlib.util.find_spec(self._mod_name) is not None
        return self._cached

OPEN3D = _LazyModuleCheck("open3d")
PYVISTA = _LazyModuleCheck("pyvista")

from geokitpy import geotensors as gkp
from geokitpy import geogis

PointCollection = Union[List[Point], Tuple[Point, ...],
                        MultiPoint, np.ndarray, pd.DataFrame]

@dataclass
class Trimesh3d:
    """Dataclass container for 3D surface vertices and faces."""
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices)
        self.faces = np.asarray(self.faces)

    @property
    def triangles(self) -> np.ndarray:
        """Alias for faces to maintain compatibility with open3d and trimesh conventions."""
        return self.faces


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
    new_coords = rotate_flat_polygon_strike_dip(x=x, y=y, strike=plane.strike, dip=plane.dip)
    # breakpoint()
    new_coords[:,0] += center[0]
    new_coords[:,1] += center[1]
    new_coords[:,2] += center[2]
    return new_coords

@dataclass
class CylinderAlongPath:
    position_disks:Literal['start', 'middle', 'end', 'end-to-end']
    radius_disks:ArrayLike
    num_sides_disks:int
    strike_disks:ArrayLike|None=None
    dip_disks:ArrayLike|None=None
    path:pd.DataFrame|None=None
    cap_ends: bool = False

def generate_disks_around_path(cylinder:CylinderAlongPath)->list:
    cyl=cylinder
    
    path_coords = cyl.path.loc[:,['x','y','z']]
    path_arr = path_coords.to_numpy()
    if (cylinder.strike_disks is not None) & (cylinder.dip_disks is not None):
        if not (np.asarray(cylinder.strike_disks).shape==np.asarray(cylinder.dip_disks).shape):
            raise ValueError("Input strike and dip must have the same length")
        planes = gkp.Plane(cylinder.strike_disks, cylinder.dip_disks)
    else:
        planes = gkp.Vector.from_path(path_arr).view(gkp.Plane)
    
    match cyl.position_disks:
        case 'start':
            centers=path_arr[:-1,:]
        case 'middle':
            centers=path_arr[:-1,:]+planes/2.
        case 'end':
            centers=path_arr[1:,:]
        case 'end-to-end':
            centers = path_arr
            planes= np.append(planes,
                              planes[-1,:][np.newaxis],
                              axis=0).view(gkp.Plane)
            
                
        case _:
            raise ValueError('position_disks must be either '
                             '"start, "middle" or "end". '
                             f'{cyl.position_disks} was given')   
    radius_array = np.asarray(cyl.radius_disks)
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

    iterator = zip(np.tile(planes,(len(centers),1)),
                   centers,
                   radii,
                   itertools.repeat(cyl.num_sides_disks, len(centers)))
    # breakpoint()
    disks = [build_disk_parallel_to_plane(plane=pl,
                                     center=center,
                                     radius=rad,
                                     num_sides=num_sides) for
             pl, center, rad, num_sides in iterator]
    return disks


def build_surface_from_disks(
        disks: Sequence[ArrayLike],
        *,
        cap_ends: bool = False,
        ) -> Trimesh3d:
    """Build a triangulated surface by lofting ordered polygonal rings.

    Each disk must contain the same number of perimeter points. Point ``j``
    on one ring is connected to point ``j`` on the adjacent ring.

    Args:
        disks: Ordered disk perimeter coordinates. Each item must have shape
            ``(n_points, 3)``.
        cap_ends: If ``True``, add triangle-fan caps to the first and last
            rings.

    Returns:
        Trimesh3d: A triangulated surface mesh containing the lofted lateral surface and,
        optionally, end caps.

    Raises:
        ValueError: If the rings are empty, invalid, mismatched, or contain
            non-finite coordinates.
    """
    if len(disks) < 2:
        raise ValueError("At least two disks are required")

    rings = []
    for disk in disks:
        ring = np.asarray(disk, dtype=float)
        if ring.ndim != 2 or ring.shape[1] != 3:
            raise ValueError("Each disk must have shape (n_points, 3)")
        if not np.isfinite(ring).all():
            raise ValueError("Disk coordinates must be finite")

        # Disk perimeters generated with np.linspace include the first point
        # again at 2*pi; cyclic connectivity should not include that duplicate.
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(ring) < 3:
            raise ValueError("Each disk must contain at least three points")
        rings.append(ring)

    n_points = len(rings[0])
    if any(len(ring) != n_points for ring in rings[1:]):
        raise ValueError("All disks must have the same number of points")

    vertices = np.vstack(rings)
    ring_centers = np.asarray([ring.mean(axis=0) for ring in rings])
    faces = []
    
    

    def add_oriented_face(a: int, b: int, c: int, radial: np.ndarray) -> None:
        triangle = vertices[[a, b, c]]
        normal = np.cross(triangle[1] - triangle[0],
                          triangle[2] - triangle[0])
        if np.linalg.norm(normal) == 0:
            raise ValueError("Disks produce a degenerate triangle")
        if np.dot(normal, radial) < 0:
            faces.append([a, c, b])
        else:
            faces.append([a, b, c])

    for ring_index in range(len(rings) - 1):
        next_ring_index = ring_index + 1
        for point_index in range(n_points):
            next_point_index = (point_index + 1) % n_points
            current = ring_index * n_points + point_index
            current_next = ring_index * n_points + next_point_index
            following = next_ring_index * n_points + point_index
            following_next = next_ring_index * n_points + next_point_index
            radial = (
                (vertices[current] + vertices[current_next]) / 2
                - ring_centers[ring_index]
            )

            add_oriented_face(current, current_next, following, radial)
            add_oriented_face(current_next, following_next, following, radial)

    if cap_ends:
        for ring_index, reverse in ((0, True), (len(rings) - 1, False)):
            center_index = len(vertices)
            vertices = np.vstack((vertices, ring_centers[ring_index]))
            for point_index in range(n_points):
                next_point_index = (point_index + 1) % n_points
                point = ring_index * n_points + point_index
                point_next = ring_index * n_points + next_point_index
                face = [center_index, point_next, point] if reverse else [
                    center_index, point, point_next
                ]
                faces.append(face)

    return Trimesh3d(
        vertices=vertices,
        faces=np.asarray(faces, dtype=int),
    )
    
def build_cylinder_around_path(cylinder:CylinderAlongPath,
                               )->Trimesh3d:
    
    rings = generate_disks_around_path(cylinder)
    
    cylinder_trimesh = build_surface_from_disks(
            rings,
            cap_ends=cylinder.cap_ends,
            )  
  
    
    return cylinder_trimesh    
    
    

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
                           ) -> Trimesh3d:
    """Builds a vertical 3D surface (mesh) by extruding a 2D path.

    Args:
        surface_path (shapely.Geometry): The 2D path (LineString, etc.).
        max_segment_length (float, optional): Maximum length for segmentizing. Defaults to 100.
        xsection_top (float, optional): The top Z elevation. Defaults to 900.
        xsection_bottom (float, optional): The bottom Z elevation. Defaults to -4000.

    Returns:
        Trimesh3d: The resulting vertical surface mesh.
        
    Raises:
        ValueError: If geometry type is unsupported.
    """
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
    surface = Trimesh3d(vertices=vertices, faces=faces)
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


def project_points_to_view(
    points: ArrayLike,
    eye: Union[Dict[str, float], Sequence[float]],
    center: Union[Dict[str, float], Sequence[float]] = (0.0, 0.0, 0.0),
    up: Union[Dict[str, float], Sequence[float]] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Project 3D points onto a 2D camera view plane.

    Args:
        points (ArrayLike): (N, 3) or (3,) points in 3D space.
        eye (Union[Dict[str, float], Sequence[float]]): Camera eye position {x, y, z} or tuple.
        center (Union[Dict[str, float], Sequence[float]], optional): Camera target {x, y, z} or tuple. Defaults to (0,0,0).
        up (Union[Dict[str, float], Sequence[float]], optional): Camera up direction {x, y, z} or tuple. Defaults to (0,0,1).

    Returns:
        np.ndarray: (N, 2) array of coordinates projected onto the camera viewing plane (u, v).
    """
    pts = np.atleast_2d(points)

    def _to_vec(val):
        if isinstance(val, dict):
            return np.array([val.get("x", 0.0), val.get("y", 0.0), val.get("z", 0.0)], dtype=float)
        return np.array(val, dtype=float)

    e = _to_vec(eye)
    c = _to_vec(center)
    u_vec = _to_vec(up)

    f = c - e
    norm_f = np.linalg.norm(f)
    if norm_f < 1e-12:
        f = np.array([0.0, 0.0, -1.0])
    else:
        f = f / norm_f

    s = np.cross(f, u_vec)
    norm_s = np.linalg.norm(s)
    if norm_s < 1e-12:
        fallback_up = np.array([0.0, 1.0, 0.0]) if abs(f[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        s = np.cross(f, fallback_up)
        s = s / np.linalg.norm(s)
    else:
        s = s / norm_s

    v = np.cross(s, f)

    shifted = pts - c
    proj_u = shifted @ s
    proj_v = shifted @ v

    result = np.column_stack((proj_u, proj_v))
    if np.asarray(points).ndim == 1:
        return result[0]
    return result


def generate_stereonet_grid_lonlat(
    step_deg: int = 10,
    npoints: int = 100
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """Generate longitude and latitude coordinates (in radians) for a stereonet grid.

    Args:
        step_deg (int, optional): Angular spacing between grid lines in degrees. Defaults to 10.
        npoints (int, optional): Number of points per curve. Defaults to 100.

    Returns:
        List[Tuple[np.ndarray, np.ndarray, str]]: List of (lons, lats, kind) tuples in radians.
    """
    grid_lines = []

    # 1. Meridians (great circles) of constant longitude
    lons_deg = np.arange(-90 + step_deg, 90, step_deg)
    lat_range = np.radians(np.linspace(-90, 90, npoints))
    for lon_deg in lons_deg:
        lon_rad = np.full(npoints, np.radians(lon_deg))
        grid_lines.append((lon_rad, lat_range, "meridian"))

    # 2. Parallels (small circles) of constant latitude
    lats_deg = np.arange(-90 + step_deg, 90, step_deg)
    lon_range = np.radians(np.linspace(-90, 90, npoints))
    for lat_deg in lats_deg:
        lat_rad = np.full(npoints, np.radians(lat_deg))
        grid_lines.append((lon_range, lat_rad, "parallel"))

    # 3. Primitive bounding circle (outer boundary of visible hemisphere: lon = ±90°)
    lat_east = np.linspace(-np.pi / 2, np.pi / 2, npoints)
    lon_east = np.full(npoints, np.pi / 2)
    lat_west = np.linspace(np.pi / 2, -np.pi / 2, npoints)
    lon_west = np.full(npoints, -np.pi / 2)
    prim_lon = np.concatenate([lon_east, lon_west])
    prim_lat = np.concatenate([lat_east, lat_west])
    grid_lines.append((prim_lon, prim_lat, "primitive"))

    # 4. Central crosshairs
    grid_lines.append((np.zeros(npoints), np.radians(np.linspace(-90, 90, npoints)), "crosshair"))
    grid_lines.append((np.radians(np.linspace(-90, 90, npoints)), np.zeros(npoints), "crosshair"))

    return grid_lines


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


def compute_surface_boundary_polygon(
    points_2d: np.ndarray,
    boundary: Literal["convex", "concave"] = "convex",
    concave_ratio: float = 0.3,
    spline: bool = False,
    n_spline_pts: int = 100,
) -> np.ndarray:
    """Computes the 2D boundary polygon (convex or concave) enclosing a set of 2D points.

    Args:
        points_2d (np.ndarray): (N, 2) array of coordinates in the projection plane.
        boundary (Literal["convex", "concave"], optional): Type of boundary hull. Defaults to "convex".
        concave_ratio (float, optional): Concavity ratio (0.0 to 1.0) when boundary='concave'. Defaults to 0.3.
        spline (bool, optional): Whether to smooth the boundary polygon with a B-spline. Defaults to False.
        n_spline_pts (int, optional): Number of spline points if smoothed. Defaults to 100.

    Returns:
        np.ndarray: (M, 2) array of boundary polygon vertices.
    """
    if boundary == "concave":
        poly_coords = geogis.concave_hull_2d(points_2d, ratio=concave_ratio, as_array=True)
    elif boundary == "convex":
        poly_coords = geogis.convex_hull_2d(points_2d, as_array=True)
    else:
        raise ValueError(f"Unknown boundary type: '{boundary}'. Expected 'convex' or 'concave'.")

    if len(poly_coords) < 3:
        raise ValueError(f"Boundary polygon must have at least 3 vertices, got {len(poly_coords)}.")

    if spline and len(poly_coords) >= 4:
        try:
            coords_for_spline = poly_coords[:-1] if np.allclose(poly_coords[0], poly_coords[-1]) else poly_coords
            tck, u_spl = splprep(coords_for_spline.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u_spl.min(), u_spl.max(), n_spline_pts)
            x_new, y_new = splev(u_new, tck, der=0)
            poly_coords = np.column_stack((x_new, y_new))
        except (TypeError, ValueError, LinAlgError):
            pass

    return poly_coords


def generate_masked_surface_grid(
    boundary_polygon: np.ndarray,
    spacing: Optional[float] = None,
    n_elements: Optional[int] = None,
    default_n_elements: int = 50,
) -> np.ndarray:
    """Generates a regular 2D grid of points bounded within a polygon.

    Args:
        boundary_polygon (np.ndarray): (K, 2) boundary polygon vertices.
        spacing (Optional[float], optional): Target grid node spacing.
        n_elements (Optional[int], optional): Number of grid subdivisions per axis.
        default_n_elements (int, optional): Fallback grid resolution if neither spacing
            nor n_elements is specified. Defaults to 50.

    Returns:
        np.ndarray: (M, 2) array of 2D grid points strictly inside the polygon.
    """
    xmin, ymin = boundary_polygon[:, 0].min(), boundary_polygon[:, 1].min()
    xmax, ymax = boundary_polygon[:, 0].max(), boundary_polygon[:, 1].max()

    dx = xmax - xmin
    dy = ymax - ymin

    if spacing is not None and spacing > 0:
        n_elem_x = max(2, int(np.ceil(dx / spacing)) + 1)
        n_elem_y = max(2, int(np.ceil(dy / spacing)) + 1)
    elif n_elements is not None and n_elements > 0:
        n_elem_x = max(2, int(n_elements))
        n_elem_y = max(2, int(n_elements))
    else:
        n_elem_x = default_n_elements
        n_elem_y = default_n_elements

    xgrid, ygrid = np.mgrid[xmin:xmax:n_elem_x * 1j, ymin:ymax:n_elem_y * 1j]
    grid = np.column_stack((xgrid.ravel(), ygrid.ravel()))

    path = mplpath.Path(boundary_polygon)
    inside = path.contains_points(grid)
    grid_inside = grid[inside]

    if len(grid_inside) < 3:
        raise ValueError(
            f"Fewer than 3 grid points fell inside boundary polygon. "
            f"Consider decreasing spacing or increasing n_elements."
        )

    return grid_inside


def fit_surface_z_weighted_avg(
    sample_xy: np.ndarray,
    sample_z: np.ndarray,
    query_xy: np.ndarray,
    k_neighbors: int = 10,
    weighting: Literal["idw", "gaussian", "exponential", "uniform"] = "idw",
    power: float = 2.0,
    sigma: Optional[float] = None,
    scale: Optional[float] = None,
    max_distance: Optional[float] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits surface Z elevations on 2D query locations using distance-weighted averaging.

    Queries nearest neighbors in compiled C++ via scipy.spatial.cKDTree and evaluates
    weights strictly using 2D in-plane Euclidean distances.

    Args:
        sample_xy (np.ndarray): (N, 2) in-plane coordinates of sample points.
        sample_z (np.ndarray): (N,) elevations of sample points.
        query_xy (np.ndarray): (M, 2) in-plane coordinates of query grid nodes.
        k_neighbors (int, optional): Number of nearest neighbors to query. Defaults to 10.
        weighting (Literal["idw", "gaussian", "exponential", "uniform"], optional):
            Weighting kernel. Defaults to "idw".
        power (float, optional): Distance exponent for IDW weighting (w = 1 / d^power). Defaults to 2.0.
        sigma (Optional[float], optional): Standard deviation for Gaussian kernel. Defaults to None (median distance).
        scale (Optional[float], optional): Decay length for exponential kernel. Defaults to None (median distance).
        max_distance (Optional[float], optional): Maximum search radius. Query nodes without any sample
            points within this radius are excluded. Defaults to None.
        eps (float, optional): Epsilon to prevent division by zero at exact sample locations. Defaults to 1e-12.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (valid_query_xy, fitted_z) where valid_query_xy has shape (M_valid, 2)
        and fitted_z has shape (M_valid,).
    """
    k = min(len(sample_xy), max(1, k_neighbors))
    tree = cKDTree(sample_xy)
    distances, indices = tree.query(query_xy, k=k)

    if k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    if max_distance is not None:
        valid_mask = distances[:, 0] <= max_distance
        if not np.any(valid_mask):
            raise ValueError(f"No query points have sample points within max_distance={max_distance}.")
        query_xy = query_xy[valid_mask]
        distances = distances[valid_mask]
        indices = indices[valid_mask]
        if len(query_xy) < 3:
            raise ValueError(f"Fewer than 3 query points remain within max_distance={max_distance}.")

    z_neighbors = sample_z[indices]

    if weighting == "uniform":
        z_interp = np.mean(z_neighbors, axis=1)
    elif weighting == "gaussian":
        if sigma is None or sigma <= 0:
            sigma = float(np.median(distances[:, -1]))
            if sigma <= 0:
                sigma = 1.0
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        z_interp = np.sum(z_neighbors * (weights / weights_sum), axis=1)
    elif weighting == "exponential":
        if scale is None or scale <= 0:
            scale = float(np.median(distances[:, -1]))
            if scale <= 0:
                scale = 1.0
        weights = np.exp(-distances / scale)
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        z_interp = np.sum(z_neighbors * (weights / weights_sum), axis=1)
    elif weighting == "idw":
        exact_matches = distances < eps
        weights = 1.0 / np.maximum(distances, eps) ** power
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        z_interp = np.sum(z_neighbors * (weights / weights_sum), axis=1)
        has_exact = np.any(exact_matches, axis=1)
        if np.any(has_exact):
            exact_col = np.argmax(exact_matches[has_exact], axis=1)
            z_interp[has_exact] = z_neighbors[has_exact, exact_col]
    else:
        raise ValueError(f"Unknown weighting kernel: '{weighting}'. Expected 'idw', 'gaussian', 'exponential', or 'uniform'.")

    return query_xy, z_interp


def fit_surface_z_quadric(
    sample_xy: np.ndarray,
    sample_z: np.ndarray,
    query_xy: np.ndarray,
    degree: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits a polynomial (quadric or cubic) surface in 2D coordinates and evaluates it on query points.

    For degree=2 (quadric):
        z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
    For degree=3 (cubic):
        includes degree 3 cross-terms: x^3, x^2*y, x*y^2, y^3

    Args:
        sample_xy (np.ndarray): (N, 2) in-plane sample coordinates.
        sample_z (np.ndarray): (N,) sample elevations.
        query_xy (np.ndarray): (M, 2) query grid coordinates.
        degree (int, optional): Polynomial degree (2 or 3). Defaults to 2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (query_xy, fitted_z).
    """
    x, y = sample_xy[:, 0], sample_xy[:, 1]
    qx, qy = query_xy[:, 0], query_xy[:, 1]

    if degree == 2:
        cols_sample = [np.ones_like(x), x, y, x**2, x * y, y**2]
        cols_query = [np.ones_like(qx), qx, qy, qx**2, qx * qy, qy**2]
    elif degree == 3:
        cols_sample = [
            np.ones_like(x), x, y, x**2, x * y, y**2,
            x**3, (x**2) * y, x * (y**2), y**3
        ]
        cols_query = [
            np.ones_like(qx), qx, qy, qx**2, qx * qy, qy**2,
            qx**3, (qx**2) * qy, qx * (qy**2), qy**3
        ]
    else:
        raise ValueError(f"Degree must be 2 or 3, got {degree}.")

    A_sample = np.column_stack(cols_sample)
    A_query = np.column_stack(cols_query)

    coeffs, _, _, _ = np.linalg.lstsq(A_sample, sample_z, rcond=None)
    z_interp = A_query @ coeffs
    return query_xy, z_interp


def fit_surface_z_rbf(
    sample_xy: np.ndarray,
    sample_z: np.ndarray,
    query_xy: np.ndarray,
    function: str = "thin_plate",
    smooth: float = 0.0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits surface Z using Radial Basis Functions with optional smoothing.

    Args:
        sample_xy (np.ndarray): (N, 2) sample points.
        sample_z (np.ndarray): (N,) sample elevations.
        query_xy (np.ndarray): (M, 2) query grid coordinates.
        function (str, optional): Radial basis function type ('thin_plate', 'multiquadric',
            'linear', 'cubic', 'gaussian'). Defaults to 'thin_plate'.
        smooth (float, optional): Smoothing factor (>= 0). Values > 0 regularize the fit,
            preventing overfitting on noisy data. Defaults to 0.0.
        **kwargs: Extra parameters passed to scipy.interpolate.Rbf.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (query_xy, fitted_z).
    """
    rbf_model = Rbf(
        sample_xy[:, 0], sample_xy[:, 1], sample_z,
        function=function,
        smooth=smooth,
        **kwargs,
    )
    z_interp = rbf_model(query_xy[:, 0], query_xy[:, 1])
    return query_xy, z_interp


def fit_surface_z_bspline(
    sample_xy: np.ndarray,
    sample_z: np.ndarray,
    query_xy: np.ndarray,
    s: Optional[float] = None,
    kx: int = 3,
    ky: int = 3,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits surface Z using SmoothBivariateSpline with noise smoothing.

    Args:
        sample_xy (np.ndarray): (N, 2) sample points.
        sample_z (np.ndarray): (N,) sample elevations.
        query_xy (np.ndarray): (M, 2) query grid coordinates.
        s (Optional[float], optional): Smoothing factor determining the trade-off
            between smoothness and closeness of fit. If None, uses default. Defaults to None.
        kx (int, optional): Degree of the spline in x. Defaults to 3 (bicubic).
        ky (int, optional): Degree of the spline in y. Defaults to 3 (bicubic).
        **kwargs: Extra parameters passed to SmoothBivariateSpline.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (query_xy, fitted_z).
    """
    spline_model = SmoothBivariateSpline(
        sample_xy[:, 0], sample_xy[:, 1], sample_z,
        s=s,
        kx=kx,
        ky=ky,
        **kwargs,
    )
    z_interp = spline_model.ev(query_xy[:, 0], query_xy[:, 1])
    return query_xy, z_interp


# Backward-compatibility aliases for module-level functions
interpolate_surface_z_weighted_avg = fit_surface_z_weighted_avg
interpolate_surface_z_quadric = fit_surface_z_quadric
interpolate_surface_z_rbf = fit_surface_z_rbf
interpolate_surface_z_bspline = fit_surface_z_bspline


def build_transformed_trimesh3d(
    grid_xy: np.ndarray,
    grid_z: np.ndarray,
    plane: gkp.Plane,
    centroid: np.ndarray,
) -> Trimesh3d:
    """Builds a 2D Delaunay triangulation and back-transforms vertices to world 3D coordinates.

    Args:
        grid_xy (np.ndarray): (M, 2) in-plane grid node coordinates.
        grid_z (np.ndarray): (M,) interpolated elevations in plane reference frame.
        plane (gkp.Plane): Fitting plane carrying orientation axes.
        centroid (np.ndarray): 3D centroid vector used during forward centering.

    Returns:
        Trimesh3d: Reconstructed triangulated 3D surface mesh in original world coordinates.
    """
    triangulation = tri.Triangulation(grid_xy[:, 0], grid_xy[:, 1])
    local_xyz = np.column_stack((grid_xy, grid_z))
    world_vertices = (local_xyz @ plane.axes.as_matrix) + np.asarray(centroid)
    faces = triangulation.triangles.astype(np.int64)
    return Trimesh3d(vertices=world_vertices, faces=faces)


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
    
    
    def pca(self, n_components: int = 3, **kwargs) -> PCA:
        """Calculates the Principal Component Analysis (PCA) of the point cloud coordinates.

        Args:
            n_components (int, optional): Number of components to keep. Defaults to 3.
            **kwargs: Additional keyword arguments passed to sklearn.decomposition.PCA.

        Returns:
            PCA: The fitted PCA object from scikit-learn.

        Raises:
            ValueError: If there are fewer points than n_components.
        """
        points = self.coord_values()
        if len(points) < n_components:
            raise ValueError(
                f"Not enough points for PCA with {n_components} components; got {len(points)} points."
            )
        pca_model = PCA(n_components=n_components, **kwargs)
        pca_model.fit(points)
        return pca_model

    def least_squares(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        """Calculates the ordinary least squares fit of a plane (z = ax + by + c) to the point cloud.

        Args:
            **kwargs: Additional keyword arguments passed to np.linalg.lstsq.

        Returns:
            Tuple[np.ndarray, np.ndarray, int, np.ndarray]: The fit coefficients, residual array,
                matrix rank, and singular values.
        """
        points = self.coord_values()
        A = np.array(np.hstack((points[:, 0:2], np.ones((points.shape[0], 1)))))
        b = np.array(points[:, 2].reshape((points.shape[0], 1)))
        fit, residual, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        return fit, residual, rank, singular_values

    def svd(self, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates the Singular Value Decomposition (SVD) of centered point coordinates.

        Accesses coordinates directly from self.data.

        Args:
            **kwargs: Additional keyword arguments passed to np.linalg.svd.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Left singular vectors (U),
                singular values (s), and right singular vectors (Vt).
        """
        points = self.coord_values()
        centered = points - self.centroid().values
        u, s, vt = np.linalg.svd(centered, full_matrices=False, **kwargs)
        return u, s, vt

    def ransac(
        self,
        max_iterations: int = 1000,
        distance_threshold: float = 0.05,
        random_state: Optional[int] = None,
        residual_type: Literal["orthogonal_sum", "orthogonal_rmse"] = "orthogonal_sum",
        return_inliers: bool = False,
        **kwargs,
    ) -> Union[Tuple[gkp.Plane, float], Tuple[gkp.Plane, float, pd.Series]]:
        """Fits a plane to the point cloud using 3D geometric RANSAC consensus.

        Accesses coordinates directly from self.data. Iteratively samples 3 random
        non-collinear points, counts inliers within distance_threshold, and refines
        the consensus plane using SVD on the inliers.

        Args:
            max_iterations (int, optional): Maximum number of RANSAC iterations. Defaults to 1000.
            distance_threshold (float, optional): Orthogonal distance threshold for inlier
                classification. Defaults to 0.05.
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            residual_type (Literal["orthogonal_sum", "orthogonal_rmse"], optional):
                Metric for the returned residual ('orthogonal_sum' or 'orthogonal_rmse').
                Defaults to "orthogonal_sum".
            return_inliers (bool, optional): If True, returns (plane, residual, inlier_series).
                If False, returns (plane, residual). Defaults to False.
            **kwargs: Additional parameters.

        Returns:
            Union[Tuple[gkp.Plane, float], Tuple[gkp.Plane, float, pd.Series]]:
                - If return_inliers is False: (plane, residual)
                - If return_inliers is True: (plane, residual, inlier_series) where inlier_series
                  is a boolean pd.Series indexed matching self.data.

        Raises:
            ValueError: If there are fewer than 3 points.
        """
        points = self.coord_values()
        n_points = len(points)
        if n_points < 3:
            raise ValueError("Need at least 3 points for RANSAC plane fitting.")

        rng = np.random.default_rng(random_state)
        best_inlier_mask = None
        best_inlier_count = -1

        for _ in range(max_iterations):
            idx = rng.choice(n_points, size=3, replace=False)
            p0, p1, p2 = points[idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            dists = np.abs((points - p0) @ normal)
            inliers = dists <= distance_threshold
            inlier_count = int(np.sum(inliers))
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_mask = inliers
                if inlier_count == n_points:
                    break

        if best_inlier_mask is None or best_inlier_count < 3:
            plane, residual = self.fit_plane_to_points(method="pca", residual_type=residual_type)
            inlier_mask = np.ones(n_points, dtype=bool)
        else:
            inlier_points = points[best_inlier_mask]
            centroid = np.mean(inlier_points, axis=0)
            _, _, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
            normal_vector = vt[-1, :].copy()
            if normal_vector[2] < 0:
                normal_vector = -normal_vector
            elif np.isclose(normal_vector[2], 0.0) and normal_vector[1] < 0:
                normal_vector = -normal_vector

            normal_unit = normal_vector / np.linalg.norm(normal_vector)
            plane = gkp.Vector(normal_unit).unit.view(gkp.Plane)

            inlier_dists = np.abs((inlier_points - centroid) @ normal_unit)
            if residual_type == "orthogonal_rmse":
                residual = float(np.sqrt(np.mean(inlier_dists ** 2)))
            else:
                residual = float(np.sum(inlier_dists))
            inlier_mask = best_inlier_mask

        if return_inliers:
            inlier_series = pd.Series(inlier_mask, index=self.data.index, name="inlier")
            return plane, residual, inlier_series

        return plane, residual

    fit_plane_ransac = ransac
    _fit_plane_ransac = ransac

    def fit_plane_to_points(
        self,
        method: Literal["pca", "least_squares", "svd", "ransac"] = "pca",
        residual_type: Literal["orthogonal_sum", "orthogonal_rmse"] = "orthogonal_sum",
        **kwargs,
    ) -> Tuple[gkp.Plane, float]:
        """Fits a plane to the point cloud using PCA, least squares, SVD, or RANSAC.

        Args:
            method (Literal["pca", "least_squares", "svd", "ransac"], optional):
                Fitting method:
                - 'pca': Total least squares via Principal Component Analysis.
                - 'least_squares': Ordinary least squares in Z (z = ax + by + c).
                - 'svd': Total least squares via SVD on centered points.
                - 'ransac': 3D geometric RANSAC consensus fitting (robust to outliers).
                Defaults to 'pca'.
            residual_type (Literal["orthogonal_sum", "orthogonal_rmse"], optional):
                Residual metric to return:
                - 'orthogonal_sum': Sum of perpendicular Euclidean distances.
                - 'orthogonal_rmse': Root mean square perpendicular distance.
                Defaults to 'orthogonal_sum'.
            **kwargs: Additional parameters passed to the chosen method:
                - For 'ransac': max_iterations (int, default 1000), distance_threshold (float, default 0.05),
                  random_state (int, optional).
                - For 'pca': passed to sklearn.decomposition.PCA.
                - For 'least_squares': passed to np.linalg.lstsq.

        Returns:
            Tuple[gkp.Plane, float]: The fitted plane (with upper-hemisphere normal nz >= 0)
                and the scalar orthogonal residual.

        Raises:
            ValueError: If there are fewer than 3 points or an unknown method is specified.
        """
        points = self.coord_values()
        if points.shape[0] < 3:
            raise ValueError("Not enough points to get plane (minimum 3 required)")

        if method == "least_squares":
            fit, _, _, _ = self.least_squares(**kwargs)
            normal_vector = np.array([-fit[0, 0], -fit[1, 0], 1.0], dtype=float)
            plane_point = self.centroid().values

        elif method == "pca":
            pca_model = self.pca(n_components=3, **kwargs)
            normal_vector = pca_model.components_[-1].copy()
            plane_point = pca_model.mean_

        elif method == "svd":
            plane_point = self.centroid().values
            _, _, vt = self.svd(**kwargs)
            normal_vector = vt[-1, :].copy()

        elif method == "ransac":
            max_iterations = kwargs.pop("max_iterations", 1000)
            distance_threshold = kwargs.pop("distance_threshold", 0.05)
            random_state = kwargs.pop("random_state", None)
            return self.ransac(
                max_iterations=max_iterations,
                distance_threshold=distance_threshold,
                random_state=random_state,
                residual_type=residual_type,
                return_inliers=False,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown plane fitting method: '{method}'. "
                f"Supported methods: 'pca', 'least_squares', 'svd', 'ransac'."
            )

        # Standardize normal vector to upper hemisphere (nz >= 0)
        if normal_vector[2] < 0:
            normal_vector = -normal_vector
        elif np.isclose(normal_vector[2], 0.0) and normal_vector[1] < 0:
            normal_vector = -normal_vector

        norm = np.linalg.norm(normal_vector)
        if norm == 0:
            raise ValueError("Degenerate points: plane normal vector has zero norm.")
        normal_unit = normal_vector / norm
        plane = gkp.Vector(normal_unit).unit.view(gkp.Plane)

        # Compute vectorized orthogonal perpendicular distances
        distances = np.abs((points - plane_point) @ normal_unit)
        if residual_type == "orthogonal_rmse":
            residual = float(np.sqrt(np.mean(distances ** 2)))
        else:
            residual = float(np.sum(distances))

        return plane, residual

    def distance_to_plane(
        self,
        plane: gkp.Plane,
        signed: bool = True,
        plane_point: Optional[ArrayLike] = None,
    ) -> pd.Series:
        """Calculates orthogonal perpendicular distances from all points to a given plane.

        This method does not modify self.data.

        Args:
            plane (gkp.Plane): The reference plane.
            signed (bool, optional): If True, returns signed orthogonal distances
                (+ along normal, - opposite). If False, returns absolute distances. Defaults to True.
            plane_point (Optional[ArrayLike], optional): A point known to lie on the plane.
                If None, uses the centroid of this point cloud. Defaults to None.

        Returns:
            pd.Series: Series of orthogonal distances indexed by DataFrame index.
        """
        points = self.coord_values()
        if plane_point is None:
            p0 = self.centroid().values
        else:
            p0 = np.asarray(plane_point, dtype=float)

        normal = np.asarray(plane, dtype=float)
        norm = np.linalg.norm(normal)
        if norm == 0:
            raise ValueError("Plane normal has zero length.")
        normal_unit = normal / norm

        dists = (points - p0) @ normal_unit
        if not signed:
            dists = np.abs(dists)

        return pd.Series(dists, index=self.data.index, name="distance_to_plane")

    def evaluate_fit_plane(
        self,
        plane: gkp.Plane,
        plane_point: Optional[ArrayLike] = None,
    ) -> Dict[str, float]:
        """Evaluates statistical fit quality metrics of a plane against the point cloud.

        This method does not modify self.data.

        Args:
            plane (gkp.Plane): The fitted plane.
            plane_point (Optional[ArrayLike], optional): Point on the plane. Defaults to centroid.

        Returns:
            Dict[str, float]: Statistical metrics:
                - 'rmse': Root Mean Square Error (orthogonal)
                - 'mae': Mean Absolute Error (orthogonal)
                - 'mad': Median Absolute Deviation
                - 'max': Maximum absolute orthogonal error
                - 'sum': Sum of absolute orthogonal errors
        """
        dists = self.distance_to_plane(plane, signed=False, plane_point=plane_point).values
        mad = float(np.median(np.abs(dists - np.median(dists))))
        return {
            "rmse": float(np.sqrt(np.mean(dists ** 2))),
            "mae": float(np.mean(dists)),
            "mad": mad,
            "max": float(np.max(dists)),
            "sum": float(np.sum(dists)),
        }

    def distance_to_surface(
        self,
        surface: Trimesh3d,
    ) -> pd.Series:
        """Calculates Euclidean distances from each point in the cloud to the nearest vertex of the surface mesh.

        This method does not modify self.data.

        Args:
            surface (Trimesh3d): The surface mesh to measure distances against.

        Returns:
            pd.Series: Series of nearest-vertex Euclidean distances indexed by DataFrame index.
        """
        tree = cKDTree(surface.vertices)
        dists, _ = tree.query(self.coord_values())
        return pd.Series(dists, index=self.data.index, name="distance_to_surface")

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
    
    
    def _fit_surface_pipeline(
        self,
        interp_func: Callable[..., Tuple[np.ndarray, np.ndarray]],
        plane: Optional[gkp.Plane] = None,
        spacing: Optional[float] = None,
        n_elements: Optional[int] = None,
        boundary: Literal["convex", "concave"] = "convex",
        concave_ratio: float = 0.3,
        spline: bool = False,
        **interp_kwargs,
    ) -> Trimesh3d:
        """Shared pipeline for fitting a 3D surface mesh to the point cloud.

        Projects points to local plane coordinates, builds boundary polygon,
        generates masked grid, calls interp_func to obtain grid Z, and reconstructs Trimesh3d.
        """
        if plane is None:
            pl, _ = self.fit_plane_to_points(method="pca")
        else:
            pl = plane

        temp_xyz = self.project_points_on_plane(pl)
        sample_xy = temp_xyz[:, :2]
        sample_z = temp_xyz[:, 2]

        boundary_poly = compute_surface_boundary_polygon(
            sample_xy,
            boundary=boundary,
            concave_ratio=concave_ratio,
            spline=spline,
        )

        grid_xy = generate_masked_surface_grid(
            boundary_poly,
            spacing=spacing,
            n_elements=n_elements,
        )

        valid_grid_xy, grid_z = interp_func(
            sample_xy=sample_xy,
            sample_z=sample_z,
            query_xy=grid_xy,
            **interp_kwargs,
        )

        mesh = build_transformed_trimesh3d(
            grid_xy=valid_grid_xy,
            grid_z=grid_z,
            plane=pl,
            centroid=self.centroid().values,
        )

        return mesh

    def fit_surface_z_quadric(
        self,
        plane: Optional[gkp.Plane] = None,
        spacing: Optional[float] = None,
        n_elements: Optional[int] = None,
        boundary: Literal["convex", "concave"] = "convex",
        concave_ratio: float = 0.3,
        spline: bool = False,
        degree: int = 2,
        **kwargs,
    ) -> Trimesh3d:
        """Fits a 3D surface mesh to the point cloud using polynomial (quadric or cubic) fitting.

        Args:
            plane (Optional[gkp.Plane], optional): Orientation plane. If None, fitted via PCA.
            spacing (Optional[float], optional): Target grid node spacing.
            n_elements (Optional[int], optional): Number of grid intervals per axis. Defaults to 50 if spacing is None.
            boundary (Literal["convex", "concave"], optional): Boundary hull type. Defaults to "convex".
            concave_ratio (float, optional): Concavity ratio when boundary='concave'. Defaults to 0.3.
            spline (bool, optional): Whether to smooth the boundary polygon. Defaults to False.
            degree (int, optional): Polynomial degree (2 for quadric, 3 for cubic). Defaults to 2.
            **kwargs: Extra parameters.

        Returns:
            Trimesh3d: Triangulated surface mesh.
        """
        return self._fit_surface_pipeline(
            interp_func=fit_surface_z_quadric,
            plane=plane,
            spacing=spacing,
            n_elements=n_elements,
            boundary=boundary,
            concave_ratio=concave_ratio,
            spline=spline,
            degree=degree,
            **kwargs,
        )

    def fit_surface_z_rbf(
        self,
        plane: Optional[gkp.Plane] = None,
        spacing: Optional[float] = None,
        n_elements: Optional[int] = None,
        boundary: Literal["convex", "concave"] = "convex",
        concave_ratio: float = 0.3,
        spline: bool = False,
        function: str = "thin_plate",
        smooth: float = 0.0,
        **kwargs,
    ) -> Trimesh3d:
        """Fits a 3D surface mesh to the point cloud using Radial Basis Functions with optional smoothing.

        Args:
            plane (Optional[gkp.Plane], optional): Orientation plane. If None, fitted via PCA.
            spacing (Optional[float], optional): Target grid node spacing.
            n_elements (Optional[int], optional): Number of grid intervals per axis. Defaults to 50 if spacing is None.
            boundary (Literal["convex", "concave"], optional): Boundary hull type. Defaults to "convex".
            concave_ratio (float, optional): Concavity ratio when boundary='concave'. Defaults to 0.3.
            spline (bool, optional): Whether to smooth the boundary polygon. Defaults to False.
            function (str, optional): RBF kernel ('thin_plate', 'multiquadric', 'linear', 'cubic', 'gaussian').
            smooth (float, optional): Smoothing parameter (>= 0). Defaults to 0.0.
            **kwargs: Extra parameters passed to RBF.

        Returns:
            Trimesh3d: Triangulated surface mesh.
        """
        return self._fit_surface_pipeline(
            interp_func=fit_surface_z_rbf,
            plane=plane,
            spacing=spacing,
            n_elements=n_elements,
            boundary=boundary,
            concave_ratio=concave_ratio,
            spline=spline,
            function=function,
            smooth=smooth,
            **kwargs,
        )

    def fit_surface_z_bspline(
        self,
        plane: Optional[gkp.Plane] = None,
        spacing: Optional[float] = None,
        n_elements: Optional[int] = None,
        boundary: Literal["convex", "concave"] = "convex",
        concave_ratio: float = 0.3,
        spline: bool = False,
        s: Optional[float] = None,
        kx: int = 3,
        ky: int = 3,
        **kwargs,
    ) -> Trimesh3d:
        """Fits a 3D surface mesh to the point cloud using SmoothBivariateSpline with noise smoothing.

        Args:
            plane (Optional[gkp.Plane], optional): Orientation plane. If None, fitted via PCA.
            spacing (Optional[float], optional): Target grid node spacing.
            n_elements (Optional[int], optional): Number of grid intervals per axis. Defaults to 50 if spacing is None.
            boundary (Literal["convex", "concave"], optional): Boundary hull type. Defaults to "convex".
            concave_ratio (float, optional): Concavity ratio when boundary='concave'. Defaults to 0.3.
            spline (bool, optional): Whether to smooth the boundary polygon. Defaults to False.
            s (Optional[float], optional): Smoothing factor.
            kx (int, optional): Spline degree in x. Defaults to 3.
            ky (int, optional): Spline degree in y. Defaults to 3.
            **kwargs: Extra parameters.

        Returns:
            Trimesh3d: Triangulated surface mesh.
        """
        return self._fit_surface_pipeline(
            interp_func=fit_surface_z_bspline,
            plane=plane,
            spacing=spacing,
            n_elements=n_elements,
            boundary=boundary,
            concave_ratio=concave_ratio,
            spline=spline,
            s=s,
            kx=kx,
            ky=ky,
            **kwargs,
        )

    def fit_surface_z_weighted_avg(
        self,
        plane: Optional[gkp.Plane] = None,
        spacing: Optional[float] = None,
        n_elements: Optional[int] = None,
        boundary: Literal["convex", "concave"] = "convex",
        concave_ratio: float = 0.3,
        spline: bool = False,
        k_neighbors: int = 10,
        weighting: Literal["idw", "gaussian", "exponential", "uniform"] = "idw",
        power: float = 2.0,
        sigma: Optional[float] = None,
        scale: Optional[float] = None,
        max_distance: Optional[float] = None,
        **kwargs,
    ) -> Trimesh3d:
        """Fits a 3D surface to the point cloud using distance-weighted averaging.

        Args:
            plane (Optional[gkp.Plane], optional): Orientation plane. If None, fitted via PCA.
            spacing (Optional[float], optional): Target grid node spacing.
            n_elements (Optional[int], optional): Number of grid intervals per axis. Defaults to 50 if spacing is None.
            boundary (Literal["convex", "concave"], optional): Boundary hull type. Defaults to "convex".
            concave_ratio (float, optional): Concavity ratio when boundary='concave'. Defaults to 0.3.
            spline (bool, optional): Whether to smooth the boundary polygon. Defaults to False.
            k_neighbors (int, optional): Number of nearest neighbors. Defaults to 10.
            weighting (Literal["idw", "gaussian", "exponential", "uniform"], optional):
                Weighting kernel. Defaults to "idw".
            power (float, optional): Exponent for IDW weighting. Defaults to 2.0.
            sigma (Optional[float], optional): Gaussian kernel width.
            scale (Optional[float], optional): Exponential kernel decay scale.
            max_distance (Optional[float], optional): Maximum search radius cutoff.
            **kwargs: Extra parameters (e.g. closest_nelems alias for k_neighbors).

        Returns:
            Trimesh3d: Triangulated surface mesh containing vertices and faces.
        """
        if "closest_nelems" in kwargs and kwargs["closest_nelems"] is not None:
            k_neighbors = kwargs.pop("closest_nelems")

        return self._fit_surface_pipeline(
            interp_func=fit_surface_z_weighted_avg,
            plane=plane,
            spacing=spacing,
            n_elements=n_elements,
            boundary=boundary,
            concave_ratio=concave_ratio,
            spline=spline,
            k_neighbors=k_neighbors,
            weighting=weighting,
            power=power,
            sigma=sigma,
            scale=scale,
            max_distance=max_distance,
            **kwargs,
        )

    # Aliases inside Pcloud with 'interpolate' and 'fit_with_' prefixes
    interpolate_surface_z_weighted_avg = fit_surface_z_weighted_avg
    interpolate_surface_z_quadric = fit_surface_z_quadric
    interpolate_surface_z_rbf = fit_surface_z_rbf
    interpolate_surface_z_bspline = fit_surface_z_bspline

    fit_with_weighted_avg = fit_surface_z_weighted_avg
    fit_with_quadric = fit_surface_z_quadric
    fit_with_rbf = fit_surface_z_rbf
    fit_with_bspline = fit_surface_z_bspline

    def fit_surface_to_points(
        self,
        method: Literal["weighted_avg", "quadric", "rbf", "bspline"] = "weighted_avg",
        plane: Optional[gkp.Plane] = None,
        spacing: Optional[float] = None,
        n_elements: Optional[int] = None,
        boundary: Literal["convex", "concave"] = "convex",
        concave_ratio: float = 0.3,
        spline: bool = False,
        **kwargs,
    ) -> Trimesh3d:
        """Fits a 3D surface mesh to the point cloud using the chosen surface interpolation method.

        Calls the corresponding fit_surface_z_* method on Pcloud.

        Args:
            method (Literal["weighted_avg", "quadric", "rbf", "bspline"], optional):
                Surface fitting algorithm to use. Defaults to "weighted_avg".
            plane (Optional[gkp.Plane], optional): Orientation plane. If None, fitted via PCA.
            spacing (Optional[float], optional): Target grid node spacing in projection plane.
            n_elements (Optional[int], optional): Number of grid intervals per axis. Defaults to 50 if spacing is None.
            boundary (Literal["convex", "concave"], optional): Boundary hull type. Defaults to "convex".
            concave_ratio (float, optional): Concavity ratio (0.0 to 1.0) when boundary='concave'. Defaults to 0.3.
            spline (bool, optional): Whether to smooth the boundary polygon with a B-spline. Defaults to False.
            **kwargs: Method-specific parameters forwarded to the chosen fit_surface_z_* method.

        Returns:
            Trimesh3d: Triangulated surface mesh containing vertices and faces.
        """
        if method in ("weighted_avg", "idw"):
            return self.fit_surface_z_weighted_avg(
                plane=plane,
                spacing=spacing,
                n_elements=n_elements,
                boundary=boundary,
                concave_ratio=concave_ratio,
                spline=spline,
                **kwargs,
            )
        elif method in ("quadric", "polynomial"):
            return self.fit_surface_z_quadric(
                plane=plane,
                spacing=spacing,
                n_elements=n_elements,
                boundary=boundary,
                concave_ratio=concave_ratio,
                spline=spline,
                **kwargs,
            )
        elif method == "rbf":
            return self.fit_surface_z_rbf(
                plane=plane,
                spacing=spacing,
                n_elements=n_elements,
                boundary=boundary,
                concave_ratio=concave_ratio,
                spline=spline,
                **kwargs,
            )
        elif method in ("bspline", "spline"):
            return self.fit_surface_z_bspline(
                plane=plane,
                spacing=spacing,
                n_elements=n_elements,
                boundary=boundary,
                concave_ratio=concave_ratio,
                spline=spline,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown surface fitting method: '{method}'. "
                f"Supported methods: 'weighted_avg', 'quadric', 'rbf', 'bspline'."
            )

if not hasattr(pd.DataFrame, "pcloud"):
    pd.api.extensions.register_dataframe_accessor("pcloud")(Pcloud)


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


if not hasattr(pd.DataFrame, "pgrid"):
    pd.api.extensions.register_dataframe_accessor("pgrid")(Pgrid)
    

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
    pass

    #%%TESTS
    
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
    
    
    #%% cylinder around 3D path
    import pandas as pd
    import plotly.graph_objects as go
    path_3d = np.asarray(
        [[0,50,75,88,99],
         [100,120,125,130,125],
         [300,320,335,350,362]]).T
    path_3d_df = pd.DataFrame(path_3d, columns=['x','y','z'])    
    
    rad = [10,15,20,15,5]
    cylinder = CylinderAlongPath(
            position_disks='end-to-end',
            radius_disks=rad,
            num_sides_disks=30,
            path=path_3d_df,
            cap_ends= False)
    list_rings = generate_disks_around_path(cylinder)
    
    disk=build_disk_parallel_to_plane(plane=gkp.Plane(0,80),
                                 center=(10,50,150),
                                 radius=10,
                                 num_sides=10)
    
    
    cylinder_trimesh=build_cylinder_around_path(cylinder)    

    #Display
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=path_3d_df.x,
                                 y=path_3d_df.y,
                                 z=path_3d_df.z,
                                 mode='lines', name='path',
                                 line=dict( width=5),))
    for ring in list_rings:
        fig.add_trace(go.Scatter3d(x=ring[:,0],
                                   y=ring[:,1],
                                   z=ring[:,2],
                                   mode='lines', name='rings',
                                   line=dict( width=5),))
    fig.add_trace(go.Scatter3d(x=disk[:,0],
                               y=disk[:,1],
                               z=disk[:,2],
                               mode='lines', name='disk',
                               line=dict(color='black', width=10),))
    fig.add_trace(go.Mesh3d(
        x=cylinder_trimesh.vertices[:, 0],
        y=cylinder_trimesh.vertices[:, 1],
        z=cylinder_trimesh.vertices[:, 2],
        i=cylinder_trimesh.faces[:, 0],
        j=cylinder_trimesh.faces[:, 1],
        k=cylinder_trimesh.faces[:, 2],
        intensity=cylinder_trimesh.vertices[:, 2],  # optional: color by Z/property  
        colorscale="Viridis",
        opacity=0.7,
        name='cylinder',
    ))
    fig.update_layout(scene_aspectmode="data")  # preserve true proportions
    fig.show()    
    
