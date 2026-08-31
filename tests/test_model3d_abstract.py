"""Tests for model3d_abstract module."""
import pytest
from geo3dmodel.model3d_abstract import Model3D_abstract


class ConcreteModel3D(Model3D_abstract):
    """A concrete implementation of Model3D_abstract for testing purposes."""
    
    def plot_borehole(self, borehole, **kwargs):
        pass
        
    def plot_rectangular_mesh(self, strike=None, dip=None, nodes=None, triangles=None,
                              center=(0,0,0), i_elements=None, i_length=None,
                              j_elements=None, j_length=None, row=1, col=1, **kwargs):
        pass
        
    def plot_plane(self, points_3d_df, c_hull_kwargs=None, trace_kwargs=None, **kwargs):
        pass
        
    def plot_point_cloud(self, df, **kwargs):
        pass
        
    def add_borehole_path(self, borehole, **kwargs):
        pass
        
    def add_boreholes_paths(self, borehole_dict, names_to_skip=None, **kwargs):
        pass
        
    def add_borehole_interval(self):
        pass
        
    def plot_log_along_borehole(self, borehole, log):
        pass
        
    def add_one_structure_as_disk(self, strike, dip, center, disk_kwargs=None, mesh_kwargs=None):
        pass


def test_model3d_abstract_cannot_be_instantiated():
    """Test that Model3D_abstract cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Model3D_abstract()


def test_concrete_model3d_can_be_instantiated():
    """Test that a concrete subclass can be instantiated."""
    model = ConcreteModel3D()
    assert isinstance(model, Model3D_abstract)
