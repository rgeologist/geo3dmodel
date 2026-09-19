# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:10:30 2025

@author: r.castilla
"""

from typing import Any

from .model3d import *
from .model3d_abstract import Model3D_abstract
from .model3d_plotly import Model3D_plotly

def __getattr__(name: str) -> Any:
    if name == "Model3D_pyvista":
        from .model3d_pyvista import Model3D_pyvista
        globals()["Model3D_pyvista"] = Model3D_pyvista
        return Model3D_pyvista
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + ["Model3D_pyvista"])
 
