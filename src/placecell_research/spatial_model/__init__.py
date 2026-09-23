"""Spatial model interfaces and builders."""

from .builder import ModelBuildContext, build_place_model
from .contract import build_model_contract, write_parameter_shapes_csv
from .loading import select_place_model_checkpoint
from .protocol import AuxiliaryHead, PlaceModel
from .types import ModuleOutputs, RepresentationBundle

__all__ = [
    "AuxiliaryHead",
    "ModelBuildContext",
    "ModuleOutputs",
    "PlaceModel",
    "RepresentationBundle",
    "build_model_contract",
    "build_place_model",
    "select_place_model_checkpoint",
    "write_parameter_shapes_csv",
]
