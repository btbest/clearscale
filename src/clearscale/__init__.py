"""Multiscale abstractions for clearly coded metadata provenance"""

from clearscale import ome_zarr
from clearscale._spatial_relations import SpatialRelation, PermutationTo, ProjectionTo
from clearscale._affines import Affine, Coefficient, Linear
from clearscale._axis_values import Factor, PixelOffset, PixelSize, Shape, Translation, Unit
from clearscale._collections import OmeZarrGroup, GroupKind, ZarrGroup, ChildRef
from clearscale._multiscale import (
    BlueprintFactors,
    BlueprintShapes,
    DuplicatePolicy,
    Multiscale,
    Scale,
)
from clearscale._scene import Scene
from clearscale._transforms import FileRef
from clearscale._translation_shift import (
    TranslationShiftFunction,
    discrete_bin_center,
    half_pixel_space_preservation,
    first_value_decimation,
    detect_translation_shift,
)
from clearscale._version import __version__

__all__ = [
    "__version__",
    "SpatialRelation",
    "PermutationTo",
    "ProjectionTo",
    "BlueprintFactors",
    "BlueprintShapes",
    "DuplicatePolicy",
    "Factor",
    "Multiscale",
    "ome_zarr",
    "OmeZarrGroup",
    "ZarrGroup",
    "ChildRef",
    "FileRef",
    "GroupKind",
    "PixelOffset",
    "PixelSize",
    "Scale",
    "Scene",
    "Shape",
    "Translation",
    "Unit",
    "discrete_bin_center",
    "half_pixel_space_preservation",
    "first_value_decimation",
    "detect_translation_shift",
    "Affine",
    "Coefficient",
    "Linear",
]
