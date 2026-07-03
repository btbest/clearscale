"""Multiscale abstractions for clearly coded metadata provenance"""

from clearscale._axis_values import Factor, PixelOffset, PixelSize, Shape, Translation, Unit
from clearscale._multiscale import (
    BlueprintFactors,
    BlueprintShapes,
    DuplicatePolicy,
    Multiscale,
    Scale,
    discrete_bin_center,
    half_pixel_space_preservation,
    first_value_decimation,
)
from clearscale._scene import Scene

__all__ = [
    "BlueprintFactors",
    "BlueprintShapes",
    "DuplicatePolicy",
    "Factor",
    "IdentityTransform",
    "Multiscale",
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
]
