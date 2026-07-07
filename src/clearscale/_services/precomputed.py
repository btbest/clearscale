from typing import Any, Mapping, Sequence, Tuple

from clearscale._axis_values import AxisKey, PixelSize

SCALES_DICT = Mapping[str, Any]
"""
Expected:
"key" (str), "size" (List[int]), "resolution" (List[float]), "voxel_offset" (List[int])
"""
INFO_DICT = Mapping[str, Any]
"""
Expected:
"scales" (List[SCALES_DICT]), "num_channels" (int)
"""


def zero_resolution_axes(resolution: Sequence[float], axes: Sequence[AxisKey]) -> Tuple[AxisKey, ...]:
    return tuple(axis for axis, value in zip(axes, resolution) if value == 0)


def pixel_size_from_resolution(resolution: Sequence[float], axes: Sequence[AxisKey]) -> PixelSize:
    normalized_resolution = [PixelSize._default() if value == 0 else value for value in resolution]
    return PixelSize(zip(axes, [1.0] + list(reversed(normalized_resolution))))


def validate_info_dict(info_dict: INFO_DICT) -> None:
    if "scales" not in info_dict:
        raise ValueError("Precomputed info JSON must contain 'scales' field")

    scales_list = info_dict["scales"]
    if not isinstance(scales_list, list) or not scales_list:
        raise ValueError("Precomputed info JSON 'scales' must be a non-empty list")

    required_keys = ("key", "size", "resolution")

    for s in scales_list:
        if any(k not in s for k in required_keys):
            raise ValueError("Precomputed info JSON has invalid scale metadata (missing key, size or resolution).")
