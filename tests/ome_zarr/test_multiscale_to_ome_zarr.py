import pytest
from clearscale._axis_values import PixelSize, Shape
from clearscale._multiscale import Multiscale, Scale


def _multiscale(axes, size=4, pixel_size=None):
    shape = Shape(zip(axes, [size] * len(axes)))
    ps = PixelSize(zip(axes, pixel_size)) if pixel_size else None
    return Multiscale({"s0": Scale(shape=shape, pixel_size=ps)})


def _global_scale(result):
    (scale_dict,) = [t for t in result.get("coordinateTransformations", []) if t["type"] == "scale"]
    return scale_dict["scale"]


def _dataset_scale(result, key="s0"):
    dataset = next(d for d in result["datasets"] if d["path"] == key)
    (scale_dict,) = [t for t in dataset["coordinateTransformations"] if t["type"] == "scale"]
    return scale_dict["scale"]


def test_multiscale_to_ome_zarr_synthesizes_global_t_scale_when_uniform():
    ms = Multiscale(
        {
            "s0": Scale(shape=Shape(t=4, z=2, y=2, x=2), pixel_size=PixelSize(t=0.5, z=0.3, y=20.0, x=30.0)),
            "s1": Scale(shape=Shape(t=4, z=1, y=1, x=1), pixel_size=PixelSize(t=0.5, z=0.6, y=40.0, x=60.0)),
        },
        _legacy_convention_global_t_scale=0.5,
    )

    result = ms.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" in result
    assert _global_scale(result) == [0.5, 1.0, 1.0, 1.0]
    assert _dataset_scale(result, "s0") == [1.0, 0.3, 20.0, 30.0]
    assert _dataset_scale(result, "s1") == [1.0, 0.6, 40.0, 60.0]


def test_multiscale_to_ome_zarr_falls_back_on_nonuniform_t_scale():
    # Pathological: Claims to use the global-t-scale convention but actually doesn't (s1[t] != s0[t])
    ms = Multiscale(
        {
            "s0": Scale(shape=Shape(t=4, z=2, y=2, x=2), pixel_size=PixelSize(t=0.5, z=0.3, y=20.0, x=30.0)),
            "s1": Scale(shape=Shape(t=4, z=1, y=1, x=1), pixel_size=PixelSize(t=0.9, z=0.6, y=40.0, x=60.0)),
        },
        _legacy_convention_global_t_scale=0.5,
    )

    with pytest.warns(UserWarning, match="Multiscale claims to use legacy t convention but has non-uniform pixel_size"):
        result = ms.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" not in result
    assert _dataset_scale(result, "s0") == [0.5, 0.3, 20.0, 30.0]
    assert _dataset_scale(result, "s1") == [0.9, 0.6, 40.0, 60.0]
