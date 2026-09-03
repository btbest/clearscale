import pytest
from clearscale._axis_values import PixelSize, Shape, Factor, Translation
from clearscale._multiscale import Multiscale, Scale
from clearscale._spatial_relations import AxisRearrangementTo


def _multiscale(axes, size=4, pixel_size=None):
    shape = Shape(zip(axes, [size] * len(axes)))
    ps = PixelSize(zip(axes, pixel_size)) if pixel_size else None
    return Multiscale({"s0": Scale(shape=shape, pixel_size=ps)})


def _global_scale(result):
    scale_dicts = [t for t in result.get("coordinateTransformations", []) if t["type"] == "scale"]
    return scale_dicts[0]["scale"]


def _global_scale_and_translation(result):
    translation_dicts = [t for t in result.get("coordinateTransformations", []) if t["type"] == "translation"]
    return _global_scale(result), translation_dicts[0]["translation"] if translation_dicts else None


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

    result = ms.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" not in result
    assert _dataset_scale(result, "s0") == [0.5, 0.3, 20.0, 30.0]
    assert _dataset_scale(result, "s1") == [0.9, 0.6, 40.0, 60.0]


@pytest.mark.parametrize(
    "relation, expected_global_scale_and_translation",
    [
        (Factor(y=2, x=2), ([0.5, 0.5], None)),
        (Translation(y=3, x=4), ([1.0, 1.0], [-3.0, -4.0])),
        ([Factor(y=2, x=2), Translation(y=3, x=4)], ([0.5, 0.5], [-3.0, -4.0])),
        ([Translation(y=3, x=4), Factor(y=2, x=2)], ([2.0, 2.0], [3.0, 4.0])),
        ([AxisRearrangementTo("zyx"), Translation(z=0, y=3, x=4)], ([1.0, 1.0], [-3.0, -4.0])),
        ([Translation(y=3, x=4), AxisRearrangementTo("x")], ([1.0, 1.0], [-3.0, -4.0])),
    ],
)
def test_multiscale_to_ome_zarr_serializes_compatible_coordinate_system_as_legacy_transforms(
    relation, expected_global_scale_and_translation
):
    ms = _multiscale("yx").with_coordinate_system("world", reached_by=relation)

    result = ms.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" in result
    assert _global_scale_and_translation(result) == expected_global_scale_and_translation


def test_multiscale_to_ome_zarr_serializes_coordinate_system_connected_by_reverse_drop():
    source = _multiscale("zyx")
    # The source--drop-->derived connection can only be stored in reverse
    # (normally external systems on `derived` are connected as `derived --t--> system`)
    derived_ms = _multiscale("yx").as_derived_from(source, by=[AxisRearrangementTo("yx"), Translation(y=3, x=4)])
    expected_global_scale_and_translation = ([1.0, 1.0], [3.0, 4.0])

    result = derived_ms.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" in result
    assert _global_scale_and_translation(result) == expected_global_scale_and_translation


def test_multiscale_to_ome_zarr_does_not_serialize_compatible_coordinate_system_with_identity():
    ms = _multiscale("yx").with_coordinate_system("world", reached_by=None)  # None = identity

    result = ms.to_ome_zarr(version="0.4")

    # Writing a nondescript identity on the global level doesn't contribute anything useful
    assert "coordinateTransformations" not in result


def test_multiscale_to_ome_zarr_t_scale_convention_overrides_compatible_coordinate_system():
    """
    The "global t-scale" convention splits what is supposed to be the *dataset transforms* across
    both dataset and global transforms. This leaves no room for an additional transform to an undefined
    external reference space.
    Ideally, we would want to distinguish coordinate systems added by `as_derived_from` from those added by
    `with_coordinate_system`:
    - as_derived_from transfers the convention, but also generates an external system, potentially without
      the user's awareness -> maintain convention. If they loaded nifti-zarr (which uses the convention),
      they expect to write nifti-zarr.
    - with_coordinate_system is an explicit call to add an external reference space -> override convention. If
      they made the effort to add an external system, they expect it to be saved. But, they are almost certainly
      working with OME-Zarr 0.6, which has no "global t-scale" convention.
    Hence, in absense of a good way to distinguish the two: Convention overrides external system when writing legacy metadata.
    """
    ms = Multiscale(
        {
            "s0": Scale(shape=Shape(t=4, y=2, x=2), pixel_size=PixelSize(t=0.5, y=20.0, x=30.0)),
            "s1": Scale(shape=Shape(t=4, y=1, x=1), pixel_size=PixelSize(t=0.5, y=40.0, x=60.0)),
        },
        _legacy_convention_global_t_scale=0.5,
    )
    ms = ms.with_coordinate_system("world", reached_by=Factor(t=4.0, y=1.0, x=1.0))

    result = ms.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" in result
    expected_global_transform = ([0.5, 1.0, 1.0], None)  # pixel size, not the reached_by factor
    assert _global_scale_and_translation(result) == expected_global_transform
    assert _dataset_scale(result, "s0") == [1.0, 20.0, 30.0]  # pixel size in global transforms, not here
    assert _dataset_scale(result, "s1") == [1.0, 40.0, 60.0]


def test_multiscale_to_ome_zarr_does_not_override_t_scale_convention_when_compatible_coordinate_system_with_identity():
    ms = Multiscale(
        {
            "s0": Scale(shape=Shape(t=4, y=2, x=2), pixel_size=PixelSize(t=0.5, y=20.0, x=30.0)),
            "s1": Scale(shape=Shape(t=4, y=1, x=1), pixel_size=PixelSize(t=0.5, y=40.0, x=60.0)),
        },
        _legacy_convention_global_t_scale=0.5,
    )
    ms = ms.with_coordinate_system("world", reached_by=None)  # None = identity

    result = ms.to_ome_zarr(version="0.4")

    # Writing a nondescript identity on the global level doesn't contribute anything useful.
    # More important to maintain convention.
    assert "coordinateTransformations" in result
    # time pixel size follows convention and is written on global level
    assert _global_scale_and_translation(result) == ([0.5, 1.0, 1.0], None)
    assert _dataset_scale(result, "s0") == [1.0, 20.0, 30.0]
    assert _dataset_scale(result, "s1") == [1.0, 40.0, 60.0]
