from clearscale._multiscale import Multiscale
from clearscale._spatial_relations import ProjectionTo


def _global_scale(result):
    (scale_dict,) = [t for t in result.get("coordinateTransformations", []) if t["type"] == "scale"]
    return scale_dict["scale"]


def _dataset_scale(result, key="s0"):
    dataset = next(d for d in result["datasets"] if d["path"] == key)
    (scale_dict,) = [t for t in dataset["coordinateTransformations"] if t["type"] == "scale"]
    return scale_dict["scale"]


def test_full_flow_carries_axis_insertion_relation_into_legacy_output():
    raw_source = {
        "version": "0.4",
        "axes": [{"name": a} for a in "zyx"],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [0.3, 20.0, 30.0]}]},
            {"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [0.6, 40.0, 60.0]}]},
        ],
    }
    shapes = {"s0": (4, 4, 4), "s1": (2, 2, 2)}
    source = Multiscale.from_ome_zarr(raw_source, shape_source=lambda p: shapes[p])
    assert source._legacy_convention_global_t_scale is None

    derived = Multiscale({key: source[key].with_axes("tczyx") for key in source.keys()})
    derived = derived.with_coordinate_systems_of(source, derived_by=ProjectionTo(derived.axes()))

    result = derived.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" not in result
    assert _dataset_scale(result, "s0") == [1.0, 1.0, 0.3, 20.0, 30.0]
    assert _dataset_scale(result, "s1") == [1.0, 1.0, 0.6, 40.0, 60.0]


def test_full_flow_preserves_t_scale_convention_after_axis_insertion():
    # 0.4 metadata using the global-t-scale convention: multiscale-level transform carries
    # the t component ([0.5, 1.0, 1.0, 1.0]) of pixel size.
    raw_source = {
        "version": "0.4",
        "axes": [{"name": a} for a in "tzyx"],
        "coordinateTransformations": [{"type": "scale", "scale": [0.5, 1.0, 1.0, 1.0]}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.3, 20.0, 30.0]}]},
            {"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.6, 40.0, 60.0]}]},
        ],
    }
    shapes = {"s0": (4, 4, 4, 4), "s1": (4, 2, 2, 2)}
    source = Multiscale.from_ome_zarr(raw_source, shape_source=lambda p: shapes[p])
    assert source._legacy_convention_global_t_scale == 0.5
    assert source["s0"].pixel_size["t"] == 0.5  # global t folded into the Scale pixel size on read
    assert source["s1"].pixel_size["t"] == 0.5

    # Derive a Multiscale that only adds a channel axis
    derived = Multiscale({key: source[key].with_axes("tczyx") for key in source.keys()})
    derived = derived.with_coordinate_systems_of(source, derived_by=ProjectionTo(derived.axes()))
    assert derived._legacy_convention_global_t_scale == 0.5  # inherited via identity space

    result = derived.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" in result
    assert _global_scale(result) == [0.5, 1.0, 1.0, 1.0, 1.0]
    assert _dataset_scale(result, "s0") == [1.0, 1.0, 0.3, 20.0, 30.0]  # pixel_size[t] decomposed to global
    assert _dataset_scale(result, "s1") == [1.0, 1.0, 0.6, 40.0, 60.0]


def test_full_flow_carries_axis_insertion_relation_into_generic_global_transforms():
    # Plain 0.4 metadata, no multiscale-level transform, ordinary per-dataset scale.
    raw_source = {
        "version": "0.4",
        "axes": [{"name": a} for a in "yx"],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [20.0, 30.0]}]},
            {"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [40.0, 60.0]}]},
        ],
        "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, 1.0]},
            {"type": "translation", "translation": [0.0, 0.0]},
        ],
    }
    shapes = {"s0": (4, 4), "s1": (2, 2)}
    source = Multiscale.from_ome_zarr(raw_source, shape_source=lambda p: shapes[p])
    assert source._legacy_convention_global_t_scale is None

    derived = Multiscale({key: source[key].with_axes("tczyx") for key in source.keys()})
    derived = derived.with_coordinate_systems_of(source, derived_by=ProjectionTo(derived.axes()))

    result = derived.to_ome_zarr(version="0.4")

    assert "coordinateTransformations" in result
    assert _global_scale(result) == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert _dataset_scale(result, "s0") == [1.0, 1.0, 1.0, 20.0, 30.0]
    assert _dataset_scale(result, "s1") == [1.0, 1.0, 1.0, 40.0, 60.0]
