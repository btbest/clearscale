import pytest

from clearscale import Scene, Multiscale, Scale, Shape
from clearscale._transforms import CoordinateSystem, TranslationTransform, _UnresolvedRef, TransformGraph

from tests.ome_zarr.scene_examples import (
    all_invalid_scene_examples,
    all_valid_scene_examples,
    scene_registration,
    scene_stitching,
)


def _multiscale():
    ms = Multiscale({"s0": Scale(Shape(y=2, x=3))})
    object.__setattr__(ms._intrinsic_ref, "name", "physical")
    return ms


def test_stitching_example():
    scene = Scene.from_ome_zarr(scene_stitching())
    assert scene.unresolved_paths == ["tile_0", "tile_1", "tile_2", "tile_3"]


def test_registration_example():
    scene = Scene.from_ome_zarr(scene_registration())
    assert scene.unresolved_paths == ["JRC2018F", "FCWB"]


@pytest.mark.parametrize("meta", all_valid_scene_examples())
def test_scene_public_examples_roundtrip(meta):
    # https://ngff.openmicroscopy.org/specifications/dev/examples/transformations/transformations.html
    roundtrip = Scene.from_ome_zarr(meta).to_ome_zarr(version="0.6.rc0")
    assert "coordinateTransformations" in roundtrip
    assert "coordinateTransformations" in meta
    assert len(roundtrip["coordinateTransformations"]) == len(meta["coordinateTransformations"])
    for actual, expected in zip(roundtrip["coordinateTransformations"], meta["coordinateTransformations"]):
        # compare individually for easier debugging
        assert actual == expected
    # Full check because zip can miss one containing more items than the other
    assert roundtrip == meta


@pytest.mark.parametrize("meta", all_invalid_scene_examples())
def test_scene_invalid_public_examples(meta):
    with pytest.raises(ValueError, match="ByDimensionTransform target axes must be globally unique"):
        _ = Scene.from_ome_zarr(meta)


def test_load_then_resolve():
    paths = {path: _multiscale() for path in ["tile_0", "tile_1", "tile_2", "tile_3"]}
    scene = Scene.from_ome_zarr(scene_stitching())
    assert not scene.is_fully_resolved

    resolved_scene = scene.with_resolved(paths)
    assert resolved_scene.is_fully_resolved


def test_with_resolved_remembers_only_paths_that_resolved_transform_endpoints():
    """
    This ensures Scene._full_graph cannot produce a disjunct graph.
    Could probably be relaxed since it's private anyway.
    """
    used = _multiscale()
    unused = _multiscale()
    scene = Scene.from_ome_zarr(
        {
            "coordinateSystems": [
                {
                    "name": "world",
                    "axes": [{"name": "y"}, {"name": "x"}],
                }
            ],
            "coordinateTransformations": [
                {
                    "type": "translation",
                    "input": {"path": "tile_0", "name": "physical"},
                    "output": {"name": "world"},
                    "translation": [0, 0],
                }
            ],
        }
    )

    resolved = scene.with_resolved({"tile_0": used, "unused": unused})

    assert resolved._multiscale_paths == {"tile_0": used}


def test_with_resolved_does_not_resolve_by_name():
    multiscale = _multiscale()
    world = CoordinateSystem.without_semantics("yx").as_ref("world")
    transform = TranslationTransform(
        translation=(0, 0),
        source=_UnresolvedRef(name="world"),
        target=_UnresolvedRef(path="tile_0", name="physical"),
    )
    scene = Scene(TransformGraph([transform], system_refs=(world,)), _multiscale_paths={})
    resolved = scene.with_resolved({"tile_0": multiscale})

    assert isinstance(resolved._internal_graph.transforms[0].source, _UnresolvedRef)
    assert resolved._internal_graph.transforms[0].target == multiscale.as_ref("physical")
