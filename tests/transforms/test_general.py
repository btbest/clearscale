import pytest
from clearscale import Multiscale, Scale, Shape, FileRef
from clearscale._transforms import (
    CoordinateSystem,
    Transform,
    NodeRef,
    TranslationTransform,
    _UnresolvedRef,
    IdentityTransform,
    ScaleTransform,
    AffineTransform,
    RotationTransform,
    CoordinatesTransform,
    DisplacementsTransform,
    TransformSequence,
    MapAxisTransform,
    ProjectAxisTransform,
    BijectionTransform,
    ByDimensionTransform,
    TransformGraph,
)


def _sys_ref(name, axes):
    return CoordinateSystem.without_semantics(axes).as_ref(name)


def test_transform_name_round_trips():
    transform = Transform.from_ome_zarr(
        {
            "type": "scale",
            "name": "pixel-size",
            "scale": [2.0],
            "input": {"name": "source"},
            "output": {"name": "target"},
        }
    )

    assert transform._ome_zarr_name == "pixel-size"
    assert transform.to_ome_zarr("0.6.rc0") == {
        "type": "scale",
        "scale": [2.0],
        "name": "pixel-size",
        "input": {"name": "source"},
        "output": {"name": "target"},
    }


def test_resolving_transform_revalidates_endpoint_axes():

    def _ref(axes: str, name: str) -> NodeRef[CoordinateSystem]:
        return CoordinateSystem.without_semantics(axes).as_ref(name)

    multiscale = Multiscale({"s0": Scale(Shape(z=1, y=2, x=3))}, _intrinsic_ref=_ref("yx", "physical"))
    world = _sys_ref("world", "yx")
    transform = TranslationTransform(
        translation=(0, 0),
        source=_UnresolvedRef(file=FileRef.from_string("tile_0"), name="physical"),
        target=world,
    )

    with pytest.raises(ValueError, match="TranslationTransform expects 2 source axes"):
        transform.with_resolved({"tile_0": multiscale})


def test_with_resolved_by_name_does_not_resolve_path_refs():
    world = CoordinateSystem.without_semantics("yx").as_ref("world")
    original_target = _UnresolvedRef(file=FileRef.from_string("tile_0"), name="world")
    transform = TranslationTransform(translation=(0, 0), source=_UnresolvedRef(name="world"), target=original_target)

    resolved = transform.with_resolved_by_name((world,))

    assert resolved.source is world
    assert resolved.target is original_target


@pytest.mark.parametrize(
    "transform",
    [
        IdentityTransform(),
        ScaleTransform(scale=(0.5, 0.25)),
        TranslationTransform(translation=(0.5, 0.25)),
        RotationTransform(
            rotation=(
                (1, 0, 0, 0),
                (0, 0, 0, -1),
                (0, 0, 1, 0),
                (0, 1, 0, 0),
            )
        ),
        MapAxisTransform(map_axis=(3, 0, 2, 1)),
        AffineTransform(affine=((2, 0, 3), (0, 4, 8))),
    ],
)
def test_composed_with_inverse_preserves_type_and_simplifies_to_identity(transform):
    compose_inverse_with_self = transform.inverted().composed_with(transform)
    compose_self_with_inverse = transform.composed_with(transform.inverted())
    assert isinstance(compose_inverse_with_self, type(transform))
    assert isinstance(compose_self_with_inverse, type(transform))
    assert isinstance(compose_inverse_with_self.simplified(), IdentityTransform)
    assert isinstance(compose_self_with_inverse.simplified(), IdentityTransform)


@pytest.mark.parametrize(
    ("ome_dict", "expected_type", "expected_roundtrip"),
    [
        pytest.param(
            {"type": "identity"},
            IdentityTransform,
            {"type": "identity"},
            id="identity",
        ),
        pytest.param(
            {"type": "scale", "scale": [1, 0]},
            ScaleTransform,
            {"type": "scale", "scale": [1.0, 0.0]},  # include a pathological zero-scale
            id="scale",
        ),
        pytest.param(
            {"type": "translation", "translation": [1, 0]},
            TranslationTransform,
            {"type": "translation", "translation": [1.0, 0.0]},
            id="translation",
        ),
        pytest.param(
            {"type": "affine", "affine": [[1, 0, 2], [0, 1, 3]]},
            AffineTransform,
            {"type": "affine", "affine": [[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]]},
            id="affine",
        ),
        pytest.param(
            {"type": "rotation", "rotation": [[0, -1], [1, 0]]},
            RotationTransform,
            {"type": "rotation", "rotation": [[0.0, -1.0], [1.0, 0.0]]},
            id="rotation",
        ),
        pytest.param(
            {"type": "coordinates", "path": "coordinateTransformations/coords", "interpolation": "linear"},
            CoordinatesTransform,
            {"type": "coordinates", "path": "coordinateTransformations/coords", "interpolation": "linear"},
            id="coordinates",
        ),
        pytest.param(
            {"type": "displacements", "path": "coordinateTransformations/displacements"},
            DisplacementsTransform,
            {"type": "displacements", "path": "coordinateTransformations/displacements"},
            id="displacements",
        ),
        pytest.param(
            {"type": "sequence", "transformations": [{"type": "identity"}]},
            TransformSequence,
            {"type": "sequence", "transformations": [{"type": "identity"}]},
            id="sequence",
        ),
        pytest.param(
            {"type": "mapAxis", "mapAxis": [1, 0]},
            MapAxisTransform,
            {"type": "mapAxis", "mapAxis": [1, 0]},
            id="mapAxis",
        ),
        pytest.param(
            {"type": "projectAxis", "droppedInputs": [0], "createdOutputs": [0]},
            ProjectAxisTransform,
            {"type": "projectAxis", "droppedInputs": [0], "createdOutputs": [0]},
            id="projectAxis",
        ),
        pytest.param(
            {
                "type": "bijection",
                "input": {"name": "source"},
                "output": {"name": "target"},
                "forward": {"type": "scale", "path": "coordinateTransformations/forward"},
                "inverse": {"type": "scale", "path": "coordinateTransformations/inverse"},
            },
            BijectionTransform,
            {
                "type": "bijection",
                "input": {"name": "source"},
                "output": {"name": "target"},
                "forward": {"type": "scale", "path": "coordinateTransformations/forward"},
                "inverse": {"type": "scale", "path": "coordinateTransformations/inverse"},
            },
            id="bijection",
        ),
        pytest.param(
            {
                "type": "byDimension",
                "transformations": [
                    {"inputAxes": [0], "outputAxes": [0], "transformation": {"type": "scale", "scale": [2.0]}},
                    {
                        "inputAxes": [1],
                        "outputAxes": [1],
                        "transformation": {"type": "translation", "translation": [3.0]},
                    },
                ],
            },
            ByDimensionTransform,
            {
                "type": "byDimension",
                "transformations": [
                    {"inputAxes": [0], "outputAxes": [0], "transformation": {"type": "scale", "scale": [2.0]}},
                    {
                        "inputAxes": [1],
                        "outputAxes": [1],
                        "transformation": {"type": "translation", "translation": [3.0]},
                    },
                ],
            },
            id="byDimension",
        ),
    ],
)
def test_transform_from_ome_zarr_parses_all_transform_types(ome_dict, expected_type, expected_roundtrip):
    """Note that without `input` and `output`, all of these examples are actually invalid.
    Outside a TransformGraph, we can read and round-trip them anyway."""
    transform = Transform.from_ome_zarr(ome_dict)

    assert isinstance(transform, expected_type)
    assert transform.to_ome_zarr("0.6.rc0") == expected_roundtrip


def test_transform_graph_rejects_unbound_transforms():
    with pytest.raises(ValueError, match="Graph transforms must have bound endpoints"):
        TransformGraph([ScaleTransform(scale=(1, 1))])


def test_transform_graph_keeps_bound_transforms_from_generator():
    world = _sys_ref("world", "yx")
    transform = ScaleTransform(scale=(1, 1), source=world, target=world)

    graph = TransformGraph(t for t in (transform,))

    assert graph.transforms == (transform,)
