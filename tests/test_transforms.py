import re

import pytest

from clearscale import Multiscale, Scale, Shape
from clearscale._transforms import (
    CoordinateSystem,
    IdentityTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    ScaleTransform,
    TransformSequence,
    TranslationTransform,
    TransformGraph,
    _UnresolvedRef,
)


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
    assert transform.to_ome_zarr("0.6.dev4") == {
        "type": "scale",
        "scale": [2.0],
        "name": "pixel-size",
        "input": {"name": "source"},
        "output": {"name": "target"},
    }


def _sys_ref(name, axes):
    return CoordinateSystem.without_semantics(axes).as_ref(name)


def test_with_resolved_by_name_does_not_resolve_path_refs():
    world = CoordinateSystem.without_semantics("yx").as_ref("world")
    original_target = _UnresolvedRef(path="tile_0", name="world")
    transform = TranslationTransform(translation=(0, 0), source=_UnresolvedRef(name="world"), target=original_target)

    resolved = transform.with_resolved_by_name((world,))

    assert resolved.source is world
    assert resolved.target is original_target


def test_bound_scale_rejects_axis_count_mismatch():
    world = _sys_ref("world", "yx")

    with pytest.raises(ValueError, match="ScaleTransform expects 3 source axes"):
        ScaleTransform(scale=(1, 1, 1)).bound(source=world, target=world)


def test_bound_translation_rejects_endpoint_axis_count_mismatch():
    source = _sys_ref("source", "yx")
    target = _sys_ref("target", "zyx")

    with pytest.raises(ValueError, match="TranslationTransform expects 2 target axes"):
        TranslationTransform(translation=(0, 0)).bound(source=source, target=target)


def test_path_backed_scale_round_trips_and_no_ndim():
    scale = Transform.from_ome_zarr({"type": "scale", "path": "coordinateTransformations/scale"})

    assert isinstance(scale, ScaleTransform)
    assert scale._ndim_by_payload() is None
    assert not scale.is_invertible
    assert scale.to_ome_zarr("0.6.dev4") == {"type": "scale", "path": "coordinateTransformations/scale"}


def test_path_backed_translation_round_trips_and_no_ndim():
    translation = Transform.from_ome_zarr({"type": "translation", "path": "coordinateTransformations/translation"})
    assert isinstance(translation, TranslationTransform)
    assert translation._ndim_by_payload() is None
    assert not translation.is_invertible
    assert translation.to_ome_zarr("0.6.dev4") == {
        "type": "translation",
        "path": "coordinateTransformations/translation",
    }


def test_bound_identity_rejects_endpoint_axis_count_mismatch():
    source = _sys_ref("source", "yx")
    target = _sys_ref("target", "zyx")

    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        IdentityTransform().bound(source=source, target=target)


def test_resolving_transform_revalidates_endpoint_axes():
    multiscale = Multiscale({"s0": Scale(Shape(z=1, y=2, x=3))})
    world = _sys_ref("world", "yx")
    transform = TranslationTransform(
        translation=(0, 0),
        source=_UnresolvedRef(path="tile_0", name="physical"),
        target=world,
    )

    with pytest.raises(ValueError, match="TranslationTransform expects 2 source axes"):
        transform.with_resolved({"tile_0": multiscale})


@pytest.mark.parametrize(
    "original, inverse",
    [
        ((0,), (0,)),
        ((0, 1), (0, 1)),
        ((1, 0), (1, 0)),
        ((0, 1, 2), (0, 1, 2)),
        ((0, 2, 1), (0, 2, 1)),
        ((1, 0, 2), (1, 0, 2)),
        ((1, 2, 0), (2, 0, 1)),
        ((2, 0, 1), (1, 2, 0)),
        ((2, 1, 0), (2, 1, 0)),
        ((0, 1, 2, 3), (0, 1, 2, 3)),
        ((3, 0, 1, 2), (1, 2, 3, 0)),
        ((2, 3, 0, 1), (2, 3, 0, 1)),
        ((0, 2, 1, 3), (0, 2, 1, 3)),
    ],
)
def test_map_axis_inverted(original, inverse):
    """
    Basically whenever it's a flip (0, 2, 1), the inverse is the same flip.
    If it's a shift but same order (1, 2, 3, 0), the inverse is the opposite shift
    """
    transform = MapAxisTransform(original)

    assert transform.inverted().map_axis == inverse


@pytest.mark.parametrize(
    "earlier, later, composed",
    [
        ((0,), (0,), (0,)),
        ((0, 1), (0, 1), (0, 1)),
        ((0, 1), (1, 0), (1, 0)),
        ((1, 0), (0, 1), (1, 0)),
        ((1, 0), (1, 0), (0, 1)),
        ((0, 1, 2), (0, 1, 2), (0, 1, 2)),
        ((0, 2, 1), (0, 2, 1), (0, 1, 2)),  # flip + inverse flip
        ((1, 0, 2), (0, 2, 1), (1, 2, 0)),  # flip + flip
        ((1, 2, 0), (0, 2, 1), (1, 0, 2)),  # shift + flip
        ((1, 2, 0), (1, 2, 0), (2, 0, 1)),  # shift + shift
        ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3)),
        ((3, 0, 1, 2), (1, 2, 3, 0), (0, 1, 2, 3)),
        ((0, 2, 1, 3), (0, 2, 1, 3), (0, 1, 2, 3)),
        ((3, 0, 1, 2), (3, 0, 1, 2), (2, 3, 0, 1)),  # shift + shift
        ((3, 0, 1, 2), (0, 2, 1, 3), (3, 1, 0, 2)),  # shift + flip
    ],
)
def test_map_axis_composed(earlier, later, composed):
    """
    Composing a map-axis with its inverse results in identity (0, 1, 2, ...).
    Otherwise, combine the respective flips and shifts.
    """
    earlier = MapAxisTransform(earlier)
    later = MapAxisTransform(later)

    assert later.composed_with(earlier) == MapAxisTransform(composed)


def test_map_axis_rejects_missing_transpose():
    with pytest.raises(ValueError, match="must include all zero-based indices"):
        _ = MapAxisTransform((0, 2))  # 1 is missing (MapAxis isn't allowed to drop)


def test_map_axis_rejects_mismatching_dims():
    source = _sys_ref("source", "cyx")
    bad_target = _sys_ref("target", "yx")

    with pytest.raises(ValueError, match="expects 3 target axes"):
        _ = MapAxisTransform((0, 1, 2), source=source, target=bad_target)


def test_project_axis_matches_differing_endpoint_ndim():
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "ij")

    _ = ProjectAxisTransform(drops=(0,), source=source, target=target)


@pytest.mark.parametrize(
    "earlier, later, composed",
    [  # three tuples of (drop, create)
        (((), ()), ((), ()), ((), ())),
        (((0,), ()), ((), ()), ((0,), ())),
        (((), (2,)), ((), ()), ((), (2,))),
        (((), ()), ((1,), ()), ((1,), ())),
        (((), ()), ((), (3,)), ((), (3,))),
        (((0,), ()), ((0,), ()), ((0, 1), ())),
        (((), (0,)), ((), (0,)), ((), (0, 1))),
        (((), (0,)), ((), (3,)), ((), (0, 3))),
        (((), (1,)), ((), (5,)), ((), (1, 5))),
        (((), ()), ((3,), ()), ((3,), ())),
        (((2,), ()), ((), (4,)), ((2,), (4,))),
        (((0,), (0,)), ((0,), (0,)), ((0,), (0,))),
        (((0,), (0,)), ((0,), ()), ((0,), ())),
        (((3,), (1,)), ((0,), (0,)), ((0, 3), (0, 1))),
        (((1,), (3,)), ((0,), (3,)), ((0, 1), (2, 3))),
        (((1,), (3,)), ((1,), (3,)), ((1, 2), (2, 3))),
        (((0, 1, 4), (3,)), ((3, 4), ()), ((0, 1, 4, 6), ())),
        (((), (0, 1)), ((), (0, 5)), ((), (0, 1, 2, 5))),
    ],
)
def test_project_axis_composed(earlier, later, composed):
    """
    Composing a project-axis needs to trace `earlier`'s source axes through its dropping and
    insertions, to determine which of them are subsequently dropped by `later`,
    and which axes of the final result are newly created across both transforms.
    """
    earlier_t = ProjectAxisTransform(drops=earlier[0], inserts=earlier[1])
    later_t = ProjectAxisTransform(drops=later[0], inserts=later[1])

    composed_t = later_t.composed_with(earlier_t)
    assert isinstance(composed_t, ProjectAxisTransform)
    assert composed_t.drops == composed[0]
    assert composed_t.inserts == composed[1]


def test_project_axis_rejects_duplicates():
    with pytest.raises(ValueError, match="Expected unique indices"):
        _ = ProjectAxisTransform(drops=(1, 1))
    with pytest.raises(ValueError, match="Expected unique indices"):
        _ = ProjectAxisTransform(inserts=(1, 1))


def test_project_axis_rejects_mismatching_dims():
    source = _sys_ref("source", "cyx")
    bad_target = _sys_ref("target", "yx")

    with pytest.raises(ValueError, match="expects 3 target axes"):
        ProjectAxisTransform(drops=(0,), inserts=(0,), source=source, target=bad_target)


def test_project_axis_rejects_create_index_out_of_bounds():
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "yxi")

    with pytest.raises(ValueError, match="inserts output index outside target axes"):
        _ = ProjectAxisTransform(inserts=(2, 3), source=source, target=target)


def test_project_axis_rejects_drop_index_out_of_bounds():
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "yx")

    with pytest.raises(ValueError, match="drops input index outside source axes"):
        _ = ProjectAxisTransform(drops=(3,), source=source, target=target)


def test_transform_sequence_rejects_mismatched_axis_value_counts():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Transform chain dimensionality mismatches: ScaleTransform(target_ndim=2) != TranslationTransform(source_ndim=3)"
        ),
    ):
        TransformSequence((ScaleTransform(scale=(1, 1)), TranslationTransform(translation=(0, 0, 0))))


def test_bound_transform_sequence_rejects_endpoint_axis_count_mismatch():
    source = _sys_ref("source", "yx")
    target = _sys_ref("target", "zyx")
    sequence = TransformSequence((IdentityTransform(), IdentityTransform()))

    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        sequence.bound(source=source, target=target)


def test_transform_sequence_infers_endpoint_dimensionality_through_identity():
    source = _sys_ref("source", "yx")
    target = _sys_ref("target", "yx")
    sequence = TransformSequence((IdentityTransform(), IdentityTransform(), TranslationTransform(translation=(0, 0))))

    bound = sequence.bound(source=source, target=target)

    assert bound.source == source
    assert bound.target == target


def test_transform_sequence_rejects_endpoint_mismatch_inferred_through_identity():
    source = _sys_ref("source", "zyx")
    target = _sys_ref("target", "yx")
    sequence = TransformSequence((IdentityTransform(), IdentityTransform(), TranslationTransform(translation=(0, 0))))

    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        sequence.bound(source=source, target=target)


def test_transform_sequence_accepts_consistent_dimensionality_through_identity():
    _ = TransformSequence(
        (
            ScaleTransform((0.25, 0.25)),
            IdentityTransform(),
            IdentityTransform(),
            TranslationTransform(translation=(0, 0)),
        )
    )


def test_transform_sequence_allows_dimension_change_through_project_axis():
    _ = TransformSequence(
        (
            ScaleTransform((1, 1, 1)),
            ProjectAxisTransform(drops=(0,)),
            ScaleTransform((2, 2)),
        )
    )


def test_transform_sequence_rejects_inconsistent_dimensionality_through_identity():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Transform chain dimensionality mismatches: ScaleTransform(target_ndim=3) != TranslationTransform(source_ndim=2)"
        ),
    ):
        _ = TransformSequence(
            (
                ScaleTransform((0.25, 0.25, 0.25)),
                IdentityTransform(),
                IdentityTransform(),
                TranslationTransform(translation=(0, 0)),
            )
        )


def test_collapsed_scale_sequence_preserves_bound_endpoints():
    source = _sys_ref("source", "yx")
    middle = _sys_ref("middle", "yx")
    target = _sys_ref("target", "yx")
    sequence = TransformSequence(
        (
            ScaleTransform(scale=(2, 3), source=source, target=middle),
            ScaleTransform(scale=(5, 7), source=middle, target=target),
        )
    )

    collapsed = sequence.collapsed()

    assert isinstance(collapsed, ScaleTransform)
    assert collapsed.source == source
    assert collapsed.target == target
    assert collapsed.scale == (10, 21)


def test_transform_graph_rejects_unbound_transforms():
    with pytest.raises(ValueError, match="Graph transforms must have bound endpoints"):
        TransformGraph([ScaleTransform(scale=(1, 1))])


def test_transform_graph_keeps_bound_transforms_from_generator():
    world = _sys_ref("world", "yx")
    transform = ScaleTransform(scale=(1, 1), source=world, target=world)

    graph = TransformGraph(t for t in (transform,))

    assert graph.transforms == (transform,)
