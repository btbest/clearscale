import re
from typing import cast

import pytest

from clearscale import Multiscale, Scale, Shape
from clearscale._transforms import (
    BijectionTransform,
    ByDimensionTransform,
    _ByDimensionChild,
    CoordinateSystem,
    IdentityTransform,
    MapAxisTransform,
    ProjectAxisTransform,
    ScaleTransform,
    RotationTransform,
    AffineTransform,
    CoordinatesTransform,
    DisplacementsTransform,
    Transform,
    TransformSequence,
    TranslationTransform,
    TransformGraph,
    _UnresolvedRef,
    NodeRef,
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
    assert transform.to_ome_zarr("0.6.rc0") == {
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
    assert scale.to_ome_zarr("0.6.rc0") == {"type": "scale", "path": "coordinateTransformations/scale"}


def test_path_backed_translation_round_trips_and_no_ndim():
    translation = Transform.from_ome_zarr({"type": "translation", "path": "coordinateTransformations/translation"})
    assert isinstance(translation, TranslationTransform)
    assert translation._ndim_by_payload() is None
    assert not translation.is_invertible
    assert translation.to_ome_zarr("0.6.rc0") == {
        "type": "translation",
        "path": "coordinateTransformations/translation",
    }


def test_bound_identity_rejects_endpoint_axis_count_mismatch():
    source = _sys_ref("source", "yx")
    target = _sys_ref("target", "zyx")

    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        IdentityTransform().bound(source=source, target=target)


def test_resolving_transform_revalidates_endpoint_axes():

    def _ref(axes: str, name: str) -> NodeRef[CoordinateSystem]:
        return CoordinateSystem.without_semantics(axes).as_ref(name)

    multiscale = Multiscale({"s0": Scale(Shape(z=1, y=2, x=3))}, _intrinsic_ref=_ref("yx", "physical"))
    world = _sys_ref("world", "yx")
    transform = TranslationTransform(
        translation=(0, 0),
        source=_UnresolvedRef(path="tile_0", name="physical"),
        target=world,
    )

    with pytest.raises(ValueError, match="TranslationTransform expects 2 source axes"):
        transform.with_resolved({"tile_0": multiscale})


def test_path_backed_rotation_round_trips_and_no_ndim():
    rotation = Transform.from_ome_zarr({"type": "rotation", "path": "coordinateTransformations/rotation"})
    assert isinstance(rotation, RotationTransform)
    assert rotation._ndim_by_payload() is None
    assert not rotation.is_invertible
    assert rotation.to_ome_zarr("0.6.rc0") == {
        "type": "rotation",
        "path": "coordinateTransformations/rotation",
    }


@pytest.mark.parametrize(
    "matrix, match",
    [
        (
            (
                (2, 0),
                (0, 0.5),
            ),
            "must define a rotation",  # every row product must be 1 (unit length vector)
        ),
        (
            (
                (1, 0, 0),
                (0, 1, 0),
            ),
            "must be square",
        ),
        (
            (
                (1, 0),
                (0, -1),
            ),
            "must define a rotation",  # This would be a reflection
        ),
    ],
)
def test_rotation_rejects_invalid_matrix(matrix, match):
    with pytest.raises(ValueError, match=match):
        RotationTransform(rotation=matrix)


@pytest.mark.parametrize(
    "rotation, inverse",
    [
        (((1, 0), (0, 1)), ((1, 0), (0, 1))),
        (((0, -1), (1, 0)), ((0, 1), (-1, 0))),
        (
            (
                (1, 0, 0, 0),
                (0, 0, 0, -1),
                (0, 0, 1, 0),
                (0, 1, 0, 0),
            ),
            (
                (1, 0, 0, 0),
                (0, 0, 0, 1),
                (0, 0, 1, 0),
                (0, -1, 0, 0),
            ),
        ),
    ],
)
def test_rotation_inverts_matrix(rotation, inverse):
    transform = RotationTransform(rotation=rotation)
    assert transform.inverted().rotation == inverse, "did not invert as expected"
    assert transform.inverted().inverted() == transform, "double inversion is always eq input"


@pytest.mark.parametrize(
    "earlier, later, expected",
    [
        pytest.param(
            ((0, -1), (1, 0)),
            ((0, -1), (1, 0)),
            ((-1, 0), (0, -1)),
            id="+90+90=+180",
        ),
        pytest.param(
            ((0, -1), (1, 0)),
            ((-1, 0), (0, -1)),  # +180 and -180 are identical
            ((0, 1), (-1, 0)),  # +270 and -90 are identical
            id="+90+180=+270",
        ),
        pytest.param(
            ((-1, 0), (0, -1)),
            ((0, -1), (1, 0)),
            ((0, 1), (-1, 0)),
            id="+180+90=+270",
        ),
    ],
)
def test_rotation_composition(earlier, later, expected):
    composed = RotationTransform(rotation=later).composed_with(RotationTransform(rotation=earlier))

    assert cast(RotationTransform, composed).rotation == expected


@pytest.mark.parametrize(
    "earlier",
    [
        ScaleTransform(scale=(2, 2)),
        TranslationTransform((1, 2)),
        AffineTransform(((1, 0, 0), (0, 1, 0))),
    ],
)
def test_rotation_only_composes_with_rotation(earlier):
    assert RotationTransform(((1, 0), (0, 1))).composed_with(earlier) is None


def test_path_backed_affine_round_trips_and_no_ndim():
    affine = Transform.from_ome_zarr({"type": "affine", "path": "coordinateTransformations/affine"})
    assert isinstance(affine, AffineTransform)
    assert affine._ndim_by_payload() is None
    assert not affine.is_invertible
    assert affine.to_ome_zarr("0.6.rc0") == {
        "type": "affine",
        "path": "coordinateTransformations/affine",
    }


@pytest.mark.parametrize(
    "matrix",
    [
        ((0, 0),),  # 1d
        ((0, 0, 0), (0, 0, 0)),  # 2d
        ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),  # 3d
        ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),  # 4d
    ],
)
def test_affine_instantiates_with_abbreviated_form(matrix):
    """The bottom row (0,...0, 1) of the homogenous form should not be included"""
    _ = AffineTransform(affine=matrix)


@pytest.mark.parametrize(
    "matrix, expected_error",
    [
        ((0,), "Expected 2D array"),
        (((0,),), "at least one input dimension and one offset column"),
        (((0, 0, 0),), "upper portion of the homogenous"),
        (((0,), (0,)), "at least one input dimension and one offset column"),
        (((0,), (0, 1)), "Expected rectangular 2D array"),
        (
            (
                (0, 0),
                (0, 1),
            ),
            "upper portion of the homogenous",
        ),
        (
            (
                (0, 0),
                (0, 0),
                (0, 1),
            ),
            "upper portion of the homogenous",
        ),
        (
            (
                (0, 0),
                (0, 0, 0),
                (0, 1),
            ),
            "Expected rectangular 2D array",
        ),
        (
            (
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 1),
            ),
            "upper portion of the homogenous",
        ),
        (
            (
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 1),
            ),
            "upper portion of the homogenous",
        ),
    ],
)
def test_affine_rejects_anything_but_ndim_by_ndim_plus1(matrix, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        _ = AffineTransform(affine=matrix)


def test_affine_transform_inverted():
    transform = AffineTransform(affine=((2, 0, 3), (0, 4, 8)))
    assert transform.inverted().affine == ((0.5, 0.0, -1.5), (0.0, 0.25, -2.0))
    assert transform.inverted().inverted() == transform, "double inversion is always eq input"


@pytest.mark.parametrize(
    "matrix",
    [
        (
            (1, 0, 0),
            (0, 0, 0),
        ),
        (
            (1, 2, 0),
            (2, 4, 0),
        ),
    ],
)
def test_affine_noninvertible(matrix):
    transform = AffineTransform(affine=matrix)
    assert not transform.is_invertible

    with pytest.raises(ValueError, match="not invertible"):
        transform.inverted()


@pytest.mark.parametrize(
    "earlier,later,expected",
    [
        pytest.param(
            ((2, 0, 0), (0, 3, 0)),
            ((4, 0, 0), (0, 5, 0)),
            ScaleTransform((8, 15)),
            id="scale only",
        ),
        pytest.param(
            ((0.25, 0, 7), (0, 1, 3)),
            ((4, 0, 2.2), (0, 1, 5)),
            TranslationTransform((30.2, 8.0)),  # (4*7 + 2.2, 1*3 + 5)
            id="scale composes to identity",
        ),
        pytest.param(
            ((1, 0, 7), (0, 2, 3)),
            ((4, 0, -28), (0, 1, -3)),  # (4*7 - 28 = 0, 1*3 - 3 = 0)
            ScaleTransform((4, 2)),
            id="translation composes to identity + scale",
        ),
        pytest.param(
            ((0, -1, 7), (1, 0, 3)),
            ((1, 0, -7), (0, 1, -3)),
            RotationTransform(((0, -1), (1, 0))),
            id="translation composes to identity + rotation",
        ),
    ],
)
def test_affine_composed_with_simplifies(earlier, later, expected):
    earlier_t = AffineTransform(affine=earlier)
    later_t = AffineTransform(affine=later)
    composed = later_t.composed_with(earlier_t)

    assert isinstance(composed, type(expected))
    assert composed == expected


@pytest.mark.parametrize(
    "earlier,later,expected",
    [
        pytest.param(
            ((2, 0, 3), (0, 2, 5)),
            ((3, 0, 7), (0, 3, 11)),
            ((6, 0, 16), (0, 6, 26)),
            id="scale+translation",
        ),
        pytest.param(
            ((0, -1, 0), (1, 0, 0)),
            ((1, 0, 2), (0, 1, 3)),
            ((0, -1, 2), (1, 0, 3)),
            id="rotation then translation",
        ),
        pytest.param(
            ((1, 0.3, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0.3, 0), (0, 1, 0)),
            id="shear+identity",
        ),
    ],
)
def test_affine_composed_with_does_not_simplify_combinations(earlier, later, expected):
    composed = cast(AffineTransform, AffineTransform(affine=later).composed_with(AffineTransform(affine=earlier)))

    assert composed.affine == expected


def test_affines_with_dimension_mismatch_do_not_compose():
    earlier_2d = AffineTransform(
        affine=(
            (1, 0, 0),
            (0, 1, 0),
        )
    )
    later_3d = AffineTransform(
        affine=(
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
        )
    )

    assert later_3d.composed_with(earlier_2d) is None
    assert earlier_2d.composed_with(later_3d) is None


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
        AffineTransform(affine=((2, 0, 3), (0, 4, 8))),
    ],
)
def test_affine_composed_with_inverse_simplifies_to_identity(transform):
    assert isinstance(transform.inverted().composed_with(transform), IdentityTransform)
    assert isinstance(transform.composed_with(transform.inverted()), IdentityTransform)


def test_coordinates_and_displacements_reject_non_strings():
    with pytest.raises(ValueError, match="requires a non-empty path"):
        _ = CoordinatesTransform(path=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interpolation must be a non-empty string"):
        _ = CoordinatesTransform(path="t/coords.zarr", interpolation=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Expected non-empty path"):
        _ = CoordinatesTransform.from_ome_zarr({"path": 1})
    with pytest.raises(ValueError, match="Expected interpolation string"):
        _ = CoordinatesTransform.from_ome_zarr({"path": "t/coords.zarr", "interpolation": 1})
    with pytest.raises(ValueError, match="requires a non-empty path"):
        _ = DisplacementsTransform(path=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="interpolation must be a non-empty string"):
        _ = DisplacementsTransform(path="t/vectors.zarr", interpolation=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Expected non-empty path"):
        _ = DisplacementsTransform.from_ome_zarr({"path": 1})
    with pytest.raises(ValueError, match="Expected interpolation string"):
        _ = DisplacementsTransform.from_ome_zarr({"path": "t/vectors.zarr", "interpolation": 1})


def test_coordinates_and_displacements_cannot_invert_or_compose():
    coords = CoordinatesTransform(path="t/coords.zarr", interpolation="linear")
    assert not coords.is_invertible
    with pytest.raises(ValueError, match="generally not invertible"):
        _ = coords.inverted()
    assert coords.composed_with(IdentityTransform()) is None

    displacements = DisplacementsTransform(path="t/vectors.zarr", interpolation="linear")
    assert not displacements.is_invertible
    with pytest.raises(ValueError, match="generally not invertible"):
        _ = displacements.inverted()
    assert displacements.composed_with(IdentityTransform()) is None


def test_coordinates_can_be_bound_and_chained_arbitrarily():
    coords1 = CoordinatesTransform(path="t/coords1.zarr", interpolation="linear")
    coords2 = CoordinatesTransform(path="t/coords2.zarr", interpolation="linear")
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "tzyx")

    _bound = coords1.bound(source=source, target=target)

    seq = TransformSequence((coords1, coords2))
    _bound_seq = seq.bound(source=source, target=target)

    # Identity normally requires equal source and target dimensionality.
    # Coords is "everything goes", so it removes this constraint when chained with identity.
    seq2 = TransformSequence((IdentityTransform(), coords1, coords2))
    _bound_seq2 = seq2.bound(source=source, target=target)

    seq3 = TransformSequence((coords1, coords2, IdentityTransform()))
    _bound_seq3 = seq3.bound(source=source, target=target)


def test_displacements_bound_reject_changed_axes():
    displacements = DisplacementsTransform(path="t/vectors.zarr", interpolation="linear")
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "tzyx")

    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        _ = displacements.bound(source=source, target=target)


def test_displacements_bound_reject_changed_axes_within_sequence():
    displacements = DisplacementsTransform(path="t/vectors.zarr", interpolation="linear")
    displacements2 = DisplacementsTransform(path="t/vectors2.zarr", interpolation="linear")
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "tzyx")

    seq = TransformSequence((displacements, displacements2))
    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        _ = seq.bound(source=source, target=target)


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


def test_bijection_infers_endpoints_from_child():
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "cyx")
    forward = ScaleTransform((1.0, 2.0, 2.0))
    inverse = ScaleTransform((1.0, 0.5, 0.5))

    forward_with_source = forward.bound(source=source, target=None)
    forward_with_target = forward.bound(source=None, target=target)
    source_from_fwd = BijectionTransform(forward=forward_with_source, inverse=inverse, target=target)
    target_from_fwd = BijectionTransform(forward=forward_with_target, inverse=inverse, source=source)
    assert source_from_fwd.source == target_from_fwd.source
    assert source_from_fwd.target == target_from_fwd.target

    neither_from_child = BijectionTransform(forward=forward, inverse=inverse, target=target, source=source)
    assert source_from_fwd.source == neither_from_child.source
    assert source_from_fwd.target == neither_from_child.target

    inverse_with_source = inverse.bound(source=target, target=None)
    inverse_with_target = inverse.bound(source=None, target=source)
    source_from_inv = BijectionTransform(forward=forward, inverse=inverse_with_target, target=target)
    assert source_from_fwd.source == source_from_inv.source
    target_from_inv = BijectionTransform(forward=forward, inverse=inverse_with_source, source=source)
    assert source_from_fwd.target == target_from_inv.target


def test_bijection_composed_with_scale():
    earlier = ScaleTransform((2.0,))
    bijection = BijectionTransform(forward=ScaleTransform((2.0,)), inverse=ScaleTransform((0.5,)))

    composed = bijection.composed_with(earlier)
    assert composed == BijectionTransform(forward=ScaleTransform((4.0,)), inverse=ScaleTransform((0.25,)))


def test_bijection_composed_with_bijection():
    earlier = BijectionTransform(forward=ScaleTransform((2.0,)), inverse=ScaleTransform((0.5,)))
    later = BijectionTransform(forward=ScaleTransform((2.0,)), inverse=ScaleTransform((0.5,)))

    composed = later.composed_with(earlier)
    assert isinstance(composed, BijectionTransform)
    assert composed.forward == later.forward.composed_with(earlier.forward)
    assert composed.inverse == earlier.inverse.composed_with(later.inverse)


def test_bijection_composition_commutes_with_inversion():
    """(E∘F) ** −1 = (F ** −1) ∘ (E ** −1) (This should really hold for any combination of composable transforms)"""
    earlier = BijectionTransform(forward=ScaleTransform((2.0,)), inverse=ScaleTransform((0.5,)))
    later = BijectionTransform(forward=ScaleTransform((3.0,)), inverse=ScaleTransform((1 / 3,)))

    composed = later.composed_with(earlier)
    assert composed is not None

    compose_then_invert = composed.inverted()
    invert_then_compose = earlier.inverted().composed_with(later.inverted())

    assert compose_then_invert == invert_then_compose


def test_bijection_rejects_mismatching_child_endpoint():
    source = _sys_ref("source", "cyx")
    other_source = _sys_ref("source", "cyx")  # value-equal but not identical

    with pytest.raises(ValueError, match="BijectionTransform endpoint does not match parent endpoint"):
        _ = BijectionTransform(
            forward=IdentityTransform(source=source), inverse=IdentityTransform(), source=other_source
        )


def test_bijection_transform_validates_inverse_dimensionality():
    with pytest.raises(ValueError, match="forward and inverse dimensionality disagree"):
        BijectionTransform(
            forward=ScaleTransform(scale=(1, 1, 1)),
            inverse=ScaleTransform(scale=(1, 1)),
        )


def test_bijection_rejects_project_axis():
    with pytest.raises(ValueError, match="ProjectAxisTransforms cannot be used in BijectionTransform"):
        _ = BijectionTransform(forward=ProjectAxisTransform(), inverse=ProjectAxisTransform())


@pytest.mark.parametrize(
    "children",
    [
        pytest.param(
            (
                _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=IdentityTransform()),
                _ByDimensionChild(source_indices=(0,), target_indices=(1,), transform=IdentityTransform()),
            ),
            id="duplicate-source-axis",
        ),
        pytest.param(
            (
                _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=IdentityTransform()),
                _ByDimensionChild(source_indices=(2,), target_indices=(1,), transform=IdentityTransform()),
            ),
            id="non-contiguous-source-axes",
        ),
    ],
)
def test_by_dimension_accepts_duplicate_or_non_consecutive_source_indices(children):
    _ = ByDimensionTransform(transforms=children)


@pytest.mark.parametrize(
    ("children", "expected_error"),
    [
        pytest.param((), "requires at least one child transformation", id="empty"),
        pytest.param(
            (
                _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=IdentityTransform()),
                _ByDimensionChild(source_indices=(1,), target_indices=(0,), transform=IdentityTransform()),
            ),
            "must be globally unique",
            id="duplicate-target-axis",
        ),
        pytest.param(
            (
                _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=IdentityTransform()),
                _ByDimensionChild(source_indices=(1,), target_indices=(2,), transform=IdentityTransform()),
            ),
            "must include all zero-based indices",
            id="non-contiguous-target-axes",
        ),
    ],
)
def test_by_dimension_rejects_duplicate_or_non_consecutive_target_indices(children, expected_error):
    with pytest.raises(ValueError, match=str(expected_error)):
        _ = ByDimensionTransform(transforms=children)


@pytest.mark.parametrize(
    ("source_indices", "target_indices", "transform"),
    [
        pytest.param((0, 0), (0, 1), IdentityTransform(), id="duplicate sources"),
        pytest.param((0, 1), (0, 0), IdentityTransform(), id="duplicate targets"),
        pytest.param((0,), (0,), ScaleTransform((1.0, 2.0)), id="source ndim too small"),
        pytest.param((0, 1, 2), (0, 1), ScaleTransform((1.0, 2.0)), id="source ndim too large"),
        pytest.param((0, 1), (0,), ScaleTransform((1.0, 2.0)), id="target ndim too small"),
        pytest.param((0, 1), (0, 1, 2), ScaleTransform((1.0, 2.0)), id="target ndim too large"),
    ],
)
def test_by_dimension_item_rejects_axes_mismatching_child(source_indices, target_indices, transform):
    # TODO: Here we would in particular want to test ProjectAxisTransform because its ndim delta needs unique
    #  validation - but that validation doesn't exist yet.
    with pytest.raises(ValueError):
        _ByDimensionChild(
            source_indices=source_indices,
            target_indices=target_indices,
            transform=cast(Transform, transform),
        )


def test_by_dimension_accepts_excess_axes_on_bound_source():
    """Dropping axes is canonically done through ProjectAxisTransform,
    but simply ignoring source axes in ByDimension is also valid.
    The only requirement is that all *target* axes must be produced by a child."""
    source = _sys_ref("source", "zyx")
    target = _sys_ref("target", "yx")

    item = _ByDimensionChild(source_indices=(1, 2), target_indices=(0, 1), transform=IdentityTransform())
    _ = ByDimensionTransform(transforms=(item,), source=source, target=target)


def test_by_dimension_rejects_sourcing_more_axes_than_bound():
    bad_source = _sys_ref("source", "x")
    target = _sys_ref("target", "yx")
    item = _ByDimensionChild(source_indices=(1, 2), target_indices=(0, 1), transform=ScaleTransform(scale=(1, 1)))

    with pytest.raises(ValueError, match="input axis outside source axes"):
        ByDimensionTransform(transforms=(item,), source=bad_source, target=target)


def test_by_dimension_rejects_targeting_fewer_axes_than_bound():
    source = _sys_ref("source", "zyx")
    bad_target = _sys_ref("target", "zyx")
    item = _ByDimensionChild(source_indices=(1, 2), target_indices=(0, 1), transform=ScaleTransform(scale=(1, 1)))

    with pytest.raises(ValueError, match="must cover target axes"):
        ByDimensionTransform(transforms=(item,), source=source, target=bad_target)


def test_by_dimension_inverted_swaps_item_axes_and_inverts_nested_transforms():
    transform = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(0, 1), target_indices=(0, 1), transform=ScaleTransform((2.0, 3.0))),
            _ByDimensionChild(source_indices=(2,), target_indices=(2,), transform=TranslationTransform((5.0,))),
            _ByDimensionChild(source_indices=(3, 4), target_indices=(4, 3), transform=IdentityTransform()),
        ),
    )

    assert transform.inverted() == ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(0, 1), target_indices=(0, 1), transform=ScaleTransform((0.5, 1 / 3))),
            _ByDimensionChild(source_indices=(2,), target_indices=(2,), transform=TranslationTransform((-5.0,))),
            _ByDimensionChild(source_indices=(4, 3), target_indices=(3, 4), transform=IdentityTransform()),
        ),
    )


def test_by_dimension_inverted_rejects_non_invertible_transform():
    transform = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(
                source_indices=(0, 1),
                target_indices=(0,),
                transform=ProjectAxisTransform(drops=(0,), inserts=()),
            ),
        ),
    )

    with pytest.raises(ValueError, match="not invertible"):
        transform.inverted()


def test_by_dimension_double_inversion_returns_original_transform():
    transform = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(0, 1), target_indices=(0, 1), transform=ScaleTransform((2.0, 3.0))),
            _ByDimensionChild(source_indices=(2,), target_indices=(2,), transform=TranslationTransform((5.0,))),
        ),
    )

    assert transform.inverted().inverted() == transform


def test_by_dimension_composes_itemwise_when_matching():
    earlier = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=IdentityTransform()),
            _ByDimensionChild(source_indices=(1,), target_indices=(1,), transform=ScaleTransform((2.0,))),
            _ByDimensionChild(source_indices=(2,), target_indices=(2,), transform=TranslationTransform((2.0,))),
            _ByDimensionChild(
                source_indices=(3,), target_indices=(3, 4), transform=ProjectAxisTransform(drops=(), inserts=(1,))
            ),
        ),
    )

    later = ByDimensionTransform(
        transforms=(  # order of the children shouldn't matter; earlier's targets and later's sources match
            _ByDimensionChild(source_indices=(2,), target_indices=(2,), transform=TranslationTransform((3.0,))),
            _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=IdentityTransform()),
            _ByDimensionChild(
                source_indices=(3, 4), target_indices=(3,), transform=ProjectAxisTransform(drops=(1,), inserts=())
            ),
            _ByDimensionChild(source_indices=(1,), target_indices=(1,), transform=ScaleTransform((3.0,))),
        ),
    )

    expected = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(2,), target_indices=(2,), transform=TranslationTransform((5.0,))),
            _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=IdentityTransform()),
            _ByDimensionChild(
                source_indices=(3,), target_indices=(3,), transform=ProjectAxisTransform(drops=(), inserts=())
            ),
            _ByDimensionChild(source_indices=(1,), target_indices=(1,), transform=ScaleTransform((6.0,))),
        ),
    )

    assert later.composed_with(earlier) == expected


@pytest.mark.parametrize(
    "earlier",
    [
        pytest.param(
            ByDimensionTransform(
                transforms=(_ByDimensionChild((0,), (0,), ScaleTransform((2.0,))),),
            ),
            id="missing matching item",
        ),
        pytest.param(
            ByDimensionTransform(
                transforms=(
                    _ByDimensionChild((0,), (0,), IdentityTransform()),
                    _ByDimensionChild((1,), (1,), IdentityTransform()),
                    _ByDimensionChild((2,), (2,), IdentityTransform()),
                ),
            ),
            id="unused earlier item",
        ),
        pytest.param(
            ByDimensionTransform(
                transforms=(
                    _ByDimensionChild((0,), (0,), ProjectAxisTransform(drops=(0,), inserts=(0,))),
                    _ByDimensionChild((1,), (1,), IdentityTransform()),
                ),
            ),
            id="project axis does not compose with scale",
        ),
    ],
)
def test_by_dimension_fails_invalid_itemwise_composition(earlier):
    later = ByDimensionTransform(
        transforms=(
            _ByDimensionChild((0,), (0,), ScaleTransform((2.0,))),
            _ByDimensionChild((1,), (1,), IdentityTransform()),
        ),
    )

    assert later.composed_with(cast(ByDimensionTransform, earlier)) is None


def test_by_dimension_fails_composition_of_project_target_into_non_identity():
    earlier = ProjectAxisTransform(drops=(), inserts=(1,))
    later = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(
                source_indices=(0, 1),  # sources an axis inserted by projectAxis
                target_indices=(0, 1),
                transform=ScaleTransform((2.0, 3.0)),
            ),
        ),
    )
    # The equivalent composition would be as below. This isn't necessarily better or simpler, so instead we refuse.
    # composed = ByDimensionTransform(
    #     transforms=(
    #         _ByDimensionChild(
    #             source_indices=(0,),
    #             target_indices=(0, 1),
    #             transform=TransformSequence(
    #                 (
    #                     ProjectAxisTransform(drops=(), inserts=(1,)),
    #                     ScaleTransform((2.0, 3.0)),
    #                 )
    #             ),
    #         ),
    #     ),
    # )
    assert later.composed_with(earlier) is None


@pytest.mark.parametrize(
    ("earlier", "later", "expected"),
    [
        pytest.param(
            ProjectAxisTransform(drops=(0,), inserts=()),
            ByDimensionTransform((_ByDimensionChild((0, 1), (0, 1), ScaleTransform((2, 3))),)),
            ByDimensionTransform((_ByDimensionChild((1, 2), (0, 1), ScaleTransform((2, 3))),)),
            id="drop leading axis increments all source indices",
        ),
        pytest.param(
            ProjectAxisTransform(drops=(1,), inserts=()),
            ByDimensionTransform((_ByDimensionChild((0, 1), (0, 1), ScaleTransform((2, 3))),)),
            ByDimensionTransform((_ByDimensionChild((0, 2), (0, 1), ScaleTransform((2, 3))),)),
            id="drop middle axis increments only subsequent source indices",
        ),
        pytest.param(
            ProjectAxisTransform(drops=(1, 3), inserts=()),
            ByDimensionTransform((_ByDimensionChild((0, 1, 2), (0, 1, 2), ScaleTransform((2, 3, 4))),)),
            ByDimensionTransform((_ByDimensionChild((0, 2, 4), (0, 1, 2), ScaleTransform((2, 3, 4))),)),
            id="multiple drops increment source indices cumulatively",
        ),
        pytest.param(
            # Composing this insert before byDim with only (0, 1) targets implies dropping the just-inserted axis
            ProjectAxisTransform(drops=(), inserts=(0,)),
            ByDimensionTransform((_ByDimensionChild((1, 2), (0, 1), ScaleTransform((2, 3))),)),
            ByDimensionTransform((_ByDimensionChild((0, 1), (0, 1), ScaleTransform((2, 3))),)),
            id="insert leading axis decrements all source indices",
        ),
        pytest.param(
            ProjectAxisTransform(drops=(), inserts=(1,)),
            ByDimensionTransform((_ByDimensionChild((0, 2), (0, 1), ScaleTransform((2, 3))),)),
            ByDimensionTransform((_ByDimensionChild((0, 1), (0, 1), ScaleTransform((2, 3))),)),
            id="insert middle axis decrements only subsequent source indices",
        ),
        pytest.param(
            ProjectAxisTransform(drops=(), inserts=(1, 3)),
            ByDimensionTransform((_ByDimensionChild((0, 2, 4), (0, 1, 2), ScaleTransform((2, 3, 4))),)),
            ByDimensionTransform((_ByDimensionChild((0, 1, 2), (0, 1, 2), ScaleTransform((2, 3, 4))),)),
            id="multiple inserts decrement source indices cumulatively",
        ),
        pytest.param(
            ProjectAxisTransform(drops=(), inserts=(1, 3, 5)),
            ByDimensionTransform(
                (
                    _ByDimensionChild((0, 2, 4), (0, 1, 3), ScaleTransform((2, 3, 4))),
                    _ByDimensionChild((1, 5), (2, 4), IdentityTransform()),
                )
            ),
            ByDimensionTransform(
                (
                    _ByDimensionChild((0, 1, 2), (0, 1, 3), ScaleTransform((2, 3, 4))),
                    _ByDimensionChild((), (2, 4), ProjectAxisTransform(inserts=(0, 1))),
                )
            ),
            id="inserts sourced by later identity remain inserts in composition",
        ),
    ],
)
def test_by_dimension_composes_after_project_axis(earlier, later, expected):
    later = cast(ByDimensionTransform, later)
    earlier = cast(ByDimensionTransform, earlier)
    assert later.composed_with(earlier) == expected


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
