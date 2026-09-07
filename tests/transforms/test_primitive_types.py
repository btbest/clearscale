from typing import cast, Tuple

import pytest
from clearscale._services.ome_zarr import MultiscaleTransforms
from clearscale._transforms import (
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
    NodeRef,
)
from clearscale._transforms._transform_types import IDENTITY_TOLERANCE


def _sys_ref(name, axes) -> NodeRef[CoordinateSystem]:
    return CoordinateSystem.without_semantics(axes).as_ref(name)


def test_identity_bound_rejects_endpoint_axis_count_mismatch():
    source = _sys_ref("source", "yx")
    target = _sys_ref("target", "zyx")

    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        IdentityTransform().bound(source=source, target=target)


def test_path_backed_scale_round_trips_and_no_ndim():
    scale = Transform.from_ome_zarr({"type": "scale", "path": "coordinateTransformations/scale"})

    assert isinstance(scale, ScaleTransform)
    assert scale._ndim_by_payload().delta == 0
    assert not scale.is_invertible
    assert scale.to_ome_zarr("0.6.rc0") == {"type": "scale", "path": "coordinateTransformations/scale"}


def test_scale_bound_rejects_axis_count_mismatch():
    world = _sys_ref("world", "yx")

    with pytest.raises(ValueError, match="ScaleTransform expects 3 source axes"):
        ScaleTransform(scale=(1, 1, 1)).bound(source=world, target=world)


def test_scale_composed_with_identity():
    earlier_t = IdentityTransform()
    later_t = ScaleTransform(scale=(5, 7, 11))

    composed = later_t.composed_with(earlier_t)
    assert isinstance(composed, ScaleTransform)
    assert composed == later_t


def test_scale_composed_with_scale():
    earlier_t = ScaleTransform(scale=(2, 3, 4))
    later_t = ScaleTransform(scale=(5, 7, 11))

    composed = later_t.composed_with(earlier_t)
    assert isinstance(composed, ScaleTransform)
    assert composed == ScaleTransform(scale=(10, 21, 44))


def test_scale_composed_with_translation():
    earlier_t = TranslationTransform(translation=(5, 7))
    later_t = ScaleTransform(scale=(2, 3))

    composed = later_t.composed_with(earlier_t)
    assert isinstance(composed, AffineTransform)
    assert composed == AffineTransform(affine=((2, 0, 10), (0, 3, 21)))


def test_scale_composed_with_rotation():
    earlier_t = RotationTransform(rotation=((0, -1), (1, 0)))
    later_t = ScaleTransform(scale=(2, 3))

    composed = later_t.composed_with(earlier_t)
    assert isinstance(composed, AffineTransform)
    assert composed == AffineTransform(affine=((0, -2, 0), (3, 0, 0)))


@pytest.mark.parametrize(
    "map_axis,scale,expected",
    [
        pytest.param((1, 0), (2, 3), ((0, 2, 0), (3, 0, 0)), id="2d swap"),
        pytest.param((2, 0, 1), (2, 3, 5), ((0, 0, 2, 0), (3, 0, 0, 0), (0, 5, 0, 0)), id="3d cycle"),
        pytest.param((0, 2, 1), (2, 3, 5), ((2, 0, 0, 0), (0, 0, 3, 0), (0, 5, 0, 0)), id="3d swap"),
    ],
)
def test_scale_composed_with_map_axis(map_axis, scale, expected):
    earlier_t = MapAxisTransform(map_axis=map_axis)
    later_t = ScaleTransform(scale=scale)

    composed = later_t.composed_with(earlier_t)
    assert isinstance(composed, AffineTransform)
    assert composed == AffineTransform(affine=expected)


@pytest.mark.parametrize(
    "drops,inserts,scale,expected",
    [
        pytest.param((), (), (2, 3), ((2, 0, 0), (0, 3, 0)), id="noop"),
        pytest.param((1,), (), (2, 3), ((2, 0, 0, 0), (0, 0, 3, 0)), id="drop"),
        pytest.param((), (0,), (2, 3, 5), ((0, 0, 0), (3, 0, 0), (0, 5, 0)), id="insert"),
        pytest.param((1,), (0,), (2, 3, 5), ((0, 0, 0, 0), (3, 0, 0, 0), (0, 0, 5, 0)), id="drop and insert"),
    ],
)
def test_scale_composed_with_project_axis(drops, inserts, scale, expected):
    earlier_t = ProjectAxisTransform(drops=drops, inserts=inserts)
    later_t = ScaleTransform(scale=scale)

    composed = later_t.composed_with(earlier_t)

    assert composed == AffineTransform(affine=expected)


@pytest.mark.parametrize(
    "earlier,later",
    [
        pytest.param(
            ScaleTransform(scale=(), _ome_zarr_path="foo"), ScaleTransform(scale=(2,)), id="earlier unloaded scale"
        ),
        pytest.param(
            ScaleTransform(scale=(2,)), ScaleTransform(scale=(), _ome_zarr_path="foo"), id="later unloaded scale"
        ),
        pytest.param(
            TranslationTransform(translation=(), _ome_zarr_path="foo"),
            ScaleTransform(scale=(2,)),
            id="earlier unloaded translation",
        ),
        pytest.param(
            RotationTransform(_ome_zarr_path="foo"), ScaleTransform(scale=(2,)), id="earlier unloaded rotation"
        ),
        pytest.param(ScaleTransform(scale=(2, 3)), ScaleTransform(scale=(5,)), id="scale ndim mismatch"),
        pytest.param(
            TranslationTransform(translation=(2, 3)), ScaleTransform(scale=(5,)), id="translation ndim mismatch"
        ),
        pytest.param(
            RotationTransform(rotation=((1, 0), (0, 1))), ScaleTransform(scale=(5,)), id="rotation ndim mismatch"
        ),
        pytest.param(
            CoordinatesTransform(path="coords.zarr"), ScaleTransform(scale=(2,)), id="some other transform type"
        ),
    ],
)
def test_scale_composed_with_rejects_unsupported_or_invalid(earlier: Transform, later: Transform):
    assert later.composed_with(earlier) is None


def test_scale_simplified_within_tolerance_identity():
    transform = ScaleTransform(scale=(1, 1 + IDENTITY_TOLERANCE / 2, 1 - IDENTITY_TOLERANCE / 2))
    simplified = transform.simplified()
    assert simplified == IdentityTransform()


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param((1 + IDENTITY_TOLERANCE * 2,), id="outside tolerance above"),
        pytest.param((1 - IDENTITY_TOLERANCE * 2,), id="outside tolerance below"),
        pytest.param((1, 2), id="one identity one nonidentity"),
    ],
)
def test_scale_simplified_keeps_nonidentity(scale):
    transform = ScaleTransform(scale=scale)
    simplified = transform.simplified()
    assert simplified is transform


def test_scale_zero_does_not_roundtrip_through_affine():
    zero_scale = ScaleTransform((0,))
    assert zero_scale._to_affine_transform().simplified() == ProjectAxisTransform(drops=(0,), inserts=(0,))


def test_translation_bound_rejects_endpoint_axis_count_mismatch():
    source = _sys_ref("source", "yx")
    target = _sys_ref("target", "zyx")

    with pytest.raises(ValueError, match="TranslationTransform expects 2 target axes"):
        TranslationTransform(translation=(0, 0)).bound(source=source, target=target)


def test_path_backed_translation_round_trips_and_no_ndim():
    translation = Transform.from_ome_zarr({"type": "translation", "path": "coordinateTransformations/translation"})
    assert isinstance(translation, TranslationTransform)
    assert translation._ndim_by_payload().delta == 0
    assert not translation.is_invertible
    assert translation.to_ome_zarr("0.6.rc0") == {
        "type": "translation",
        "path": "coordinateTransformations/translation",
    }


def test_translation_composed_with_translation():
    earlier_t = TranslationTransform(translation=(2, 3, 5))
    later_t = TranslationTransform(translation=(7, 11, 13))

    composed = later_t.composed_with(earlier_t)
    assert isinstance(composed, TranslationTransform)
    assert composed == TranslationTransform(translation=(9, 14, 18))


def test_translation_composed_with_scale():
    earlier_t = ScaleTransform(scale=(2, 3))
    later_t = TranslationTransform(translation=(5, 7))

    composed = later_t.composed_with(earlier_t)
    assert isinstance(composed, AffineTransform)
    assert composed == AffineTransform(affine=((2, 0, 5), (0, 3, 7)))


def test_translation_composed_with_rotation():
    earlier_t = RotationTransform(rotation=((0, -1), (1, 0)))
    later_t = TranslationTransform(translation=(3, 5))

    composed = later_t.composed_with(earlier_t)
    assert isinstance(composed, AffineTransform)
    assert composed == AffineTransform(affine=((0, -1, 3), (1, 0, 5)))


@pytest.mark.parametrize(
    "earlier,later",
    [
        pytest.param(
            TranslationTransform(translation=(), _ome_zarr_path="foo"),
            TranslationTransform(translation=(2,)),
            id="earlier unloaded translation",
        ),
        pytest.param(
            TranslationTransform(translation=(2,)),
            TranslationTransform(translation=(), _ome_zarr_path="foo"),
            id="later unloaded translation",
        ),
        pytest.param(
            ScaleTransform(scale=(), _ome_zarr_path="foo"),
            TranslationTransform(translation=(2,)),
            id="earlier unloaded scale",
        ),
        pytest.param(
            RotationTransform(_ome_zarr_path="foo"),
            TranslationTransform(translation=(2,)),
            id="earlier unloaded rotation",
        ),
        pytest.param(
            TranslationTransform(translation=(2, 3)),
            TranslationTransform(translation=(5,)),
            id="translation ndim mismatch",
        ),
        pytest.param(ScaleTransform(scale=(2, 3)), TranslationTransform(translation=(5,)), id="scale ndim mismatch"),
        pytest.param(
            RotationTransform(rotation=((1, 0), (0, 1))),
            TranslationTransform(translation=(5,)),
            id="rotation ndim mismatch",
        ),
        pytest.param(
            CoordinatesTransform(path="coords.zarr"),
            TranslationTransform(translation=(2,)),
            id="some other transform type",
        ),
    ],
)
def test_translation_composed_with_rejects_unsupported_or_invalid(earlier: Transform, later: Transform):
    assert later.composed_with(earlier) is None


def test_translation_simplified_within_tolerance_identity():
    transform = TranslationTransform(translation=(0, IDENTITY_TOLERANCE / 2, IDENTITY_TOLERANCE / -2))
    simplified = transform.simplified()
    assert simplified == IdentityTransform()


@pytest.mark.parametrize(
    "translation",
    [
        pytest.param((IDENTITY_TOLERANCE * 2,), id="outside tolerance above"),
        pytest.param((IDENTITY_TOLERANCE * -2,), id="outside tolerance below"),
        pytest.param((0, -1), id="one identity one nonidentity"),
    ],
)
def test_translation_simplified_keeps_nonidentity(translation):
    transform = TranslationTransform(translation=translation)
    simplified = transform.simplified()
    assert simplified is transform


def test_rotation_bound_rejects_axis_count_mismatch():
    world = _sys_ref("world", "yx")

    with pytest.raises(ValueError, match="RotationTransform expects 3 source axes"):
        RotationTransform(rotation=((0, 1, 0), (-1, 0, 0), (0, 0, 1))).bound(source=world, target=world)


def test_path_backed_rotation_round_trips_and_no_ndim():
    rotation = Transform.from_ome_zarr({"type": "rotation", "path": "coordinateTransformations/rotation"})
    assert isinstance(rotation, RotationTransform)
    assert rotation._ndim_by_payload().delta == 0
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
def test_rotation_composed_with_rotation(earlier, later, expected):
    composed = RotationTransform(rotation=later).composed_with(RotationTransform(rotation=earlier))

    assert cast(RotationTransform, composed).rotation == expected


@pytest.mark.parametrize(
    "earlier, expected",
    [
        (ScaleTransform(scale=(2, 3)), ((0, -3, 0), (2, 0, 0))),
        (TranslationTransform((1, 2)), ((0, -1, -2), (1, 0, 1))),
        (AffineTransform(((2, 0, 3), (0, 4, 5))), ((0, -4, -5), (2, 0, 3))),
    ],
)
def test_rotation_composed_with_other_affine_representable_transforms(earlier, expected):
    composed = RotationTransform(((0, -1), (1, 0))).composed_with(earlier)
    assert isinstance(composed, AffineTransform)


def test_path_backed_affine_round_trips_and_no_ndim():
    affine = Transform.from_ome_zarr({"type": "affine", "path": "coordinateTransformations/affine"})
    assert isinstance(affine, AffineTransform)
    assert affine._ndim_by_payload().is_unconstrained()
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
        ((0, 0, 0),),  # 2d -> 1d
        (
            (0, 0),
            (0, 0),
        ),  # 1d -> 2d
        (
            (0, 0),
            (0, 0),
            (0, 0),
        ),  # 1d -> 3d
        (
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        ),  # 2d -> 3d
        (
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
        ),  # 2d -> 4d
    ],
)
def test_affine_instantiates_with_any_rectangular(matrix):
    """The bottom row (0,...0, 1) of the homogenous form should not be included"""
    _ = AffineTransform(affine=matrix)


@pytest.mark.parametrize(
    "matrix, expected_error",
    [
        ((0,), "Expected 2D array"),
        (((0,),), "at least one input dimension and one offset column"),
        (((0,), (0,)), "at least one input dimension and one offset column"),
        (((0,), (0, 1)), "Expected rectangular 2D array"),
        (
            (
                (0, 0),
                (0, 0, 0),
                (0, 1),
            ),
            "Expected rectangular 2D array",
        ),
    ],
)
def test_affine_rejects_non_rectangular_or_too_few_cols(matrix, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        _ = AffineTransform(affine=matrix)


@pytest.mark.parametrize(
    "matrix, source_axes, target_axes",
    [
        (((0, 0),), "x", "x"),
        (((0, 0, 0), (0, 0, 0)), "xy", "xy"),
        (((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), "xyz", "xyz"),
        (((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)), "cxyz", "cxyz"),
        (((0, 0, 0),), "xy", "x"),
        (((0, 0), (0, 0)), "x", "xy"),
        (((0, 0), (0, 0), (0, 0)), "x", "xyz"),
        (((0, 0, 0, 0),), "xyz", "x"),
        (((0, 0, 0), (0, 0, 0), (0, 0, 0)), "xy", "xyz"),
        (((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)), "xy", "cxyz"),
    ],
)
def test_affine_bound_accepts_matching_axis_counts(matrix, source_axes, target_axes):
    source = _sys_ref("source", source_axes)
    target = _sys_ref("target", target_axes)
    transform = AffineTransform(affine=matrix).bound(source=source, target=target)
    assert transform.source is source
    assert transform.target == target


@pytest.mark.parametrize(
    "matrix, source_axes, target_axes, expected_err",
    [
        (((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), "x", "xyz", "3 source"),
        (((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), "xyz", "x", "3 target"),
        (((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), "cxyz", "xy", "3 source"),
        (((0, 0), (0, 0), (0, 0)), "xy", "xyz", "1 source"),
        (((0, 0), (0, 0), (0, 0)), "x", "xy", "3 target"),
        (((0, 0), (0, 0), (0, 0)), "xy", "xy", "1 source"),
        (((0, 0), (0, 0), (0, 0)), "cxyz", "cxyz", "1 source"),
        (((0, 0, 0, 0),), "xy", "x", "3 source"),
        (((0, 0, 0, 0),), "xyz", "xy", "1 target"),
        (((0, 0, 0, 0),), "xy", "xy", "3 source"),
        (((0, 0, 0, 0),), "xy", "xy", "3 source"),
    ],
)
def test_affine_bound_rejects_mismatching_axis_counts(matrix, source_axes, target_axes, expected_err):
    source = _sys_ref("source", source_axes)
    target = _sys_ref("target", target_axes)
    with pytest.raises(ValueError, match=f"AffineTransform expects {expected_err}"):
        AffineTransform(affine=matrix).bound(source=source, target=target)


def test_affine_transform_inverted():
    transform = AffineTransform(affine=((2, 0, 3), (0, 4, 8)))
    assert transform.inverted().affine == ((0.5, 0.0, -1.5), (0.0, 0.25, -2.0))
    assert transform.inverted().inverted() == transform, "double inversion is always eq input"


@pytest.mark.parametrize(
    "matrix",
    [
        ((1, 0, 0), (0, 0, 0)),
        ((1, 2, 0), (2, 4, 0)),
        ((1, 0, 0),),
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
            ((1, 0), (0, 1)),
            ((1, 0, 0),),
            ((1, 0),),
            id="1->2 then 2->1",
        ),
        pytest.param(
            ((1, 0), (0, 0), (0, 0)),
            ((1, 0, 0, 0), (0, 0, 0, 0)),
            ((1, 0), (0, 0)),
            id="1->3 then 3->2",
        ),
        pytest.param(
            ((1, 0, 0),),
            ((1, 0), (0, 1), (0, 0)),
            ((1, 0, 0), (0, 0, 1), (0, 0, 0)),
            id="2->1 then 1->3",
        ),
        pytest.param(
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0)),
            id="square",
        ),
    ],
)
def test_affine_composed_with_matches_earlier_output_to_later_input_ndim(earlier, later, expected):
    earlier_t = AffineTransform(affine=earlier)
    later_t = AffineTransform(affine=later)

    composed = later_t.composed_with(earlier_t)

    assert composed is not None


@pytest.mark.parametrize(
    "earlier,later",
    [
        pytest.param(
            ((1, 0), (0, 1)),
            ((1, 0), (0, 1)),
            id="1->2 then 1->2",
        ),
        pytest.param(
            ((1, 0), (0, 1), (0, 0)),
            ((1, 0), (0, 1)),
            id="1->3 then 1->2",
        ),
        pytest.param(
            ((1, 0, 0),),
            ((1, 0, 0),),
            id="2->1 then 2->1",
        ),
        pytest.param(
            ((1, 0, 0), (0, 1, 0)),
            ((1, 0), (0, 1), (0, 0)),
            id="2->2 then 1->3",
        ),
    ],
)
def test_affine_composed_with_rejects_mismatching_dims(earlier, later):
    earlier_t = AffineTransform(affine=earlier)
    later_t = AffineTransform(affine=later)

    assert later_t.composed_with(earlier_t) is None


@pytest.mark.parametrize(
    "earlier,later,expected",
    [
        pytest.param(
            ((2, 0, 0), (0, 3, 0)),
            ((4, 0, 0), (0, 5, 0)),
            ScaleTransform((8, 15)),
            id="scale_only",
        ),
        pytest.param(
            ((0.25, 0, 7), (0, 1, 3)),
            ((4, 0, 2.2), (0, 1, 5)),
            TranslationTransform((30.2, 8.0)),  # (4*7 + 2.2, 1*3 + 5)
            id="scale_composes_to_identity",
        ),
        pytest.param(
            ((1, 0, 7), (0, 2, 3)),
            ((4, 0, -28), (0, 1, -3)),  # (4*7 - 28 = 0, 1*3 - 3 = 0)
            ScaleTransform((4, 2)),
            id="translation_composes_to_identity_plus_scale",
        ),
        pytest.param(
            ((0, -1, 7), (1, 0, 3)),
            ((1, 0, -7), (0, 1, -3)),
            RotationTransform(((0, -1), (1, 0))),
            id="translation_composes_to_identity_plus_rotation",
        ),
        pytest.param(
            ((1, 0, 0, 0), (0, 0, 1, 0)),
            ((1, 0, 0), (0, 1, 0)),
            ProjectAxisTransform(drops=(1,)),
            id="project_axis",
        ),
        pytest.param(
            ((0, 1, 0), (1, 0, 0)),
            ((1, 0, 0), (0, 1, 0)),
            MapAxisTransform((1, 0)),
            id="map_axis",
        ),
    ],
)
def test_affine_composed_with_preserves_affine_and_simplified_returns_special_case(earlier, later, expected):
    earlier_t = AffineTransform(affine=earlier)
    later_t = AffineTransform(affine=later)
    composed = later_t.composed_with(earlier_t)

    assert isinstance(composed, AffineTransform)
    assert composed.simplified() == expected


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


@pytest.mark.parametrize(
    "earlier,later,expected",
    [
        pytest.param(
            MapAxisTransform((1, 0)),
            AffineTransform(((2, 0, 5), (0, 3, 7))),
            AffineTransform(((0, 2, 5), (3, 0, 7))),
            id="map then affine",
        ),
        pytest.param(
            AffineTransform(((2, 0, 5), (0, 3, 7))),
            MapAxisTransform((1, 0)),
            AffineTransform(((0, 3, 7), (2, 0, 5))),
            id="affine then map",
        ),
        pytest.param(
            ProjectAxisTransform(drops=(0, 2)),
            AffineTransform(((2, 0, 5), (0, 3, 7))),
            AffineTransform(((0, 2, 0, 0, 5), (0, 0, 0, 3, 7))),
            id="project then affine",
        ),
        pytest.param(
            AffineTransform(((2, 0, 0, 5), (0, 3, 0, 7), (0, 0, 5, 11))),
            ProjectAxisTransform(drops=(1,)),
            AffineTransform(((2, 0, 0, 5), (0, 0, 5, 11))),
            id="affine then project",
        ),
    ],
)
def test_affine_map_and_project_axis_compose_in_both_directions(earlier: Transform, later: Transform, expected):
    assert later.composed_with(earlier) == expected


@pytest.mark.parametrize(
    "linear,expected_simplified",
    [
        pytest.param(
            ((1, 0, 0), (0, 0, 1)),
            ProjectAxisTransform(drops=(1,)),
            id="non_square_missing_row_matching_zero_col",
        ),
        pytest.param(
            ((1, 0, 0), (0, 1, 1)),
            TransformSequence(
                (AffineTransform(((1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0))), ProjectAxisTransform(drops=(2,)))
            ),
            id="non_square_missing_row_no_matching_zero_col",
        ),
        pytest.param(
            ((1, 0, 0), (0, 0, 0)),
            ProjectAxisTransform(drops=(1, 2), inserts=(1,)),
            id="non_square_missing_row_excess_zero_col",
        ),
        pytest.param(
            ((1, 0), (0, 0), (0, 1)),
            ProjectAxisTransform(inserts=(1,)),
            id="non_square_missing_col_matching_zero_row",
        ),
        pytest.param(
            ((1, 0), (1, 0), (0, 1)),
            TransformSequence(
                (ProjectAxisTransform(inserts=(2,)), AffineTransform(((1, 0, 0, 0), (1, 0, 0, 0), (0, 1, 1, 0))))
            ),
            id="non_square_missing_col_no_matching_zero_row",
        ),
        pytest.param(
            ((1, 0), (0, 0), (0, 0)),
            ProjectAxisTransform(drops=(1,), inserts=(1, 2)),
            id="non_square_missing_col_excess_zero_row",
        ),
        pytest.param(
            ((1, 0, 0), (0, 0, 0), (0, 0, 1)),
            ProjectAxisTransform(drops=(1,), inserts=(1,)),
            id="square_zero_scale",
        ),
        pytest.param(
            ((1, 0, 0), (0, 0, 1), (0, 0, 1)),
            AffineTransform(((1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0))),
            id="square_zero_col_only",
        ),
        pytest.param(
            ((1, 0, 0), (0, 0, 0), (0, 1, 1)),
            AffineTransform(((1, 0, 0, 0), (0, 0, 0, 0), (0, 1, 1, 0))),
            id="square_zero_row_only",
        ),
        pytest.param(
            ((1, 0, 0), (0, 0, 0), (1, 0, 0)),
            AffineTransform(((1, 0, 0, 0), (0, 0, 0, 0), (1, 0, 0, 0))),
            id="square_zero_scale_excess_col",
        ),
        pytest.param(
            ((1, 0, 1), (0, 0, 0), (0, 0, 0)),
            AffineTransform(((1, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))),
            id="square_zero_scale_excess_row",
        ),
    ],
)
def test_affine_simplified_decomposes_project_axis(linear, expected_simplified):
    affine = tuple(row + (0,) for row in linear)  # add 0 translations
    simplified = AffineTransform(affine).simplified()
    assert simplified == expected_simplified


@pytest.mark.parametrize(
    "linear",
    [
        pytest.param(
            ((1, 0, 0), (0, 0, 0), (0, 1, 1)),
            id="square_zero_row_only",
        ),
        pytest.param(
            ((1, 0, 0), (0, 0, 1), (0, 0, 1)),
            id="square_zero_col_only",
        ),
        pytest.param(
            ((1, 0, 0), (0, 0, 0), (1, 0, 0)),
            id="square_zero_scale_excess_col",
        ),
        pytest.param(
            ((1, 0, 1), (0, 0, 0), (0, 0, 0)),
            id="square_zero_scale_excess_row",
        ),
    ],
)
def test_affine_simplified_does_not_decompose_square_except_zero_scale(linear):
    # Additional variants for test_affine_simplified_decomposes_project_axis,
    # but these do *not* decompose a ProjectAxis out of the affine.
    # For the "excess" cases, one could decompose a `drop - affine - insert` sequence,
    # and the remaining middle affine would be a reduced square. Could be useful, maybe tbd.
    affine = tuple(row + (0,) for row in linear)  # add 0 translations
    affine_transform = AffineTransform(affine)
    simplified = affine_transform.simplified()
    assert simplified is affine_transform


@pytest.mark.parametrize(
    "affine,expected_simplified",
    [
        pytest.param(
            ((2, 0, 3), (0, 5, 7)),
            TransformSequence((ScaleTransform((2, 5)), TranslationTransform((3, 7)))),
            id="scale and translation",
        ),
        pytest.param(
            ((0, -1, 3), (1, 0, 7)),
            TransformSequence((RotationTransform(((0, -1), (1, 0))), TranslationTransform((3, 7)))),
            id="rotation and translation",
        ),
        pytest.param(
            ((-1, 0, 2), (0, -1, 0)),
            TransformSequence((ScaleTransform((-1, -1)), TranslationTransform((2, 0)))),
            id="reflection-scale and translation",
        ),
        pytest.param(
            ((0, -1, 3), (-1, 0, 7)),
            TransformSequence(
                (RotationTransform(((0, 1), (-1, 0))), ScaleTransform((-1, 1)), TranslationTransform((3, 7)))
            ),
            id="rotation reflection and translation",
        ),
        pytest.param(
            ((0, -2, 3), (5, 0, 7)),
            TransformSequence(
                (
                    RotationTransform(((0, -1), (1, 0))),
                    ScaleTransform((2, 5)),
                    TranslationTransform((3, 7)),
                )
            ),
            id="rotation scale and translation",
        ),
        pytest.param(
            ((0, 1, 3), (1, 0, 7)),
            TransformSequence((MapAxisTransform((1, 0)), TranslationTransform((3, 7)))),
            id="map and translation",
        ),
        pytest.param(
            ((0, 2, 3), (5, 0, 7)),
            TransformSequence((MapAxisTransform((1, 0)), ScaleTransform((2, 5)), TranslationTransform((3, 7)))),
            id="map scale and translation",
        ),
        pytest.param(
            ((1, 0, 0, 3), (0, 0, 1, 7)),
            TransformSequence((ProjectAxisTransform(drops=(1,)), TranslationTransform((3, 7)))),
            id="project and translation",
        ),
        pytest.param(
            ((0, 0, 2, 3), (5, 0, 0, 7)),
            TransformSequence(
                (
                    ProjectAxisTransform(drops=(1,)),
                    MapAxisTransform((1, 0)),
                    ScaleTransform((2, 5)),
                    TranslationTransform((3, 7)),
                )
            ),
            id="project map scale and translation",
        ),
        pytest.param(
            ((3, 0, -4, 3), (4, 0, 3, 7)),
            TransformSequence(
                (
                    ProjectAxisTransform(drops=(1,)),
                    RotationTransform(((0.6, -0.8), (0.8, 0.6))),
                    ScaleTransform((5, 5)),
                    TranslationTransform((3, 7)),
                )
            ),
            id="project rotation scale and translation",
        ),
        pytest.param(
            ((0, 0, 2), (3, 0, 3), (0, 5, 7)),
            TransformSequence(
                (
                    ProjectAxisTransform(inserts=(0,)),
                    ScaleTransform((1, 3, 5)),
                    TranslationTransform((2, 3, 7)),
                )
            ),
            id="insert scale and translation",
        ),
        pytest.param(
            ((1, 1, 3),),
            TransformSequence(
                (
                    AffineTransform(((1, 1, 0), (0, 1, 0))),
                    ProjectAxisTransform(drops=(1,)),
                    TranslationTransform((3,)),
                )
            ),
            id="general affine then dimensionality reduction",
        ),
        pytest.param(
            ((1, 3), (1, 7)),
            TransformSequence(
                (
                    ProjectAxisTransform(inserts=(1,)),
                    AffineTransform(((1, 0, 0), (1, 1, 0))),
                    TranslationTransform((3, 7)),
                )
            ),
            id="dimensionality expansion then general affine",
        ),
        pytest.param(
            ((0, 0, 0, 10), (2, 0, 0, 20), (0, 0, 3, 30)),
            TransformSequence(
                (
                    ProjectAxisTransform(drops=(1,), inserts=(0,)),
                    ScaleTransform((1, 2, 3)),
                    TranslationTransform((10, 20, 30)),
                )
            ),
            id="square drop scale insert and translation",
        ),
    ],
)
def test_affine_simplified_decomposes_sequence_and_collapsed_roundtrips(affine, expected_simplified):
    simplified = AffineTransform(affine).simplified()
    assert simplified == expected_simplified

    recomposed = expected_simplified.collapsed(raise_uncollapsed=True)
    assert isinstance(recomposed, AffineTransform)
    assert recomposed.affine is not None
    for actual_row, expected_row in zip(recomposed.affine, affine):
        assert actual_row == pytest.approx(expected_row)


@pytest.mark.parametrize(
    "affine",
    [
        pytest.param(AffineTransform(((1, 0.25, 0), (0, 1, 0))), id="shear"),
        pytest.param(AffineTransform(((1, 1, 0), (0, 0, 0))), id="zero scale is_diagonal false"),
        pytest.param(AffineTransform(_ome_zarr_path="affine.zarr"), id="path-backed scale"),
        pytest.param(AffineTransform(((0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 2, 0))), id="square but drops != inserts"),
    ],
)
def test_affine_simplified_keeps_shear_zero_scale_and_path_backed_payloads(affine: AffineTransform):
    """
    - Shears have no OME-Zarr representation.
    - Zero-scale treatment is subject to debate/change; they are technically allowed ('SHOULD be non-zero').
    - Collapsing axes (effectively dropping) is in general reserved for ProjectAxis and is not
      simplified/decomposed out of Affines. Except for the special case where the Affine's linear
      component is square, but it encodes an equal number of drops and inserts.
    """
    assert affine.simplified() is affine


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


def test_project_inverse_composes_with_self_to_identity():
    # ProjectAxis does not symmetrically invert and compose like the other transforms in the test above,
    # because it is non-invertible in one direction.
    project = ProjectAxisTransform(inserts=(0, 3))

    # Inserting some axes, then dropping them, is overall a noop
    insert_then_drop = project.inverted().composed_with(project)
    assert isinstance(insert_then_drop, ProjectAxisTransform)
    assert insert_then_drop == ProjectAxisTransform(drops=(), inserts=())
    assert isinstance(insert_then_drop.simplified(), IdentityTransform)

    # Dropping some axes, then inserting new ones in the same index, destroys information
    drop_then_insert = project.composed_with(project.inverted())
    assert isinstance(drop_then_insert, ProjectAxisTransform)
    assert drop_then_insert.simplified() == ProjectAxisTransform(drops=(0, 3), inserts=(0, 3))


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


def test_displacements_bound_rejects_changed_axes():
    displacements = DisplacementsTransform(path="t/vectors.zarr", interpolation="linear")
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "tzyx")

    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        _ = displacements.bound(source=source, target=target)


def test_displacements_bound_rejects_changed_axes_within_sequence():
    displacements = DisplacementsTransform(path="t/vectors.zarr", interpolation="linear")
    displacements2 = DisplacementsTransform(path="t/vectors2.zarr", interpolation="linear")
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "tzyx")

    seq = TransformSequence((displacements, displacements2))
    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        _ = seq.bound(source=source, target=target)


def test_map_axis_bound_rejects_changed_axes():
    transform = MapAxisTransform((4, 0, 2, 3, 1))
    source = _sys_ref("source", "tzcyx")
    target = _sys_ref("target", "tzyx")

    with pytest.raises(ValueError, match="MapAxisTransform expects 5 target axes"):
        _ = transform.bound(source=source, target=target)


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


def test_map_axis_transform_simplified_returns_identity_only_for_noops():
    assert MapAxisTransform((0, 1)).simplified() == IdentityTransform()
    assert MapAxisTransform((1, 0)).simplified() == MapAxisTransform((1, 0))
    assert ProjectAxisTransform().simplified() == IdentityTransform()
    project = ProjectAxisTransform(drops=(0,))
    assert project.simplified() is project


def test_map_axis_rejects_missing_transpose():
    with pytest.raises(ValueError, match="must include all zero-based indices"):
        _ = MapAxisTransform((0, 2))  # 1 is missing (MapAxis isn't allowed to drop)


def test_map_axis_rejects_mismatching_dims():
    source = _sys_ref("source", "cyx")
    bad_target = _sys_ref("target", "yx")

    with pytest.raises(ValueError, match="expects 3 target axes"):
        _ = MapAxisTransform((0, 1, 2), source=source, target=bad_target)


@pytest.mark.parametrize(
    "drops,inserts,source_axes,target_axes",
    [
        pytest.param((2,), (2,), "xyz", "xyz", id="noop 3d at min"),
        pytest.param((), (), "xyz", "xyz", id="noop 3d above min"),
        pytest.param((2,), (), "xyz", "xy", id="drop 1 source at min"),
        pytest.param((2,), (), "tcxyz", "tcyz", id="drop 1 source above min"),
        pytest.param((), (2,), "xy", "xyz", id="insert 1 target at min"),
        pytest.param((), (2,), "tcxy", "tczxy", id="insert 1 target above min"),
    ],
)
def test_project_axis_bound_accepts_ndim_at_and_above_highest_index(drops, inserts, source_axes, target_axes):
    source = _sys_ref("source", source_axes)
    target = _sys_ref("target", target_axes)

    transform = ProjectAxisTransform(drops=drops, inserts=inserts, source=source, target=target)

    assert transform.is_fully_bound


@pytest.mark.parametrize(
    "drops,inserts,source_axes,target_axes",
    [
        pytest.param((2,), (2,), "xy", "xy", id="noop 3d"),
        pytest.param((2,), (), "xy", "x", id="drop 1 source"),
        pytest.param((), (2,), "x", "xy", id="insert 1 target"),
    ],
)
def test_project_axis_bound_rejects_ndim_below_highest_index(drops, inserts, source_axes, target_axes):
    source = _sys_ref("source", source_axes)
    target = _sys_ref("target", target_axes)

    with pytest.raises(ValueError, match="ProjectAxisTransform requires at least 3"):
        _ = ProjectAxisTransform(drops=drops, inserts=inserts, source=source, target=target)


@pytest.mark.parametrize(
    "drops,inserts,source_axes,target_axes,expected_err",
    [
        ((2,), (2,), "txyz", "txy", "same dimensionality"),
        ((2,), (2,), "txy", "txyz", "same dimensionality"),
        ((2,), (), "xyz", "xyz", "target - source ndim = -1"),
        ((), (2,), "xyz", "xyz", "target - source ndim = 1"),
    ],
)
def test_project_axis_bound_rejects_ndim_delta_mismatch(drops, inserts, source_axes, target_axes, expected_err):
    source = _sys_ref("source", source_axes)
    target = _sys_ref("target", target_axes)

    with pytest.raises(ValueError, match=expected_err):
        _ = ProjectAxisTransform(drops=drops, inserts=inserts, source=source, target=target)


@pytest.mark.parametrize(
    "given_inserts",
    [(), (0,), (1,), (1, 3, 7)],
)
def test_project_axis_inverted(given_inserts: Tuple[int, ...]):
    project = ProjectAxisTransform(inserts=given_inserts)
    assert project.is_invertible
    inverse = project.inverted()
    assert isinstance(inverse, ProjectAxisTransform)
    assert inverse.drops == project.inserts


def test_project_axis_with_drops_is_not_invertible():
    project = ProjectAxisTransform(drops=(0,))
    assert not project.is_invertible
    with pytest.raises(ValueError, match="Axis dropping is not invertible"):
        _ = project.inverted()


@pytest.mark.parametrize(
    "earlier, later, composed",
    [  # tuples of (drops, inserts)
        pytest.param(((), ()), ((), ()), ((), ()), id="all ident"),
        pytest.param(((0,), ()), ((), ()), ((0,), ()), id="drop then ident"),
        pytest.param(((), (2,)), ((), ()), ((), (2,)), id="insert then ident"),
        pytest.param(((), ()), ((1,), ()), ((1,), ()), id="ident then drop"),
        pytest.param(((), ()), ((), (3,)), ((), (3,)), id="ident then insert"),
        pytest.param(((0,), ()), ((0,), ()), ((0, 1), ()), id="consecutive drop"),
        pytest.param(((), (0,)), ((), (0,)), ((), (0, 1)), id="consecutive insert"),
        pytest.param(((), (0,)), ((), (3,)), ((), (0, 3)), id="insert then insert higher"),
        pytest.param(((), (3,)), ((), (1,)), ((), (1, 4)), id="insert then insert lower"),
        pytest.param(((2,), ()), ((), (4,)), ((2,), (4,)), id="drop then insert"),
        pytest.param(((0,), (0,)), ((0,), (0,)), ((0,), (0,)), id="repeatedly drop-insert same axis"),
        pytest.param(((0,), (0,)), ((0,), ()), ((0,), ()), id="drop intermediately inserted axis"),
        pytest.param(((3,), (1,)), ((0,), (0,)), ((0, 3), (0, 1)), id="later swaps unrelated axis"),
        pytest.param(((1,), (3,)), ((0,), (3,)), ((0, 1), (2, 3)), id="later inserts higher axis"),
        pytest.param(((1,), (3,)), ((1,), (3,)), ((1, 2), (2, 3)), id="stack same indices"),
        pytest.param(((0, 1, 4), (3,)), ((3, 4), ()), ((0, 1, 4, 6), ()), id="later drops earlier insert"),
        pytest.param(((), (0, 1)), ((), (0, 5)), ((), (0, 1, 2, 5)), id="accumulate inserts"),
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

    with pytest.raises(ValueError, match="source and target must be same dimensionality"):
        ProjectAxisTransform(drops=(0,), inserts=(0,), source=source, target=bad_target)


def test_project_axis_rejects_create_index_out_of_bounds():
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "yxi")

    with pytest.raises(ValueError, match="ProjectAxisTransform requires at least 4 target axes"):
        _ = ProjectAxisTransform(inserts=(2, 3), source=source, target=target)


def test_project_axis_rejects_drop_index_out_of_bounds():
    source = _sys_ref("source", "cyx")
    target = _sys_ref("target", "yx")

    with pytest.raises(ValueError, match="ProjectAxisTransform requires at least 4 source axes"):
        _ = ProjectAxisTransform(drops=(3,), source=source, target=target)


def test_ome_zarr_multiscale_transforms_composed_with_rescales_translations():
    earlier = MultiscaleTransforms((ScaleTransform((2.0,)), TranslationTransform((3.0,))))
    later = MultiscaleTransforms((ScaleTransform((4.0,)), TranslationTransform((5.0,))))

    composed = later.composed_with(earlier)

    assert isinstance(composed, MultiscaleTransforms)
    assert composed.scale_transform is not None
    assert composed.scale_transform.scale == (8.0,)
    assert composed.translation_transform is not None
    assert composed.translation_transform.translation == ((5.0 * 2.0) + (3.0 * 4.0),)


@pytest.mark.parametrize(
    "earlier_transl, later_transl",
    [
        (None, TranslationTransform((0.0,))),
        (TranslationTransform((0.0,)), None),
        (TranslationTransform((0.0,)), TranslationTransform((0.0,))),
    ],
)
def test_ome_zarr_multiscale_transforms_composed_with_preserves_explicit_identity_translations(
    earlier_transl, later_transl
):
    earlier_ts = (ScaleTransform((2.0,)), earlier_transl) if earlier_transl is not None else (ScaleTransform((2.0,)),)
    later_ts = (ScaleTransform((4.0,)), later_transl) if later_transl is not None else (ScaleTransform((4.0,)),)
    earlier = MultiscaleTransforms(earlier_ts)
    later = MultiscaleTransforms(later_ts)

    composed = later.composed_with(earlier)

    assert isinstance(composed, MultiscaleTransforms)
    assert composed.scale_transform is not None
    assert composed.scale_transform.scale == (8.0,)
    assert composed.translation_transform is not None
    assert composed.translation_transform.translation == (0.0,)
