import re
from typing import cast

import pytest
from clearscale._transforms import (
    CoordinateSystem,
    BijectionTransform,
    ByDimensionTransform,
    _ByDimensionChild,
    IdentityTransform,
    ProjectAxisTransform,
    ScaleTransform,
    RotationTransform,
    CoordinatesTransform,
    Transform,
    TransformSequence,
    TranslationTransform,
)


def _sys_ref(name, axes):
    return CoordinateSystem.without_semantics(axes).as_ref(name)


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


def test_bijection_bound_rejects_mismatching_dims():
    source = _sys_ref("source", "yx")
    target = _sys_ref("target", "cyx")
    bijection = BijectionTransform(forward=ScaleTransform((0.5, 0.5)), inverse=ScaleTransform((2.0, 2.0)))

    with pytest.raises(ValueError, match="BijectionTransform expects 2 target axes"):
        bijection.bound(source=source, target=target)


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


@pytest.mark.parametrize(
    "forward,inverse",
    [
        pytest.param(ScaleTransform(scale=(1,)), TranslationTransform(translation=(0,)), id="scale and translation"),
        pytest.param(
            RotationTransform(rotation=((1, 0), (0, 1))), ScaleTransform(scale=(1, 1)), id="rotation and scale"
        ),
    ],
)
def test_bijection_simplified_detects_both_children_simplified(forward: Transform, inverse: Transform):
    transform = BijectionTransform(forward=forward, inverse=inverse)
    simplified = transform.simplified()
    assert isinstance(simplified, IdentityTransform)


@pytest.mark.parametrize(
    "forward,inverse,expected",
    [
        pytest.param(
            ScaleTransform(scale=(1,)),
            TranslationTransform(translation=(3,)),
            BijectionTransform(forward=IdentityTransform(), inverse=TranslationTransform(translation=(3,))),
            id="forward only",
        ),
        pytest.param(
            ScaleTransform(scale=(2,)),
            TranslationTransform(translation=(0,)),
            BijectionTransform(forward=ScaleTransform(scale=(2,)), inverse=IdentityTransform()),
            id="inverse only",
        ),
    ],
)
def test_bijection_simplified_replaces_changed_children(forward: Transform, inverse: Transform, expected: Transform):
    transform = BijectionTransform(forward=forward, inverse=inverse)
    simplified = transform.simplified()
    assert simplified == expected


@pytest.mark.parametrize(
    "forward,inverse",
    [
        pytest.param(ScaleTransform(scale=(2,)), TranslationTransform(translation=(3,)), id="scale and translation"),
        pytest.param(
            RotationTransform(rotation=((0, -1), (1, 0))), ScaleTransform(scale=(2, 3)), id="rotation and scale"
        ),
    ],
)
def test_bijection_simplified_detects_no_simplification(forward: Transform, inverse: Transform):
    transform = BijectionTransform(forward=forward, inverse=inverse)
    simplified = transform.simplified()
    assert simplified is transform


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
                    _ByDimensionChild((0,), (0,), CoordinatesTransform(path="foo.zarr")),
                    _ByDimensionChild((1,), (1,), IdentityTransform()),
                ),
            ),
            id="child cannot compose",
        ),
    ],
)
def test_by_dimension_fails_invalid_itemwise_composition(earlier: ByDimensionTransform):
    later = ByDimensionTransform(
        transforms=(
            _ByDimensionChild((0,), (0,), ScaleTransform((2.0,))),
            _ByDimensionChild((1,), (1,), IdentityTransform()),
        ),
    )

    assert later.composed_with(earlier) is None


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


def test_by_dimension_simplified_detects_all_subspaces_identity():
    transform = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=ScaleTransform(scale=(1,))),
            _ByDimensionChild(
                source_indices=(1,), target_indices=(1,), transform=TranslationTransform(translation=(0,))
            ),
        )
    )
    simplified = transform.simplified()
    assert isinstance(simplified, IdentityTransform)


def test_by_dimension_simplified_replaces_changed_children():
    transform = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=ScaleTransform(scale=(1,))),
            _ByDimensionChild(
                source_indices=(1,), target_indices=(1,), transform=TranslationTransform(translation=(3,))
            ),
        )
    )
    expected = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=IdentityTransform()),
            _ByDimensionChild(
                source_indices=(1,), target_indices=(1,), transform=TranslationTransform(translation=(3,))
            ),
        )
    )
    simplified = transform.simplified()
    assert simplified == expected


def test_by_dimension_simplified_detects_no_simplification():
    transform = ByDimensionTransform(
        transforms=(
            _ByDimensionChild(source_indices=(0,), target_indices=(0,), transform=ScaleTransform(scale=(2,))),
            _ByDimensionChild(
                source_indices=(1,), target_indices=(1,), transform=TranslationTransform(translation=(3,))
            ),
        )
    )
    simplified = transform.simplified()
    assert simplified is transform


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


@pytest.mark.parametrize(
    "transforms",
    [
        pytest.param(
            (
                IdentityTransform(),
                ScaleTransform(scale=(2,)),
                TranslationTransform(translation=(3,)),
            ),
            id="identity removed from front",
        ),
        pytest.param(
            (
                ScaleTransform(scale=(2,)),
                TranslationTransform(translation=(3,)),
                IdentityTransform(),
            ),
            id="identity removed from back",
        ),
        pytest.param(
            (
                IdentityTransform(),
                ScaleTransform(scale=(2,)),
                IdentityTransform(),
            ),
            id="single child remains",
        ),
    ],
)
def test_transform_sequence_simplified_maintains_sequence_endpoints_on_children(transforms):
    source = _sys_ref("source", "x")
    intermediate1 = _sys_ref("intermediate1", "x")
    intermediate2 = _sys_ref("intermediate2", "x")
    target = _sys_ref("target", "x")

    bound_chain = []
    refs = [source, intermediate1, intermediate2, target]
    for i, transform in enumerate(transforms):
        bound_chain.append(transform.bound(source=refs[i], target=refs[i + 1]))

    simplified = TransformSequence(tuple(bound_chain), source=source, target=target).simplified()

    assert simplified.source is source
    assert simplified.target is target
    if isinstance(simplified, TransformSequence):
        assert simplified.transforms[0].source is source
        assert simplified.transforms[-1].target is target
