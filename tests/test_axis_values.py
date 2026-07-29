import math
import pytest
from typing import cast

from clearscale import Factor, PixelOffset, PixelSize, Shape, Translation


def assert_axis_values(value, expected_type, expected_items):
    assert type(value) is expected_type
    assert list(value.items()) == expected_items


def test_factor_multiplies_factor_axis_wise():
    result = Factor([("y", 2.0), ("x", 4.0)]) * Factor([("y", 3.0), ("x", 0.5)])

    assert_axis_values(result, Factor, [("y", 6.0), ("x", 2.0)])


def test_factor_divides_factor_axis_wise():
    result = Factor([("y", 6.0), ("x", 3.0)]) / Factor([("y", 2.0), ("x", 1.5)])

    assert_axis_values(result, Factor, [("y", 3.0), ("x", 2.0)])


def test_factor_and_pixel_size_multiply_using_pixel_size_scaling_rules():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor(x=2.0)

    assert_axis_values(factor * pixel_size, PixelSize, [("y", 2.0), ("x", 8.0)])
    assert_axis_values(pixel_size * factor, PixelSize, [("y", 2.0), ("x", 8.0)])


def test_pixel_size_rejects_zero_values():
    with pytest.raises(ValueError, match="Pixel size cannot be 0 or negative"):
        PixelSize(c=0.0)


def test_pixel_size_rejects_negative_values():
    with pytest.raises(ValueError, match="Pixel size cannot be 0 or negative"):
        PixelSize(c=-1.0)


def test_pixel_size_divides_by_factor_using_inverted_scaling_rules():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor(x=2.0)

    assert_axis_values(pixel_size / factor, PixelSize, [("y", 2.0), ("x", 2.0)])


def test_pixel_size_mul_factor_rejects_extra_factor_axes():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor([("y", 2.0), ("x", 2.0), ("z", 2.0)])

    with pytest.raises(ValueError, match="Attempted to scale axes with no base pixel size"):
        _ = factor * pixel_size


def test_factor_mul_pixel_size_rejects_extra_factor_axes():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor([("y", 2.0), ("x", 2.0), ("z", 2.0)])

    with pytest.raises(ValueError, match="Attempted to scale axes with no base pixel size"):
        _ = pixel_size * factor


def test_pixel_size_div_factor_rejects_extra_factor_axes():
    pixel_size = PixelSize([("y", 2.0), ("x", 4.0)])
    factor = Factor([("y", 2.0), ("x", 2.0), ("z", 2.0)])

    with pytest.raises(ValueError, match="Attempted to scale axes with no base pixel size"):
        _ = pixel_size / factor


def test_pixel_size_divides_pixel_size_to_factor():
    target_pixel_size = PixelSize([("y", 4.0), ("x", 1.0)])
    base_pixel_size = PixelSize([("y", 2.0), ("x", 0.5)])

    result = target_pixel_size / base_pixel_size

    assert_axis_values(result, Factor, [("y", 2.0), ("x", 2.0)])


def test_translation_adds_and_subtracts_translation_axis_wise():
    left = Translation([("y", 1.0), ("x", 5.0)])
    right = Translation([("y", 2.0), ("x", 3.0)])

    assert_axis_values(left + right, Translation, [("y", 3.0), ("x", 8.0)])
    assert_axis_values(left - right, Translation, [("y", -1.0), ("x", 2.0)])


def test_pixel_offset_adds_and_subtracts_pixel_offset_axis_wise():
    left = PixelOffset([("y", 10), ("x", 3)])
    right = PixelOffset([("y", 4), ("x", 8)])

    assert_axis_values(left + right, PixelOffset, [("y", 14), ("x", 11)])
    assert_axis_values(left - right, PixelOffset, [("y", 6), ("x", -5)])


def test_pixel_offset_and_pixel_size_multiply_to_translation_with_extra_pixel_size_axes_ignored():
    offset = PixelOffset([("y", 2), ("x", 4)])
    pixel_size = PixelSize([("z", 10.0), ("y", 0.5), ("x", 2.0)])

    assert_axis_values(offset * pixel_size, Translation, [("y", 1.0), ("x", 8.0)])
    assert_axis_values(pixel_size * offset, Translation, [("y", 1.0), ("x", 8.0)])


def test_offset_mul_pixel_size_multiplication_requires_offset_axes():
    offset = PixelOffset([("y", 2), ("x", 4)])
    pixel_size = PixelSize(y=0.5)

    with pytest.raises(ValueError, match="must contain all axes"):
        _ = offset * pixel_size


def test_pixel_size_mul_offset_multiplication_requires_offset_axes():
    offset = PixelOffset([("y", 2), ("x", 4)])
    pixel_size = PixelSize(y=0.5)

    with pytest.raises(ValueError, match="must contain all axes"):
        _ = pixel_size * offset


def test_shape_divides_shape_to_factor_using_scaling_to_semantics():
    original_shape = Shape([("y", 100), ("x", 50)])
    resized_shape = Shape([("y", 25), ("x", 100)])

    result = original_shape / resized_shape

    assert_axis_values(result, Factor, [("y", 4.0), ("x", 0.5)])


@pytest.mark.parametrize(
    ("translation", "axes", "expected"),
    [
        pytest.param(Translation.identity(("x", "y")), ("x", "y"), True, id="identity"),
        pytest.param(Translation({"x": 1.0, "y": 0.0}), ("y",), True, id="identity along subset"),
        pytest.param(Translation({"x": 1.0, "y": 0.0}), ("x",), False, id="non-identity along subset"),
        pytest.param(Translation({"x": 1.0, "y": 2.0}), ("x", "y"), False, id="non-identity"),
        pytest.param(Translation({"x": 1.0, "y": 2.0}), (), True, id="empty axes"),
    ],
)
def test_translation_is_identity_along(translation, axes, expected):
    assert cast(Translation, translation).is_identity_along(axes) is expected


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [
        ("ceil", [("y", 3), ("x", 2)]),
        ("floor", [("y", 2), ("x", 1)]),
        ("round", [("y", 3), ("x", 2)]),
        (lambda value: math.floor(value) + 10, [("y", 12), ("x", 11)]),
    ],
)
def test_translation_to_pixel_offset_rounds_and_ignores_extra_pixel_size_axes(rounding, expected):
    translation = Translation([("y", 2.6), ("x", 3.6)])
    pixel_size = PixelSize([("z", 100.0), ("y", 1.0), ("x", 2.0)])

    result = translation.to_pixel_offset(pixel_size, rounding=rounding)

    assert_axis_values(result, PixelOffset, expected)


def test_translation_to_pixel_offset_requires_translation_axes():
    translation = Translation([("y", 2.6), ("x", 3.6)])

    with pytest.raises(ValueError, match="must contain all axes"):
        translation.to_pixel_offset(PixelSize(y=1.0), rounding="round")


@pytest.mark.parametrize(
    ("operation", "left", "right"),
    [
        (lambda left, right: left * right, Factor(y=2.0, x=3.0), Factor(x=4.0, y=5.0)),
        (lambda left, right: left / right, Factor(y=2.0, x=3.0), Factor(x=4.0, y=5.0)),
        (lambda left, right: left / right, PixelSize(y=2.0, x=3.0), PixelSize(x=4.0, y=5.0)),
        (lambda left, right: left + right, Translation(y=2.0, x=3.0), Translation(x=4.0, y=5.0)),
        (lambda left, right: left - right, Translation(y=2.0, x=3.0), Translation(x=4.0, y=5.0)),
        (lambda left, right: left + right, PixelOffset(y=2, x=3), PixelOffset(x=4, y=5)),
        (lambda left, right: left - right, PixelOffset(y=2, x=3), PixelOffset(x=4, y=5)),
        (lambda left, right: left / right, Shape(y=20, x=30), Shape(x=40, y=50)),
    ],
)
def test_same_kind_operations_reject_incompatible_axis_order(operation, left, right):
    with pytest.raises(ValueError, match="Incompatible axes"):
        operation(left, right)


@pytest.mark.parametrize(
    ("operation", "left", "right"),
    [
        (lambda left, right: left * right, Factor(y=2.0, x=3.0), Factor(y=4.0, z=5.0)),
        (lambda left, right: left / right, Factor(y=2.0, x=3.0), Factor(y=4.0, z=5.0)),
        (lambda left, right: left / right, PixelSize(y=2.0, x=3.0), PixelSize(y=4.0, z=5.0)),
        (lambda left, right: left + right, Translation(y=2.0, x=3.0), Translation(y=4.0, z=5.0)),
        (lambda left, right: left - right, Translation(y=2.0, x=3.0), Translation(y=4.0, z=5.0)),
        (lambda left, right: left + right, PixelOffset(y=2, x=3), PixelOffset(y=4, z=5)),
        (lambda left, right: left - right, PixelOffset(y=2, x=3), PixelOffset(y=4, z=5)),
        (lambda left, right: left / right, Shape(y=20, x=30), Shape(y=40, z=50)),
    ],
)
def test_same_kind_operations_reject_incompatible_axes(operation, left, right):
    with pytest.raises(ValueError, match="Incompatible axes"):
        operation(left, right)


def test_implemented_dunders_return_not_implemented_for_unsupported_operands():
    assert Factor(y=2.0).__mul__(Shape(y=2)) is NotImplemented
    assert Factor(y=2.0).__truediv__(PixelSize(y=1.0)) is NotImplemented
    assert PixelSize(y=1.0).__mul__(PixelSize(y=2.0)) is NotImplemented
    assert PixelSize(y=1.0).__truediv__(Shape(y=2)) is NotImplemented
    assert Translation(y=1.0).__add__(PixelOffset(y=1)) is NotImplemented
    assert Translation(y=1.0).__sub__(PixelOffset(y=1)) is NotImplemented
    assert PixelOffset(y=1).__add__(Shape(y=1)) is NotImplemented
    assert PixelOffset(y=1).__sub__(Shape(y=1)) is NotImplemented
    assert PixelOffset(y=1).__mul__(Factor(y=1.0)) is NotImplemented
    assert Shape(y=1).__truediv__(Factor(y=1.0)) is NotImplemented


@pytest.mark.parametrize(
    "operation",
    [
        lambda: Shape(y=4) * Factor(y=2.0),  # type: ignore[operator]
        lambda: Shape(y=4) / Factor(y=2.0),  # type: ignore[operator]
        lambda: Translation(y=1.0) / PixelSize(y=1.0),  # type: ignore[operator]
        lambda: Shape(y=4) * PixelSize(y=1.0),  # type: ignore[operator]
        lambda: PixelOffset(y=1) + Shape(y=1),  # type: ignore[operator]
        lambda: Shape(y=1) + PixelOffset(y=1),  # type: ignore[operator]
        lambda: Factor(y=1.0) + Factor(y=1.0),  # type: ignore[operator]
        lambda: Factor(y=1.0) - Factor(y=1.0),  # type: ignore[operator]
        lambda: PixelSize(y=1.0) + PixelSize(y=1.0),  # type: ignore[operator]
        lambda: PixelSize(y=1.0) - PixelSize(y=1.0),  # type: ignore[operator]
        lambda: Shape(y=1) + Shape(y=1),  # type: ignore[operator]
        lambda: Shape(y=1) - Shape(y=1),  # type: ignore[operator]
        lambda: Translation(y=1.0) * Factor(y=1.0),  # type: ignore[operator]
        lambda: PixelOffset(y=1) / Factor(y=1.0),  # type: ignore[operator]
    ],
)
def test_unsupported_arithmetic_raises_type_error(operation):
    with pytest.raises(TypeError, match="unsupported operand type"):
        operation()
