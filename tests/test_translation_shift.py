import math

import pytest

from clearscale import (
    PixelSize,
    Scale,
    Shape,
    Translation,
    detect_translation_shift,
    discrete_bin_center,
    first_value_decimation,
    half_pixel_space_preservation,
)
from clearscale._translation_shift import known_shift_functions


def _scale(pixel_size_items):
    """Shape should be irrelevant for all calculations.
    Pixel size will always be the precise source of information for what scaling was done."""
    ps = PixelSize(pixel_size_items)
    sh = Shape.all_singletons(ps)
    return Scale(shape=sh, pixel_size=ps)


def test_half_pixel_space_preservation_computes_axis_wise_shift():
    base = _scale([("t", 12.0), ("z", 0.4), ("y", 0.6), ("x", 2.0)])
    target = _scale([("t", 12.0), ("z", 0.2), ("y", 1.5), ("x", 6.0)])

    shift = half_pixel_space_preservation(base, target)

    assert shift == Translation([("t", 0.0), ("z", -0.1), ("y", 0.45), ("x", 2.0)])


def test_discrete_bin_center_computes_center_of_first_implicit_bin():
    base = _scale([("z", 0.4), ("y", 0.6), ("x", 2.0)])
    target = _scale([("z", 0.2), ("y", 1.5), ("x", 6.0)])

    shift = discrete_bin_center(base, target)

    assert shift == Translation([("z", 0.0), ("y", 0.6), ("x", 2.0)])


@pytest.mark.parametrize(
    ("target_pixel_size", "expected_shift"),
    [
        (0.2, 0.0),
        (0.4, 0.0),
        (0.8, 0.2),
        (1.0, 0.4),
        (1.01, 0.4),
        (1.21, 0.6000000000000001),
    ],
)
def test_discrete_bin_center_uses_ceiled_implicit_bin_size(target_pixel_size, expected_shift):
    base = _scale([("x", 0.4)])
    target = _scale([("x", target_pixel_size)])

    shift = discrete_bin_center(base, target)

    assert shift == Translation(x=expected_shift)


def test_first_value_decimation_returns_identity_translation():
    base = _scale([("cookies", 0.5), ("y", 1.0), ("x", 2.0)])
    target = _scale([("cookies", 0.0000212), ("y", 126.0), ("x", 4.0)])

    shift = first_value_decimation(base, target)

    assert shift == Translation([("cookies", 0.0), ("y", 0.0), ("x", 0.0)])


@pytest.mark.parametrize("shift_function", known_shift_functions)
@pytest.mark.parametrize(
    "target",
    [
        _scale([("x", 4.0), ("y", 2.0)]),
        _scale([("y", 2.0), ("z", 4.0)]),
    ],
)
def test_all_shift_functions_reject_axis_mismatches(shift_function, target):
    base = _scale([("y", 1.0), ("x", 2.0)])

    with pytest.raises(ValueError, match="Axis mismatch"):
        shift_function(base, target)


def _linear_scale_values_starting_at(*, first_coordinate, target_length, source_length):
    """Mock scaling implementation with an artificial offset"""
    sample_spacing = source_length / target_length
    return [first_coordinate + sample_spacing * index for index in range(target_length)]


@pytest.mark.parametrize(
    ("expected_function", "first_coordinate"),
    [
        (half_pixel_space_preservation, 0.5 * (1025 / 256 - 1.0)),
        (discrete_bin_center, 0.5 * (math.ceil(1025 / 256) - 1)),
        (first_value_decimation, 0.0),
    ],
)
def test_detect_translation_shift_matches_known_conventions(expected_function, first_coordinate):
    def scaling_function(source):
        return _linear_scale_values_starting_at(
            first_coordinate=first_coordinate,
            target_length=256,
            source_length=len(source),
        )

    matching_function, error = detect_translation_shift(scaling_function)

    assert matching_function is expected_function
    assert error < 1e-12


def test_detect_translation_shift_is_resistant_to_edge_errors():
    source_length = 1025
    target_length = 256
    first_coordinate = 0.5 * (source_length / target_length - 1.0)

    def scaling_function(source):
        values = _linear_scale_values_starting_at(
            first_coordinate=first_coordinate,
            target_length=target_length,
            source_length=len(source),
        )
        for index in range(0, target_length // 4):
            values[index] = -1000.0
        for index in range(target_length - target_length // 4, target_length):
            values[index] = 1000.0
        return values

    matching_function, error = detect_translation_shift(scaling_function)

    assert matching_function is half_pixel_space_preservation
    assert error < 1e-14


def test_detect_translation_shift_reports_error_from_closest_known_convention():
    source_length = 1025
    target_length = 256
    correct_first_coordinate = 0.5 * (source_length / target_length - 1.0)

    # half_pixel_space_preservation shift is 1.502 (the "correct" first coordinate)
    # discrete_bin_center shift is 2.0
    # As long as 1.502 + artificial_error is closer to 1.502 than 2.0,
    # half_pixel_space_preservation is the closest match
    artificial_error = 0.13

    def scaling_function(source):
        return _linear_scale_values_starting_at(
            first_coordinate=correct_first_coordinate + artificial_error,
            target_length=target_length,
            source_length=len(source),
        )

    matching_function, error = detect_translation_shift(scaling_function)

    assert matching_function is half_pixel_space_preservation
    assert abs(error - artificial_error) < 1e-14


@pytest.mark.parametrize(
    ("scaling_function", "expected_error"),
    [
        (lambda source: [[0.0] * 200], "must return a one-dimensional array"),
        (lambda source: [float(index) for index in range(1025)], "actually do scaling"),
        (lambda source: [float(index) for index in range(149)], "at least 150"),
    ],
)
def test_detect_translation_shift_rejects_invalid_scaling_function_outputs(scaling_function, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        detect_translation_shift(scaling_function)


def test_ilastik_op_resize_is_half_pixel_space_preserving():
    op_resize = pytest.importorskip("lazyflow.operators.opResize")
    graph = pytest.importorskip("lazyflow.graph")
    vigra = pytest.importorskip("vigra")
    np = pytest.importorskip("numpy")

    def resize_with_op(x):
        op_scale = op_resize.OpResize(
            graph=graph.Graph(),
            RawImage=vigra.taggedView(np.asarray(x), "x"),
            TargetShape=(327,),
            InterpolationOrder=1,
        )
        return op_scale.ResizedImage[:].wait()

    matching_function, error = detect_translation_shift(resize_with_op)
    assert matching_function is half_pixel_space_preservation
    assert error < 1e-12
