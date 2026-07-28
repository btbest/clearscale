import pytest
from clearscale import (
    BlueprintShapes,
    Multiscale,
    PixelSize,
    Scale,
    Shape,
    Translation,
    Unit,
    discrete_bin_center,
    half_pixel_space_preservation,
    BlueprintFactors,
    Factor,
)
from clearscale._transforms import CoordinateSystem, NodeRef


def _ref(axes: str, name: str) -> NodeRef[CoordinateSystem]:
    return CoordinateSystem.without_semantics(axes).as_ref(name)


def test_blueprint_hash_matches_value_equality():
    left = BlueprintShapes({"s0": Shape(y=2, x=3)})
    right = BlueprintShapes({"s0": Shape(y=2, x=3)})

    assert left == right
    assert hash(left) == hash(right)


def test_multiscale_equality_and_hash_are_value_based():
    left = Multiscale({"s0": Scale(Shape(y=2, x=3))}, _intrinsic_ref=_ref("yx", "physical"))
    right = Multiscale({"s0": Scale(Shape(y=2, x=3))}, _intrinsic_ref=_ref("yx", "physical"))

    assert left == right
    assert {left, right} == {left}, "Value hash should lead to collapse in sets"


def test_multiscale_refs_are_hashable():
    left = Multiscale({"s0": Scale(Shape(y=2, x=3))}, _intrinsic_ref=_ref("yx", "physical"))
    right = Multiscale({"s0": Scale(Shape(y=2, x=3))}, _intrinsic_ref=_ref("yx", "physical"))

    assert len({left.as_ref("physical"), right.as_ref("physical")}) == 2


def test_with_sizes_broadcasts_single_shape_to_all_scales():
    shapes = BlueprintShapes({"s0": Shape(x=10, y=20), "s1": Shape(x=30, y=40)})

    result = shapes.with_sizes({"x": 100})
    assert result == BlueprintShapes({"s0": Shape(x=100, y=20), "s1": Shape(x=100, y=40)})


def test_with_sizes_updates_only_specified_scales():
    shapes = BlueprintShapes({"s0": Shape(x=10, y=20), "s1": Shape(x=30, y=40)})

    result = shapes.with_sizes({"s1": {"y": 99}})
    assert result == BlueprintShapes({"s0": Shape(x=10, y=20), "s1": Shape(x=30, y=99)})


def test_with_sizes_updates_multiple_scales_independently():
    shapes = BlueprintShapes({"s0": Shape(x=10, y=20), "s1": Shape(x=30, y=40)})

    result = shapes.with_sizes({"s0": {"x": 1}, "s1": {"y": 2}})
    assert result == BlueprintShapes({"s0": Shape(x=1, y=20), "s1": Shape(x=30, y=2)})


def test_with_sizes_ignores_unknown_scale_keys():
    shapes = BlueprintShapes({"s0": Shape(x=10, y=20), "s1": Shape(x=30, y=40)})

    result = shapes.with_sizes({"unknown": {"z": 1}})
    assert result == shapes


def test_with_sizes_only_axes_limits_broadcast_update():
    shapes = BlueprintShapes({"s0": Shape(x=10, y=20), "s1": Shape(x=30, y=40)})

    result = shapes.with_sizes({"x": 5, "y": 6}, only_axes="x")
    assert result == BlueprintShapes({"s0": Shape(x=5, y=20), "s1": Shape(x=5, y=40)})


def test_with_sizes_only_axes_limits_nested_update():
    shapes = BlueprintShapes({"s0": Shape(x=10, y=20), "s1": Shape(x=30, y=40)})

    result = shapes.with_sizes({"s0": {"x": 5, "y": 6}}, only_axes="x")
    assert result == BlueprintShapes({"s0": Shape(x=5, y=20), "s1": Shape(x=30, y=40)})


def test_with_factors_works_like_with_sizes():
    shapes = BlueprintFactors({"s0": Factor(x=1.0, y=2.0), "s1": Factor(x=3.0, y=4.0)})

    result = shapes.with_factors({"s0": {"x": 10.0, "y": 11.0}}, only_axes="x")
    assert result == BlueprintFactors({"s0": Factor(x=10.0, y=2.0), "s1": Factor(x=3.0, y=4.0)})


def test_blueprint_shapes_apply_to_scale_derives_scale_metadata_from_base():
    blueprint = BlueprintShapes({"s0": Shape(c=3, y=8, x=12), "s1": Shape(c=3, y=4, x=3)})
    base = Scale(
        shape=Shape(c=3, y=8, x=12),
        pixel_size=PixelSize(c=1.0, y=0.5, x=2.0),
        unit=Unit(c="", y="um", x="um"),
        translation=Translation(c=0.0, y=1.0, x=2.0),
    )

    multiscale = blueprint.apply_to_scale(base)

    assert multiscale == Multiscale(
        {
            "s0": base,
            "s1": Scale(
                shape=Shape(c=3, y=4, x=3),
                pixel_size=PixelSize(c=1.0, y=1.0, x=8.0),
                unit=base.unit,
                translation=base.translation,
            ),
        }
    )


def test_blueprint_shapes_apply_to_scale_can_apply_half_pixel_shift():
    blueprint = BlueprintShapes({"s0": Shape(y=8, x=8), "s1": Shape(y=4, x=2)})
    base = Scale(
        shape=Shape(y=8, x=8),
        pixel_size=PixelSize(y=2.0, x=3.0),
        translation=Translation(y=10.0, x=-5.0),
    )

    multiscale = blueprint.apply_to_scale(base, translation_shift_func=half_pixel_space_preservation)

    # Along y: 8 -> 4 px = factor 2. Pixel size 2.0 * 2 = 4.0
    #   s0 data space begins at 10.0-(2.0/2) = 9.0
    #   s1 first pixel coordinate is at 9.0 + (4.0 / 2) = 11.0
    # Along x: 8 -> 2 px = factor 4. Pixel size 3.0 * 4 = 12.0
    #   s0 data space begins at -5.0-(3.0/2) = -6.5
    #   s1 first pixel coordinate is at -6.5 + (12.0 / 2) = -0.5
    assert multiscale["s1"].pixel_size == PixelSize(y=4.0, x=12.0)
    assert multiscale["s1"].translation == Translation(y=11.0, x=-0.5)


def test_blueprint_shapes_apply_to_scale_can_apply_bin_center_shift():
    blueprint = BlueprintShapes({"s0": Shape(y=5, x=8), "s1": Shape(y=2, x=4)})
    base = Scale(
        shape=Shape(y=5, x=8),
        pixel_size=PixelSize(y=0.6, x=2.0),
        translation=Translation(y=10.0, x=-5.0),
    )

    multiscale = blueprint.apply_to_scale(base, translation_shift_func=discrete_bin_center)

    # Along y: 5 -> 2 px = factor 2.5 (Pixel size 0.6 * 2.5 = 1.5)
    #   Implicit bin size = ceil(2.5) = 3
    #   In this case first scaled pixel coordinate = middle bin coordinate = 10.0 + 0.6
    #   (or: bin space start: 10.0 - 0.6/2 = 9.7; bin extent = 0.6 * 3 = 1.8; bin center = 9.7 + 1.8/2 = 10.6)
    # Along x: 8 -> 4 px = factor 2 (Pixel size 2.0 * 2 = 4.0)
    #   Implicit bin size = 2
    #   First bin coordinates = -5.0 and -3.0; bin center = -4.0
    assert multiscale["s1"].pixel_size == PixelSize(y=1.5, x=4.0)
    assert multiscale["s1"].translation == Translation(y=10.6, x=-4.0)


@pytest.mark.parametrize("shift_func", [(lambda param1: True), (lambda scale1, scale2: True)])
def test_blueprint_shapes_apply_to_scale_rejects_malformed_shift_functions(shift_func):
    bp = BlueprintShapes({"s0": Shape(x=2)})
    base = Scale(shape=Shape(x=1))

    with pytest.raises(TypeError, match="See clearscale.half_pixel_shift for an example implementation"):
        _ = bp.apply_to_scale(base, translation_shift_func=shift_func)  # noqa


def test_proportional_blueprint_from_multiscale_template():
    ms = Multiscale(
        {
            "s0": Scale(
                shape=Shape(c=3, y=8, x=12),
                pixel_size=PixelSize(c=1.0, y=1.0, x=2.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
            "s1": Scale(
                shape=Shape(c=3, y=4, x=3),
                pixel_size=PixelSize(c=1.0, y=2.0, x=8.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
        }
    )

    target_shape = Shape(c=3, y=2, x=6)
    bp = BlueprintShapes.from_multiscale_rescaled(ms, target_shape=target_shape, rounding="floor")

    assert bp == BlueprintShapes({"s0": target_shape, "s1": Shape(c=3, y=1, x=1)})


def test_proportional_blueprint_rebased_on_downscale():
    ms = Multiscale(
        {
            "s0": Scale(
                shape=Shape(c=3, y=8, x=12),
                pixel_size=PixelSize(c=1.0, y=1.0, x=2.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
            "s1": Scale(
                shape=Shape(c=3, y=4, x=3),
                pixel_size=PixelSize(c=1.0, y=2.0, x=8.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
        }
    )

    target_shape = Shape(c=3, y=2, x=6)
    bp = BlueprintShapes.from_multiscale_rescaled(ms, target_shape=target_shape, rounding="floor", source_key="s1")

    assert bp == BlueprintShapes({"s0": Shape(c=3, y=4, x=24), "s1": target_shape})


def test_proportional_blueprint_restricted_scaling_axes():
    ms = Multiscale(
        {
            "s0": Scale(
                shape=Shape(c=3, y=8, x=12),
                pixel_size=PixelSize(c=1.0, y=1.0, x=2.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
            "s1": Scale(
                shape=Shape(c=3, y=4, x=3),
                pixel_size=PixelSize(c=1.0, y=2.0, x=8.0),
                unit=Unit(c="", y="nm", x="nm"),
                translation=Translation.identity("cyx"),
            ),
        }
    )

    target_shape = Shape(c=3, y=2, x=6)
    bp = BlueprintShapes.from_multiscale_rescaled(ms, target_shape=target_shape, rounding="floor", scaled_axes="y")

    assert bp == BlueprintShapes({"s0": target_shape, "s1": Shape(c=3, y=1, x=6)})
