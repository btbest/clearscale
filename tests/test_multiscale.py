import re

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
from clearscale._transforms import CoordinateSystem, IdentityTransform, TransformGraph, _UnresolvedRef, FileRef, NodeRef


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


def _multiscale(axes="yx", size=4) -> Multiscale:
    return Multiscale({"s0": Scale(shape=Shape(zip(axes, [size] * len(axes))))})


def test_multiscale_ome_properties_separate_across_instances():
    ms1 = _multiscale()
    ms2 = _multiscale()
    assert id(ms1.ome) != id(ms2.ome)
    assert ms1.ome is not ms2.ome


def _with_intrinsic_system_name(ms: Multiscale, name: str) -> Multiscale:
    """Creates modified `ms` *with empty graph*, so this helper must be used *before* other helpers that modify the graph"""
    return Multiscale(ms.items(), _intrinsic_ref=ms._intrinsic_ref.owner.as_ref(name))


def _with_extra_system(ms: Multiscale, name: str) -> Multiscale:
    """Attach one additional named coordinate system to `ms` via an identity edge from its intrinsic ref."""
    extra_ref = CoordinateSystem.without_semantics(tuple(ms.axes())).as_ref(name)
    edge = IdentityTransform().bound(source=ms._intrinsic_ref, target=extra_ref)
    graph = TransformGraph(transforms=ms._transform_graph.transforms + (edge,))
    return Multiscale(ms.items(), _transform_graph=graph, _intrinsic_ref=ms._intrinsic_ref)


def _with_path_bound_edge(ms: Multiscale, path: str) -> Multiscale:
    """Attach a path-bound edge to `ms`, simulating a link to a label-Multiscale overlay."""
    edge = IdentityTransform().bound(
        source=ms._intrinsic_ref, target=_UnresolvedRef(name=path, file=FileRef.from_string(path))
    )
    graph = TransformGraph(transforms=ms._transform_graph.transforms + (edge,))
    return Multiscale(ms.items(), _transform_graph=graph, _intrinsic_ref=ms._intrinsic_ref)


def test_multiscale_with_coordinate_systems_of_replaces_intrinsic():
    source_ms = _with_extra_system(_multiscale(), "world")

    derived_ms = _multiscale()
    result = derived_ms.with_coordinate_systems_of(source_ms)

    assert source_ms._intrinsic_ref not in result._transform_graph.all_system_refs
    assert derived_ms._intrinsic_ref in result._transform_graph.all_system_refs
    assert "world" in result.coordinate_systems


def test_multiscale_with_coordinate_systems_of_does_not_accumulate_on_repetition():
    source_ms = _with_extra_system(_multiscale(), "world")

    middle_ms = _multiscale().with_coordinate_systems_of(source_ms)
    derived_ms = _multiscale().with_coordinate_systems_of(middle_ms)

    assert source_ms._intrinsic_ref not in derived_ms._transform_graph.all_system_refs
    assert middle_ms._intrinsic_ref not in derived_ms._transform_graph.all_system_refs
    assert len(derived_ms._transform_graph.transforms) == len(middle_ms._transform_graph.transforms) == 1
    assert "world" in derived_ms.coordinate_systems


def test_multiscale_with_coordinate_systems_of_preserves_existing_systems_on_caller():
    caller_ms = _with_extra_system(_multiscale(), "caller_space")
    donor_ms = _with_extra_system(_multiscale(), "world")

    result = caller_ms.with_coordinate_systems_of(donor_ms)

    assert "caller_space" in result.coordinate_systems
    assert "world" in result.coordinate_systems


def test_multiscale_with_coordinate_systems_of_preserves_caller_on_noop():
    caller_ms = _with_extra_system(_multiscale(), "only_space")
    donor_ms = _multiscale()  # plain, isolated: nothing to contribute

    result = caller_ms.with_coordinate_systems_of(donor_ms)

    assert result is caller_ms


def test_multiscale_with_coordinate_systems_of_does_not_port_path_bound_transforms():
    source_ms = _with_extra_system(_multiscale(), "world")
    source_ms = _with_path_bound_edge(source_ms, "labels")

    derived_ms = _multiscale()
    result = derived_ms.with_coordinate_systems_of(source_ms)

    assert "world" in result.coordinate_systems
    assert len(result._transform_graph.transforms) == 1
    assert not any(Multiscale._is_transform_path_bound(t) for t in result._transform_graph.transforms)


def test_multiscale_with_coordinate_systems_of_transfers_t_scale_convention_even_when_graph_unchanged():
    # caller pixel_size[t] == donor's global legacy t-scale. Satisfies that caller is really derived, so it should
    # also follow donor's serialization convention.
    caller_ms = Multiscale({"s0": Scale(shape=Shape(t=4, y=4, x=4), pixel_size=PixelSize(t=0.5, y=1.0, x=1.0))})
    donor_ms = Multiscale({"s0": Scale(shape=Shape(t=4, y=4, x=4))}, _legacy_convention_global_t_scale=0.5)

    result = caller_ms.with_coordinate_systems_of(donor_ms)

    assert result is not caller_ms, "should not short-cut and return self"
    assert caller_ms._legacy_convention_global_t_scale is None, "must not modify original"
    assert result._legacy_convention_global_t_scale == 0.5


def test_multiscale_with_coordinate_systems_of_does_not_port_t_scale_when_mismatching():
    """
    Calling with_coordinate_systems_of() claims caller and donor share a space, but caller's
    pixel_size["t"] != donor's global t-scale. This is deliberately not an error and does not
    get auto-corrected via an inferred Factor: fabricating a t-scale relationship from two
    numbers that happen to both be named "t" would assert a registration fact nothing here
    can actually verify. If the two really are related by a scale factor, the caller should
    say so explicitly via `derived_by=Factor(t=...)`. Absent that, the convention just doesn't
    transfer, same as other cases where the donor's metadata doesn't apply
    (like path-bound transforms in the donor graph).
    """
    caller_ms = Multiscale({"s0": Scale(shape=Shape(t=4, y=4, x=4), pixel_size=PixelSize(t=0.9, y=1.0, x=1.0))})
    donor_ms = Multiscale({"s0": Scale(shape=Shape(t=4, y=4, x=4))}, _legacy_convention_global_t_scale=0.5)

    result = caller_ms.with_coordinate_systems_of(donor_ms)

    assert result._legacy_convention_global_t_scale is None


def test_multiscale_with_coordinate_systems_of_with_derivation_retains_other():
    source_ms = _with_extra_system(_multiscale(), "world")

    derived_ms = _multiscale()
    result = derived_ms.with_coordinate_systems_of(source_ms, derived_by=Factor.identity(derived_ms.axes()))

    assert source_ms._intrinsic_ref in result._transform_graph.all_system_refs
    assert derived_ms._intrinsic_ref in result._transform_graph.all_system_refs
    assert "world" in result.coordinate_systems


def test_multiscale_with_coordinate_systems_of_with_derivation_renames_on_duplicate_intrinsic_system_name():
    derived_ms = _with_intrinsic_system_name(_multiscale(), "physical")
    source_ms = _with_intrinsic_system_name(_multiscale(), "physical")

    result = derived_ms.with_coordinate_systems_of(source_ms, derived_by=Factor.identity(derived_ms.axes()))

    assert len(result._transform_graph.all_system_refs) == 2
    assert derived_ms._intrinsic_ref in result._transform_graph.all_system_refs
    assert source_ms._intrinsic_ref not in result._transform_graph.all_system_refs
    assert "physical" in result.coordinate_systems
    assert "physical-1" in result.coordinate_systems


def test_multiscale_with_coordinate_systems_of_rejects_differing_axes_without_derivation():
    caller_ms = _multiscale("xy")
    donor_ms = _multiscale("xyz")

    with pytest.raises(
        ValueError, match=re.escape("Cannot transfer coordinate systems from source with axes ('x', 'y', 'z')")
    ):
        caller_ms.with_coordinate_systems_of(donor_ms)


def test_multiscale_with_coordinate_systems_of_rejects_mismatching_derivation():
    caller_ms = _multiscale("xyz")
    donor_ms = _multiscale("xyz")

    with pytest.raises(ValueError, match="Incompatible axes/order"):
        caller_ms.with_coordinate_systems_of(donor_ms, derived_by=Factor.identity("xy"))


def test_multiscale_with_coordinate_systems_of_rejects_duplicate_external_system_name():
    caller_ms = _with_extra_system(_multiscale(), "world")
    donor_ms = _with_extra_system(_multiscale(), "world")

    with pytest.raises(ValueError, match="Cannot transfer coordinate systems {'world'}"):
        caller_ms.with_coordinate_systems_of(donor_ms)
