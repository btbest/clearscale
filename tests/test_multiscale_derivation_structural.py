import pytest

from clearscale._axis_values import Shape
from clearscale._multiscale import Multiscale, Scale
from clearscale._spatial_relations import ProjectionTo
from clearscale._transforms import CoordinateSystem, IdentityTransform, ProjectAxisTransform, TransformGraph


def _multiscale(axes="zyx", size=4):
    return Multiscale({"s0": Scale(shape=Shape(zip(axes, [size] * len(axes))))})


def _with_edge(ms, name, target_axes, *, direction="forward", via=None):
    target_ref = CoordinateSystem.without_semantics(target_axes).as_ref(name)
    transform = via if via is not None else IdentityTransform()
    edge = (
        transform.bound(source=ms._intrinsic_ref, target=target_ref)
        if direction == "forward"
        else transform.bound(source=target_ref, target=ms._intrinsic_ref)
    )
    graph = TransformGraph(transforms=ms._transform_graph.transforms + (edge,))
    return Multiscale(ms.items(), _transform_graph=graph, _intrinsic_ref=ms._intrinsic_ref)


def _source_plain():
    return _multiscale("zyx"), None


def _source_invertible_forward():
    return _with_edge(_multiscale("zyx"), "system", "zyx", direction="forward"), "system"


def _source_invertible_reverse():
    return _with_edge(_multiscale("zyx"), "system", "zyx", direction="reverse"), "system"


def _source_drop_forward():
    # source(zyx) -> system(yx): system is a projection of source, drop stored source->system
    via = ProjectAxisTransform(drops=(0,))  # drop 'z'
    return _with_edge(_multiscale("zyx"), "system", "yx", direction="forward", via=via), "system"


def _source_drop_reverse():
    # system(tzyx) -> source(zyx): source is a projection of system, drop stored system->source
    via = ProjectAxisTransform(drops=(0,))  # drop 't'
    return _with_edge(_multiscale("zyx"), "system", "tzyx", direction="reverse", via=via), "system"


SOURCE_VARIANTS = {
    "plain": _source_plain,
    "source_to_system_invertible": _source_invertible_forward,
    "system_to_source_invertible": _source_invertible_reverse,
    "source_to_system_drop": _source_drop_forward,
    "system_to_source_drop": _source_drop_reverse,
}


DERIVED_BY_VARIANTS = {
    "none": (None, "zyx"),
    "insert": (ProjectionTo("tczyx"), "tczyx"),
    "drop": (ProjectionTo("yx"), "yx"),
}


@pytest.mark.parametrize("derivation_variant", DERIVED_BY_VARIANTS.keys())
@pytest.mark.parametrize("source_variant", SOURCE_VARIANTS.keys())
def test_as_derived_from_carries_source_system_per_structural_combination(source_variant, derivation_variant, recwarn):
    if (source_variant, derivation_variant) == ("source_to_system_drop", "drop"):
        return  # See test below; this is the only combination that cannot be carried over yet
    source_ms, system_name = SOURCE_VARIANTS[source_variant]()
    by, derived_axes = DERIVED_BY_VARIANTS[derivation_variant]
    derived_ms = _multiscale(derived_axes)

    result = derived_ms.as_derived_from(source_ms, by=by)
    assert len(recwarn) == 0
    if system_name is not None:
        assert system_name in result.coordinate_systems


def test_as_derived_from_cannot_carry_over_axis_dropping_system_when_derivation_also_drops():
    """
    `derived_ms <--dropA-- source_ms --dropB--> satellite_system`
    Tough: Neither edge can be inverted to form a unidirectional path, so preserving this
    means keeping `source_ms` in the graph. We don't want that.
    The direct `derived_ms --X--> satellite_system` *can* be computed when dropB is a subset of dropA
    (or vice versa).
    Implementing this is deferred until someone needs it.
    """
    source_variant, derivation_variant = ("source_to_system_drop", "drop")
    source_ms, system_name = SOURCE_VARIANTS[source_variant]()
    by, derived_axes = DERIVED_BY_VARIANTS[derivation_variant]
    derived_ms = _multiscale(derived_axes)

    with pytest.warns(UserWarning, match="Cannot carry over"):
        result = derived_ms.as_derived_from(source_ms, by=by)
    assert system_name not in result.coordinate_systems


def test_as_derived_from_preserves_derived_ms_own_existing_system():
    source_ms, _ = _source_invertible_forward()
    derived_ms = _with_edge(_multiscale("tczyx"), "derived_own", "tczyx", direction="forward")

    result = derived_ms.as_derived_from(source_ms, by=ProjectionTo("tczyx"))

    assert "derived_own" in result.coordinate_systems
    assert "system" in result.coordinate_systems


def test_as_derived_from_is_idempotent_when_source_intrinsic_already_connected():
    source_ms, _ = _source_plain()
    by = ProjectionTo("tczyx")

    once = _multiscale("tczyx").as_derived_from(source_ms, by=by)
    twice = once.as_derived_from(source_ms, by=by)

    assert twice._transform_graph.transforms == once._transform_graph.transforms
    assert len(twice._transform_graph.all_system_refs) == len(once._transform_graph.all_system_refs)


def test_as_derived_from_is_idempotent_when_source_other_system_already_connected():
    source_ms, system_name = _source_invertible_forward()
    by = ProjectionTo("tczyx")

    once = _multiscale("tczyx").as_derived_from(source_ms, by=by)
    twice = once.as_derived_from(source_ms, by=by)

    assert twice._transform_graph.transforms == once._transform_graph.transforms
    system_ref = next(r for r in twice._transform_graph.all_system_refs if r.name == system_name)
    edges_touching_system = [t for t in twice._transform_graph.transforms if system_ref in (t.source, t.target)]
    assert len(edges_touching_system) == 1, "second call must not duplicate"
