from typing import List, Sequence, Tuple

from clearscale._axis_values import AxisKey, Factor, Translation
from clearscale._spatial_relations import PermutationTo, ProjectionTo, SpatialRelation, AxisRearrangementTo
from clearscale._transforms._base import Transform, TransformSequence
from clearscale._transforms._transform_types import (
    MapAxisTransform,
    ProjectAxisTransform,
    ScaleTransform,
    TranslationTransform,
)
from clearscale.types import OrderedAxes


def relation_to_transform(relation: SpatialRelation, source_axes: OrderedAxes) -> Transform:
    """Build one Transform for a single SpatialRelation hop. Validate that `relation`
    can operate on `source_axes` and if so, reshape it to produce a Transform payload that fits `source_axes`."""
    source_axes = tuple(source_axes)

    if isinstance(relation, Factor):
        _ensure_compatible = relation.target_axes(source_axes)
        # Inversion: When acting as a SpatialRelation (an image operation), the Factor is
        # exactly the inverse of what the coordinate transformation needs to be.
        # `derived_ms = B.as_derived_from(A, by=Factor(x=2.0))`
        # means "B was derived by downscaling A by factor 2".
        # But if downscaling A by Factor(2.0) produces B, then A[2] == B[1],
        # i.e. to obtain B-coordinates, A-coordinates must be halved.
        # hence, A--Factor(2.0)-->B is the coordinate transform A--Scale(0.5)-->B.
        # Likewise:
        # `embedded_ms = A.with_coordinate_system("B", reached_by=Factor(x=2.0))`
        # means "B is reached by downscaling A by factor 2".
        # Again, if downscaling A by Factor(2.0) produces B, then A[2] == B[1].
        return ScaleTransform(scale=tuple(relation.inverted().values()))

    if isinstance(relation, Translation):
        _ensure_compatible = relation.target_axes(source_axes)
        # Inversion: same as for Factor
        # `derived_ms = B.as_derived_from(A, by=Translation(x=1.0))`
        # means "B was derived by shifting A by 1".
        # Then A[0] == B[-1] (A's origin is outside B, because B is shifted), and B[0] == A[1]
        # i.e. to obtain B-coordinates, A-coordinates must have 1 subtracted (-1 added).
        # hence, A--Translation(1.0)-->B is the coordinate transform A--TranslationTransform(-1.0)-->B.
        # Likewise:
        # `embedded_ms = A.with_coordinate_system("B", reached_by=Translation(x=1.0))`
        # means "B is reached by shifting A by 1".
        # Again, A[0] == B[-1] - if shifting A produces B, then A's origin is outside B.
        return TranslationTransform(translation=tuple(relation.inverted().values()))

    if isinstance(relation, ProjectionTo):
        target_axes = relation.target_axes(source_axes)
        dropped = set(source_axes) - set(target_axes)
        inserted = set(target_axes) - set(source_axes)
        return ProjectAxisTransform(
            drops=tuple(i for i, axis in enumerate(source_axes) if axis in dropped),
            inserts=tuple(i for i, axis in enumerate(target_axes) if axis in inserted),
        )

    if isinstance(relation, PermutationTo):
        target_axes = relation.target_axes(source_axes)
        return MapAxisTransform(map_axis=tuple(source_axes.index(axis) for axis in target_axes))

    if isinstance(relation, AxisRearrangementTo):
        target_axes = relation.target_axes(source_axes)
        if not relation.dropped_axes(source_axes) and not relation.inserted_axes(source_axes):
            return relation_to_transform(PermutationTo(target_axes), source_axes)
        try:
            return relation_to_transform(ProjectionTo(target_axes), source_axes)
        except ValueError:
            pass
        intermediate_axes = relation.retained_axes(source_axes) + relation.inserted_axes(source_axes)
        project_axes = relation_to_transform(ProjectionTo(intermediate_axes), source_axes)
        map_axis = relation_to_transform(PermutationTo(target_axes), intermediate_axes)
        return TransformSequence((project_axes, map_axis))

    raise NotImplementedError(f"Conversion to Transform not yet implemented for {relation.__class__.__name__}")


def relations_to_transform(relations: Sequence[SpatialRelation], source_axes: OrderedAxes) -> Transform:
    """Compose a left-to-right chain of SpatialRelations into one Transform
    (a TransformSequence for more than one hop). Each relation's target axes become
    the next relation's source axes."""
    if not relations:
        raise ValueError("relations_to_transform requires at least one SpatialRelation.")
    current_axes = tuple(source_axes)
    steps: List[Transform] = []
    for relation in relations:
        steps.append(relation_to_transform(relation, current_axes))
        current_axes = relation.target_axes(current_axes)
    return TransformSequence(tuple(steps)).canonicalized()


def relation_chain_target_axes(relations: Sequence[SpatialRelation], source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
    axes = tuple(source_axes)
    for relation in relations:
        axes = relation.target_axes(axes)
    return axes
