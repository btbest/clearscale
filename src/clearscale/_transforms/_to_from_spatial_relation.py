from typing import List, Sequence, Tuple

from clearscale._axis_values import AxisKey, Factor, Translation
from clearscale._spatial_relations import PermutationTo, ProjectionTo, SpatialRelation
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
        return ScaleTransform(scale=tuple(relation.values()))

    if isinstance(relation, Translation):
        _ensure_compatible = relation.target_axes(source_axes)
        return TranslationTransform(translation=tuple(relation.values()))

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
    return TransformSequence(tuple(steps)).collapsed()


def relation_chain_target_axes(relations: Sequence[SpatialRelation], source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
    axes = tuple(source_axes)
    for relation in relations:
        axes = relation.target_axes(axes)
    return axes
