from clearscale._axis_values import Factor, Translation
from clearscale._spatial_relations import PermutationTo, ProjectionTo, SpatialRelation
from clearscale._transforms._base import Transform
from clearscale._transforms._transform_types import (
    MapAxisTransform,
    ProjectAxisTransform,
    ScaleTransform,
    TranslationTransform,
)
from clearscale.types import OrderedAxes


def relation_to_transform(relation: SpatialRelation, source_axes: OrderedAxes) -> Transform:
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
