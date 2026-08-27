from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Tuple

from clearscale.types import AxisKey, OrderedAxes


class SpatialRelation(ABC):
    """Any object that can describe how a new image was derived from an existing one is a spatial relation."""

    @abstractmethod
    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        """Raise if self is not compatible with `source_axes`, and return the axes this relation results in when applied to source_axes."""


@dataclass(frozen=True, slots=True)
class PermutationTo(SpatialRelation):
    _order: Tuple[AxisKey, ...]

    def __init__(self, target_axes: OrderedAxes):
        target_axes = tuple(target_axes)
        if not target_axes:
            raise ValueError("PermutationTo requires at least one axis.")
        if len(set(target_axes)) != len(target_axes):
            raise ValueError(f"target_axes must be unique. Received: {target_axes!r}")
        object.__setattr__(self, "_order", target_axes)

    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        source_axes = tuple(source_axes)
        if set(source_axes) != set(self._order):
            raise ValueError(
                f"PermutationTo cannot insert or drop axes, only reorder. "
                f"Source axes {source_axes!r} are not the same set as target axes {self._order!r}."
            )
        return self._order


@dataclass(frozen=True, slots=True)
class ProjectionTo(SpatialRelation):
    _targets: Tuple[AxisKey, ...]

    def __init__(self, target_axes: OrderedAxes):
        target_axes = tuple(target_axes)
        if not target_axes:
            raise ValueError("ProjectionTo requires at least one axis.")
        if len(set(target_axes)) != len(target_axes):
            raise ValueError(f"target_axes must be unique. Received: {target_axes!r}")
        object.__setattr__(self, "_targets", target_axes)

    def target_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        self._require_retained_axes_not_reordered(source_axes)
        return self._targets

    def dropped_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        source_axes = tuple(source_axes)
        self._require_retained_axes_not_reordered(source_axes)
        return tuple(a for a in source_axes if a not in self._targets)

    def inserted_axes(self, source_axes: OrderedAxes) -> Tuple[AxisKey, ...]:
        source_axes = tuple(source_axes)
        self._require_retained_axes_not_reordered(source_axes)
        return tuple(a for a in self._targets if a not in source_axes)

    def _require_retained_axes_not_reordered(self, source_axes: OrderedAxes) -> None:
        shared_source = tuple(a for a in source_axes if a in self._targets)
        shared_target = tuple(a for a in self._targets if a in source_axes)
        if shared_source != shared_target:
            raise ValueError(
                f"Projection cannot reorder retained axes ({shared_source} -> {shared_target}). "
                f"Source axes: {source_axes!r}; target axes: {self._targets!r}."
            )
