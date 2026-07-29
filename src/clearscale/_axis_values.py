import math
import numbers
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Collection, Iterator, Mapping as ABCMapping
from types import NotImplementedType
from typing import (
    Any,
    Mapping,
    Generic,
    Protocol,
    TypeVar,
    Union,
    Callable,
    Optional,
    List,
    Literal,
    Hashable,
    ItemsView,
    KeysView,
    ValuesView,
    Tuple,
)

AxisKey = Hashable
AxisKeyT = TypeVar("AxisKeyT", bound=AxisKey)
AxisKeyT_co = TypeVar("AxisKeyT_co", bound=AxisKey, covariant=True)
AxisMappedHashable = TypeVar("AxisMappedHashable", bound=Hashable)
AxisMappedPrimitive = TypeVar("AxisMappedPrimitive", int, float, str)
Axes = Union[Collection[AxisKey], str]
Scalar = Union[int, float, numbers.Real]
ShapeLike = Union["Shape", Mapping[AxisKeyT, int]]
FactorLike = Union["Factor", Mapping[AxisKeyT, float]]
RoundingFunction = Callable[[float], int]
RoundingMethod = Union[Literal["ceil"], Literal["floor"], Literal["round"], RoundingFunction]
_AxisMappingSelf = TypeVar("_AxisMappingSelf", bound="_AxisMapping[Any, Any]")
_AxisValuesSelf = TypeVar("_AxisValuesSelf", bound="_AxisValues[Any, Any]")


class _Ordered(Protocol[AxisKeyT_co]):
    """Defined order (unlike Iterable, Collection), but not necessarily indexable (unlike Sequence).
    To match e.g. strings, but also odict_keys and of course _AxisMapping"""

    def __iter__(self) -> Iterator[AxisKeyT_co]: ...
    def __len__(self) -> int: ...


OrderedAxesT = _Ordered[AxisKeyT]
OrderedAxes = _Ordered[AxisKey]


def _axis_in(axis: AxisKey, axes: Axes) -> bool:
    """Required whenever `axes` should tolerate strings like "xyz",
    because `axis` can be any Hashable, and `hashable_obj in 'tczyx'` raises TypeError"""
    if isinstance(axes, str):
        return isinstance(axis, str) and axis in axes
    return axis in axes


class _AxisMapping(ABCMapping[AxisKeyT, AxisMappedHashable], Generic[AxisKeyT, AxisMappedHashable]):
    """
    Base class for "tagged dictionaries" that map axis keys to values (like shape, resolution, unit).
    Instantiation and usage of the subclasses should work pretty much like usual dicts, but with
    guaranteed (and explicit) order of the elements and immutability.

    Currently just wraps an OrderedDict.
    Custom classes are used instead of plain OrderedDicts to:
    - hide the internal implementation from the consumer in case this might need to change in the future
    - enforce immutability
    - control which parts of the OrderedDict API are exposed
    """

    def __init__(self, *args, **kwargs):
        self._mapping: OrderedDict[AxisKeyT, AxisMappedHashable] = OrderedDict(*args, **kwargs)
        if not self._mapping:
            raise ValueError(f"Empty {self.__class__.__name__}. Received: {args=}, {kwargs=}")
        if None in self._mapping.keys():
            raise ValueError(f"None keys not allowed. Received: {list(self._mapping.keys())}")
        if None in self._mapping.values():
            raise ValueError(f"None values not allowed. Received: {list(self._mapping.values())}")

    def __repr__(self):
        map_substr = self._mapping.__repr__()[len(type(self._mapping).__name__) :]
        return str(self.__class__.__name__) + map_substr

    def __getitem__(self, key: AxisKeyT) -> AxisMappedHashable:
        return self._mapping[key]

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[AxisKeyT]:
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def keys(self) -> KeysView[AxisKeyT]:
        return self._mapping.keys()

    def values(self) -> ValuesView[AxisMappedHashable]:
        return self._mapping.values()

    def items(self) -> ItemsView[AxisKeyT, AxisMappedHashable]:
        return self._mapping.items()

    def __hash__(self):
        return hash(tuple(self._mapping.items()))

    def __eq__(self, other):
        if isinstance(other, _AxisMapping) and self.__class__ is other.__class__:
            return self._mapping == other._mapping
        if isinstance(other, OrderedDict) or isinstance(other, dict):
            return self._mapping == other
        return False

    def copy(self: _AxisMappingSelf) -> _AxisMappingSelf:
        """Doesn't provide much value, since everything is immutable anyway.
        This is just to stay consistent with the plain dict interface."""
        return self.__class__(self._mapping)

    def to_tuple(self) -> Tuple[AxisMappedHashable, ...]:
        return tuple(self.values())

    def to_list(self) -> List[AxisMappedHashable]:
        return list(self.values())


class _AxisValues(ABC, _AxisMapping[AxisKeyT, AxisMappedPrimitive], Generic[AxisKeyT, AxisMappedPrimitive]):
    @classmethod
    @abstractmethod
    def _default(cls) -> AxisMappedPrimitive: ...

    """Subclasses provide the default value for axes where no value is provided."""

    @classmethod
    def fromkeys(cls: type[_AxisValuesSelf], keys: OrderedAxesT) -> _AxisValuesSelf:
        return cls(zip(keys, [cls._default()] * len(keys)))

    def is_default(self) -> bool:
        """Check if all values in this metadata are the default value."""
        return all(v == self._default() for v in self.values())

    def with_axes(self: _AxisValuesSelf, axes: OrderedAxesT) -> _AxisValuesSelf:
        """Order like axes. Drop axes, or insert new axes with default value if necessary."""
        if not axes:
            raise ValueError(f"Cannot create empty {self.__class__.__name__}. Attempted reorder to: {axes!r}")
        reordered_items = [(a, self[a] if a in self else self._default()) for a in axes]
        return self.__class__(reordered_items)

    def with_axes_order(self: _AxisValuesSelf, axes: OrderedAxesT) -> _AxisValuesSelf:
        """Order like given axes (but no new insertions or drops)."""
        if not axes:
            raise ValueError(f"Cannot create empty {self.__class__.__name__}. Attempted reorder to {axes!r}.")
        would_drop = tuple(a for a in self if a not in axes)
        if would_drop:
            raise ValueError(
                f"Cannot reorder {self.__class__.__name__} to axes {axes!r}. This would drop: {would_drop!r}."
            )
        reordered_items = [(a, self[a]) for a in axes if a in self]
        return self.__class__(reordered_items)

    def without_axes_except(self: _AxisValuesSelf, axes: Axes) -> _AxisValuesSelf:
        """Keep only given axes (no reordering)."""
        kept_items = [(a, self[a]) for a in self if _axis_in(a, axes)]
        if not kept_items:
            raise ValueError(
                f"Cannot create empty {self.__class__.__name__}. "
                f"None of the specified axes {axes!r} are present in {list(self.keys())}."
            )
        return self.__class__(kept_items)

    def without_axes(self: _AxisValuesSelf, axes: Axes) -> _AxisValuesSelf:
        """Drop given axes."""
        kept_items = [(a, self[a]) for a in self if not _axis_in(a, axes)]
        if not kept_items:
            raise ValueError(f"Cannot create empty {self.__class__.__name__}. Removing {axes!r} would leave no axes.")
        return self.__class__(kept_items)

    def without_default_values(self: _AxisValuesSelf) -> _AxisValuesSelf:
        """Drop axes with default value"""
        if self.is_default():
            raise ValueError(
                f"Cannot create empty {self.__class__.__name__}. Removing all defaults would leave no axes."
            )
        return self.without_axes(tuple(a for a in self if self[a] == self._default()))

    def with_default(self: _AxisValuesSelf, axes: Axes) -> _AxisValuesSelf:
        """
        Reset the values for `axes` to the type's default value, keeping the rest unchanged.

        Equivalent to pandas.DataFrame.mask(self in axes, other=self._default()).
        """
        reset_items = [(a, self._default() if _axis_in(a, axes) else self[a]) for a in self]
        return self.__class__(reset_items)

    def with_default_except(self: _AxisValuesSelf, axes: Axes) -> _AxisValuesSelf:
        """
        Keep the values for `axes` and reset the remaining values to the type's default value.

        Equivalent to pandas.DataFrame.where(self in axes, other=self._default()).
        """
        keep_items = [(a, self[a] if _axis_in(a, axes) else self._default()) for a in self]
        return self.__class__(keep_items)

    def with_values(
        self: _AxisValuesSelf, other: Mapping[AxisKeyT, Union[Scalar, str]], *, only: Optional[Axes] = None
    ) -> _AxisValuesSelf:
        if not isinstance(other, ABCMapping):
            raise TypeError(f"Pass {{axis: value}} mapping. Received: {other!r}")
        if not other or (only is not None and not only):
            return self
        replaced_items = []
        for a, old_v in self.items():
            new_v: Union[Scalar, str] = old_v
            if (only is None or _axis_in(a, only)) and a in other and other[a] is not None:
                new_v = other[a]
            if type(old_v) is type(new_v):
                pass
            elif isinstance(old_v, bool) and isinstance(new_v, numbers.Integral):
                new_v = bool(new_v)
            elif isinstance(old_v, int) and not isinstance(old_v, bool) and isinstance(new_v, numbers.Integral):
                # bools are instances of int
                new_v = int(new_v)
            elif isinstance(old_v, float) and isinstance(new_v, numbers.Real):
                new_v = float(new_v)
            else:
                raise TypeError(
                    f"Invalid attempt to replace values in {type(self).__name__}: Cannot cast "
                    f"{new_v!r} of type {type(new_v).__name__} to {type(old_v).__name__}"
                )
            replaced_items.append((a, new_v))

        return self.__class__(replaced_items)


def _require_identical_axes(left: Mapping[AxisKey, object], right: Mapping[AxisKey, object]) -> None:
    left_axes = list(left)
    right_axes = list(right)
    if left_axes != right_axes:
        raise ValueError(f"Incompatible axes/order: {left_axes} vs {right_axes}")


def _require_axes_present(
    container: Mapping[AxisKey, object],
    required: Mapping[AxisKey, object],
    *,
    container_name: str,
    required_name: str,
) -> None:
    missing_axes = [axis for axis in required if axis not in container]
    if missing_axes:
        raise ValueError(
            f"{container_name} must contain all axes from {required_name}. "
            f"Missing axes: {missing_axes}; available axes: {list(container)}"
        )


def _normalize_rounding(rounding: RoundingMethod) -> RoundingFunction:
    if rounding == "ceil":
        return math.ceil
    if rounding == "floor":
        return int
    if rounding == "round":
        return round
    return rounding


class _AxisFloats(_AxisValues[AxisKey, float], ABC):
    """
    Base for Scaling, Resolution and Translation. Ensures values are floats.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if isinstance(value, numbers.Integral) or isinstance(value, numbers.Real):
                self._mapping[axis] = float(value)
                continue
            if not isinstance(value, float):
                raise TypeError(f"All values must be float. Got {type(value).__name__} for axis '{axis}'.")


class Factor(_AxisFloats):
    """
    Describes relative scaling factors from some shape to another.
    The values are in units of "raw pixels per scaled pixel".
    This makes them divisors for the original shape:
    ```
    scaled_shape = Shape(x=10).scaled_by(Factor(x=2.0), rounding="floor")
    assert scaled_shape == Shape(x=5)
    ```
    """

    @classmethod
    def _default(cls) -> float:
        return 1.0

    def with_axes(self, axes: OrderedAxes) -> "Factor":
        """
        Reorder to `axes`.

        Inserts 1.0 for target axes that this Factor doesn't have yet.

        Examples:
            >>> res = Factor(z=0.25, y=120., x=120., t=0.1)
            >>> res.with_axes("tczyx")
            Factor(t=0.1, c=1.0, z=0.25, y=120.0, x=120.0)
        """
        return super().with_axes(axes)

    @classmethod
    def identity(cls, axes: OrderedAxes) -> "Factor":
        """Create a new identity Scaling (1.0 along all axes) with `axes`."""
        return super().fromkeys(axes)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if value <= 0:
                raise ValueError(f"Scaling factor cannot be 0 or negative (got {value} for axis '{axis}').")

    @classmethod
    def uniform(cls, axes: OrderedAxes, factor: Scalar) -> "Factor":
        """Create a new Scaling with `axes` and all values being `factor`."""
        return Factor(zip(axes, [factor] * len(axes)))

    def magnitude(self):
        return math.prod(self.values())

    def is_identity(self) -> bool:
        """True if this Factor is the identity scaling (1.0 along all axes)."""
        return super().is_default()

    def is_downscaling(self) -> bool:
        """True if the product across all axes is greater than 1.
        Note: Factors act as divisors for shape (e.g. 1024 / 2 = 512)."""
        return math.prod(self.values()) > 1

    def is_upscaling(self) -> bool:
        """True if the product across all axes is lesser than 1.
        Note: Factors act as divisors for shape (e.g. 1024 / 0.5 = 2048)."""
        return math.prod(self.values()) < 1

    def inverted(self) -> "Factor":
        """Axis-wise 1/factor."""
        inverted_items = [(a, 1 / self[a]) for a in self]
        return self.__class__(inverted_items)

    def __mul__(self, other: object) -> Union["Factor", "PixelSize", NotImplementedType]:
        if isinstance(other, Factor):
            _require_identical_axes(self, other)
            return Factor((a, self[a] * other[a]) for a in self)
        if isinstance(other, PixelSize):
            return other.scaled_by(self)
        return NotImplemented

    def __truediv__(self, other: object) -> Union["Factor", NotImplementedType]:
        if not isinstance(other, Factor):
            return NotImplemented
        _require_identical_axes(self, other)
        return Factor((a, self[a] / other[a]) for a in self)

    def with_identity(self, axes: Axes) -> "Factor":
        """Reset the values for `axes` to 1.0."""
        return super().with_default(axes)

    def with_identity_except(self, axes: Axes) -> "Factor":
        """Reset the values for all axes except `axes` to 1.0."""
        return super().with_default_except(axes)

    def to_physical(self, base: "PixelSize") -> "PixelSize":
        """
        Convert relative scaling factor to absolute physical pixel size.
        Identical to base.scaled_by(self).
        """
        return PixelSize(base).scaled_by(self)


class PixelSize(_AxisFloats):
    """
    Describes absolute scaling factors, i.e. physical pixel size.
    The values are in "units (e.g. nanometer) per pixel".
    """

    @classmethod
    def _default(cls) -> float:
        return 1.0

    @classmethod
    def identity(cls, axes: OrderedAxes) -> "PixelSize":
        """Create a new identity PixelSize (1.0 along all axes) with `axes`."""
        return super().fromkeys(axes)

    @classmethod
    def from_vigra(cls, vigra_axistags: Any) -> "PixelSize":
        """Extract AxisInfo.resolution from the provided `vigra.AxisTags`"""
        vigra_default_resolution = 0.0
        axes = []
        resolutions = []
        for tag in vigra_axistags:
            axes.append(tag.key)
            resolutions.append(tag.resolution if tag.resolution != vigra_default_resolution else cls._default())
        return cls(zip(axes, resolutions))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if value <= 0:
                raise ValueError(f"Pixel size cannot be 0 or negative (got {value} for axis '{axis}').")

    def is_identity(self) -> bool:
        """True if this PixelSize is the unit size (1.0 along all axes)."""
        return super().is_default()

    def with_identity(self, axes: Axes) -> "PixelSize":
        """Reset the values for `axes` to 1.0."""
        return super().with_default(axes)

    def with_identity_except(self, axes: Axes) -> "PixelSize":
        """Reset the values for all axes except `axes` to 1.0."""
        return super().with_default_except(axes)

    def to_vigra(self, axistags: Optional[Any]) -> Any:
        """
        Creates or modifies `vigra.AxisTags` with the values in this PixelSize.
        :param axistags: (Optional) Existing AxisTags to modify.
            Only transfers values for axes shared between the AxisTags and this PixelSize.
        :return: Returns the newly created or the modified AxisTags.
        """
        try:
            import vigra  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                'This function requires the package "vigra". '
                "Please install it using e.g. `conda install -c conda-forge vigra`"
            ) from e
        tags = axistags if axistags else vigra.defaultAxistags("".join(str(axis) for axis in self.keys()))
        for tag in tags:
            if tag.key in self and self[tag.key] != self._default():
                tags.setResolution(tag.key, self[tag.key])
        return tags

    def scaled_by(self, factor: Union[Factor, Mapping[AxisKey, float], Scalar]) -> "PixelSize":
        """
        Scale this PixelSize by factor to obtain a scaled PixelSize.
        This is an axis-wise operation:
        - Missing axes in `factor` default to 1.0 (no change)
        - Passing a scalar (float/int) applies it uniformly to all axes
        - Extra axes in `factor` are rejected
        Note if passing scalar: factor 2.0 means double pixel size = half resolution.
        """
        if isinstance(factor, Factor):
            pass
        elif isinstance(factor, numbers.Real):
            factor = Factor.uniform(self, float(factor))
        else:
            factor = Factor(factor)
        base_axes = set(self.keys())
        factor_axes = set(factor.keys())
        invalid_axes = factor_axes - base_axes
        if invalid_axes:
            raise ValueError(
                f"Attempted to scale axes with no base pixel size: "
                f"{sorted(invalid_axes, key=str)} not present in {sorted(base_axes, key=str)}"
            )
        reordered = factor.with_axes(self)
        scaled_items = [(a, reordered[a] * self[a]) for a in self]
        return PixelSize(scaled_items)

    def __mul__(self, other: object) -> Union["PixelSize", "Translation", NotImplementedType]:
        if isinstance(other, Factor):
            return self.scaled_by(other)
        if isinstance(other, PixelOffset):
            return other.to_physical(self)
        return NotImplemented

    def __truediv__(self, other: object) -> Union["PixelSize", "Factor", NotImplementedType]:
        if isinstance(other, Factor):
            return self.scaled_by(other.inverted())
        if isinstance(other, PixelSize):
            _require_identical_axes(self, other)
            return Factor((a, self[a] / other[a]) for a in self)
        return NotImplemented


class Translation(_AxisFloats):
    """Describes a shift in physical units."""

    @classmethod
    def _default(cls) -> float:
        return 0.0

    def with_axes(self, axes: OrderedAxes) -> "Translation":
        """
        Reorder to `axes`.

        Inserts 0.0 for target axes that this Translation doesn't have yet.

        Examples:
            >>> translate = Translation(y=0.5, x=0.5, t=0.3)
            >>> translate.with_axes("tczyx")
            Translation(t=0.3, c=0.0, z=0.0, y=0.5, x=0.5)
        """
        return super().with_axes(axes)

    @classmethod
    def identity(cls, axes: OrderedAxes) -> "Translation":
        """Create a new identity Translation (0.0 along all axes) with `axes`."""
        return super().fromkeys(axes)

    def is_identity(self) -> bool:
        """True if this Translation is the identity translation (0.0 along all axes)."""
        return super().is_default()

    def is_identity_along(self, axes: Axes) -> bool:
        """True if this Translation is 0.0 for `axes`."""
        if not axes:
            return True
        return self.with_axes(axes).is_identity()

    def __add__(self, other: object) -> Union["Translation", NotImplementedType]:
        if not isinstance(other, Translation):
            return NotImplemented
        _require_identical_axes(self, other)
        return Translation((a, self[a] + other[a]) for a in self)

    def __sub__(self, other: object) -> Union["Translation", NotImplementedType]:
        if not isinstance(other, Translation):
            return NotImplemented
        _require_identical_axes(self, other)
        return Translation((a, self[a] - other[a]) for a in self)

    def with_identity(self, axes: Axes) -> "Translation":
        """Reset the values for `axes` to 0.0."""
        return super().with_default(axes)

    def with_identity_except(self, axes: Axes) -> "Translation":
        """Reset the values for all axes except `axes` to 0.0."""
        return super().with_default_except(axes)

    def to_pixel_offset(
        self,
        pixel_size: Union[PixelSize, Mapping[AxisKey, float]],
        *,
        rounding: RoundingMethod,
    ) -> "PixelOffset":
        """
        Convert this Translation from physical units to a PixelOffset.

        Extra axes in `pixel_size` are ignored, but all Translation axes must be present.
        """
        _require_axes_present(pixel_size, self, container_name="pixel_size", required_name="Translation")
        rounding = _normalize_rounding(rounding)
        return PixelOffset((a, rounding(self[a] / pixel_size[a])) for a in self)


class Unit(_AxisValues[AxisKey, str]):
    """
    Describes the physical units of the image's axes.
    Together with a Resolution, this can form a complete pixel size description.
    Example: {"t": "seconds", "x": "nm, "y": "nm, "c": ""}.
    """

    @classmethod
    def _default(cls) -> str:
        return ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if not isinstance(value, str):
                raise TypeError(f"All values must be strings. Got {type(value).__name__} for axis '{axis}'.")

    def with_axes(self, axes: OrderedAxes) -> "Unit":
        """
        Reorder to `axes`.

        Inserts "" for target axes that this Unit doesn't have yet.

        Examples:
            >>> unit = Unit(y="nm", x="nm", t="sec")
            >>> unit.with_axes("tczyx")
            Unit(t="sec", c="", z="", ="nm", x="nm")
        """
        return super().with_axes(axes)

    @classmethod
    def empty(cls, axes: OrderedAxes) -> "Unit":
        """Create a new Unit with `axes` and empty string values."""
        return super().fromkeys(axes)


class PixelOffset(_AxisValues[AxisKey, int]):
    """
    Describes the number of pixels of distance from some reference point to another.
    """

    @classmethod
    def _default(cls) -> int:
        return 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if not isinstance(value, numbers.Integral):
                raise TypeError(f"All values must be integer. Got {type(value).__name__} for axis '{axis}'.")
            self._mapping[axis] = int(value)

    def with_axes(self, axes: OrderedAxes) -> "PixelOffset":
        """
        Reorder to `axes`.

        Inserts 0 for target axes that this PixelOffset doesn't have yet.

        Examples:
            >>> crop_offset = PixelOffset(y=15, x=37, t=23)
            >>> crop_offset.with_axes("tczyx")
            PixelOffset(t=23, c=0, z=0, y=15, x=37)
        """
        return super().with_axes(axes)

    def to_physical(self, pixel_size: Union[PixelSize, Mapping[AxisKey, float]]) -> Translation:
        """
        Multiply with `resolution` to obtain this PixelOffset as a Translation in physical units.
        """
        _require_axes_present(pixel_size, self, container_name="pixel_size", required_name="PixelOffset")
        items_in_physical_units = [(a, self[a] * pixel_size[a]) for a in self]
        return Translation(items_in_physical_units)

    def __add__(self, other: object) -> Union["PixelOffset", NotImplementedType]:
        if not isinstance(other, PixelOffset):
            return NotImplemented
        _require_identical_axes(self, other)
        return PixelOffset((a, self[a] + other[a]) for a in self)

    def __sub__(self, other: object) -> Union["PixelOffset", NotImplementedType]:
        if not isinstance(other, PixelOffset):
            return NotImplemented
        _require_identical_axes(self, other)
        return PixelOffset((a, self[a] - other[a]) for a in self)

    def __mul__(self, other: object) -> Union["Translation", NotImplementedType]:
        if not isinstance(other, PixelSize):
            return NotImplemented
        return self.to_physical(other)


class Shape(_AxisValues[AxisKey, int]):
    """
    Describes the number of pixels in the image.
    """

    @classmethod
    def _default(cls) -> int:
        return 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for axis, value in self._mapping.items():
            if not isinstance(value, numbers.Integral):
                raise TypeError(f"All values must be integer. Got {type(value).__name__} for axis '{axis}'.")
            if value < 1:
                raise ValueError(f"Shape cannot be lower than 1 (got {value} for axis '{axis}').")
            self._mapping[axis] = int(value)

    @classmethod
    def all_singletons(cls, axes: OrderedAxes) -> "Shape":
        return super().fromkeys(axes)

    def with_axes(self, axes: OrderedAxes) -> "Shape":
        """
        Reorder to `axes`.

        Inserts 1 for target axes that this Shape doesn't have yet.

        Examples:
            >>> shape = Shape(y=256, x=256, t=23)
            >>> shape.with_axes("tczyx")
            Shape(t=23, c=1, z=1, y=256, x=256)
        """
        return super().with_axes(axes)

    def with_ones(self, axes: Axes) -> "Shape":
        """Reset the values for `axes` to 1."""
        return super().with_default(axes)

    def without_singletons(self):
        """Drop all axes where the shape is 1."""
        return self.without_default_values()

    def matches(self, other: ShapeLike, *, only: Optional[Axes] = None) -> bool:
        """Permissive value matching.
        True if shapes are equal in all shared axes, optionally further constrained to `only`."""
        shared = set(self.keys()) & set(other.keys())
        if only:
            shared &= set(only)
        return all(self[axis] == other[axis] for axis in shared)

    def non_singleton_axes(self, axes: Optional[Axes] = None) -> List[AxisKey]:
        """Return axes along which this Shape is singleton (value is 1).

        :param axes: (Optional) Return subset of `axes` along which this Shape is singleton."""
        axes = axes if axes is not None else self.keys()
        return [a for a in axes if a in self and self[a] != self._default()]

    def scaled_by(self, factor: Union[Factor, Mapping[AxisKey, float]], *, rounding: RoundingMethod) -> "Shape":
        """
        Returns the Shape of this image when scaled by `factor`.

        :param factor:
        :param rounding: Specify how the scaling implementation used to scale the image data treats uneven cases.
            Allowed: "ceil", "floor", or a function that takes a float and returns an int.
            Example: When scaling 11 pixels by factor 2, does your method produce 5 or 6 pixels?
            Common examples:
             - skimage.transform.rescale: `rounding=round`.
             - skimage.transform.downscale_local_mean: `rounding="ceil"`.
        """
        rounding = _normalize_rounding(rounding)
        factor = Factor(factor).with_axes(self)

        def _rescale_size(s: int, f: float) -> int:
            """
            Rescale a single dimension of a shape.
            Floor-round to match behavior of OpResize, and ensure minimum size is 1.
            """
            return max(rounding(s / f), self._default())

        scaled_shape = self.__class__([(a, _rescale_size(size, factor[a])) for a, size in self.items()])
        return scaled_shape

    def scaling_to(self, resized: "Shape") -> "Factor":
        """
        Returns the Scaling factors of this Shape that would produce the `resized` shape.
        """
        _require_identical_axes(self, resized)
        # In multiscale image context, scaling "factors" are technically divisors for the shape
        # (factor 2.0 means half the shape).
        return Factor((a, self[a] / resized[a]) for a in self)

    def __truediv__(self, other: object) -> Union["Factor", NotImplementedType]:
        if not isinstance(other, Shape):
            return NotImplemented
        return self.scaling_to(other)
