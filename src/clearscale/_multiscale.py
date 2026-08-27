import warnings
from abc import ABC
from collections import OrderedDict, defaultdict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import (
    Optional,
    TypeVar,
    Mapping,
    Generic,
    Union,
    Sequence,
    Set,
    Callable,
    Iterable,
    List,
    Tuple,
    Literal,
    Dict,
    Any,
    Hashable,
    Iterator,
    ItemsView,
    KeysView,
    ValuesView,
    Collection,
)

from clearscale._axis_values import (
    Shape,
    Factor,
    PixelSize,
    Unit,
    Translation,
    PixelOffset,
    ShapeLike,
    FactorLike,
    Axes,
    RoundingMethod,
    OrderedAxes,
    AxisKey,
)
from clearscale._errors import NoSuchCoordinateSystemError
from clearscale._spatial_relations import SpatialRelation
from clearscale._transforms import (
    CoordinateSystemName,
    CoordinateSystem,
    TransformGraph,
    NodeRef,
    IdentityTransform,
    TransformGraphNode,
    PRE_TRANSFORMS_VERSIONS,
    Transform,
    _UnresolvedRef,
    relation_to_transform,
    ScaleTransform,
    TransformSequence,
    ProjectAxisTransform,
    TranslationTransform,
    MapAxisTransform,
)
from clearscale._services import ome_zarr, precomputed

ScaleKey = str
ValueType = TypeVar("ValueType", Shape, Factor, "Scale")
AxisValuesType = TypeVar("AxisValuesType", Shape, Factor)
_ScaleMappingSelf = TypeVar("_ScaleMappingSelf", bound="_ScaleMapping[Any]")
_ScaledAxisValuesSelf = TypeVar("_ScaledAxisValuesSelf", bound="_ScaledAxisValues[Any]")
DEFAULT_NAME_PATTERN = "s{}"

TranslationShiftFunction = Callable[["Scale", "Scale"], "Translation"]
"""
base_scale: the reference scale being transformed from
target_scale: the new scale being created (with 0 translation)
Returns: target_scale's translation
"""


class DuplicatePolicy(str, Enum):
    ERROR = "error"
    KEEP_ALL = "keep_all"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"


@dataclass(frozen=True, slots=True, init=False)
class Scale:
    shape: Shape
    pixel_size: PixelSize
    unit: Unit
    translation: Translation

    def __init__(
        self,
        shape: ShapeLike,
        pixel_size: Optional[Union[PixelSize, Mapping[AxisKey, float]]] = None,
        unit: Optional[Union[Unit, Mapping[AxisKey, str]]] = None,
        translation: Optional[Union[Translation, Mapping[AxisKey, float]]] = None,
    ):
        shape = Shape(shape)
        pixel_size = PixelSize.fromkeys(shape) if pixel_size is None else PixelSize(pixel_size)
        unit = Unit.fromkeys(shape) if unit is None else Unit(unit)
        translation = Translation.fromkeys(shape) if translation is None else Translation(translation)
        if shape.keys() != pixel_size.keys() or shape.keys() != unit.keys() or shape.keys() != translation.keys():
            raise ValueError(
                f"Tried to set up invalid scale: Axiskeys differ "
                f"(shape={list(shape.keys())}, "
                f"pixel_size={list(pixel_size.keys())}, "
                f"translation={list(translation.keys())}, "
                f"unit={list(unit.keys())})"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "pixel_size", pixel_size)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "translation", translation)

    def with_axes(self, axes: OrderedAxes) -> "Scale":
        """Build a Scale with all properties produced by their respective `.with_axes`."""
        if not axes:
            raise ValueError(f"Cannot create empty {self.__class__.__name__}. Attempted reorder to: {axes!r}")
        return Scale(
            shape=self.shape.with_axes(axes),
            pixel_size=self.pixel_size.with_axes(axes),
            unit=self.unit.with_axes(axes),
            translation=self.translation.with_axes(axes),
        )

    def has_physical_meta(self):
        return not self.unit.is_default() or not self.pixel_size.is_default()

    def to_display_string(self, name=""):
        shape = ", ".join(f"{axis}: {size}" for axis, size in self.shape.items())
        name_and_shape = f'"{name}" ({shape})' if name else f"{shape}"
        pixel_size = ""
        if self.has_physical_meta():
            axis_strings = []
            for axis in self.shape.keys():
                if axis == "c":
                    continue
                pixel_size = self.pixel_size[axis]
                unit = ""
                if self.unit[axis]:
                    unit = f" {self.unit[axis]}"
                elif axis != "t":
                    unit = " px"
                axis_strings.append(f"{axis}: {pixel_size:g}{unit}")
            pixel_size = " at pixel size: " + ", ".join(axis_strings)
        return f"{name_and_shape}{pixel_size}"


class _ScaleMapping(ABC, ABCMapping[ScaleKey, ValueType], Generic[ValueType]):
    """Common base class for Multiscale, BlueprintShapes and BlueprintFactors"""

    def __init__(self, *args, **kwargs):
        self._mapping = OrderedDict(*args, **kwargs)
        if not self._mapping:
            raise ValueError(f"Cannot instantiate empty {self.__class__.__name__}")
        if any(v is None for v in self._mapping.values()):
            raise ValueError(f"None values not allowed. Received: {list(self._mapping.values())}")

    def __repr__(self):
        map_substr = self._mapping.__repr__()[len(type(self._mapping).__name__) :]
        return str(self.__class__.__name__) + map_substr

    def __getitem__(self, key: ScaleKey) -> ValueType:
        if key not in self:
            raise KeyError(f"No such scale: '{key}' (available: {list(self.keys())})")
        return self._mapping[key]

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __iter__(self) -> Iterator[ScaleKey]:
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def keys(self) -> KeysView[ScaleKey]:
        return self._mapping.keys()

    def values(self) -> ValuesView[ValueType]:
        return self._mapping.values()

    def first_value(self) -> ValueType:
        return next(iter(self.values()))

    def items(self) -> ItemsView[ScaleKey, ValueType]:
        return self._mapping.items()

    def __hash__(self):
        return hash(tuple(self._mapping.items()))

    def __eq__(self, other):
        if isinstance(other, _ScaleMapping):
            return self._mapping == other._mapping
        if isinstance(other, ABCMapping):
            return self._mapping == other
        return False

    def copy(self: _ScaleMappingSelf) -> _ScaleMappingSelf:
        return self.__class__(self._mapping)

    def filter_items(self: _ScaleMappingSelf, keep_func: Callable[[ScaleKey, ValueType], bool]) -> _ScaleMappingSelf:
        items = [(k, v) for k, v in self.items() if keep_func(k, v)]
        return self.__class__(items)

    def with_keys(
        self: _ScaleMappingSelf,
        keys_pattern_or_func: Union[Sequence[ScaleKey], str, Callable[[int, ScaleKey, ValueType], ScaleKey]],
    ) -> _ScaleMappingSelf:
        """
        Assign new scale keys using one of:
        - a sequence of new scale keys (one per current scale, unique)
        - a string format pattern with placeholder for the scale's int index
        - a function that takes the int index, the old scale key, and the Scale object, and returns a new key
        """
        if isinstance(keys_pattern_or_func, str):
            pattern = keys_pattern_or_func
            if pattern.format(0) == pattern:
                raise ValueError(
                    f"Name pattern must contain exactly one placeholder for scale index (received: '{pattern}')"
                )
            items = [(keys_pattern_or_func.format(i), v) for i, v in enumerate(self.values())]
            return self.__class__(items)
        elif callable(keys_pattern_or_func):
            new_keys = self._generate_and_validate_new_keys(keys_pattern_or_func)
            return self.__class__(zip(new_keys, self.values()))
        else:
            new_keys = keys_pattern_or_func
            if len(new_keys) != len(self):
                raise ValueError(
                    f"Must provide a new key for every current key: {list(self.keys())}. Received: {new_keys}"
                )
            if not self._all_unique(new_keys):
                raise ValueError(f"All new scale keys must be unique. Received: {new_keys}")
            return self.__class__(zip(new_keys, self.values()))

    def drop_before(self: _ScaleMappingSelf, key: ScaleKey, inclusive=False) -> _ScaleMappingSelf:
        keys = list(self.keys())
        if key not in keys:
            raise KeyError(f"No such scale: '{key}' (available: {keys})")

        start_idx = keys.index(key)
        if inclusive:
            start_idx += 1

        items = [(k, v) for k, v in self.items() if k in keys[start_idx:]]
        return self.__class__(items)

    def _generate_and_validate_new_keys(self, keys_pattern_or_func: Callable) -> List[ScaleKey]:
        new_keys = []
        for i, (key, value) in enumerate(self.items()):
            try:
                new_key = keys_pattern_or_func(i, key, value)
            except TypeError as e:
                if "positional argument" in str(e):
                    raise TypeError(
                        "Key-generating function must accept scale's integer index, "
                        "the old scale key, and the corresponding value object, e.g.: "
                        "lambda i, old_key, factor: f\"scale{i}-{factor['x']}\""
                    ) from e
                raise e
            new_keys.append(new_key)
        if not self._all_unique(new_keys):
            raise ValueError(f"All new scale keys must be unique. Generated: {new_keys}")
        return new_keys

    @staticmethod
    def _all_unique(things: Sequence[Hashable]) -> bool:
        seen = set()
        for item in things:
            if item in seen:
                return False
            seen.add(item)
        return True


class _ScaledAxisValues(_ScaleMapping[AxisValuesType], Generic[AxisValuesType]):
    """Base class for BlueprintShapes and BlueprintFactors"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._mapping:
            return

        if len(set(self._mapping.keys())) != len(self._mapping.keys()):
            raise ValueError(f"Scale keys must be unique. Received: {list(self._mapping.keys())}")

        axes = next(iter(self._mapping.values())).keys()
        for k, v in self._mapping.items():
            if v.keys() != axes:
                raise ValueError(
                    f"All values must have the same axes. (Expected {axes}, received {v.keys()} for key '{k}')"
                )

    def to_dict(self) -> OrderedDict[str, OrderedDict[AxisKey, Union[int, float]]]:
        return OrderedDict([(scale_key, OrderedDict(axis_values)) for scale_key, axis_values in self.items()])

    def _with_values(self: _ScaledAxisValuesSelf, values: Sequence[AxisValuesType]) -> _ScaledAxisValuesSelf:
        return self.__class__(zip(self.keys(), values))

    def _with_values_by_axes(
        self: _ScaledAxisValuesSelf,
        other: Union[ShapeLike, Mapping[ScaleKey, ShapeLike], FactorLike, Mapping[ScaleKey, FactorLike]],
        *,
        only_axes: Optional[Axes] = None,
    ) -> _ScaledAxisValuesSelf:
        if not isinstance(other, ABCMapping):
            raise TypeError(f"Pass {{axis: value}} or {{scale_key: {{axis: value}}}} mapping. Received: {other!r}")
        if not other or (only_axes is not None and not only_axes):
            return self
        is_other_nested = isinstance(next(iter(other.values())), ABCMapping)
        if is_other_nested:
            new_values = [
                axis_values.with_values(other[scale_key], only=only_axes) if scale_key in other else axis_values
                for scale_key, axis_values in self.items()
            ]
            return self._with_values(new_values)
        # Broadcast across scales
        return self._with_values([axis_values.with_values(other, only=only_axes) for axis_values in self.values()])

    def with_axes(self: _ScaledAxisValuesSelf, axes: OrderedAxes) -> _ScaledAxisValuesSelf:
        return self._with_values([value.with_axes(axes) for value in self.values()])

    @staticmethod
    def _resolve_duplicates(
        raw_items: Iterable[Tuple[ScaleKey, AxisValuesType]],
        on_duplicate: DuplicatePolicy,
        on_duplicate_prefer: Optional[ScaleKey],
    ) -> List[Tuple[ScaleKey, AxisValuesType]]:
        """
        Ensure raw_items contains no duplicate values. Resolve duplicates according to on_duplicate:
        "error": Raise error if there are duplicates.
        "keep_all": Skip (return raw_items as list)
        "keep_first": Keep the first key seen with any particular value.
        "keep_last": Keep the last key seen with any particular value.
        The two "keep" policies can be combined with `on_duplicate_prefer`.
        In this case, if the `on_duplicate_prefer` key is involved in a duplication, it has priority over first/last.
        """
        raw_items = list(raw_items)
        if on_duplicate == DuplicatePolicy.KEEP_ALL:
            return raw_items

        by_value = defaultdict(list)
        for k, v in raw_items:
            by_value[tuple(v.items())].append(k)
        duplicates = {tuple(ks): v for v, ks in by_value.items() if len(ks) > 1}

        if duplicates and on_duplicate == DuplicatePolicy.ERROR:
            raise ValueError(f"Duplicate values not allowed. Collisions: {duplicates}")

        pop_keys: List[ScaleKey] = []
        for dup_keys in duplicates:
            if on_duplicate_prefer is not None and on_duplicate_prefer in dup_keys:
                keep = on_duplicate_prefer
            elif on_duplicate == DuplicatePolicy.KEEP_FIRST:
                keep = dup_keys[0]
            elif on_duplicate == DuplicatePolicy.KEEP_LAST:
                keep = dup_keys[-1]
            else:
                raise AssertionError(f"Invalid duplicate scale policy: '{on_duplicate}'")

            pop_keys.extend(k for k in dup_keys if k != keep)

        return [(k, v) for k, v in raw_items if k not in pop_keys]


class BlueprintShapes(_ScaledAxisValues[Shape]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._mapping.items():
            self._mapping[k] = Shape(v)

    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale") -> "BlueprintShapes":
        return cls([(key, scale.shape) for key, scale in multiscale.items()])

    @classmethod
    def from_multiscale_rescaled(
        cls,
        multiscale: "Multiscale",
        *,
        target_shape: ShapeLike,
        rounding: RoundingMethod,
        source_key: Optional[ScaleKey] = None,
        scaled_axes: Optional[Axes] = None,
    ) -> "BlueprintShapes":
        """
        Build a blueprint rescaling shapes from `multiscale`
        such that the shape at `source_key` matches `target_shape`.
        If no `source_key`, `target_shape` becomes the blueprint's base shape.
        All other shapes are rescaled from `target_shape` according to their relative factor to `source_key`
        """
        resolved_source_key = source_key if source_key is not None else next(iter(multiscale.keys()))
        source_shape = multiscale[resolved_source_key].shape

        factors = BlueprintFactors.from_multiscale(multiscale, reference=source_shape)
        if scaled_axes:
            factors = factors.with_identity_except(scaled_axes)

        return factors.to_shapes(reference=target_shape, rounding=rounding)

    @classmethod
    def uniform_steps(
        cls,
        *,
        step: Union[int, float],
        base_shape: Shape,
        rounding: RoundingMethod,
        shape_limit: Optional[ShapeLike] = None,
        scaled_axes: Optional[Axes] = None,
        max_levels=42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
        on_duplicate_prefer: Optional[ScaleKey] = None,
    ) -> "BlueprintShapes":
        """Generate Blueprint where each scale is a `step` downsampling of the previous scale.
        Applies scaling uniformly to all axes until they become singleton."""
        cls._validate_resampling_step(step)
        if step == 1:
            return cls({name_pattern.format(0): base_shape})

        if scaled_axes is None:
            scaled_axes = base_shape.keys()
        scaled_axes = [a for a in scaled_axes if a in base_shape]
        if not scaled_axes:
            return cls({name_pattern.format(0): base_shape})

        shape_limit = shape_limit or base_shape.with_ones(scaled_axes)

        cls._validate_shape_limit(base_shape, scaled_axes, shape_limit, max_levels, step)
        shape_limit = Shape(shape_limit).without_axes_except(base_shape)

        scales_items = []
        for i in range(0, max_levels):
            scale_key = name_pattern.format(i)
            scale_factor = step**i
            scaling = Factor.uniform(base_shape, scale_factor).with_identity_except(scaled_axes)
            scaled_shape = base_shape.scaled_by(scaling, rounding=rounding)
            scales_items.append((scale_key, scaled_shape))
            if (step > 1 and all(scaled_shape[axis] <= shape_limit[axis] for axis in scaled_axes)) or (
                step < 1 and all(scaled_shape[axis] >= shape_limit[axis] for axis in scaled_axes)
            ):
                break
        scales_items = cls._resolve_duplicates(scales_items, on_duplicate, on_duplicate_prefer)
        bp = cls(scales_items)
        return bp.with_keys(name_pattern)

    @classmethod
    def downscale_powers_of_2_xyz(
        cls,
        *,
        base_shape: Shape,
        rounding: RoundingMethod,
        shape_limit: Optional[ShapeLike] = None,
        max_levels: int = 42,
        name_pattern=DEFAULT_NAME_PATTERN,
        on_duplicate=DuplicatePolicy.KEEP_FIRST,
        on_duplicate_prefer: Optional[ScaleKey] = None,
    ):
        return cls.uniform_steps(
            step=2,
            scaled_axes="xyz",
            base_shape=base_shape,
            rounding=rounding,
            shape_limit=shape_limit,
            max_levels=max_levels,
            name_pattern=name_pattern,
            on_duplicate=on_duplicate,
            on_duplicate_prefer=on_duplicate_prefer,
        )

    def axes(self) -> Iterable[AxisKey]:
        return self.first_value().keys()

    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where shapes differ across scales."""
        if len(self) < 2:
            return ()

        shapes = list(self.values())
        first_shape = shapes[0]
        scaled = []

        for axis in first_shape.keys():
            first_value = first_shape[axis]
            if any(shape[axis] != first_value for shape in shapes[1:]):
                scaled.append(axis)

        return tuple(scaled)

    def with_sizes(
        self, other: Union[ShapeLike, Mapping[ScaleKey, ShapeLike]], *, only_axes: Optional[Axes] = None
    ) -> "BlueprintShapes":
        """
        Replace shape values in this blueprint.
        Provide either
        - a blueprint-like mapping {scale_key: {axis: size}}
          to replace the values for those axes at that scale
        - a shape-like mapping {axis: size}
          to replace the values for those axes in *all* scales.
        Optionally limit the replacing to `only` axes (and ignore other axes in the provided mapping).
        """
        return super()._with_values_by_axes(other, only_axes=only_axes)

    def to_factors(self, reference: Optional[Shape] = None) -> "BlueprintFactors":
        if reference is None:
            reference = self.first_value()
        factors = [Shape(reference).scaling_to(scale_shape) for scale_shape in self.values()]
        return BlueprintFactors(zip(self.keys(), factors))

    def apply_to_scale(
        self, base: Scale, *, translation_shift_func: Optional[TranslationShiftFunction] = None
    ) -> "Multiscale":
        if list(self.first_value().keys()) != list(base.shape.keys()):
            raise ValueError(
                f"Cannot apply blueprint with axes {list(self.first_value().keys())} "
                f"to base scale with axes {list(base.shape.keys())}. "
                "Axes must match exactly. Maybe blueprint.with_axes(base.shape) first?"
            )

        scales = []
        for scale_key, target_shape in self.items():
            factor = base.shape.scaling_to(target_shape)
            new_pixel_size = base.pixel_size.scaled_by(factor)

            if translation_shift_func is not None:
                target_scale_pre_shift = Scale(
                    shape=target_shape, pixel_size=new_pixel_size, unit=base.unit, translation=base.translation
                )
                shift = self._compute_and_validate_shift(translation_shift_func, base, target_scale_pre_shift)
                new_translation = base.translation + shift
            else:
                new_translation = base.translation

            scales.append(
                (
                    scale_key,
                    Scale(shape=target_shape, pixel_size=new_pixel_size, unit=base.unit, translation=new_translation),
                )
            )

        return Multiscale(scales)

    @staticmethod
    def _validate_resampling_step(step: Union[int, float]):
        if step <= 0:
            raise ValueError(f"Cannot downsample by a negative step size (received: {step})")

    @staticmethod
    def _validate_shape_limit(
        base_shape: Shape, scaled_axes: Axes, shape_limit: ShapeLike, max_levels, step: Union[int, float]
    ):
        applicable_limit_axes = [a for a in shape_limit if a in base_shape]
        if not applicable_limit_axes:
            raise ValueError(
                f"Cannot scale to limit if none of the axes in shape_limit "
                f"({list(shape_limit.keys())}) are in base_shape ({list(base_shape.keys())})."
            )
        if step < 1 and set(scaled_axes) != set(applicable_limit_axes) and not max_levels:
            raise ValueError(
                f"When upscaling, either max_levels must be set, or shape_limit must limit all axes in `scaled_axes`. "
                f"Received: {scaled_axes=}, {max_levels=}, {shape_limit=}"
            )
        for axis in scaled_axes:
            if axis not in applicable_limit_axes:
                continue
            if step > 1 and shape_limit[axis] > base_shape[axis]:
                raise ValueError(f"Cannot limit downsampling to a shape larger than the base (along {axis}).")
            if step < 1 and shape_limit[axis] < base_shape[axis]:
                raise ValueError(f"Cannot limit upsampling to a shape smaller than the base (along {axis}).")

    @staticmethod
    def _compute_and_validate_shift(translation_shift_func, base, target_scale_pre_shift):
        try:
            shift = translation_shift_func(base, target_scale_pre_shift)
        except TypeError as e:
            if "argument" in str(e):
                raise TypeError(
                    "translation_shift_func must accept two positional arguments (base and target scale). "
                    "See clearscale.half_pixel_shift for an example implementation."
                ) from e
            raise e
        if not isinstance(shift, Translation):
            raise TypeError(
                f"translation_shift_func must return a Translation, got {type(shift).__name__}. "
                "See clearscale.half_pixel_shift for an example implementation."
            )
        if list(shift.keys()) != list(target_scale_pre_shift.shape.keys()):
            raise ValueError(
                f"translation_shift_func returned Translation with axes {list(shift.keys())}, "
                f"but target scale has axes {list(target_scale_pre_shift.shape.keys())}."
            )
        return shift


class BlueprintFactors(_ScaledAxisValues[Factor]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self._mapping.items():
            self._mapping[k] = Factor(v)

    @classmethod
    def from_shapes(cls, shapes: Mapping[ScaleKey, ShapeLike], reference: Shape) -> "BlueprintFactors":
        return BlueprintShapes(shapes).to_factors(reference)

    @classmethod
    def from_multiscale(cls, multiscale: "Multiscale", reference: Shape) -> "BlueprintFactors":
        return BlueprintShapes.from_multiscale(multiscale).to_factors(reference)

    def axes(self) -> Iterable[AxisKey]:
        return self.first_value().keys()

    @property
    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where any factor is not 1.0."""
        if len(self) < 2:
            return ()

        scaled: Set[AxisKey] = set()
        for factor in self.values():
            scaled.update(axis for axis, value in factor.items() if value != 1.0)

        all_axes = next(iter(self.values())).keys()
        return tuple(axis for axis in all_axes if axis in scaled)

    def with_factors(
        self, other: Union[FactorLike, Mapping[ScaleKey, FactorLike]], *, only_axes: Optional[Axes] = None
    ) -> "BlueprintFactors":
        """
        Replace factor values in this blueprint.
        Provide either
        - a blueprint-like mapping {scale_key: {axis: factor}}
          to replace the values for those axes at that scale
        - a shape-like mapping {axis: factor}
          to replace the values for those axes in *all* scales.
        Optionally limit the replacing to `only` axes (and ignore other axes in the provided mapping).
        """
        return super()._with_values_by_axes(other, only_axes=only_axes)

    def to_shapes(self, reference: ShapeLike, *, rounding: RoundingMethod) -> BlueprintShapes:
        ref = Shape(reference)
        shapes = [ref.scaled_by(scale_factor, rounding=rounding) for scale_factor in self.values()]
        return BlueprintShapes(zip(self.keys(), shapes))

    def apply_to_scale(
        self,
        scale: Scale,
        *,
        rounding: RoundingMethod,
        translation_shift_func: Optional[TranslationShiftFunction] = None,
    ) -> "Multiscale":
        shapes = self.to_shapes(scale.shape, rounding=rounding)
        return shapes.apply_to_scale(scale, translation_shift_func=translation_shift_func)

    def with_identity(self, axes: Axes) -> "BlueprintFactors":
        return self._with_values([factor.with_identity(axes) for factor in self.values()])

    def with_identity_except(self, axes: Axes) -> "BlueprintFactors":
        return self._with_values([factor.with_identity_except(axes) for factor in self.values()])


def _random_multiscale_name(seed: int | str | None = None) -> str:
    from clearscale._services.animal_names import generate_random_animal_name

    return generate_random_animal_name(seed)


class Multiscale(_ScaleMapping[Scale], TransformGraphNode):
    _transform_graph: TransformGraph
    """Transform graph that by default consists only of one isolated node: _intrinsic_ref."""
    _intrinsic_ref: NodeRef[CoordinateSystem]
    """The system in which the Scales' shape, pixel size, translation etc. are correct."""
    _zero_scale_axes_by_key: Mapping[str, Tuple[AxisKey, ...]]
    """Dataset scale axes that were read as 0.0 from loaded meta; kept for as-read round-trip."""
    _legacy_convention_global_t_scale: Optional[float]
    """Special-case for a legacy OME-Zarr convention where time step size (t-scale) is written as a global scale 
    transform. Serializes like a `ScaleTransform.bound(source=self._intrinsic_ref, target=synthetic_external)`, 
    but is redacted from self._transform_graph and instead stored on every scale as `self[].pixel_size['t']`."""
    has_shapes: bool
    """If False, this indicates the Multiscale was generated with fake (all-singleton) shapes."""
    ome: ome_zarr.OmeMultiscaleProperties
    """Additional props specific to OME-Zarr, customizable by user (mutable!)."""

    def __init__(
        self,
        *args,
        ome: Optional[ome_zarr.OmeMultiscaleProperties] = None,
        _transform_graph: Optional[TransformGraph] = None,
        _intrinsic_ref: Optional[NodeRef[CoordinateSystem]] = None,
        _zero_scale_axes_by_key: Optional[Mapping[str, Tuple[AxisKey, ...]]] = None,
        _legacy_convention_global_t_scale=None,
        has_shapes=True,
        **kwargs,
    ):
        """
        Multiscales can be constructed from a `scale_key : Scale` mapping, but this should be avoided.
        Multiscale objects should reflect either metadata read from a file (`.from_ome_zarr`, `.from_precomputed`),
        or expand a single Scale according to a scaling blueprint (`.from_shapes`, `.from_factors`).
        """
        super().__init__(*args, **kwargs)
        for key, scale in self._mapping.items():
            if scale.shape.keys() != self.axes():
                raise ValueError(
                    f"All Scales must have identical axes. Scale at '{key}' has {list(scale.shape.keys())}"
                )

        if _intrinsic_ref is None:
            if _transform_graph:
                raise AssertionError("Must specify _intrinsic_ref when _transform_graph is given.")
            self._transform_graph = self._make_single_system_graph()
            self._intrinsic_ref = next(iter(self._transform_graph.system_refs))
        else:
            transform_graph = _transform_graph or self._make_single_system_graph(_intrinsic_ref)
            if _intrinsic_ref not in transform_graph.all_system_refs:
                raise AssertionError("_intrinsic_ref must be inside _transform_graph")
            self._transform_graph = transform_graph
            self._intrinsic_ref = _intrinsic_ref
        zero_scale_axes_by_key = {}
        if _zero_scale_axes_by_key:
            available_axes = set(self.axes())
            for key, axes in _zero_scale_axes_by_key.items():
                if key not in self:
                    continue
                kept_axes = tuple(axis for axis in axes if axis in available_axes)
                if kept_axes:
                    zero_scale_axes_by_key[key] = kept_axes
        self._zero_scale_axes_by_key = zero_scale_axes_by_key
        self._legacy_convention_global_t_scale = _legacy_convention_global_t_scale
        self.has_shapes = has_shapes
        self.ome = ome if isinstance(ome, ome_zarr.OmeMultiscaleProperties) else ome_zarr.OmeMultiscaleProperties()

    def __eq__(self, other):
        return _ScaleMapping.__eq__(self, other)

    def __hash__(self):
        return _ScaleMapping.__hash__(self)

    def copy(self) -> "Multiscale":
        raise NotImplementedError(
            "Shallow copying of Multiscale is likely to cause surprising behavior. "
            "Please use the `copy` module and consider using `copy.deepcopy()` to avoid mutating the original."
        )

    @staticmethod
    def from_shapes(
        blueprint: Mapping[ScaleKey, ShapeLike],
        *,
        base: Optional[Scale] = None,
        translation_shift_func: Optional[TranslationShiftFunction] = None,
    ):
        bp = BlueprintShapes(blueprint)
        base = base or Scale(shape=bp.first_value())
        return bp.apply_to_scale(base, translation_shift_func=translation_shift_func)

    @staticmethod
    def from_factors(blueprint: BlueprintFactors, base: Scale, *, rounding: RoundingMethod):
        return blueprint.apply_to_scale(base, rounding=rounding)

    @classmethod
    def from_ome_zarr(
        cls,
        multiscale_dict: ome_zarr.OME_ZARR_MULTISCALE,
        *,
        shape_source: ome_zarr.ShapeSource,
    ):
        ome_zarr.require_dataset_paths(multiscale_dict)
        get_shape = ome_zarr.normalize_shape_source_to_callable(shape_source, multiscale_dict)
        try:
            transform_graph, intrinsic_system_ref = ome_zarr.extract_multiscale_graph(multiscale_dict)
            global_t_scale = None
        except ValueError:
            intrinsic_system_name = _random_multiscale_name()
            transform_graph, intrinsic_system_ref, global_t_scale = ome_zarr.multiscale_graph_from_legacy(
                multiscale_dict, name=intrinsic_system_name
            )
        assert intrinsic_system_ref.owner, "dev error: must reference intrinsic system"
        axis_keys = list(intrinsic_system_ref.owner.axes())
        unit = intrinsic_system_ref.owner.get_unit()
        datasets = multiscale_dict["datasets"]
        base_shape = None
        scales_items = []
        zero_scale_axes_by_key = {}
        for dataset in datasets:
            scale_key = dataset["path"]
            scale_shape = Shape(zip(axis_keys, get_shape(scale_key)))
            if base_shape is None:
                base_shape = scale_shape
            relative_scale_factor = base_shape.scaling_to(scale_shape)
            relative_scale_pixel_size = PixelSize.identity(axis_keys).scaled_by(relative_scale_factor)
            dataset_transforms = dataset.get("coordinateTransformations")
            scale_pixel_size, scale_translation, zero_scale_axes, did_merge_t_scale = (
                ome_zarr.scale_meta_from_dataset_transforms(
                    axis_keys, global_t_scale, relative_scale_pixel_size, dataset_transforms
                )
            )
            if zero_scale_axes:
                zero_scale_axes_by_key[scale_key] = zero_scale_axes
            scales_items.append(
                (
                    scale_key,
                    Scale(shape=scale_shape, pixel_size=scale_pixel_size, translation=scale_translation, unit=unit),
                )
            )

        return cls(
            scales_items,
            _transform_graph=transform_graph,
            _intrinsic_ref=intrinsic_system_ref,
            _zero_scale_axes_by_key=zero_scale_axes_by_key,
            _legacy_convention_global_t_scale=global_t_scale,
            has_shapes=shape_source != "singletons",
            ome=ome_zarr.OmeMultiscaleProperties.from_ome_zarr(multiscale_dict),
        )

    @classmethod
    def from_precomputed(cls, info_dict: precomputed.INFO_DICT):
        precomputed.validate_info_dict(info_dict)
        scales_list = info_dict["scales"]
        num_channels = info_dict.get("num_channels", 1)
        axis_keys = ["c", "z", "y", "x"]  # Precomputed is always czyx (x varies fastest)

        scales_items = []
        zero_scale_axes_by_key = {}
        for scale_dict in scales_list:
            scale_key = scale_dict["key"]

            size = scale_dict["size"]
            if len(size) != 3:
                raise ValueError(f"Scale {scale_key!r} must have 'size' as [x, y, z]")
            shape = Shape(zip(axis_keys, [num_channels] + list(reversed(size))))

            resolution = scale_dict["resolution"]
            if len(resolution) != 3:
                raise ValueError(f"Scale {scale_key!r} must have 'resolution' as [x, y, z]")
            zero_axes = precomputed.zero_resolution_axes(resolution, "zyx")
            if zero_axes:
                zero_scale_axes_by_key[scale_key] = zero_axes
            pixel_size = precomputed.pixel_size_from_resolution(resolution, axis_keys)

            voxel_offset = scale_dict.get("voxel_offset", [0, 0, 0])
            if len(voxel_offset) != 3:
                warnings.warn(f"Scale {scale_key!r} has invalid voxel_offset. Using [0, 0, 0].")
                voxel_offset = [0, 0, 0]
            offset = PixelOffset(zip(axis_keys, [0] + list(reversed(voxel_offset))))
            translation = offset.to_physical(pixel_size)

            unit = Unit(zip(axis_keys, ["", "nm", "nm", "nm"]))

            scale = Scale(shape, pixel_size, unit, translation)
            scales_items.append((scale_key, scale))

        return cls(scales_items, _zero_scale_axes_by_key=zero_scale_axes_by_key)

    def axes(self) -> OrderedAxes:
        return self.first_value().shape.keys()

    def scaled_axes(self) -> Tuple[AxisKey, ...]:
        """Axes where pixel_sizes differ across scales."""
        if len(self) < 2:
            return ()

        pixel_sizes = list(scale.pixel_size for scale in self.values())
        first_pixel_size = pixel_sizes[0]
        scaled = []

        for axis in first_pixel_size.keys():
            first_value = first_pixel_size[axis]
            if any(pixel_size[axis] != first_value for pixel_size in pixel_sizes[1:]):
                scaled.append(axis)

        return tuple(scaled)

    @cached_property
    def coordinate_systems(self) -> Tuple[str, ...]:
        """Spaces, other than the multiscale's own, into which it can be transformed."""
        return tuple(ref.name for ref in self._transform_graph.all_system_refs)

    @cached_property
    def keys_by_shape(self) -> Mapping[Shape, Tuple[ScaleKey, ...]]:
        grouped = defaultdict(list)
        for key, scale in self.items():
            grouped[scale.shape].append(key)
        return {shape: tuple(keys) for shape, keys in grouped.items()}

    def with_coordinate_systems_of(
        self, other: "Multiscale", *, derived_by: Optional[SpatialRelation] = None
    ) -> "Multiscale":
        """Transfer `other`'s coordinate systems onto self.
        This is appropriate when self is derived from `other`.
        Optionally provide `derived_by`: The relation by which self was derived from `other`."""
        source_axes = tuple(other.axes())
        target_axes = tuple(self.axes())
        if derived_by is None:
            if source_axes != target_axes:
                raise ValueError(
                    f"Cannot transfer coordinate systems from source with axes {source_axes!r} to Multiscale "
                    f"with axes {target_axes!r}. Use `derived_by` to specify how the new axes were obtained. "
                    f"E.g. `ProjectionTo(result_axes)` for inserted or dropped axes."
                )
        else:
            derived_axes = derived_by.target_axes(source_axes)
            if derived_axes != target_axes:
                raise ValueError(
                    f"Incompatible derivation: Provided {derived_by.__class__.__name__} would produce axes "
                    f"{derived_axes!r} from {source_axes!r}, but this Multiscale has {target_axes!r}. {derived_by=!r}"
                )

        # Exclude other's intrinsic name from collision check: Will either be rebased, or renamed
        incoming_names = {ref.name for ref in other._transform_graph.all_system_refs if ref != other._intrinsic_ref}
        existing_names = {ref.name for ref in self._transform_graph.all_system_refs}
        collisions = incoming_names & existing_names
        if collisions:
            raise ValueError(
                f"Cannot transfer coordinate systems {collisions} into Multiscale that already has {existing_names}."
            )

        incoming_transforms = tuple(
            t for t in other._transform_graph.transforms if not self._is_transform_path_bound(t)
        )
        if derived_by is None:
            # Identity with other: rebase other's graph onto our intrinsic to avoid identity chaining
            merged = tuple(
                self._replace_transform_ref(t, other._intrinsic_ref, self._intrinsic_ref) for t in incoming_transforms
            )
        else:
            incoming_with_unique_names = incoming_transforms
            other_intrinsic_unique = other._intrinsic_ref
            if other._intrinsic_ref.name in existing_names:
                other_intrinsic_unique = self._make_ref_unique(other_intrinsic_unique, existing_names)
                incoming_with_unique_names = tuple(
                    self._replace_transform_ref(t, other._intrinsic_ref, other_intrinsic_unique)
                    for t in incoming_transforms
                )
            derivation_transform = relation_to_transform(derived_by, source_axes)
            merged = incoming_with_unique_names + (
                derivation_transform.bound(source=other_intrinsic_unique, target=self._intrinsic_ref),
            )

        candidate_t_scale = self._legacy_convention_global_t_scale or (
            "t" in self.axes() and other._legacy_convention_global_t_scale
        )
        transfers_cleanly = not candidate_t_scale or all(
            scale.pixel_size["t"] == candidate_t_scale for scale in self.values()
        )
        transferred_global_t_scale = candidate_t_scale if transfers_cleanly else None

        if (
            not merged
            and not incoming_names
            and bool(transferred_global_t_scale) == bool(self._legacy_convention_global_t_scale)
        ):
            return self

        new_graph = TransformGraph(
            transforms=self._transform_graph.transforms + merged,
            system_refs=self._transform_graph.system_refs,
        )
        return Multiscale(
            self.items(),
            _transform_graph=new_graph,
            _intrinsic_ref=self._intrinsic_ref,
            _zero_scale_axes_by_key=self._zero_scale_axes_by_key,
            _legacy_convention_global_t_scale=transferred_global_t_scale,
        )

    # Ignore narrowing of `version: str` to Literal (nicer to be explicit)
    def to_ome_zarr(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        version: Literal["0.4", "0.5", "0.6.rc0"],
        name: Optional[str] = None,
        axis_types: Union[None, Literal["infer"], Mapping[AxisKey, Literal["space", "time", "channel"]]] = None,
    ) -> Dict[str, Any]:
        if version not in ome_zarr.SUPPORTED_OME_ZARR_VERSIONS_WRITE:
            raise ValueError("Cannot write OME-Zarr versions other than 0.4, 0.5 and 0.6.rc0.")
        ome_zarr.validate_multiscale(self)
        result: Dict[str, Any] = {"version": version, "datasets": []}
        if self.ome:
            result.update(self.ome.to_ome_zarr())

        if name:
            result["name"] = name

        # Modern: Multiscale is graph + datasets
        if version not in PRE_TRANSFORMS_VERSIONS:
            result.update(self._transform_graph.to_ome_zarr(version=version))
            for key, scale in self.items():
                dataset = ome_zarr.build_dataset_dict(
                    version,
                    key,
                    scale.pixel_size,
                    scale.translation,
                    self._intrinsic_ref,
                    self._zero_scale_axes_by_key.get(key, ()),
                )
                result["datasets"].append(dataset)
            return result

        # Legacy: Determine axes, coordinateTransformations, and datasets, and handle legacy t-scale convention
        assert self._intrinsic_ref.owner, "dev error: must always have intrinsic"
        intrinsic_system_dict = self._intrinsic_ref.owner.to_ome_zarr(
            name="", version=version, axis_types=axis_types, unit=self.first_value().unit
        )
        result["axes"] = intrinsic_system_dict["axes"]

        multiscale_transforms = self._legacy_multiscale_transforms()
        all_scale_t_matching = "t" in self.axes() and all(
            scale.pixel_size["t"] == self._legacy_convention_global_t_scale for scale in self.values()
        )
        if not self._legacy_convention_global_t_scale or not all_scale_t_matching:
            # "Normal" case, no special treatment.
            if self._legacy_convention_global_t_scale:
                pixel_sizes = {scale_key: scale.pixel_size["t"] for scale_key, scale in self.items()}
                warnings.warn(
                    f"Dev error? Multiscale claims to use legacy t convention but has non-uniform pixel_size[t]: {pixel_sizes}"
                )
            if multiscale_transforms:
                # Unconventional use of multiscale-transforms for some transform to an undefined external reference.
                # Write as read.
                result["coordinateTransformations"] = multiscale_transforms.to_legacy_ome_zarr()
            for key, scale in self.items():
                dataset = ome_zarr.build_dataset_dict(
                    version,
                    key,
                    scale.pixel_size,
                    scale.translation,
                    serialized_zero_scale_axes=self._zero_scale_axes_by_key.get(key, ()),
                )
                result["datasets"].append(dataset)
            return result

        # Legacy convention where pixel size along t is written as a global scale transform
        axes = list(self.axes())
        global_t_scale = self._legacy_convention_global_t_scale
        assert global_t_scale and "t" in axes, "global t-scale must not be set when inapplicable"
        if multiscale_transforms is None:
            global_pixel_size = PixelSize(t=global_t_scale).with_axes(axes)
            global_scale = ScaleTransform.from_pixel_size(global_pixel_size)
            multiscale_transforms = ome_zarr.MultiscaleTransforms((global_scale,))
        else:
            global_scale_values = list(multiscale_transforms.scale_transform.scale)
            assert all(v in (PixelSize._default(), 0) for v in global_scale_values), "doesn't conform to convention"
            global_scale_values[axes.index("t")] = global_t_scale
            global_scale = ScaleTransform(scale=tuple(global_scale_values))
            tfs = (
                (global_scale,)
                if multiscale_transforms.translation_transform is None
                else (global_scale, multiscale_transforms.translation_transform)
            )
            multiscale_transforms = ome_zarr.MultiscaleTransforms(tfs)
        result["coordinateTransformations"] = multiscale_transforms.to_legacy_ome_zarr()
        for key, scale in self.items():
            dataset_scale = scale.pixel_size
            if global_t_scale:
                assert dataset_scale["t"] == global_t_scale, "doesn't conform to convention"
                # scale.pixel_size["t"] is written in global transforms in this case
                dataset_scale = scale.pixel_size.with_identity("t")
            dataset = ome_zarr.build_dataset_dict(
                version,
                key,
                dataset_scale,
                scale.translation,
                serialized_zero_scale_axes=self._zero_scale_axes_by_key.get(key, ()),
            )
            result["datasets"].append(dataset)
        return result

    def _make_single_system_graph(self, sys_ref: Optional[NodeRef[CoordinateSystem]] = None) -> TransformGraph:
        if sys_ref is None:
            intrinsic_sys = CoordinateSystem.without_semantics(list(self.axes()))
            intrinsic_name = _random_multiscale_name()
            sys_ref = intrinsic_sys.as_ref(intrinsic_name)
        return TransformGraph.single_isolated_system(sys_ref)

    def as_ref(self, name: CoordinateSystemName) -> NodeRef["Multiscale"]:
        """For Multiscale, making a ref means selecting one of their coordinate systems by name."""
        if name not in (ref.name for ref in self._transform_graph.all_system_refs):
            raise NoSuchCoordinateSystemError(name)
        return NodeRef(name=str(name), owner=self)

    def _get_interface_transform(self):
        """Allows a scene to traverse into this subgraph"""
        return IdentityTransform(source=self._intrinsic_ref, target=self.as_ref(self._intrinsic_ref.name))

    @staticmethod
    def _is_transform_path_bound(t: Transform) -> bool:
        """True for edges bound to another Multiscale's storage location (e.g. label overlays),
        which must not be carried into a Multiscale at a different location."""
        return any(isinstance(ref, _UnresolvedRef) and ref.file for ref in (t.source, t.target))

    @staticmethod
    def _replace_transform_ref(t: Transform, old: NodeRef, new: NodeRef) -> Transform:
        return t.bound(source=new if t.source == old else t.source, target=new if t.target == old else t.target)

    @staticmethod
    def _make_ref_unique(old_ref: NodeRef[CoordinateSystem], exclude: Collection[str]):
        if old_ref.name not in exclude:
            return old_ref
        i = 1
        while (new_name := f"{old_ref.name}-{i}") in exclude:
            i += 1
        return old_ref.owner.as_ref(new_name)

    def _legacy_multiscale_transforms(self) -> Optional["ome_zarr.MultiscaleTransforms"]:
        """
        Inspect graph to find MultiscaleTransforms for multiscale_dict["coordinateTransformations"].
        Preferably bound directly on this Multiscale's own intrinsic system (exact, lossless).
        Falls back to one reachable via inserts-only ProjectAxisTransform path,
        in which case the MultiscaleTransforms can be widened to self's axes with identity values.
        This padding is a deliberate approximation: the legacy format cannot express the ProjectAxisTransform
        itself, but MultiscaleTransforms along undefined axes are implicitly identities.
        """
        candidates = [t for t in self._transform_graph.transforms if isinstance(t, ome_zarr.MultiscaleTransforms)]

        for t in candidates:
            if t.source == self._intrinsic_ref:
                return t

        own_axes = tuple(self.axes())
        for t in candidates:
            assert isinstance(t.source, NodeRef), "All MultiscaleTransforms must have a Multiscale source"
            if self._is_axis_reshaping_neighbor(t.source):
                return self._reshaped_multiscale_transforms(t, target_axes=own_axes)

        return None

    def _is_axis_reshaping_neighbor(self, other_ref: NodeRef) -> bool:
        """
        True if `other_ref` connects to `self` via a single transform that is purely axis-reshaping (inserts, drops, reorders).
        We can't use path traversal because ProjectAxisTransforms are not invertible if they contain drops.
        In the typical constellation:
        `self <- projectAxis(drops) <- other_ref -> scale+[translation] -> external_system`
        there is no path `self -> external_system`
        only `external_system -> self` (which is not helpful because legacy formats can't serialize it).
        """
        return any(
            {t.source, t.target} == {self._intrinsic_ref, other_ref} and self._is_pure_axis_reshape([t])
            for t in self._transform_graph.transforms
        )

    @staticmethod
    def _is_pure_axis_reshape(path: List["Transform"]) -> bool:
        """True if `path` collapses to pure axis bookkeeping (drop/insert/reorder) with no
        physical content (scale, translation, rotation) — safe to fold into a widened/narrowed
        MultiscaleTransforms via with_axes, since nothing is being approximated away, only axes
        that were never physically related to begin with."""
        if not path:
            return True
        collapsed = TransformSequence(tuple(path)).collapsed() if len(path) > 1 else path[0]
        structural_types = (IdentityTransform, ProjectAxisTransform, MapAxisTransform)
        if isinstance(collapsed, TransformSequence):
            return all(isinstance(child, structural_types) for child in collapsed.transforms)
        return isinstance(collapsed, structural_types)

    @staticmethod
    def _reshaped_multiscale_transforms(
        t: ome_zarr.MultiscaleTransforms, *, target_axes: Tuple[AxisKey, ...]
    ) -> "ome_zarr.MultiscaleTransforms":
        source_axes = tuple(t.source.owner.axes())
        scale = ome_zarr.scale_to_pixel_size_with_normalized_zeros(t.scale_transform.scale, source_axes).with_axes(
            target_axes
        )
        widened_scale = ScaleTransform(scale=scale.to_tuple())
        if t.translation_transform is None:
            return ome_zarr.MultiscaleTransforms(transforms=(widened_scale,))
        translation = t.translation_transform.to_translation(source_axes).with_axes(target_axes)
        return ome_zarr.MultiscaleTransforms(
            transforms=(widened_scale, TranslationTransform.from_translation(translation))
        )
