import numbers
from collections import OrderedDict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, replace
from typing import Iterable, List, overload, Sequence, Tuple, Mapping, Optional, Union

from clearscale._axis_values import Axes, _axis_in, AxisKey, _AxisFloats, _AxisMapping, OrderedAxes, Scalar, Translation


class Coefficient(_AxisFloats):
    """
    A generic set of linear coefficients in a function like `X_out = a * Z_in + b * Y_in + c * X_in`.
    This represents one row of a linear transformation matrix.
    """

    @classmethod
    def _default(cls) -> float:
        return 0.0

    @classmethod
    def zeros(cls, axes: OrderedAxes) -> "Coefficient":
        """Create a new Coefficient with 0.0 for all `axes`."""
        return super().fromkeys(axes)


class Linear(_AxisMapping[AxisKey, Coefficient]):
    """
    A square linear transformation matrix, represented as a nested axis mapping:
    {target_axis : {source_axis : coefficient}}, as in
    {row_key : {col_key : coefficient}}.
    Rows are represented by the Coefficient class, mapping {axis : float}.
    """

    @staticmethod
    def _default(cols: OrderedAxes, row: AxisKey) -> Coefficient:
        """For a linear matrix, the default is an identity row."""
        return Coefficient.zeros(cols).with_values({row: 1.0})

    @classmethod
    def fromkeys(cls, keys: OrderedAxes) -> "Linear":
        return cls(zip(keys, [Coefficient.zeros(keys)] * len(keys)))

    @classmethod
    def identity(cls, axes: OrderedAxes) -> "Linear":
        axes = tuple(axes)
        return cls({axis: cls._default(axes, axis) for axis in axes})

    @classmethod
    def from_array(cls, array: Sequence[Sequence[Scalar]], axes: OrderedAxes) -> "Linear":
        axes = tuple(axes)
        if len(array) != len(axes):
            raise ValueError(f"Expected {len(axes)} rows for axes {axes!r}, received {len(array)}.")

        rows = []
        for target_axis, row in zip(axes, array):
            if len(row) != len(axes):
                raise ValueError(f"Expected {len(axes)} columns for row '{target_axis}', received {len(row)}.")
            rows.append(Coefficient(zip(axes, row)))

        return cls(zip(axes, rows))

    def __init__(self, *args, **kwargs):
        """Supports instantiation using any nested mappings like Linear({'x': {'x': 2.0}})"""
        super().__init__(*args, **kwargs)
        expected_axes = None
        for target_axis, coefficient in self._mapping.items():
            if isinstance(coefficient, ABCMapping):
                coefficient = Coefficient(coefficient)
                self._mapping[target_axis] = coefficient
            elif not isinstance(coefficient, Coefficient):
                raise TypeError(
                    f"All values must be Coefficient or Mapping. "
                    f"Got {type(coefficient).__name__} for axis '{target_axis}'."
                )
            if expected_axes is None:
                expected_axes = coefficient.keys()
            elif coefficient.keys() != expected_axes:
                raise ValueError(
                    f"All source Coefficients must have identical axes. "
                    f"Expected {tuple(expected_axes)!r}, "
                    f"received {tuple(coefficient.keys())!r} for target axis '{target_axis}'."
                )
        assert expected_axes is not None, "super.init should enforce non-empty self._mapping"
        if tuple(self.keys()) != tuple(expected_axes):
            raise ValueError(
                f"Linear must be square. Target axes {tuple(self.keys())!r} != source axes {tuple(expected_axes)!r}."
            )

    def copy(self) -> "Linear":
        """.copy doesn't provide much value since this is immutable, but Linear still behaves enough
        like a dict that it might be nice to have, for consistency with dict.copy"""
        return self.__class__((row_axis, coeff.copy()) for row_axis, coeff in self.items())

    def with_axes(self, axes: OrderedAxes) -> "Linear":
        """Order like axes. Drop axes, or insert new identity axes if necessary."""
        if not axes:
            raise ValueError(f"Cannot create empty Linear. Attempted reorder to: {axes!r}")
        identity_rows = Linear.identity(axes)
        reordered_items = [(a, self[a].with_axes(axes) if _axis_in(a, self) else identity_rows[a]) for a in axes]
        return self.__class__(reordered_items)

    def with_axes_order(self, axes: OrderedAxes) -> "Linear":
        """Order like given axes (but no new insertions or drops)."""
        if not axes:
            raise ValueError(f"Cannot create empty GeneralLinear. Attempted reorder to {axes!r}.")
        would_drop = tuple(a for a in self if a not in axes)
        if would_drop:
            raise ValueError(f"Cannot reorder GeneralLinear to axes {axes!r}. This would drop: {would_drop!r}.")
        reordered_items = [(a, self[a].with_axes_order(axes)) for a in axes if a in self]
        return self.__class__(reordered_items)

    def without_axes_except(self, axes: Axes) -> "Linear":
        """Keep only given axes (no reordering or inserts)."""
        kept_items = [(a, self[a].without_axes_except(axes)) for a in self if _axis_in(a, axes)]
        if not kept_items:
            raise ValueError(
                f"Cannot create empty Linear. "
                f"None of the specified axes {axes!r} are present in {list(self.keys())}."
            )
        return self.__class__(kept_items)

    def without_axes(self, axes: Axes) -> "Linear":
        """Drop given axes."""
        kept_items = [(a, self[a].without_axes(axes)) for a in self if not _axis_in(a, axes)]
        if not kept_items:
            raise ValueError(f"Cannot create empty {self.__class__.__name__}. Removing {axes!r} would leave no axes.")
        return self.__class__(kept_items)

    def with_identity(self, axes: Axes) -> "Linear":
        """
        Reset `axes` to identity transformations.
        This means `axes` rows become identity, and `axes` values in all other rows become 0.
        """
        if not axes:
            return self
        identity_rows = Linear.identity(self)
        new_rows = [
            (a, identity_rows[a]) if _axis_in(a, axes) else (a, row.with_default(axes)) for a, row in self.items()
        ]
        return self.__class__(new_rows)

    def with_identity_except(self, axes: Axes) -> "Linear":
        """
        Reset all axes except `axes` to identity transformations.
        This means all rows except `axes` become identity, and all columns except `axes` become 0.
        """
        if not axes:
            return Linear.identity(self)

        identity_rows = Linear.identity(self)
        new_rows = [
            (a, row.with_default_except(axes)) if _axis_in(a, axes) else (a, identity_rows[a])
            for a, row in self.items()
        ]
        return self.__class__(new_rows)

    def without_identity(self) -> "Linear":
        """Drop identity axes (identity rows whose axis is also 0 in all other rows)."""
        identity_axes = []
        for axis in self:
            if self[axis] != self._default(self.keys(), axis):
                continue
            if any(other_axis != axis and self[other_axis][axis] != 0.0 for other_axis in self):
                continue
            identity_axes.append(axis)

        if len(identity_axes) == len(self):
            raise ValueError("Cannot create empty Linear. Removing all identities would leave no axes.")

        return self.without_axes(identity_axes)

    def with_values(
        self, other: Mapping[AxisKey, Mapping[AxisKey, Scalar]], *, only: Optional[Axes] = None
    ) -> "Linear":
        if not isinstance(other, ABCMapping):
            raise TypeError(f"Pass {{target_axis: {{source_axis: value}}}} mapping. Received: {other!r}")
        if not other or (only is not None and not only):
            return self
        replaced_items = []
        for row, coeff in self.items():
            new_coefficient = coeff
            if (only is None or _axis_in(row, only)) and row in other:
                replacements = other[row]
                if not isinstance(replacements, ABCMapping):
                    raise TypeError(
                        f"Replacement for target axis {row!r} must be a mapping. Received: {replacements!r}"
                    )
                new_coefficient = coeff.with_values(replacements, only=only)
            replaced_items.append((row, new_coefficient))
        return self.__class__(replaced_items)

    def to_tuples(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(coeff.to_tuple() for coeff in self.values())

    def to_lists(self) -> List[List[float]]:
        return [coeff.to_list() for coeff in self.values()]

    def to_dict(self) -> OrderedDict[AxisKey, OrderedDict[AxisKey, float]]:
        return OrderedDict([(axis, OrderedDict(coeff)) for axis, coeff in self.items()])

    def at(self, row: AxisKey, col: AxisKey) -> float:
        return self._mapping[row][col]

    def axes(self) -> Iterable[AxisKey]:
        return self.keys()

    def is_identity(self) -> bool:
        return self == self.identity(tuple(self.axes()))

    def is_identity_along(self, axes: Axes) -> bool:
        return self == self.with_identity(axes)


@dataclass(frozen=True, slots=True, init=False)
class Affine:
    """
    A full affine transformation consists of a linear matrix and a translation.
    The perspective row required for homogenous coordinates is not represented.
    """

    linear: Linear
    translation: Translation

    @classmethod
    def identity(cls, axes: OrderedAxes) -> "Affine":
        axes = tuple(axes)
        return cls(linear=Linear.identity(axes), translation=Translation.identity(axes))

    @classmethod
    def from_linear(cls, linear: Linear) -> "Affine":
        """Explicit alias for a common upgrade path"""
        return cls(linear=linear)

    @classmethod
    def from_array(cls, array: Sequence[Sequence[Scalar]], axes: OrderedAxes) -> "Affine":
        """
        Construct an Affine from a matrix shaped either N x (N+1):
        `[ linear | translation ]`
        or homogenous form (N+1) x (N+1):
        ```
        [ linear | translation ]
        [ 0 ... 0 |     1      ]
        ```
        where N == len(axes).
        """
        axes = tuple(axes)
        ndim = len(axes)
        expected_cols = ndim + 1  # Each row is Linear + translation
        if len(array) not in (ndim, expected_cols):
            raise ValueError(
                f"Expected either {ndim} or {ndim + 1} rows for axes {axes!r}, received {len(array)}: {array!r}."
            )
        if len(array) == expected_cols:
            expected_perspective_row = [0] * ndim + [1]
            if list(array[-1]) != expected_perspective_row:
                raise ValueError(
                    "Homogeneous affine matrices must end with the perspective row "
                    f"{expected_perspective_row!r}. Received {list(array[-1])!r}."
                )
            non_homogenous = array[:-1]
        else:
            non_homogenous = array
        for i, row in enumerate(non_homogenous):
            if len(row) != expected_cols:
                raise ValueError(f"Expected {expected_cols} columns for row {i}, received {len(row)}.")
        linear = Linear.from_array([row[:ndim] for row in non_homogenous], axes)
        translation = Translation(zip(axes, (row[ndim] for row in non_homogenous)))
        return cls(linear=linear, translation=translation)

    def __init__(self, *, linear: Optional[Linear] = None, translation: Optional[Translation] = None):
        if not isinstance(linear, Linear) and not isinstance(translation, Translation):
            raise ValueError(f"Provide instances of Linear and/or Translation. Received {linear!r} and {translation!r}")
        if linear is None and isinstance(translation, Translation):
            linear = Linear.identity(translation)
        if not isinstance(linear, Linear):
            raise TypeError(f"linear must be a Linear. Received {type(linear).__name__}: {linear!r}")
        if translation is None:
            translation = Translation.identity(linear)
        if not isinstance(translation, Translation):
            raise TypeError(
                f"translation must be a Translation. " f"Received {type(translation).__name__}: {translation!r}"
            )
        if tuple(linear.keys()) != tuple(translation.keys()):
            raise ValueError(
                "Linear and Translation must have identical axes. "
                f"Linear has {tuple(linear.keys())!r}, "
                f"Translation has {tuple(translation.keys())!r}."
            )
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "translation", translation)

    def with_axes(self, axes: OrderedAxes) -> "Affine":
        """Order like axes. Drop axes, or insert new identity axes if necessary."""
        return self.__class__(
            linear=self.linear.with_axes(axes),
            translation=self.translation.with_axes(axes),
        )

    def with_axes_order(self, axes: OrderedAxes) -> "Affine":
        """Order like given axes (but no new insertions or drops)."""
        return self.__class__(
            linear=self.linear.with_axes_order(axes),
            translation=self.translation.with_axes_order(axes),
        )

    def without_axes_except(self, axes: Axes) -> "Affine":
        """Keep only given axes (no reordering or inserts)."""
        return self.__class__(
            linear=self.linear.without_axes_except(axes),
            translation=self.translation.without_axes_except(axes),
        )

    def without_axes(self, axes: Axes) -> "Affine":
        """Drop given axes."""
        return self.__class__(
            linear=self.linear.without_axes(axes),
            translation=self.translation.without_axes(axes),
        )

    def with_identity(self, axes: Axes) -> "Affine":
        """
        Reset `axes` to identity affine transformations.
        This means the corresponding linear rows become identity, their columns
        become 0 in all other rows, and the translation along those axes becomes 0.
        """
        return self.__class__(
            linear=self.linear.with_identity(axes),
            translation=self.translation.with_identity(axes),
        )

    def with_identity_except(self, axes: Axes) -> "Affine":
        """
        Reset all axes except `axes` to identity affine transformations.
        This means all other linear rows become identity, all other columns become
        0, and all other translations become 0.
        """
        return self.__class__(
            linear=self.linear.with_identity_except(axes),
            translation=self.translation.with_identity_except(axes),
        )

    def without_identity(self) -> "Affine":
        """
        Drop identity axes.
        An axis is identity if its linear component is identity and its translation is 0.
        """
        identity_axes = [
            axis
            for axis in self.linear
            if self.translation.is_identity_along((axis,)) and self.linear.is_identity_along((axis,))
        ]
        if len(identity_axes) == len(self.linear):
            raise ValueError("Cannot create empty Affine. Removing all identities would leave no axes.")
        return self.without_axes(identity_axes)

    def with_linear(self, linear: Linear) -> "Affine":
        return replace(self, linear=linear)

    def with_translation(self, translation: Translation) -> "Affine":
        return replace(self, translation=translation)

    def to_tuples(self) -> Tuple[Tuple[float, ...], ...]:
        return tuple(row.to_tuple() + (self.translation[axis],) for axis, row in self.linear.items())

    def to_tuples_homogenous(self) -> Tuple[Tuple[float, ...], ...]:
        return self.to_tuples() + ((*(0.0 for _ in self.linear), 1.0),)

    def to_lists(self) -> List[List[float]]:
        return [row.to_list() + [self.translation[axis]] for axis, row in self.linear.items()]

    def to_lists_homogenous(self) -> List[List[float]]:
        return self.to_lists() + [[*(0.0 for _ in self.linear), 1.0]]

    def axes(self) -> Iterable[AxisKey]:
        return self.linear.keys()

    def is_identity(self) -> bool:
        return self.linear.is_identity() and self.translation.is_identity()

    def is_identity_along(self, axes: Axes) -> bool:
        return self.linear.is_identity_along(axes) and self.translation.is_identity_along(axes)
