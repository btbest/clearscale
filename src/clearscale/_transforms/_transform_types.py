from abc import ABC, abstractmethod
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
import numbers
from typing import (
    Optional,
    Tuple,
    Dict,
    Mapping,
    Iterable,
    Any,
    List,
    TypeGuard,
    Union,
    TYPE_CHECKING,
)

from clearscale._axis_values import (
    AxisKey,
    PixelSize,
    Translation,
)
from clearscale._errors import CannotConvertToAffineError
from clearscale._services.matrices import (
    FloatMatrix,
    FloatVector,
    is_identity_matrix,
    is_diagonal_matrix,
    is_rotation_matrix,
    matrix_shape,
    matrix_multiply,
    matrix_vector_multiply,
    matrix_transpose,
    matrix_invert,
    matrix_determinant,
    DETERMINANT_SINGULARITY_TOLERANCE,
)
from clearscale._transforms._base import NodeRef, RelativePath, Transform

if TYPE_CHECKING:
    from ._base import TransformSequence

IDENTITY_TOLERANCE = 1e-13


def _is_number(value: Any) -> TypeGuard[numbers.Real]:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _require_numbers(value: Any, field_name: str) -> Tuple[float, ...]:
    try:
        values = tuple(value)
    except TypeError:
        values = ()
    if not values:
        raise ValueError(f"Invalid {field_name}. Expected sequence, received: {value!r}")
    if not all(_is_number(v) for v in values):
        raise ValueError(f"Invalid {field_name}. Expected sequence of numbers, received: {value!r}")
    return tuple(float(v) for v in values)


def _require_rectangular_matrix(value: Any, field_name: str) -> FloatMatrix:
    try:
        rows = tuple(tuple(row) for row in value)
    except TypeError:
        rows = ()
    if not rows:
        raise ValueError(f"Invalid matrix. Expected 2D array in {field_name!r}, received: {value!r}")
    row_length = len(rows[0])
    if row_length == 0 or any(len(row) != row_length for row in rows):
        raise ValueError(f"Invalid matrix. Expected rectangular 2D array in {field_name!r}, received: {value!r}")
    if not all(_is_number(v) for row in rows for v in row):
        raise ValueError(f"Invalid number matrix. Expected 2D numeric array in {field_name!r}, received: {value!r}")
    return tuple(tuple(float(v) for v in row) for row in rows)


def _require_path(ome_dict: Mapping[str, Any], *, transform_type: str) -> str:
    path = ome_dict.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"Invalid {transform_type} transform metadata. Expected non-empty path, received: {path!r}")
    return path


def _require_int_or_empty_tuple(value: Any, field_name: str, *, transform_type: str) -> Tuple[int, ...]:
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(
            f"Invalid {transform_type} transform metadata. Expected integer sequence in {field_name!r}, "
            f"received: {value!r}"
        )
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in items):
        raise ValueError(
            f"Invalid {transform_type} transform metadata. Expected integer sequence in {field_name!r}, "
            f"received: {value!r}"
        )
    if any(v < 0 for v in items):
        raise ValueError(
            f"Invalid {transform_type} transform metadata. Axis indices must be non-negative, received: {items!r}"
        )
    return items


def _covers_all_indices(values: Iterable[int]) -> bool:
    """Whether `values` contains every index of an array as long as itself"""
    values = tuple(values)
    return set(values) == set(range(len(values)))


def _validate_unique(values: Tuple[int, ...], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Invalid transform. Expected unique indices in {field_name}, received: {values!r}")


@dataclass(frozen=True, slots=True)
class AffineRepresentableTransform(Transform, ABC):
    """Mixin-ish for transforms that can also be represented by an affine matrix, to deduplicate composed_with logic"""

    @abstractmethod
    def _to_affine_transform(
        self, *, source_ndim: Optional[int] = None, target_ndim: Optional[int] = None
    ) -> "AffineTransform":
        """Return an AffineTransform equivalent to this transform.

        source_ndim and target_ndim are special affordances for ProjectAxisTransform, because it can be
        represented as an AffineTransform, and we want it to be composable, but its payload
        specifies only an ndim *difference*, so it can't derive the exact matrix shape just from itself.
        IdentityTransform has the same issue (source==target by definition, but no payload to know either ndim from),
        but as a noop it has other special treatments. Better not to pollute it with AffineRepresentable.
        """
        ...

    def _compose_via_affine(self, earlier: Transform) -> Optional["AffineTransform"]:
        """Compose two affine-representable transforms using their matrix forms."""
        if not isinstance(earlier, AffineRepresentableTransform):
            return None
        # For most types this is just a sanity guard;
        # for ProjectAxis x ProjectAxis, the lack of ndim info makes composition via affine impossible
        assert not isinstance(earlier, type(self)), "must not be called for same-type composition"
        try:
            self_ndim = self._ndim_by_payload()
            earlier_ndim = earlier._ndim_by_payload()
            later_affine = self._to_affine_transform(source_ndim=earlier_ndim[1] if earlier_ndim is not None else None)
            later_ndim = later_affine._ndim_by_payload()
            assert later_ndim is not None
            earlier_affine = earlier._to_affine_transform(target_ndim=later_ndim[0])
            composed = later_affine.composed_with(earlier_affine)
            return composed if isinstance(composed, AffineTransform) else None
        except CannotConvertToAffineError:
            return None


@dataclass(frozen=True, slots=True)
class IdentityTransform(Transform):
    # Note (to be deleted): Identity is actually not representable as affine
    # because it has no payload, so it does not know its ndim...
    @property
    def is_invertible(self) -> bool:
        return True

    def inverted(self) -> "IdentityTransform":
        return replace(self, source=self.target, target=self.source)

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if not self._endpoints_can_chain_after(earlier):
            return None
        return replace(earlier, source=self._composed_source(earlier), target=self._composed_target(earlier))

    def simplified(self) -> "IdentityTransform":
        return self

    def _ndim_by_payload(self) -> None:
        return None

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        return {"type": "identity"}


@dataclass(frozen=True, slots=True)
class ScaleTransform(AffineRepresentableTransform):
    scale: Tuple[float, ...]  # Can be empty if _ome_zarr_path is provided instead
    _ome_zarr_path: Optional[str] = None
    """OME-Zarr allows scale with path to a zarr instead of plain values.
    Nobody uses this in practice.
    Support round-trip of such metadata, but until anyone actually needs it, we'll accept
    that such ScaleTransform objects are useless in practice."""

    @property
    def is_invertible(self) -> bool:
        return bool(self.scale) and all(v for v in self.scale)  # Not invertible with 0 values or unloaded values

    def inverted(self) -> "ScaleTransform":
        if not self.is_invertible:
            raise ValueError("ScaleTransform is not invertible: contains zero or unloaded scale value(s).")
        scale_inverted = tuple(1 / v for v in self.scale)
        return replace(self, scale=scale_inverted, _ome_zarr_path=None, source=self.target, target=self.source)

    def _to_affine_transform(
        self, *, source_ndim: Optional[int] = None, target_ndim: Optional[int] = None
    ) -> "AffineTransform":
        if not self.scale:
            raise CannotConvertToAffineError(self)
        affine = tuple(
            tuple(self.scale[row] if row == col else 0.0 for col in range(len(self.scale))) + (0.0,)
            for row in range(len(self.scale))
        )
        return AffineTransform(affine=affine, source=self.source, target=self.target)

    def composed_with(self, earlier: "Transform") -> Union["ScaleTransform", "AffineTransform", None]:
        if not self.scale:
            return None
        if not self._endpoints_can_chain_after(earlier):
            return None
        composed_source = self._composed_source(earlier)
        composed_target = self._composed_target(earlier)

        if isinstance(earlier, IdentityTransform):
            return replace(self, source=composed_source, target=composed_target)

        if isinstance(earlier, ScaleTransform):
            if not earlier.scale:
                return None
            return replace(
                self,
                scale=tuple(a * b for a, b in zip(self.scale, earlier.scale)),
                source=composed_source,
                target=composed_target,
                _ome_zarr_path=None,
            )

        return self._compose_via_affine(earlier)

    def simplified(self) -> Union["ScaleTransform", "IdentityTransform"]:
        if self.scale and all(abs(p - 1) <= IDENTITY_TOLERANCE for p in self.scale):
            return IdentityTransform(source=self.source, target=self.target)
        return self

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        payload_dict = {"path": self._ome_zarr_path} if self._ome_zarr_path else {"scale": list(self.scale)}
        return {"type": "scale", **payload_dict}

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        if not self.scale:
            return None
        return len(self.scale), len(self.scale)

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "ScaleTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        if "scale" in ome_dict:
            return cls(
                scale=_require_numbers(ome_dict["scale"], "scale"),
                _ome_zarr_name=cls._parse_name(ome_dict),
                source=source,
                target=target,
            )
        return cls(
            scale=(),
            _ome_zarr_path=_require_path(ome_dict, transform_type="scale"),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        if not self.scale and self._ome_zarr_path is None:
            raise ValueError("ScaleTransform requires either scale values or a path.")
        if self.scale and self._ome_zarr_path is not None:
            raise ValueError("ScaleTransform requires exactly one of scale values or a path.")
        if self.scale:
            scale = _require_numbers(self.scale, "scale")
            object.__setattr__(self, "scale", scale)
        elif not isinstance(self._ome_zarr_path, str) or not self._ome_zarr_path:
            raise ValueError(f"ScaleTransform requires a non-empty path. Received: {self._ome_zarr_path!r}")
        Transform.__post_init__(self)

    @classmethod
    def from_pixel_size(cls, pixel_size: PixelSize):
        return cls(scale=tuple(pixel_size.values()))

    def to_pixel_size(self, axes: Iterable[AxisKey]) -> PixelSize:
        if not self.scale:
            raise ValueError("Cannot derive PixelSize: Values not set.")
        axes = tuple(axes)
        if len(axes) != len(self.scale):
            raise ValueError(
                f"Cannot derive PixelSize: expected {len(self.scale)} axes, received {len(axes)}: {list(axes)}"
            )
        return PixelSize(zip(axes, self.scale))


@dataclass(frozen=True, slots=True)
class TranslationTransform(AffineRepresentableTransform):
    translation: Tuple[float, ...]  # Can be empty if _ome_zarr_path is provided instead
    _ome_zarr_path: Optional[str] = None
    """OME-Zarr allows translation with path to a zarr instead of plain values.
    Nobody uses this in practice.
    Support round-trip of such metadata, but until anyone actually needs it, we'll accept
    that such TranslationTransform objects are useless in practice."""

    @property
    def is_invertible(self) -> bool:
        return bool(self.translation)

    def inverted(self) -> "TranslationTransform":
        if not self.is_invertible:
            raise ValueError("TranslationTransform is not invertible: translation values are unloaded.")
        translation_inverted = tuple(-v for v in self.translation)
        return replace(
            self, translation=translation_inverted, _ome_zarr_path=None, source=self.target, target=self.source
        )

    def _to_affine_transform(
        self, *, source_ndim: Optional[int] = None, target_ndim: Optional[int] = None
    ) -> "AffineTransform":
        if not self.translation:
            raise CannotConvertToAffineError(self)
        affine = tuple(
            tuple(1.0 if row == col else 0.0 for col in range(len(self.translation))) + (self.translation[row],)
            for row in range(len(self.translation))
        )
        return AffineTransform(affine=affine, source=self.source, target=self.target)

    def composed_with(self, earlier: "Transform") -> Union["TranslationTransform", "AffineTransform", None]:
        if not self.translation:
            return None
        if not self._endpoints_can_chain_after(earlier):
            return None
        composed_source = self._composed_source(earlier)
        composed_target = self._composed_target(earlier)

        if isinstance(earlier, IdentityTransform):
            return replace(self, source=composed_source, target=composed_target)

        if isinstance(earlier, TranslationTransform):
            if not earlier.translation:
                return None
            return replace(
                self,
                translation=tuple(a + b for a, b in zip(self.translation, earlier.translation)),
                source=composed_source,
                target=composed_target,
                _ome_zarr_path=None,
            )

        return self._compose_via_affine(earlier)

    def simplified(self) -> Union["TranslationTransform", "IdentityTransform"]:
        if self.translation and all(abs(s) <= IDENTITY_TOLERANCE for s in self.translation):
            return IdentityTransform(source=self.source, target=self.target)
        return self

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        payload_dict = {"path": self._ome_zarr_path} if self._ome_zarr_path else {"translation": list(self.translation)}
        return {"type": "translation", **payload_dict}

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        if not self.translation:
            return None
        return len(self.translation), len(self.translation)

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "TranslationTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        if "translation" in ome_dict:
            return cls(
                translation=_require_numbers(ome_dict["translation"], "translation"),
                _ome_zarr_name=cls._parse_name(ome_dict),
                source=source,
                target=target,
            )
        return cls(
            translation=(),
            _ome_zarr_path=_require_path(ome_dict, transform_type="translation"),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        if not self.translation and self._ome_zarr_path is None:
            raise ValueError("TranslationTransform requires either translation values or a path.")
        if self.translation and self._ome_zarr_path is not None:
            raise ValueError("TranslationTransform requires exactly one of translation values or a path.")
        if self.translation:
            translation = _require_numbers(self.translation, "translation")
            object.__setattr__(self, "translation", translation)
        elif not isinstance(self._ome_zarr_path, str) or not self._ome_zarr_path:
            raise ValueError(f"TranslationTransform requires a non-empty path. Received: {self._ome_zarr_path!r}")
        Transform.__post_init__(self)

    @classmethod
    def from_translation(cls, translation: Translation):
        return cls(translation=tuple(translation.values()))

    def to_translation(self, axes: Iterable[AxisKey]) -> Translation:
        if not self.translation:
            raise ValueError("Cannot derive Translation: Values not set")
        axes = tuple(axes)
        if len(axes) != len(self.translation):
            raise ValueError(
                f"Cannot derive Translation: expected {len(self.translation)} axes, "
                f"received {len(axes)}: {list(axes)}"
            )
        return Translation(zip(axes, self.translation))


@dataclass(frozen=True, slots=True)
class RotationTransform(AffineRepresentableTransform):
    rotation: Optional[FloatMatrix] = None
    _ome_zarr_path: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        return self.rotation is not None

    def inverted(self) -> "RotationTransform":
        if self.rotation is None:
            raise ValueError("Path-backed RotationTransform cannot be inverted.")
        return replace(
            self,
            rotation=matrix_transpose(self.rotation),
            _ome_zarr_path=None,
            source=self.target,
            target=self.source,
        )

    def _to_affine_transform(
        self, *, source_ndim: Optional[int] = None, target_ndim: Optional[int] = None
    ) -> "AffineTransform":
        if self.rotation is None:
            raise CannotConvertToAffineError(self)
        affine = tuple(row + (0.0,) for row in self.rotation)
        return AffineTransform(affine=affine, source=self.source, target=self.target)

    def composed_with(self, earlier: "Transform") -> Union["RotationTransform", "AffineTransform", None]:
        if self.rotation is None:
            return None
        if not self._endpoints_can_chain_after(earlier):
            return None
        composed_source = self._composed_source(earlier)
        composed_target = self._composed_target(earlier)

        if isinstance(earlier, IdentityTransform):
            return replace(self, source=composed_source, target=composed_target)

        if isinstance(earlier, RotationTransform):
            if earlier.rotation is None:
                return None
            composed_rotation = matrix_multiply(self.rotation, earlier.rotation)
            return replace(
                self,
                rotation=composed_rotation,
                _ome_zarr_path=None,
                source=composed_source,
                target=composed_target,
            )

        return self._compose_via_affine(earlier)

    def simplified(self) -> Union["RotationTransform", "IdentityTransform"]:
        if self.rotation and is_identity_matrix(self.rotation, tolerance=IDENTITY_TOLERANCE):
            return IdentityTransform(source=self.source, target=self.target)
        return self

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        if self._ome_zarr_path is not None:
            return {"type": "rotation", "path": self._ome_zarr_path}
        assert self.rotation is not None
        return {"type": "rotation", "rotation": [list(row) for row in self.rotation]}

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        if self.rotation is None:
            return None
        rows, cols = matrix_shape(self.rotation)
        return cols, rows

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "RotationTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        if "rotation" in ome_dict:
            return cls(
                rotation=_require_rectangular_matrix(ome_dict["rotation"], "rotation"),
                _ome_zarr_name=cls._parse_name(ome_dict),
                source=source,
                target=target,
            )
        return cls(
            _ome_zarr_path=_require_path(ome_dict, transform_type="rotation"),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        if self.rotation is None and self._ome_zarr_path is None:
            raise ValueError("RotationTransform requires either rotation values or path.")
        if self.rotation is not None and self._ome_zarr_path is not None:
            raise ValueError("RotationTransform cannot have both rotation values and path.")
        if self.rotation is not None:
            rotation = _require_rectangular_matrix(self.rotation, "rotation")
            rows, cols = matrix_shape(rotation)
            if rows != cols:
                raise ValueError(f"RotationTransform matrix must be square. Received shape: {(rows, cols)}")
            if not is_rotation_matrix(rotation):
                raise ValueError(f"RotationTransform matrix must define a rotation. Received: {self.rotation!r}.")
            object.__setattr__(self, "rotation", rotation)
        elif not isinstance(self._ome_zarr_path, str) or not self._ome_zarr_path:
            raise ValueError(f"RotationTransform without values requires path. Received: {self._ome_zarr_path!r}")
        Transform.__post_init__(self)


@dataclass(frozen=True, slots=True)
class AffineTransform(AffineRepresentableTransform):
    affine: Optional[FloatMatrix] = None
    _ome_zarr_path: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        if self.affine is None:
            return False
        source_ndim, target_ndim = self._ndim_by_payload() or (None, None)
        if source_ndim != target_ndim:
            return False
        return abs(matrix_determinant(self._linear())) > DETERMINANT_SINGULARITY_TOLERANCE

    def inverted(self) -> "AffineTransform":
        if not self.is_invertible:
            raise ValueError("This AffineTransform is not invertible.")
        inverse_linear = matrix_invert(self._linear())
        inverse_offset = tuple(-v for v in matrix_vector_multiply(inverse_linear, self._translation()))
        inverse_affine = tuple(row + (inverse_offset[i],) for i, row in enumerate(inverse_linear))
        return replace(
            self,
            affine=inverse_affine,
            _ome_zarr_path=None,
            source=self.target,
            target=self.source,
        )

    def _to_affine_transform(
        self, *, source_ndim: Optional[int] = None, target_ndim: Optional[int] = None
    ) -> "AffineTransform":
        if self.affine is None:
            raise CannotConvertToAffineError(self)
        return self

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if self.affine is None:
            return None
        if not self._endpoints_can_chain_after(earlier):
            return None
        if isinstance(earlier, IdentityTransform):
            return replace(
                self,
                source=self._composed_source(earlier),
                target=self._composed_target(earlier),
            )
        if not isinstance(earlier, AffineRepresentableTransform):
            return None
        self_ndim = self._ndim_by_payload()
        assert self_ndim is not None
        try:
            earlier_affine = earlier._to_affine_transform(target_ndim=self_ndim[0])
        except CannotConvertToAffineError:
            return None
        earlier_ndim = earlier_affine._ndim_by_payload()
        if earlier_ndim is None or self_ndim[0] != earlier_ndim[1]:
            return None
        new_linear = matrix_multiply(self._linear(), earlier_affine._linear())
        new_t = tuple(
            a + b
            for a, b in zip(matrix_vector_multiply(self._linear(), earlier_affine._translation()), self._translation())
        )
        affine = tuple(row + (new_t[i],) for i, row in enumerate(new_linear))
        return replace(
            self,
            affine=affine,
            _ome_zarr_path=None,
            source=self._composed_source(earlier),
            target=self._composed_target(earlier),
        )

    def simplified(
        self,
    ) -> Union[
        "AffineTransform",
        "TransformSequence",
        "RotationTransform",
        "ScaleTransform",
        "TranslationTransform",
        "IdentityTransform",
    ]:
        if self.affine is None:
            return self
        linear = self._linear()
        translation = self._translation()
        target_ndim, source_ndim = matrix_shape(linear)
        if target_ndim == source_ndim:
            is_linear_identity = is_identity_matrix(linear, tolerance=IDENTITY_TOLERANCE)
            is_translation_identity = all(abs(t) <= IDENTITY_TOLERANCE for t in translation)
            if is_linear_identity and is_translation_identity:
                return IdentityTransform(source=self.source, target=self.target)
            if is_linear_identity:
                return TranslationTransform(translation=translation, source=self.source, target=self.target)
            is_linear_scale_only = is_diagonal_matrix(linear, tolerance=IDENTITY_TOLERANCE)
            if is_translation_identity:
                if is_linear_scale_only:
                    return ScaleTransform(
                        scale=tuple(linear[i][i] for i in range(len(linear))),
                        source=self.source,
                        target=self.target,
                    )
                if is_rotation_matrix(linear, tolerance=IDENTITY_TOLERANCE):
                    return RotationTransform(rotation=linear, source=self.source, target=self.target)
            if is_linear_scale_only:
                from ._base import TransformSequence

                return TransformSequence(
                    (
                        ScaleTransform(scale=tuple(linear[i][i] for i in range(len(linear)))),
                        TranslationTransform(translation=translation),
                    ),
                    source=self.source,
                    target=self.target,
                )
            if is_rotation_matrix(linear, tolerance=IDENTITY_TOLERANCE):
                from ._base import TransformSequence

                return TransformSequence(
                    (
                        RotationTransform(rotation=linear),
                        TranslationTransform(translation=translation),
                    ),
                    source=self.source,
                    target=self.target,
                )
            scale = [sum(value * value for value in row) ** 0.5 for row in linear]
            if all(abs(value) > IDENTITY_TOLERANCE for value in scale):
                rotation = tuple(tuple(value / scale[i] for value in row) for i, row in enumerate(linear))
                if matrix_determinant(rotation) < 0:
                    scale[0] *= -1
                    rotation = (tuple(-value for value in rotation[0]),) + rotation[1:]
                if is_rotation_matrix(rotation, tolerance=IDENTITY_TOLERANCE):
                    from ._base import TransformSequence

                    return TransformSequence(
                        (
                            RotationTransform(rotation=rotation),
                            ScaleTransform(scale=tuple(scale)),
                            TranslationTransform(translation=translation),
                        ),
                        source=self.source,
                        target=self.target,
                    ).simplified()
        return self

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        if self._ome_zarr_path is not None:
            return {"type": "affine", "path": self._ome_zarr_path}
        assert self.affine is not None
        return {"type": "affine", "affine": [list(row) for row in self.affine]}

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        if self.affine is None:
            return None
        rows, cols = matrix_shape(self.affine)
        return cols - 1, rows

    def _source_ndim_must_eq_target_ndim(self) -> bool:
        return False

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "AffineTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        if "affine" in ome_dict:
            return cls(
                affine=_require_rectangular_matrix(ome_dict["affine"], "affine"),
                _ome_zarr_name=cls._parse_name(ome_dict),
                source=source,
                target=target,
            )
        return cls(
            _ome_zarr_path=_require_path(ome_dict, transform_type="affine"),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        if self.affine is None and self._ome_zarr_path is None:
            raise ValueError("AffineTransform requires either affine values or a path.")
        if self.affine is not None and self._ome_zarr_path is not None:
            raise ValueError("AffineTransform requires exactly one of affine values or a path.")
        if self.affine is not None:
            affine = _require_rectangular_matrix(self.affine, "affine")
            _rows, cols = matrix_shape(affine)
            if cols < 2:
                raise ValueError("AffineTransform matrix must have at least one input dimension and one offset column.")
            # Arbitrary rectangles permitted. Spec says: Interpret the shape as (output_ndim, input_ndim + 1).
            object.__setattr__(self, "affine", affine)
        elif not isinstance(self._ome_zarr_path, str) or not self._ome_zarr_path:
            raise ValueError(f"AffineTransform requires a non-empty path. Received: {self._ome_zarr_path!r}")
        Transform.__post_init__(self)

    def _linear(self) -> FloatMatrix:
        assert self.affine, "Ensure `self.affine is not None` before calling"
        return tuple(row[:-1] for row in self.affine)

    def _translation(self) -> FloatVector:
        assert self.affine, "Ensure `self.affine is not None` before calling"
        return tuple(row[-1] for row in self.affine)


@dataclass(frozen=True, slots=True)
class CoordinatesTransform(Transform):
    path: RelativePath
    interpolation: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        return False

    def inverted(self) -> "Transform":
        raise ValueError("CoordinatesTransform is generally not invertible.")

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        return None

    def simplified(self) -> "CoordinatesTransform":
        return self

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": "coordinates", "path": self.path}
        if self.interpolation is not None:
            result["interpolation"] = self.interpolation
        return result

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        return None

    def _source_ndim_must_eq_target_ndim(self) -> bool:
        return False

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "CoordinatesTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        return cls(
            path=_require_path(ome_dict, transform_type="coordinates"),
            interpolation=cls._parse_interpolation(ome_dict),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"CoordinatesTransform requires a non-empty path. Received: {self.path!r}")
        if self.interpolation is not None and (not isinstance(self.interpolation, str) or not self.interpolation):
            raise ValueError(
                f"CoordinatesTransform interpolation must be a non-empty string. Received: {self.interpolation!r}"
            )
        Transform.__post_init__(self)

    def _validate_bound_axes(self) -> None:
        return None

    @staticmethod
    def _parse_interpolation(ome_dict: Mapping[str, Any]) -> Optional[str]:
        interpolation = ome_dict.get("interpolation")
        if interpolation is None:
            return None
        if not isinstance(interpolation, str) or not interpolation:
            raise ValueError(
                f"Invalid coordinates transform metadata. Expected interpolation string, received: {interpolation!r}"
            )
        return interpolation


@dataclass(frozen=True, slots=True)
class DisplacementsTransform(Transform):
    path: RelativePath
    interpolation: Optional[str] = None

    @property
    def is_invertible(self) -> bool:
        return False

    def inverted(self) -> "Transform":
        raise ValueError("DisplacementsTransform is generally not invertible.")

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        return None

    def simplified(self) -> "DisplacementsTransform":
        return self

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": "displacements", "path": self.path}
        if self.interpolation is not None:
            result["interpolation"] = self.interpolation
        return result

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        return None

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "DisplacementsTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        return cls(
            path=_require_path(ome_dict, transform_type="displacements"),
            interpolation=cls._parse_interpolation(ome_dict),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"DisplacementsTransform requires a non-empty path. Received: {self.path!r}")
        if self.interpolation is not None and (not isinstance(self.interpolation, str) or not self.interpolation):
            raise ValueError(
                f"DisplacementsTransform interpolation must be a non-empty string. Received: {self.interpolation!r}"
            )
        Transform.__post_init__(self)

    @staticmethod
    def _parse_interpolation(ome_dict: Mapping[str, Any]) -> Optional[str]:
        interpolation = ome_dict.get("interpolation")
        if interpolation is None:
            return None
        if not isinstance(interpolation, str) or not interpolation:
            raise ValueError(
                f"Invalid displacements transform metadata. Expected interpolation string, received: {interpolation!r}"
            )
        return interpolation


@dataclass(frozen=True, slots=True)
class MapAxisTransform(AffineRepresentableTransform):
    """MapAxisTransform represents a pure transposition (no drops or inserts)"""

    map_axis: Tuple[int, ...]

    @property
    def is_invertible(self) -> bool:
        """Map-axis is not allowed to drop axes, so it's always invertible"""
        return True

    def inverted(self) -> "MapAxisTransform":
        inverted_map = [0] * len(self.map_axis)
        for output_index, input_index in enumerate(self.map_axis):
            inverted_map[input_index] = output_index
        return replace(self, map_axis=tuple(inverted_map), source=self.target, target=self.source)

    def _to_affine_transform(
        self, *, source_ndim: Optional[int] = None, target_ndim: Optional[int] = None
    ) -> "AffineTransform":
        affine = tuple(
            tuple(1.0 if source_axis == mapped_source else 0.0 for source_axis in range(len(self.map_axis))) + (0.0,)
            for mapped_source in self.map_axis
        )
        return AffineTransform(affine=affine, source=self.source, target=self.target)

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if not self._endpoints_can_chain_after(earlier):
            return None
        if isinstance(earlier, IdentityTransform):
            return replace(
                self,
                source=self._composed_source(earlier),
                target=self._composed_target(earlier),
            )
        if isinstance(earlier, MapAxisTransform):
            return replace(
                self,
                map_axis=tuple(earlier.map_axis[i] for i in self.map_axis),
                source=self._composed_source(earlier),
                target=self._composed_target(earlier),
            )
        return self._compose_via_affine(earlier)

    def simplified(self) -> Union["MapAxisTransform", "IdentityTransform"]:
        if self.map_axis == tuple(range(len(self.map_axis))):
            return IdentityTransform(source=self.source, target=self.target)
        return self

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        return {"type": "mapAxis", "mapAxis": list(self.map_axis)}

    def _ndim_by_payload(self) -> Tuple[int, int]:
        return len(self.map_axis), len(self.map_axis)

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "MapAxisTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        return cls(
            map_axis=_require_int_or_empty_tuple(ome_dict.get("mapAxis", ()), "mapAxis", transform_type="mapAxis"),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        map_axis = _require_int_or_empty_tuple(self.map_axis, "map_axis", transform_type="MapAxisTransform")
        if not map_axis:
            raise ValueError("MapAxisTransform requires a non-empty axis permutation.")
        if not _covers_all_indices(map_axis):
            raise ValueError(f"MapAxisTransform must include all zero-based indices. Received: {map_axis!r}")
        object.__setattr__(self, "map_axis", map_axis)
        Transform.__post_init__(self)


@dataclass(frozen=True, slots=True)
class ProjectAxisTransform(AffineRepresentableTransform):
    """ProjectAxisTransform represents axis dropping and insertion"""

    drops: Tuple[int, ...] = field(default=())
    """Indices of source axes dropped.
    E.g. if source is 'xyz' and target is 'xz', the connecting ProjectAxisTransform has drops=(1,)"""
    inserts: Tuple[int, ...] = field(default=())
    """Indices of *target* axes that are new insertions.
    E.g. if source is 'yx' and target is 'cyx', the connecting ProjectAxisTransform has inserts=(0,)"""

    @property
    def is_invertible(self) -> bool:
        """Axis dropping is not invertible"""
        return not self.drops

    def inverted(self) -> "Transform":
        if self.drops:
            raise ValueError(f"Axis dropping is not invertible. This transform drops axes {self.drops!r}.")
        return replace(self, drops=self.inserts, inserts=(), source=self.target, target=self.source)

    def _to_affine_transform(
        self, *, source_ndim: Optional[int] = None, target_ndim: Optional[int] = None
    ) -> "AffineTransform":
        # Drawing ndim from self.source/target endpoints is not appropriate here - this is an internal
        # helper for .composed_with.
        if source_ndim is None and target_ndim is not None:
            source_ndim = target_ndim + len(self.drops) - len(self.inserts)
        if target_ndim is None and source_ndim is not None:
            target_ndim = source_ndim - len(self.drops) + len(self.inserts)
        assert (
            source_ndim is not None and target_ndim is not None
        ), f"should never be called without ndim: {source_ndim!r}, {target_ndim!r}"
        if source_ndim < 1 or target_ndim < 1:  # This might also be assert-level...
            raise CannotConvertToAffineError(self)
        # The two below are guarded by ndim validation when constructing a sequence. They could happen
        # when directly calling `projectAxis.composed_with(other)` without validating ndim, but this is not public API.
        assert target_ndim == source_ndim - len(self.drops) + len(
            self.inserts
        ), f"nonsense params: {source_ndim!r} != {target_ndim!r} - {len(self.drops)} + {len(self.inserts)}"
        assert not any(index >= source_ndim for index in self.drops) and not any(
            index >= target_ndim for index in self.inserts
        ), f"ndim mismatch: some of {self.drops} are >= {source_ndim!r} or some {self.inserts} >= {target_ndim!r}"
        retained_sources = iter(index for index in range(source_ndim) if index not in self.drops)
        affine_rows = []
        for target_axis in range(target_ndim):
            source_axis = None if target_axis in self.inserts else next(retained_sources)
            affine_rows.append(tuple(1.0 if source_axis == index else 0.0 for index in range(source_ndim)) + (0.0,))
        return AffineTransform(affine=tuple(affine_rows), source=self.source, target=self.target)

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        """
        Composing ProjectAxisTransform is done by projecting implicit earlier.source axis indices
        into the intermediate (earlier.target / self.source), then adding self's own drops/adds.
        None represents inserted axes (None because these have no corresponding source axis index).
        """
        if not self._endpoints_can_chain_after(earlier):
            return None
        if isinstance(earlier, IdentityTransform):
            return replace(
                self,
                source=self._composed_source(earlier),
                target=self._composed_target(earlier),
            )
        if not isinstance(earlier, ProjectAxisTransform):
            return self._compose_via_affine(earlier)

        highest_known_earlier_src_axis = max(earlier.drops, default=-1)
        intermediate_axes: List[Optional[int]] = [
            i for i in range(highest_known_earlier_src_axis + 1) if i not in earlier.drops
        ]
        next_source_axis = highest_known_earlier_src_axis + 1
        for inserted in sorted(earlier.inserts):
            while len(intermediate_axes) < inserted:
                # `inserted` implies additional source axes we didn't know from the drops
                intermediate_axes.append(next_source_axis)
                next_source_axis += 1
            intermediate_axes.insert(inserted, None)

        # intermediate_axes is now something like e.g. [0, None] for "drop 1, insert 1".
        dropped_from_earlier_source: List[int] = list(earlier.drops)
        for drop in self.drops:
            while len(intermediate_axes) <= drop:
                # This `drop` on self (later transform) implies there must've been more axes on earlier.source
                intermediate_axes.append(next_source_axis)
                next_source_axis += 1
            dropped_source = intermediate_axes[drop]
            if dropped_source is not None:
                dropped_from_earlier_source.append(dropped_source)

        # intermediate_axes now already contains any earlier.source axes implied by self.drops
        self_target_axes: List[Optional[int]] = [
            axis for i, axis in enumerate(intermediate_axes) if i not in self.drops
        ]
        for inserted in sorted(self.inserts):
            while len(self_target_axes) < inserted:
                self_target_axes.append(1337)  # The actual value doesn't matter anymore, just that it wasn't an insert
            self_target_axes.insert(inserted, None)
        inserted_overall = [i for i, axis in enumerate(self_target_axes) if axis is None]

        return replace(
            self,
            drops=sorted(dropped_from_earlier_source),
            inserts=inserted_overall,
            source=self._composed_source(earlier),
            target=self._composed_target(earlier),
        )

    def simplified(self) -> Union["ProjectAxisTransform", "IdentityTransform"]:
        if not self.drops and not self.inserts:
            return IdentityTransform(source=self.source, target=self.target)
        return self

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": "projectAxis"}
        if self.drops:
            result["droppedInputs"] = list(self.drops)
        if self.inserts:
            result["createdOutputs"] = list(self.inserts)
        return result

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        # The payload can only imply a *minimum* axis count. That's not helpful for callers here.
        return None

    def _source_ndim_must_eq_target_ndim(self) -> bool:
        return False

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "ProjectAxisTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        return cls(
            drops=_require_int_or_empty_tuple(
                ome_dict.get("droppedInputs", ()), "droppedInputs", transform_type="projectAxis"
            ),
            inserts=_require_int_or_empty_tuple(
                ome_dict.get("createdOutputs", ()), "createdOutputs", transform_type="projectAxis"
            ),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        drops = _require_int_or_empty_tuple(self.drops, "drops", transform_type="ProjectAxisTransform")
        inserts = _require_int_or_empty_tuple(self.inserts, "inserts", transform_type="ProjectAxisTransform")
        _validate_unique(drops, "ProjectAxisTransform.drops")
        _validate_unique(inserts, "ProjectAxisTransform.inserts")
        object.__setattr__(self, "drops", drops)
        object.__setattr__(self, "inserts", inserts)
        Transform.__post_init__(self)

    def _validate_bound_axes(self) -> None:
        source_axes = tuple(self.source.owner.axes()) if isinstance(self.source, NodeRef) else None
        target_axes = tuple(self.target.owner.axes()) if isinstance(self.target, NodeRef) else None

        if source_axes is not None:
            source_ndim = len(source_axes)
            if any(index >= source_ndim for index in self.drops):
                raise ValueError(
                    f"ProjectAxisTransform drops input index outside source axes {list(source_axes)}: "
                    f"{self.drops!r}"
                )
        else:
            source_ndim = None

        if target_axes is not None:
            target_ndim = len(target_axes)
            if any(index >= target_ndim for index in self.inserts):
                raise ValueError(
                    f"ProjectAxisTransform inserts output index outside target axes {target_axes}: {self.inserts!r}"
                )
        else:
            target_ndim = None

        if source_ndim is not None and target_ndim is not None:
            expected_target_ndim = source_ndim - len(self.drops) + len(self.inserts)
            if expected_target_ndim != target_ndim:
                raise ValueError(
                    f"ProjectAxisTransform expects {expected_target_ndim} target axes from source axes "
                    f"{source_axes} but target coordinate system has {target_ndim}: {target_axes}"
                )


@dataclass(frozen=True, slots=True)
class BijectionTransform(Transform):
    """Provides an explicit way to state the inversion of a normally not-invertible transform."""

    forward: Transform
    inverse: Transform

    @property
    def is_invertible(self) -> bool:
        return True

    def inverted(self) -> "BijectionTransform":
        return replace(
            self,
            forward=self.inverse,
            inverse=self.forward,
            source=self.target,
            target=self.source,
        )

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if not self._endpoints_can_chain_after(earlier):
            return None

        if isinstance(earlier, BijectionTransform):
            forward = self.forward.composed_with(earlier.forward)
            if forward is None:
                return None

            inverse = earlier.inverse.composed_with(self.inverse)
            if inverse is None:
                return None
        else:
            forward = self.forward.composed_with(earlier)
            if forward is None:
                return None

            inverse = earlier.inverted().composed_with(self.inverse)
            if inverse is None:
                return None

        return replace(
            self,
            forward=forward,
            inverse=inverse,
            source=self._composed_source(earlier),
            target=self._composed_target(earlier),
        )

    def simplified(self) -> Union["BijectionTransform", "IdentityTransform"]:
        forward = self.forward.simplified()
        inverse = self.inverse.simplified()
        if isinstance(forward, IdentityTransform) and isinstance(inverse, IdentityTransform):
            return IdentityTransform(source=self.source, target=self.target)
        if forward is self.forward and inverse is self.inverse:
            return self
        return replace(self, forward=forward, inverse=inverse)

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        return {
            "type": "bijection",
            "forward": self.forward.to_ome_zarr(version),
            "inverse": self.inverse.to_ome_zarr(version),
        }

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        forward_ndim = self.forward._ndim_by_payload()
        if forward_ndim is not None:
            return forward_ndim
        inverse_ndim = self.inverse._ndim_by_payload()
        if inverse_ndim is not None:
            return inverse_ndim[1], inverse_ndim[0]
        return None

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "BijectionTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        forward_dict = ome_dict.get("forward")
        inverse_dict = ome_dict.get("inverse")
        if not isinstance(forward_dict, MappingABC) or not isinstance(inverse_dict, MappingABC):
            raise ValueError(f"Invalid bijection transform metadata. Received: {ome_dict!r}")
        return cls(
            forward=Transform.from_ome_zarr(forward_dict),
            inverse=Transform.from_ome_zarr(inverse_dict),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        if not isinstance(self.forward, Transform) or not isinstance(self.inverse, Transform):
            raise ValueError("BijectionTransform forward and inverse must be Transform instances.")
        if isinstance(self.forward, ProjectAxisTransform) or isinstance(self.inverse, ProjectAxisTransform):
            # Unless the ProjectAxis is a noop, one of the directions destroys information by dropping axes.
            # Specifying them as inverses of each other is always semantically inaccurate.
            # Just forbid using it. If necessary for convenience, the noop could be allowed.
            raise ValueError("ProjectAxisTransforms cannot be used in BijectionTransform.")
        self._infer_endpoints_from_children()
        self._validate_child_endpoints()
        self._validate_child_dimensionality()
        Transform.__post_init__(self)

    def _validate_bound_axes(self) -> None:
        ndim = self._ndim_by_payload()
        if ndim is not None and ndim[0] != ndim[1]:
            raise ValueError(f"BijectionTransform requires equal input and output dimensionality. Received: {ndim!r}")

        Transform._validate_bound_axes(self)

        source_axes = tuple(self.source.owner.axes()) if isinstance(self.source, NodeRef) else None
        target_axes = tuple(self.target.owner.axes()) if isinstance(self.target, NodeRef) else None
        if source_axes is not None and target_axes is not None and len(source_axes) != len(target_axes):
            raise ValueError(
                f"BijectionTransform endpoints have incompatible dimensionality: "
                f"source {list(source_axes)} vs target {list(target_axes)}"
            )

    def _infer_endpoints_from_children(self) -> None:
        if self.source is None and self.forward.source is not None:
            object.__setattr__(self, "source", self.forward.source)
        if self.target is None and self.forward.target is not None:
            object.__setattr__(self, "target", self.forward.target)
        if self.source is None and self.inverse.target is not None:
            object.__setattr__(self, "source", self.inverse.target)
        if self.target is None and self.inverse.source is not None:
            object.__setattr__(self, "target", self.inverse.source)

    def _validate_child_endpoints(self) -> None:
        expectations = (
            ("forward input", self.forward.source, self.source),
            ("forward output", self.forward.target, self.target),
            ("inverse input", self.inverse.source, self.target),
            ("inverse output", self.inverse.target, self.source),
        )
        for label, actual, expected in expectations:
            if actual is not None and expected is not None and actual != expected:
                raise ValueError(
                    f"BijectionTransform endpoint does not match parent endpoint. "
                    f"Received {label}: (ID {id(actual)}) {actual!r}, "
                    f"vs parent: (ID {id(expected)}) {expected!r}"
                )

    def _validate_child_dimensionality(self) -> None:
        forward_ndim = self.forward._ndim_by_payload()
        inverse_ndim = self.inverse._ndim_by_payload()
        if forward_ndim is not None and inverse_ndim is not None:
            if forward_ndim[0] != inverse_ndim[1] or forward_ndim[1] != inverse_ndim[0]:
                raise ValueError(
                    "BijectionTransform forward and inverse dimensionality disagree: "
                    f"forward={forward_ndim!r}, inverse={inverse_ndim!r}"
                )


@dataclass(frozen=True, slots=True)
class _ByDimensionChild:
    source_indices: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    transform: Transform

    @classmethod
    def from_ome_zarr(cls, item_dict: Mapping[str, Any]) -> "_ByDimensionChild":
        if not isinstance(item_dict, MappingABC):
            raise ValueError(f"Invalid byDimension transform item metadata. Received: {item_dict!r}")
        transformation_dict = item_dict.get("transformation")
        if not isinstance(transformation_dict, MappingABC):
            raise ValueError(f"Invalid byDimension transform item metadata. Received: {transformation_dict!r}")
        return cls(
            source_indices=_require_int_or_empty_tuple(
                item_dict.get("inputAxes", ()), "inputAxes", transform_type="byDimension"
            ),
            target_indices=_require_int_or_empty_tuple(
                item_dict.get("outputAxes", ()), "outputAxes", transform_type="byDimension"
            ),
            transform=Transform.from_ome_zarr(transformation_dict),
        )

    def to_ome_zarr(self, version: str) -> Dict[str, Any]:
        return {
            "inputAxes": list(self.source_indices),
            "outputAxes": list(self.target_indices),
            "transformation": self.transform.to_ome_zarr(version),
        }

    def __post_init__(self):
        source_indices = _require_int_or_empty_tuple(
            self.source_indices, "source_indices", transform_type="ByDimensionTransform.child"
        )
        target_indices = _require_int_or_empty_tuple(
            self.target_indices, "target_indices", transform_type="ByDimensionTransform.child"
        )
        _validate_unique(source_indices, "ByDimensionTransform.child.source_indices")
        _validate_unique(target_indices, "ByDimensionTransform.child.target_indices")
        if not isinstance(self.transform, Transform):
            raise ValueError(f"ByDimensionChild must contain a Transform instance, not {self.transform!r}.")
        ndim = self.transform._ndim_by_payload()
        if ndim is not None:
            source_ndim, target_ndim = ndim
            if len(source_indices) != source_ndim:
                raise ValueError(
                    f"ByDimensionTransform.child.source_indices must contain {source_ndim} entries for "
                    f"{type(self.transform).__name__}. Received: {source_indices!r}"
                )
            if len(target_indices) != target_ndim:
                raise ValueError(
                    f"ByDimensionTransform.child.target_indices must contain {target_ndim} entries for "
                    f"{type(self.transform).__name__}. Received: {target_indices!r}"
                )
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "target_indices", target_indices)


@dataclass(frozen=True, slots=True)
class ByDimensionTransform(Transform):
    transforms: Tuple[_ByDimensionChild, ...]

    @property
    def is_invertible(self) -> bool:
        source_indices = tuple(axis for child in self.transforms for axis in child.source_indices)
        target_indices = tuple(axis for child in self.transforms for axis in child.target_indices)
        return (
            all(item.transform.is_invertible for item in self.transforms)
            and len(set(source_indices)) == len(source_indices)
            and _covers_all_indices(source_indices)
            and _covers_all_indices(target_indices)
        )

    def inverted(self) -> "ByDimensionTransform":
        if not self.is_invertible:
            raise ValueError("ByDimensionTransform is not invertible.")
        return replace(
            self,
            transforms=tuple(
                _ByDimensionChild(
                    source_indices=item.target_indices,
                    target_indices=item.source_indices,
                    transform=item.transform.inverted(),
                )
                for item in self.transforms
            ),
            source=self.target,
            target=self.source,
        )

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        """
        Only composes with ByDimensionTransform and ProjectAxisTransform.
        Could permit MapAxisTransform: As long as mapAxis only modifies sets of axes that are sourced by the
        same byDim child. Decided to reject this, because spec text doesn't read like reordering was intended for byDim.
        """
        if not self._endpoints_can_chain_after(earlier):
            return None
        if isinstance(earlier, ByDimensionTransform):
            earlier_by_outputs = {item.target_indices: item for item in earlier.transforms}

            new_items = []
            matched = set()

            for item in self.transforms:
                earlier_item = earlier_by_outputs.get(item.source_indices)
                if earlier_item is None:
                    return None
                matched.add(item.source_indices)

                composed = item.transform.composed_with(earlier_item.transform)
                if composed is None:
                    return None

                new_items.append(
                    _ByDimensionChild(
                        source_indices=earlier_item.source_indices,
                        target_indices=item.target_indices,
                        transform=composed,
                    )
                )

            if len(matched) != len(earlier.transforms):  # some of earlier's items had no match
                return None

            return replace(
                self,
                transforms=tuple(new_items),
                source=self._composed_source(earlier),
                target=self._composed_target(earlier),
            )

        if isinstance(earlier, ProjectAxisTransform):
            # Every drop increments bydimension items' source indices that were higher than the dropped index
            #  ex: Earlier "CYX --(drop 0)-> YX" + Later byDim "[0,1 --Scale-> 0,1]"
            #            = "CYX --> [1,2 --Scale-> 0,1] --> YX"
            # Every insert conversely decrements. This only works if all inserts are only used by identity.
            #  ex: Earlier "YX --(add 0)-> CYX" + Later byDim "[1,2 --Scale-> 1,2]+[0 --Ident-> 0]"
            #            = "YX --> [0,1 --Scale-> 1,2]+[_ --ProjectAxis-> 0] --> CYX"
            #  Note we still end up with ProjectAxis, but now it's inside the byDim and the identity is gone.
            items = []
            earlier_inserts_with_later_identity = []

            def remap_source(axis: int, dropped: Tuple[int, ...], inserted: Tuple[int, ...]) -> int:
                """Map an earlier.target axis back to the corresponding earlier.source axis.
                Every axis dropped increments eq/subsequent indices by 1, every insert decrements"""
                for d in dropped:
                    if d <= axis:
                        axis += 1
                for i in reversed(inserted):
                    if i < axis:
                        axis -= 1
                return axis

            for item in self.transforms:
                inserted_sources = tuple(axis for axis in item.source_indices if axis in earlier.inserts)
                if inserted_sources and isinstance(item.transform, IdentityTransform):
                    # Targets of self's identity children need to be collected into a new ProjectAxis child.
                    # Can't just keep inserted_sources though - the Identity might have remapped indices from src->tgt.
                    earlier_inserts_with_later_identity.extend(
                        tgt for src, tgt in zip(item.source_indices, item.target_indices) if src in earlier.inserts
                    )
                elif inserted_sources:
                    # The only way we could faithfully reproduce a non-identity that operates on an inserted axis
                    # is by nesting a sequence of [projectAxis, item.transform] inside this byDim item.
                    # That isn't necessarily better than the sequence of [projectAxis, byDim]
                    # that we're trying to collapse, so break off here.
                    return None

                unaffected_src_tgt_pairs = tuple(
                    (src, tgt)
                    for src, tgt in zip(item.source_indices, item.target_indices)
                    if src not in earlier.inserts
                )
                if unaffected_src_tgt_pairs:
                    kept_src, kept_tgt = zip(*unaffected_src_tgt_pairs)
                    items.append(
                        replace(
                            item,
                            source_indices=tuple(remap_source(ax, earlier.drops, earlier.inserts) for ax in kept_src),
                            target_indices=tuple(kept_tgt),
                        )
                    )

            if earlier_inserts_with_later_identity:
                items.append(
                    _ByDimensionChild(
                        source_indices=(),
                        target_indices=tuple(earlier_inserts_with_later_identity),
                        transform=ProjectAxisTransform(
                            drops=(),
                            inserts=tuple(range(len(earlier_inserts_with_later_identity))),
                        ),
                    )
                )

            return replace(
                self,
                transforms=tuple(items),
                source=self._composed_source(earlier),
                target=self._composed_target(earlier),
            )

        return None

    def simplified(self) -> Union["ByDimensionTransform", "IdentityTransform"]:
        """Simplify per subspace. If all subspaces are identities, unwrap the ByDimension."""
        simplified_items = tuple(replace(item, transform=item.transform.simplified()) for item in self.transforms)
        if all(
            isinstance(item.transform, IdentityTransform) and item.source_indices == item.target_indices
            for item in simplified_items
        ):
            return IdentityTransform(source=self.source, target=self.target)
        if all(new.transform is old.transform for new, old in zip(simplified_items, self.transforms)):
            return self
        return replace(self, transforms=simplified_items)

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        return {
            "type": "byDimension",
            "transformations": [item.to_ome_zarr(version) for item in self.transforms],
        }

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        # TODO: This transform type knows its output ndim, but it can't know the exact source ndim.
        #  E.g. having two children with source_indices (0,) and (0,) is valid. This means the source must have
        #  *at least* one axis. It might have more, but the others don't flow into a target, i.e. are dropped.
        return None

    def _source_ndim_must_eq_target_ndim(self) -> bool:
        return False

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "ByDimensionTransform":
        source, target = cls._parse_source_and_target(ome_dict)
        raw_transformations = ome_dict.get("transformations")
        if not isinstance(raw_transformations, list) or not raw_transformations:
            raise ValueError(f"Invalid byDimension transform metadata. Received: {ome_dict!r}")
        return cls(
            transforms=tuple(_ByDimensionChild.from_ome_zarr(item) for item in raw_transformations),
            _ome_zarr_name=cls._parse_name(ome_dict),
            source=source,
            target=target,
        )

    def __post_init__(self):
        children = tuple(self.transforms)
        if not children:
            raise ValueError("ByDimensionTransform requires at least one child transformation.")
        if any(not isinstance(c, _ByDimensionChild) for c in children):
            raise ValueError("ByDimensionTransform must only contain ByDimensionChild instances.")
        output_axes = tuple(axis for item in children for axis in item.target_indices)
        if len(set(output_axes)) != len(output_axes):
            raise ValueError(f"ByDimensionTransform target axes must be globally unique. Received: {output_axes!r}")
        if not _covers_all_indices(output_axes):
            raise ValueError(
                f"ByDimensionTransform target axes must include all zero-based indices. Received: {output_axes!r}"
            )
        object.__setattr__(self, "transforms", children)
        Transform.__post_init__(self)

    def _validate_bound_axes(self) -> None:
        source_axes = tuple(self.source.owner.axes()) if isinstance(self.source, NodeRef) else None
        target_axes = tuple(self.target.owner.axes()) if isinstance(self.target, NodeRef) else None

        if source_axes is not None:
            source_ndim = len(source_axes)
            for item in self.transforms:
                if any(axis >= source_ndim for axis in item.source_indices):
                    raise ValueError(
                        f"ByDimensionTransform input axis outside source axes {list(source_axes)}: "
                        f"{item.source_indices!r}"
                    )

        output_axes = tuple(axis for item in self.transforms for axis in item.target_indices)
        if target_axes is not None and set(output_axes) != set(range(len(target_axes))):
            raise ValueError(
                f"ByDimensionTransform output axes must cover target axes {list(target_axes)}. "
                f"Received: {output_axes!r}"
            )
