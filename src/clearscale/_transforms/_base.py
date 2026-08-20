import enum
import functools
import numbers
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace, fields
from itertools import chain
from typing import (
    Optional,
    Tuple,
    Dict,
    Mapping,
    Iterable,
    TypeVar,
    Union,
    Literal,
    Any,
    List,
    Generic,
    TypeGuard,
)

from clearscale._axis_values import (
    _AxisMapping,
    AxisKey,
    OrderedAxes,
    Unit,
)
from clearscale._errors import NoSuchCoordinateSystemError, MismatchingMultiscaleError

RelativePath = str  # 0.6.rc0: scene["coordinateTransformations"][]["input"]["path"]
CoordinateSystemName = str  # str from ["input"]["name"]
NodesByPath = Mapping[RelativePath, "TransformGraphNode"]
TransformGraphNodeT = TypeVar("TransformGraphNodeT", bound="TransformGraphNode", covariant=True)
_TransformSelf = TypeVar("_TransformSelf", bound="Transform")
_TransformSequenceSelf = TypeVar("_TransformSequenceSelf", bound="TransformSequence")
_RefT = TypeVar("_RefT")

PRE_TRANSFORMS_VERSIONS = ("0.1", "0.2", "0.3", "0.4", "0.5")
PRE_COLLECTIONS_VERSIONS = PRE_TRANSFORMS_VERSIONS + ("0.6.rc0",)


class CoordinateContinuity(str, enum.Enum):
    Categorical = enum.auto()
    Discrete = enum.auto()
    Continuous = enum.auto()


@dataclass(frozen=True, slots=True)
class AxisSemantics:
    coordinate_domain: Optional[CoordinateContinuity] = None
    _ome_zarr_type: Optional[str] = None
    _ome_zarr_unit: Optional[str] = None
    _ome_zarr_long_name: Optional[str] = None

    @classmethod
    def from_ome_zarr(cls, axis_dict: Mapping[str, Any]) -> "AxisSemantics":
        discrete = axis_dict.get("discrete")
        discrete_meaning = {None: None, False: CoordinateContinuity.Continuous, True: CoordinateContinuity.Discrete}
        coordinates = discrete_meaning.get(discrete)
        return cls(
            coordinate_domain=coordinates,
            _ome_zarr_type=axis_dict.get("type"),
            _ome_zarr_unit=axis_dict.get("unit"),
            _ome_zarr_long_name=axis_dict.get("longName"),
        )

    def __repr__(self):
        items = (f"{f.name}={getattr(self, f.name)!r}" for f in fields(self) if getattr(self, f.name) is not None)
        return f"{self.__class__.__name__}({', '.join(items)})"

    def to_ome_zarr(self, *, name: str) -> Dict[str, Any]:
        axis_dict: Dict[str, Any] = {"name": name}
        if self._ome_zarr_type:
            axis_dict["type"] = self._ome_zarr_type
        if self._ome_zarr_unit:
            axis_dict["unit"] = self._ome_zarr_unit
        if self._ome_zarr_long_name:
            axis_dict["longName"] = self._ome_zarr_long_name
        if self.coordinate_domain is not None:
            if self.coordinate_domain is CoordinateContinuity.Continuous:
                axis_dict["discrete"] = False
            else:
                axis_dict["discrete"] = True
        return axis_dict


class TransformGraphNode(ABC):
    """Mixin for classes that can own coordinate-system refs inside a TransformGraph."""

    @abstractmethod
    def axes(self) -> Iterable[AxisKey]: ...

    @abstractmethod
    def as_ref(self, name: CoordinateSystemName) -> "ResolvedRef": ...


@dataclass(frozen=True, slots=True)
class FileRef:
    """Just a path, but with the ability to answer 'What kind of path?'"""

    path: str
    """
    Path could be:
    - a relative URI like 'scales/s0' in OME-Zarr versions up to 0.6 (before RFC-8)
    - an absolute URI like 'https://...' or 'file:///C:/...'
    - a relative path like './scales/s0' or '../'
    """
    path_type: Literal["zarr", "json"] = "zarr"
    """
    'zarr': The path points to a zarr object that should be opened like 'zarr.open(path)'.
    'json': The path points to a file that should be read like 'json.loads(path)'.
    """

    def __post_init__(self):
        assert self.path_type in ("zarr", "json")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"Path must be string: {self.path}")

    @classmethod
    def from_string(cls, path: str):
        """Constructor that infers `path_type` from the provided path"""
        assert path and isinstance(path, str), f"Must call with string, received: {path!r}"
        return cls(path_type="json" if path.endswith(".json") else "zarr", path=path)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FileRef":
        path_type = value.get("type")
        if path_type not in ("zarr", "json"):
            raise ValueError(f"Invalid file reference type: {path_type!r}")
        path = value.get("path")
        if not isinstance(path, str):
            raise ValueError(f"Invalid file reference path: {path!r}")
        return cls(path_type=path_type, path=path)

    def to_ome_zarr(self, version: str = "rfc-8") -> Union[str, Dict[str, str]]:
        if version in PRE_COLLECTIONS_VERSIONS:
            if self.path_type != "zarr":
                raise ValueError(f"OME-Zarr version {version} can only reference zarr paths.")
            return self.path
        return {"type": self.path_type, "path": self.path}


@dataclass(frozen=True, slots=True)
class NodeRef(Generic[TransformGraphNodeT]):
    """
    Essentially a fancy tuple to act like dict-keys for selecting nodes inside transform graphs.
    This solves multiple problems:
    - nodes can be of different types (Multiscale or CoordinateSystem -- TransformGraphNode subclasses),
    - nodes can be absent entirely (_UnresolvedRef),
    - and node referencing must be possible via object identity and/or name
      (Scenes must be able to identify coordinate systems by name within child Multiscales,
      i.e. selection by (Multiscale or Scene, CoordinateSystemName) as a combined key)
    """

    name: CoordinateSystemName
    owner: TransformGraphNodeT
    """The Multiscale or CoordinateSystem that produced this, for identity."""

    def __post_init__(self):
        if not self.name:
            raise ValueError("Coordinate systems must always be referenced at least by name.")

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        return self.name == other.name and self.owner is other.owner

    def __hash__(self):
        return hash((type(self), self.name, id(self.owner)))

    def to_ome_zarr(self, version: str = "0.6.rc0", path: Optional[str] = None) -> Dict[str, Any]:
        if path and isinstance(path, str):
            return {"name": self.name, "path": FileRef.from_string(path).to_ome_zarr(version)}
        return {"name": self.name}


@dataclass(frozen=True, slots=True)
class _UnresolvedRef:
    """Degenerate placeholder reference without an owner.
    Enables round-trip serialization and graph traversal without fully resolved scene metadata.
    Also used for the awkwardness that multiscale-datasets have an actual zarr-array as input (path-only ref)."""

    name: Optional[CoordinateSystemName]
    """name is required. Optional only to acommodate one specific case inside OME-Zarr 0.6
    dataset['coordinateTransformations'][]['input'], where name must be null/omitted."""
    file: Optional[FileRef] = None

    def __post_init__(self):
        if not self.name and not self.file:
            raise ValueError("_UnresolvedRef requires at least one of: name, path")

    def to_ome_zarr(self, version: str = "0.6.rc0", path: Optional[FileRef] = None) -> Dict[str, Any]:
        d = {}
        if self.file is not None:
            d["path"] = self.file.to_ome_zarr(version)
        if self.name:
            d["name"] = self.name
        return d


ResolvedRef = NodeRef[TransformGraphNode]
AnyRef = Union[ResolvedRef, _UnresolvedRef]


class CoordinateSystem(_AxisMapping[AxisKey, AxisSemantics], TransformGraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        """(See __eq__)"""
        return id(self)

    def __eq__(self, other):
        """Identity-based equality and hash.
        Even content-identical coordinate systems are not necessarily the same system.
        For example, most JPEGs have content-identical coordinate systems (x, y, color), but there is no
        relationship between the coordinate systems of two different JPEG scans of paper."""
        return self is other  # even content-identical coordinate systems may not be the same system

    def axes(self) -> Iterable[AxisKey]:
        return self.keys()

    def as_ref(self, name: CoordinateSystemName) -> NodeRef["CoordinateSystem"]:
        """For CoordinateSystem, making a ref means giving the coordinate system a name."""
        return NodeRef(str(name), self)

    @classmethod
    def without_semantics(cls, axes: OrderedAxes) -> "CoordinateSystem":
        return cls([(a, AxisSemantics()) for a in axes])

    @classmethod
    def from_ome_zarr(cls, system_or_multiscale_dict: Mapping[str, Any]):
        axis_dicts = system_or_multiscale_dict.get("axes")
        if not axis_dicts:
            # v0.1 and v0.2 did not have any axis metadata
            return cls.without_semantics(["t", "c", "z", "y", "x"])
        if not isinstance(axis_dicts, list):
            raise ValueError(f"Invalid axis metadata. Received: {system_or_multiscale_dict}")
        if isinstance(axis_dicts[0], str):
            # v0.3 allowed specifying a subset of tczyx, e.g. ["t", "c", "y", "x"]
            return cls.without_semantics(axis_dicts)
        semantics_by_axis = []
        seen_axes = set()
        for axis_dict in system_or_multiscale_dict["axes"]:
            if not isinstance(axis_dict, MappingABC) or not axis_dict.get("name"):
                raise ValueError(f"Invalid axis metadata: Missing axis name. Received: {system_or_multiscale_dict}")
            if axis_dict["name"] in seen_axes:
                raise ValueError(f"Invalid axis metadata: Two axes named {axis_dict['name']}")
            seen_axes.add(axis_dict["name"])
            semantics_by_axis.append((axis_dict["name"], AxisSemantics.from_ome_zarr(axis_dict)))
        return cls(semantics_by_axis)

    def to_ome_zarr(
        self,
        *,
        name: CoordinateSystemName,
        version="0.6.rc0",
        axis_types: Union[None, Literal["infer"], Mapping[AxisKey, Literal["space", "time", "channel"]]] = None,
        unit: Optional[Unit] = None,
        long_names: Optional[Mapping[AxisKey, str]] = None,
        discrete: Optional[Mapping[AxisKey, bool]] = None,
    ) -> Dict[str, Any]:
        if not name and version not in PRE_TRANSFORMS_VERSIONS:
            raise ValueError(f"Cannot store coordinate system without name in OME-Zarr version {version}.")
        unit_map: Mapping[AxisKey, str] = unit or {}
        long_name_map = long_names or {}
        discrete_map = discrete or {}
        if axis_types is None:
            axis_types = {}
        elif axis_types == "infer":
            axis_types = {
                "t": "time",
                "time": "time",
                "timestep": "time",
                "timepoint": "time",
                "c": "channel",
                "ch": "channel",
                "channel": "channel",
                "channels": "channel",
                "z": "space",
                "y": "space",
                "x": "space",
            }
        elif axis_types and not any(ax in self.axes() for ax in axis_types):
            warnings.warn(f"Unexpected axis types provided: Did not find any axis of: {list(axis_types.keys())}")
        axis_dicts = []
        for ax, sem in self.items():
            adict = sem.to_ome_zarr(name=str(ax))
            if ax in unit_map and unit_map[ax]:
                adict["unit"] = unit_map[ax]
            if ax in axis_types and axis_types[ax]:
                adict["type"] = axis_types[ax]
            if ax in long_name_map and long_name_map[ax]:
                adict["longName"] = long_name_map[ax]
            if ax in discrete_map and discrete_map[ax]:
                adict["discrete"] = discrete_map[ax]
            axis_dicts.append(adict)
        d: Dict[str, Any] = {"axes": axis_dicts}
        if name:
            d["name"] = name
        return d

    def get_unit(self) -> Unit:
        return Unit([(a, sem._ome_zarr_unit or "") for a, sem in self.items()])  # noqa


@dataclass(frozen=True, slots=True)
class Transform(ABC):
    """
    Coordinate transformation with OME-Zarr convention for source/target coordinates:
    `source_coords x t = target_coords`
    This convention prioritises *technical simplicity*, not mathematical theory.
    Transforming array indices or slicings to meaningful physical coordinates is simple:
    `[0, 124, 124] x Scale(1, 0.2, 0.2) = [0, 24.8, 24.8]`.

    The responsibility split is: AxisValues handle all semantic operations and arithmetic related
    to the values and concepts they represent.
    Transforms should only be concerned with managing endpoints (source, target).
    They should ideally remain an implementation detail for serializing image processing concepts like
    Multiscale and Scene (spatial relationships between Multiscales) to OME-Zarr; not part of the package API.

    Example: PixelSize has to/from vigra converters and arithmetic interactions with e.g. Factor.
    ScaleTransform conceptually corresponds to pixel size, but knows nothing about axes and does not do arithmetic
    outside of e.g. dimensionality-validating its endpoints and composing inside a TransformSequence.
    """

    _ome_zarr_name: Optional[str] = field(default=None, kw_only=True)
    source: Optional[AnyRef] = field(default=None, kw_only=True)
    """The transform graph node (coordinate system) whose coordinates this transform acts on"""
    target: Optional[AnyRef] = field(default=None, kw_only=True)
    """The transform graph node (coordinate system) whose coordinates this transform produces"""

    @property
    @abstractmethod
    def is_invertible(self) -> bool: ...
    @abstractmethod
    def inverted(self) -> "Transform": ...
    @abstractmethod
    def composed_with(self, earlier: "Transform") -> Optional["Transform"]: ...
    @abstractmethod
    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        """Source and target dimensionality implied by this transform's payload.
        Return (source_ndim, target_ndim) if dimensionality can be inferred from payload.
        If this is not possible (e.g. identity: no payload, return None),
        convention is to assume source dimensionality == target dimensionality.
        """
        ...

    @abstractmethod
    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        """Must return the OME-Zarr object properties that are specific to the respective Transform type.
        At a minimum, this includes {'type': '<ome-zarr type name>'}.
        The common properties (input/output) are handled in the base class."""
        pass

    def _source_ndim_must_eq_target_ndim(self) -> bool:
        """For most transform types, this is True.
        Only Coordinates, ProjectAxis, ByDimension can modify ndim."""
        return True

    def __post_init__(self) -> None:
        if self._ome_zarr_name is not None and (not isinstance(self._ome_zarr_name, str) or not self._ome_zarr_name):
            raise ValueError(f"Transform name must be a non-empty string. Received: {self._ome_zarr_name!r}")
        self._validate_bound_axes()

    def to_ome_zarr(self, version: str, *, nodes_by_path: Optional[NodesByPath] = None) -> Dict[str, Any]:
        ome_zarr_transform_dict = self._get_subtype_ome_zarr_properties(version)
        if version in PRE_TRANSFORMS_VERSIONS:
            return ome_zarr_transform_dict
        if self._ome_zarr_name is not None:
            ome_zarr_transform_dict["name"] = self._ome_zarr_name
        source = self.source
        target = self.target
        if source is None or target is None:
            return ome_zarr_transform_dict
        input_dict = source.to_ome_zarr()
        output_dict = target.to_ome_zarr()
        for path, node in (nodes_by_path or {}).items():
            if node is None:
                continue
            if isinstance(source, NodeRef) and source.owner is node:
                input_dict = source.to_ome_zarr(path)
            if isinstance(target, NodeRef) and target.owner is node:
                output_dict = target.to_ome_zarr(path)
            if input_dict and output_dict:
                break
        input_dict = input_dict or source.to_ome_zarr()
        output_dict = output_dict or target.to_ome_zarr()
        ome_zarr_transform_dict.update({"input": input_dict, "output": output_dict})
        return ome_zarr_transform_dict

    @property
    def is_fully_bound(self) -> bool:
        return self.source is not None and self.target is not None

    @property
    def is_fully_unbound(self) -> bool:
        return self.source is None and self.target is None

    @property
    def is_fully_resolved(self) -> bool:
        return isinstance(self.source, NodeRef) and isinstance(self.target, NodeRef)

    @property
    def is_fully_unresolved(self) -> bool:
        return (self.source is None or isinstance(self.source, _UnresolvedRef)) and (
            self.target is None or isinstance(self.target, _UnresolvedRef)
        )

    def bound(self: _TransformSelf, source: Optional[AnyRef], target: Optional[AnyRef]) -> _TransformSelf:
        # binding required to use the Transform in a TransformGraph
        return replace(self, source=source, target=target)

    def with_resolved(self: _TransformSelf, path_nodes: Optional[NodesByPath]) -> _TransformSelf:
        """Resolve path-addressed _UnresolvedRef endpoints against the provided graph nodes."""
        if self.is_fully_resolved or not path_nodes:
            return self
        new_source = self._resolve_ref_by_path(self.source, path_nodes)
        new_target = self._resolve_ref_by_path(self.target, path_nodes)
        if new_source is self.source and new_target is self.target:
            return self
        return replace(self, source=new_source, target=new_target)

    def with_resolved_by_name(self: _TransformSelf, named_refs: Iterable[NodeRef]) -> _TransformSelf:
        """Resolve name-only _UnresolvedRef endpoints against refs from the same metadata batch.

        This is only for OME-Zarr parsing when coordinateSystems and coordinateTransformations
        were declared together in one metadata object. Coordinate-system names are not globally
        unique, so Scene resolution must use `with_resolved` with path-addressed Multiscales instead.
        """
        named_refs = tuple(named_refs)
        if self.is_fully_resolved or not named_refs:
            return self
        new_source = self._resolve_ref_by_name(self.source, named_refs)
        new_target = self._resolve_ref_by_name(self.target, named_refs)
        if new_source is self.source and new_target is self.target:
            return self
        return replace(self, source=new_source, target=new_target)

    @staticmethod
    def _resolve_ref_by_path(ref: Optional[AnyRef], path_nodes: NodesByPath) -> Optional[AnyRef]:
        if not isinstance(ref, _UnresolvedRef) or ref.file is None:
            return ref
        new_node = path_nodes.get(ref.file.path)
        if new_node is not None and ref.name is not None:
            try:
                return new_node.as_ref(ref.name)
            except NoSuchCoordinateSystemError:
                raise MismatchingMultiscaleError(path=ref.file.path, name=ref.name)
        return ref

    @staticmethod
    def _resolve_ref_by_name(ref: Optional[AnyRef], named_refs: Iterable[NodeRef]) -> Optional[AnyRef]:
        if not isinstance(ref, _UnresolvedRef) or not ref.name or ref.file is not None:
            return ref
        name_matches = [other for other in named_refs if other.name == ref.name]
        if len(name_matches) > 1:
            raise ValueError(
                f"Cannot resolve transform: Received multiple coordinate systems named '{ref.name}': "
                ", ".join([r.name for r in named_refs])
            )
        elif name_matches:
            return name_matches[0]
        return ref

    @classmethod
    def from_ome_zarr(cls, ome_dict: Mapping[str, Any]) -> "Transform":
        if not isinstance(ome_dict, MappingABC):
            raise ValueError(f"Invalid transform metadata. Expected mapping, received: {ome_dict!r}")
        t_type = ome_dict.get("type")
        if t_type == "identity":
            source, target = cls._parse_source_and_target(ome_dict)
            from clearscale._transforms._transform_types import IdentityTransform

            return IdentityTransform(_ome_zarr_name=cls._parse_name(ome_dict), source=source, target=target)
        elif t_type == "scale":
            from clearscale._transforms._transform_types import ScaleTransform

            return ScaleTransform.from_ome_zarr(ome_dict)
        elif t_type == "translation":
            from clearscale._transforms._transform_types import TranslationTransform

            return TranslationTransform.from_ome_zarr(ome_dict)
        elif t_type == "rotation":
            from clearscale._transforms._transform_types import RotationTransform

            return RotationTransform.from_ome_zarr(ome_dict)
        elif t_type == "affine":
            from clearscale._transforms._transform_types import AffineTransform

            return AffineTransform.from_ome_zarr(ome_dict)
        elif t_type == "coordinates":
            from clearscale._transforms._transform_types import CoordinatesTransform

            return CoordinatesTransform.from_ome_zarr(ome_dict)
        elif t_type == "displacements":
            from clearscale._transforms._transform_types import DisplacementsTransform

            return DisplacementsTransform.from_ome_zarr(ome_dict)
        elif t_type == "mapAxis":
            from clearscale._transforms._transform_types import MapAxisTransform

            return MapAxisTransform.from_ome_zarr(ome_dict)
        elif t_type == "projectAxis":
            from clearscale._transforms._transform_types import ProjectAxisTransform

            return ProjectAxisTransform.from_ome_zarr(ome_dict)
        elif t_type == "sequence":
            source, target = cls._parse_source_and_target(ome_dict)
            return TransformSequence(
                transforms=tuple(Transform.from_ome_zarr(td) for td in ome_dict["transformations"]),
                _ome_zarr_name=cls._parse_name(ome_dict),
                source=source,
                target=target,
            )
        elif t_type == "bijection":
            from clearscale._transforms._transform_types import BijectionTransform

            return BijectionTransform.from_ome_zarr(ome_dict)
        elif t_type == "byDimension":
            from clearscale._transforms._transform_types import ByDimensionTransform

            return ByDimensionTransform.from_ome_zarr(ome_dict)
        else:
            raise ValueError(f"Unknown transform type: {t_type!r}")

    @staticmethod
    def _parse_source_and_target(ome_dict: Mapping[str, Any]):
        endpoints: Dict[str, Optional[AnyRef]] = {"input": None, "output": None}
        for side in endpoints.keys():
            ref = ome_dict.get(side, {})
            if isinstance(ref, str) and ref:
                endpoints[side] = _UnresolvedRef(name=ref)
                continue
            if not isinstance(ref, dict):
                raise ValueError(f"Invalid transform endpoint metadata. Received: {ome_dict!r}")
            path = ref.get("path")
            file = None
            name = ref.get("name")
            if isinstance(path, MappingABC):
                file = FileRef.from_dict(path)
            elif isinstance(path, str):
                file = FileRef.from_string(path)
            name = name if isinstance(name, str) else None
            if file or name:
                endpoints[side] = _UnresolvedRef(file=file, name=name)
        if bool(endpoints["input"]) != bool(endpoints["output"]):
            raise ValueError(f"Invalid transform (in/out must either both be undefined or both defined): {ome_dict!r}")
        source = endpoints["input"]
        target = endpoints["output"]
        return source, target

    @staticmethod
    def _parse_name(ome_dict: Mapping[str, Any]) -> Optional[str]:
        name = ome_dict.get("name")
        if not name:
            return None
        if not isinstance(name, str):
            raise ValueError(f"Invalid metadata: Name must be string. Received: {name!r}")
        return name

    def _validate_bound_axes(self) -> None:
        source_axes = tuple(self.source.owner.axes()) if isinstance(self.source, NodeRef) else None
        target_axes = tuple(self.target.owner.axes()) if isinstance(self.target, NodeRef) else None

        ndim = self._ndim_by_payload()
        if ndim is not None:
            source_ndim, target_ndim = ndim
            if source_axes is not None and len(source_axes) != source_ndim:
                raise ValueError(
                    f"{self.__class__.__name__} expects {source_ndim} source axes, but its source "
                    f"coordinate system has {len(source_axes)}: {list(source_axes)}"
                )
            if target_axes is not None and len(target_axes) != target_ndim:
                raise ValueError(
                    f"{self.__class__.__name__} expects {target_ndim} target axes, but its target "
                    f"coordinate system has {len(target_axes)}: {list(target_axes)}"
                )

        if (
            ndim is None
            and source_axes is not None
            and target_axes is not None
            and self._source_ndim_must_eq_target_ndim()
            and len(source_axes) != len(target_axes)
        ):
            raise ValueError(
                f"{self.__class__.__name__} source and target must be same dimensionality: "
                f"source {list(source_axes)} vs target {list(target_axes)}"
            )

    def _endpoints_can_chain_after(self, earlier: "Transform") -> bool:
        if earlier.target is None or self.source is None or earlier.target == self.source:
            return True
        self_ndim = self._ndim_by_payload()
        earlier_ndim = earlier._ndim_by_payload()
        if self_ndim is None or earlier_ndim is None:
            return True
        self_source_ndim = self_ndim[0]
        earlier_target_ndim = earlier_ndim[1]
        return self_source_ndim == earlier_target_ndim

    def _composed_source(self, earlier: "Transform") -> Optional[AnyRef]:
        return earlier.source if earlier.source is not None else self.source

    def _composed_target(self, earlier: "Transform") -> Optional[AnyRef]:
        return self.target if self.target is not None else earlier.target


@dataclass(frozen=True, slots=True)
class TransformSequence(Transform):
    transforms: Tuple[Transform, ...] = field(default=())

    @property
    def is_invertible(self) -> bool:
        return all(t.is_invertible for t in self.transforms)

    def inverted(self) -> "TransformSequence":
        if not self.is_invertible:
            raise ValueError("TransformSequence is not invertible: contains non-invertible transform(s).")
        return TransformSequence(tuple(reversed([t.inverted() for t in self.transforms])))

    def composed_with(self, earlier: "Transform") -> Optional["Transform"]:
        if not self._endpoints_can_chain_after(earlier):
            return None
        source = self._composed_source(earlier)
        target = self._composed_target(earlier)
        if isinstance(earlier, TransformSequence):
            return replace(
                earlier,
                transforms=tuple(earlier.transforms + self.transforms),
                source=source,
                target=target,
            )
        return replace(self, transforms=(earlier,) + self.transforms, source=source, target=target)

    def _get_subtype_ome_zarr_properties(self, version: str) -> Dict[str, Any]:
        return {
            "type": "sequence",
            "transformations": [t.to_ome_zarr(version) for t in self.transforms],
        }

    def to_ome_zarr(self, version: str, *, nodes_by_path: Optional[NodesByPath] = None) -> Dict[str, Any]:
        if version in PRE_TRANSFORMS_VERSIONS:
            raise ValueError("TransformSequence cannot be serialized to OME-Zarr older than 0.6.rc0")
        return super(TransformSequence, self).to_ome_zarr(version, nodes_by_path=nodes_by_path)

    def __post_init__(self):
        if not self.transforms:
            raise ValueError("Cannot make empty TransformSequence.")
        if any(not isinstance(t, Transform) for t in self.transforms):
            raise ValueError("All children must be Transform instances.")
        for i, (a, b) in enumerate(zip(self.transforms, self.transforms[1:])):
            if a.target is not None and b.source is not None and a.target != b.source:
                raise ValueError(f"Transform chain broken at {i}->{i+1}: {a.target!r} != {b.source!r}")
        # Infer source/target from children if not explicitly provided
        inferred_source = self.transforms[0].source
        inferred_target = self.transforms[-1].target
        if self.source is None and inferred_source is not None:
            object.__setattr__(self, "source", inferred_source)
        if self.target is None and inferred_target is not None:
            object.__setattr__(self, "target", inferred_target)
        self._validate_child_ndim_chain()
        Transform.__post_init__(self)

    def _ndim_by_payload(self) -> Optional[Tuple[int, int]]:
        first_ndim = self.transforms[0]._ndim_by_payload()
        last_ndim = self.transforms[-1]._ndim_by_payload()
        if first_ndim is None or last_ndim is None:
            return None
        return first_ndim[0], last_ndim[1]

    def _source_ndim_must_eq_target_ndim(self) -> bool:
        return all(t._source_ndim_must_eq_target_ndim() for t in self.transforms)

    def __hash__(self):
        return hash(self.transforms)

    def __eq__(self, other):
        return isinstance(other, TransformSequence) and self.transforms == other.transforms

    def __iter__(self):
        return iter(self.transforms)

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, item):
        return self.transforms[item]

    def bound(
        self: _TransformSequenceSelf, source: Optional[AnyRef], target: Optional[AnyRef]
    ) -> _TransformSequenceSelf:
        # Override from base: Sequence needs to update endpoint transforms
        new_transforms: Tuple[Transform, ...]
        if len(self.transforms) == 1:
            first = self.transforms[0].bound(source=source, target=target)
            new_transforms = (first,)
        else:
            first = self.transforms[0].bound(source=source, target=self.transforms[0].target)
            last = self.transforms[-1].bound(source=self.transforms[-1].source, target=target)
            new_transforms = (first,) + self.transforms[1:-1] + (last,)
        return replace(self, source=source, target=target, transforms=new_transforms)

    def collapsed(self, *, raise_uncollapsed: bool = False) -> "Transform | TransformSequence":
        result: List[Transform] = [self.transforms[0]]

        for current in self.transforms[1:]:
            previous = result[-1]
            merged = current.composed_with(previous)
            if merged is not None:
                result[-1] = merged
            elif raise_uncollapsed:
                raise ValueError(f"Cannot collapse {type(previous).__name__} followed by {type(current).__name__}")
            else:
                result.append(current)

        if len(result) == 1:
            return result[0]
        return replace(self, transforms=tuple(result))

    def _validate_child_ndim_chain(self) -> None:
        earlier_target_ndim: Optional[int] = None
        earlier: Optional[Transform] = None

        for transform in self.transforms:
            ndim = transform._ndim_by_payload()
            if ndim is None:
                if not transform._source_ndim_must_eq_target_ndim():
                    earlier_target_ndim = None
                    earlier = None
                continue

            source_ndim, target_ndim = ndim
            if earlier_target_ndim is not None and source_ndim != earlier_target_ndim:
                assert earlier is not None
                raise ValueError(
                    "Transform chain dimensionality mismatches: "
                    f"{type(earlier).__name__}(target_ndim={earlier_target_ndim!r}) != "
                    f"{type(transform).__name__}(source_ndim={source_ndim!r})"
                )

            earlier_target_ndim = target_ndim
            earlier = transform


def _ordered_unique_refs(refs: Iterable[_RefT]) -> Tuple[_RefT, ...]:
    """Deduplicate graph refs while preserving order."""
    return tuple(dict.fromkeys(refs))


def _is_owner_coordinate_system(ref: AnyRef) -> TypeGuard[NodeRef["CoordinateSystem"]]:
    """Makes pyright happy with TransformGraph.connected_system_refs"""
    return isinstance(ref, NodeRef) and isinstance(ref.owner, CoordinateSystem)


@dataclass(frozen=True, init=False)
class TransformGraph:
    """
    Transform graphs consist of
    - Transforms as edges, and
    - Multiscales and CoordinateSystems as nodes.
    The TransformGraph is defined primarily via Transforms.
    Nodes are managed by the respective Transforms.

    In OME-Zarr, the TransformGraph corresponds to two metadata keys:
    {
      "coordinateSystems": [...],
      "coordinateTransformations": [...],
    }
    As present on multiscale and scene metadata.
    """

    transforms: Tuple[Transform, ...]  # This could be ~15k entries in prod
    """Transforms define the graph. Their `.source` and `.target` are the graph nodes."""
    system_refs: Tuple[NodeRef[CoordinateSystem], ...] = ()
    """Keeps references to coordinate systems on `transforms` whose order of declaration matters.
    This also enables the plain Multiscale with no transforms (only a single coordinate system).
    Must otherwise be a subset of the .source/.target nodes on `transforms`."""

    def __bool__(self):
        return bool(self.transforms) or bool(self.system_refs)

    @functools.cached_property
    def all_system_refs(self) -> Tuple[NodeRef[CoordinateSystem], ...]:
        """All CoordinateSystem instances this graph knows"""
        return _ordered_unique_refs(chain(self.system_refs, self.connected_system_refs))

    @functools.cached_property
    def connected_system_refs(self) -> Tuple[NodeRef[CoordinateSystem], ...]:
        """Only CoordinateSystem instances that can be reached by graph traversal"""
        return tuple(ref for ref in self.node_refs if _is_owner_coordinate_system(ref))

    @functools.cached_property
    def node_refs(self) -> Tuple[AnyRef, ...]:
        """All nodes (CoordinateSystems and Multiscales) that can be reached by graph traversal"""
        return _ordered_unique_refs(ref for t in self.transforms for ref in (t.source, t.target) if ref is not None)

    @functools.cached_property
    def unresolved_transforms(self) -> Tuple[Transform, ...]:
        return tuple(
            t for t in self.transforms if isinstance(t.source, _UnresolvedRef) or isinstance(t.target, _UnresolvedRef)
        )

    def __init__(
        self,
        transforms: Iterable[Transform],
        system_refs: Iterable[NodeRef[CoordinateSystem]] = (),
    ):
        transforms = tuple(transforms)
        bad_types = [t for t in transforms if not isinstance(t, Transform)]
        if bad_types:
            raise TypeError(f"Graph edges must be Transform instances: {bad_types}")
        bad = [t for t in transforms if not t.is_fully_bound]
        if bad:
            raise ValueError(f"Graph transforms must have bound endpoints: {bad}")
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "system_refs", _ordered_unique_refs(system_refs))

    @classmethod
    def single_isolated_system(cls, sys_ref: NodeRef[CoordinateSystem]):
        return cls([], system_refs=(sys_ref,))

    @classmethod
    def from_ome_zarr(cls, transform_dicts: Optional[List[Dict]], system_dicts: Optional[List[Dict]]):
        transform_dicts = transform_dicts or []
        system_dicts = system_dicts or []
        if not isinstance(transform_dicts, list) or not isinstance(system_dicts, list):
            raise ValueError(
                "Invalid graph metadata: Expected lists. "
                f"Received coordinate systems: {system_dicts!r} and transforms: {transform_dicts!r}"
            )
        named_systems: List[NodeRef[CoordinateSystem]] = []
        seen_names = set()
        for system_dict in system_dicts:
            if not isinstance(system_dict, MappingABC):
                raise ValueError(
                    f"Invalid graph metadata: Expected coordinate system dictionary. Received: {system_dict!r}"
                )
            system = CoordinateSystem.from_ome_zarr(system_dict)
            name: Optional[CoordinateSystemName] = system_dict.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Invalid metadata: Coordinate system has no name. Received: {system_dict}")
            if name in seen_names:
                raise ValueError(
                    f'Invalid metadata: Multiple coordinate systems named "{name}". Received: {system_dict}'
                )
            named_systems.append(system.as_ref(name))
            seen_names.add(name)
        transforms: List[Transform] = []
        for transform_dict in transform_dicts:
            t: Transform = Transform.from_ome_zarr(transform_dict).with_resolved_by_name(named_systems)
            if not t.is_fully_bound:
                raise ValueError(
                    f'Transform input and output must have "path", "name" or both. Received: {transform_dict}'
                )
            transforms.append(t)
        graph = TransformGraph(transforms, system_refs=tuple(named_systems))
        return graph

    def to_ome_zarr(self, version="0.6.rc0", nodes_by_path: Optional[NodesByPath] = None) -> Dict[str, Any]:
        """
        Returns dict like {
            "coordinateSystems": List[Dict] (maybe)
            "coordinateTransformations: List[Dict] (required)
        }
        """
        if version != "0.6.rc0":
            warnings.warn(
                f"Unsupported OME-Zarr version {version!r}. "
                f"This method only targets 0.6.rc0 as of 07/2026. Metadata may be invalid."
            )
        systems = [
            ref.owner.to_ome_zarr(name=ref.name, version=version)
            for ref in self.all_system_refs
            if isinstance(ref.owner, CoordinateSystem)
        ]
        transforms = [t.to_ome_zarr(version, nodes_by_path=nodes_by_path) for t in self.transforms]
        d: Dict[str, Any] = {}
        if systems:
            d["coordinateSystems"] = systems
        if transforms:
            d["coordinateTransformations"] = transforms
        return d

    def path_between(
        self,
        source: AnyRef,
        target: AnyRef,
        allow_inverse=True,
        validate_rfc5_connectedness=False,
    ) -> Optional[List[Transform]]:
        if source == target:
            return []

        # Adjacency - could be worth caching for performance
        graph = defaultdict(list)
        for t in self.transforms:
            assert t.source is not None and t.target is not None, "graphs must never contain unbound transforms."
            graph[t.source].append((t.target, t, False))  # (dest, transform, is_inverse)
            if validate_rfc5_connectedness or (allow_inverse and t.is_invertible):
                graph[t.target].append((t.source, t, True))

        # BFS tracking (predecessor, transform) instead of copying paths
        visited: Dict[AnyRef, Optional[Tuple[AnyRef, Transform]]] = {source: None}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node == target:
                break
            for neighbor, transform, is_inverse in graph[node]:
                if neighbor not in visited:
                    visited[neighbor] = (node, transform.inverted() if is_inverse else transform)
                    queue.append(neighbor)

        if target not in visited:
            return None

        # Reconstruct path
        path = []
        node = target
        step = visited[node]
        while step is not None:
            predecessor, transform = step
            path.append(transform)
            node = predecessor
            step = visited[node]
        path.reverse()
        return path
