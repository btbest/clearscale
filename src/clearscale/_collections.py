from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Dict, Literal, List, Mapping, Optional, Protocol, Tuple, Union
import warnings

from clearscale._axis_values import AxisKey
from clearscale._multiscale import Multiscale
from clearscale._scene import Scene
from clearscale._transforms import FileRef, PRE_COLLECTIONS_VERSIONS
from clearscale._services.ome_zarr import SUPPORTED_OME_ZARR_VERSIONS_WRITE, ShapeSource, ShapeSourceMap


class GroupKind(str, Enum):
    MULTISCALE = "multiscale"
    """Exactly one valid Multiscale"""
    SCENE = "scene"
    """Exactly one valid Scene and zero or more children. Children are multiscales"""
    PLATE = "plate"
    """One or more children. Children are wells"""
    WELL = "well"
    """One or more children. Children are multiscales"""
    LABELS = "labels"
    """One or more children. Children are (label) multiscales"""
    BF2RAW = "bioformats2raw"
    """Indicates the "bioformats2raw.layout" marker is present. You should try `OmeZarrGroup.from_group(bf2raw_group["OME"])`, 
    which *might* exist, and if it exists, it *might* resolve to GroupKind.BF2RAW_OME and tell you which children exist.
    If it does exist, it will report the child paths relative to itself, i.e.  "../ms_0", "../ms_1", etc.
    You will need to strip the leading "../" to use with the parent `bf2raw_group` for example like 
    `bf2raw_group[ bfw2raw_ome_group.children[0].file.path[3:] ]`.
    If it does not exist, or it does not specify children, this means there are an unknown number of multiscale children 
    at sequential number paths like `bf2raw_group["0"]`, `bf2raw_group["1"]`, `bf2raw_group["2"]`, etc."""
    BF2RAW_OME = "bioformats2raw-OME"
    """The "OME" sub-group under a BF2RAW parent group. One or more children. Children are multiscales.
    Child paths are relative to this group, i.e.  "../ms_0", "../ms_1", etc.
    You will need to strip the leading "../" to get paths relative to the BF2RAW-kind parent group."""
    COLLECTION = "collection"
    """Generic OME-Zarr container. Contains some combination of the other kinds (some mix of multiscales, scenes and/or children)."""


@dataclass(frozen=True, slots=True)
class ChildRef:
    child_type: Literal["label", "well", "multiscale"]
    file: FileRef

    def __post_init__(self):
        assert self.child_type in ("label", "well", "multiscale")

    @classmethod
    def from_string(cls, path: str, child_type: Literal["label", "well", "multiscale"]):
        return cls(file=FileRef.from_string(path), child_type=child_type)


def _children_from_attrs(attrs: Mapping[str, Any]) -> Tuple[Tuple[ChildRef, ...], Optional[str]]:
    children: List[ChildRef] = []
    version: Optional[str] = None

    labels = attrs.get("labels")
    if isinstance(labels, list) and any(labels):
        children.extend(ChildRef.from_string(path, "label") for path in labels if isinstance(path, str) and path)

    series = attrs.get("series")  # might be present in the "OME" subgroup of a bf2raw parent group
    if isinstance(series, list) and any(series):
        children.extend(
            ChildRef.from_string(
                f"../{path}",  # the strings in the "OME" subgroup's "series" attribute refer relative to the bfw2raw parent group
                "multiscale",
            )
            for path in series
            if isinstance(path, str) and path
        )

    well = attrs.get("well")
    if isinstance(well, ABCMapping):
        if version is None and isinstance(well.get("version"), str) and well.get("version"):
            version = well["version"]

        images = well.get("images")
        if isinstance(images, list) and any(images):
            children.extend(
                ChildRef.from_string(image["path"], "multiscale")
                for image in images
                if isinstance(image, ABCMapping) and isinstance(image.get("path"), str) and image.get("path")
            )

    plate = attrs.get("plate")
    if isinstance(plate, ABCMapping):
        if version is None and isinstance(plate.get("version"), str) and plate.get("version"):
            version = plate["version"]

        wells = plate.get("wells")
        if isinstance(wells, list) and any(wells):
            children.extend(
                ChildRef.from_string(well["path"], "well")
                for well in wells
                if isinstance(well, ABCMapping) and isinstance(well.get("path"), str) and well.get("path")
            )

    return tuple(children), version


class ZarrGroup(ShapeSourceMap, Protocol):
    """Matches e.g. zarr.Group (zarr-python) or z5py.Group."""

    @property
    def attrs(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class OmeZarrGroup:
    kind: Optional[GroupKind] = None
    """Indicator of this group's contents. None means empty."""
    version: Optional[str] = None
    multiscales: Tuple[Multiscale, ...] = ()
    scenes: Tuple[Scene, ...] = ()
    children: Tuple[ChildRef, ...] = ()
    """Contains *all* references to other OME-Zarr objects mentioned in this group's metadata.
    This includes references to wells, label-multiscales, and multiscales contained in wells or scenes."""
    maybe_subgroups: Tuple[str, ...] = ()
    """Contains potentially present subgroups implied by the OME-Zarr standard. "OME" for BF2RAW or 
    "labels" for groups with .multiscales."""
    _invalid_objects: Tuple[Dict[str, Any], ...] = ()

    def __post_init__(self):
        assert self.version != "", "Must not instantiate with empty version string"

        detected_kind = None
        child_types = {child.child_type for child in self.children}
        if len(self.multiscales) == 1 and not self.scenes and not self.children:
            detected_kind = GroupKind.MULTISCALE
        elif len(self.scenes) == 1 and not self.multiscales and (not self.children or child_types == {"multiscale"}):
            detected_kind = GroupKind.SCENE
        elif not self.multiscales and not self.scenes and self.children:
            detected_kind = GroupKind.COLLECTION
            if child_types == {"well"}:
                detected_kind = GroupKind.PLATE
            if child_types == {"multiscale"}:
                detected_kind = GroupKind.WELL
            if child_types == {"label"}:
                detected_kind = GroupKind.LABELS
        elif self.multiscales or self.scenes or self.children:
            detected_kind = GroupKind.COLLECTION

        if self.kind is GroupKind.BF2RAW or self.kind is GroupKind.BF2RAW_OME:
            # BF2RAW kinds are special markers that can't be detected from contents. Preset in the constructor call instead.
            detected_kind = self.kind

        if self.kind is not None and self.kind != detected_kind:
            raise ValueError(f"Group kind {self.kind!r} does not match its contents; expected {detected_kind!r}.")

        object.__setattr__(self, "kind", detected_kind)

    @classmethod
    def from_single(cls, obj: Union[Multiscale, Scene]) -> "OmeZarrGroup":
        if isinstance(obj, Multiscale):
            return cls(multiscales=(obj,))
        if isinstance(obj, Scene):
            return cls(scenes=(obj,))
        raise TypeError(f"Must be called with a Multiscale or Scene, not {obj!r}")

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any], *, shape_source: ShapeSource):
        """
        Parse the provided metadata to obtain any Multiscale and Scene definitions it contains.
        If it contains plate, well or labels metadata, collect the contained paths to multiscales.
        """
        ome_attrs = attrs.get("ome") or attrs

        version = ome_attrs.get("version")
        if not isinstance(version, str) or not version:
            version = None
        multiscale_version = scene_version = None

        invalid = []

        multiscales = []
        multiscales_json = ome_attrs.get("multiscales")
        if isinstance(multiscales_json, list):
            multiscales = []
            for ms_json in multiscales_json:
                try:
                    multiscales.append(Multiscale.from_ome_zarr(ms_json, shape_source=shape_source))
                except ValueError:
                    invalid.append(ms_json)
            if not version and multiscales:
                for ms_json in multiscales_json:
                    if (
                        isinstance(ms_json, ABCMapping)
                        and isinstance(ms_json.get("version"), str)
                        and ms_json.get("version")
                    ):
                        multiscale_version = ms_json["version"]
                        break

        scene_json = ome_attrs.get("scene")
        scenes = []
        scene_children = []
        if scene_json and isinstance(scene_json, ABCMapping):
            try:
                scene = Scene.from_ome_zarr(scene_json)
                scenes.append(scene)
                scene_children.extend(ChildRef.from_string(path, "multiscale") for path in scene.unresolved_paths)
            except ValueError:
                invalid.append(scene_json)
            if not version and isinstance(scene_json.get("version"), str) and scene_json.get("version"):
                scene_version = scene_json.get("version")

        children, child_version = _children_from_attrs(ome_attrs)

        # Kind is generally detected from content in post_init; only the bf2raw kinds can't be recognised this way and need to be preset
        kind = None
        maybe_subgroups: Tuple[str, ...] = ()
        if "bioformats2raw.layout" in ome_attrs and not children:
            # The bf2raw marker can be combined with plate metadata. The plate is supposed to take precedence if present.
            # If there is plate metadata, there are children, hence `and not children`.
            kind = GroupKind.BF2RAW
            maybe_subgroups = ("OME",)
        elif "series" in ome_attrs and children and all(child.file.path.startswith("../") for child in children):
            kind = GroupKind.BF2RAW_OME
        if multiscales:
            subgroups_dup = [str(PurePosixPath(next(iter(ms))).parent / "labels") for ms in multiscales]
            maybe_subgroups = tuple(dict.fromkeys(subgroups_dup))

        version = version or multiscale_version or scene_version or child_version

        return cls(
            kind=kind,
            version=version,
            multiscales=tuple(multiscales),
            scenes=tuple(scenes),
            children=children + tuple(scene_children),
            maybe_subgroups=maybe_subgroups,
            _invalid_objects=tuple(invalid),
        )

    @classmethod
    def from_group(cls, group: ZarrGroup, *, shape_source: Optional[ShapeSource] = None):
        """
        Parse a zarr group's metadata to obtain any Multiscale and Scene definitions it contains.
        If the group contains plate, well or labels metadata, collect the contained paths to multiscales.
        Use the provided group as the Multiscale shape_source by default.
        This can be expensive/slow when the group is on a remote server, so consider other shape_source
        options if this is an expected common case.
        """
        return cls.from_attrs(group.attrs, shape_source=shape_source or group)

    def to_attrs(
        self,
        version: Literal["0.4", "0.5", "0.6.rc0"],
        *,
        axis_types: Union[None, Literal["infer"], Mapping[AxisKey, Literal["space", "time", "channel"]]] = None,
    ) -> Dict[str, Any]:
        if version not in SUPPORTED_OME_ZARR_VERSIONS_WRITE:
            raise ValueError(f"Cannot write OME-Zarr with {version=}")
        if self.kind is None:
            return {}
        self._validate_for_version(version)
        ome: Dict[str, Any] = {}
        if self.kind is GroupKind.MULTISCALE:
            ome["multiscales"] = [self.multiscales[0].to_ome_zarr(version=version, axis_types=axis_types)]
        elif self.kind is GroupKind.SCENE:
            ome["scene"] = self.scenes[0].to_ome_zarr(version=version)
        elif self.kind is GroupKind.LABELS:
            ome["labels"] = [child.file.path for child in self.children]
        elif self.kind is GroupKind.WELL:
            # missing: objects inside "images" MUST contain "acquisition" key if more than one specified in plate
            raise NotImplementedError(
                "Writing plate and well metadata is not supported yet. Please open an issue on GitHub if you need this."
            )
        elif self.kind is GroupKind.PLATE:
            # Writing back the paths isn't sufficient.
            # "The plate object MUST contain a columns key" (... and a rows key)
            # And both columns and rows:
            # "Each [column/row] in the physical plate MUST be defined,
            # even if no wells in the [column/row] are defined."
            # "Each well object [under 'wells'] MUST contain both a rowIndex key [...] and a columnIndex key"
            raise NotImplementedError(
                "Writing plate and well metadata is not supported yet. Please open an issue on GitHub if you need this."
            )
        elif self.kind is GroupKind.COLLECTION:
            if self.multiscales and not self.scenes and not self.children:
                ome["multiscales"] = [ms.to_ome_zarr(version=version, axis_types=axis_types) for ms in self.multiscales]
            else:
                raise NotImplementedError("No version of OME-Zarr currently supports collections.")
        if version == "0.4":
            return ome
        ome["version"] = version
        return {"ome": ome}

    def _validate_for_version(self, version: Literal["0.4", "0.5", "0.6.rc0"]):
        assert self.kind is not None, "should skip if empty"
        not_implemented_kinds = (GroupKind.BF2RAW, GroupKind.BF2RAW_OME)
        if self.kind in not_implemented_kinds:
            raise NotImplementedError(
                f"Writing {self.kind.value} groups is not implemented yet. Please open an issue on GitHub."
            )
        base_kinds = (GroupKind.MULTISCALE, GroupKind.PLATE, GroupKind.WELL, GroupKind.LABELS)
        supported_kinds = {
            "0.4": base_kinds,
            "0.5": base_kinds,
            "0.6.rc0": base_kinds + (GroupKind.SCENE,),
        }
        is_multi_multiscale = (
            self.kind is GroupKind.COLLECTION and len(self.multiscales) > 1 and not self.scenes and not self.children
        )
        if self.kind not in supported_kinds[version] and not is_multi_multiscale:
            raise ValueError(
                f"Cannot write this group in OME-Zarr version {version}: {self.kind.value} groups are not supported."
            )
        elif is_multi_multiscale and version in PRE_COLLECTIONS_VERSIONS:
            warnings.warn(
                "This group consists of multiple multiscales. While this is technically valid in OME-Zarr version "
                f"{version}, support for handling multiple multiscales within a single Zarr group is sparse across "
                "the OME-Zarr tool ecosystem. Please consider storing each Multiscale in a separate OmeZarrGroup.",
                UserWarning,
            )
        if version == "0.6.rc0":
            warnings.warn(
                f"Version {version} is not a stable version. Written metadata may be ignored or invalid in the future."
            )
