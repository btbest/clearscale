from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from typing import Any, Dict, Literal, List, Mapping, Optional, Protocol, Tuple

from clearscale._multiscale import Multiscale
from clearscale._scene import Scene
from clearscale._transforms import FileRef
from clearscale._services.ome_zarr import ShapeSourceMap


@dataclass(frozen=True, slots=True)
class ChildRef:
    child_type: Literal["label", "well", "multiscale"]
    file: FileRef

    def __post_init__(self):
        assert self.child_type in ("label", "well", "multiscale")

    @classmethod
    def from_string(cls, path: str, child_type: Literal["label", "well", "multiscale"]):
        return cls(file=FileRef.from_string(path), child_type=child_type)


def _child_paths_and_version_from_attrs(attrs: Mapping[str, Any]) -> Tuple[Tuple[ChildRef, ...], Optional[str]]:
    children: List[ChildRef] = []
    version: Optional[str] = None

    labels = attrs.get("labels")
    if isinstance(labels, ABCMapping):
        if isinstance(labels.get("version"), str) and labels.get("version"):
            version = labels["version"]

        labels_list = labels.get("labels")
        if isinstance(labels_list, list):
            children.extend(
                ChildRef.from_string(path, "label") for path in labels_list if isinstance(path, str) and path
            )

    well = attrs.get("well")
    if isinstance(well, ABCMapping):
        if version is None and isinstance(well.get("version"), str) and well.get("version"):
            version = well["version"]

        images = well.get("images")
        if isinstance(images, list):
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
        if isinstance(wells, list):
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


@dataclass(frozen=True, slots=True, init=False)
class OmeZarrGroup:
    version: Optional[str] = None
    multiscales: Tuple[Multiscale, ...] = ()
    scenes: Tuple[Scene, ...] = ()
    children: Tuple[ChildRef, ...] = ()
    _invalid_objects: Tuple[Dict[str, Any], ...] = ()

    def __init__(self, group: ZarrGroup, *, shape_source: Optional[ShapeSourceMap] = None):
        """
        Parse an OME-Zarr group and construct the clearscale objects directly represented by its metadata.
        If the group contains plate, well or labels metadata, collect the contained paths to multiscales.
        """
        attrs = group.attrs
        use_shape_source = shape_source or group

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
                    multiscales.append(Multiscale.from_ome_zarr(ms_json, shape_source=use_shape_source))
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
        if scene_json and isinstance(scene_json, ABCMapping):
            try:
                scenes.append(Scene.from_ome_zarr(scene_json))
            except ValueError:
                invalid.append(scene_json)
            if not version and isinstance(scene_json.get("version"), str) and scene_json.get("version"):
                scene_version = scene_json.get("version")

        children, child_version = _child_paths_and_version_from_attrs(ome_attrs)

        version = version or multiscale_version or scene_version or child_version

        object.__setattr__(self, "version", version)
        object.__setattr__(self, "multiscales", tuple(multiscales))
        object.__setattr__(self, "scenes", tuple(scenes))
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "_invalid_objects", tuple(invalid))
