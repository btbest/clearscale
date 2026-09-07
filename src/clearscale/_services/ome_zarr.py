"""Private helpers that support Multiscale and Scene to/from_ome_zarr methods"""

import copy
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Union, Dict, Literal, List, Any, Optional, Tuple, Protocol, Iterable, TYPE_CHECKING

from clearscale._axis_values import ShapeLike, Translation, PixelSize, AxisKey, Factor
from clearscale._services.matrices import is_identity_scale, DETERMINANT_SINGULARITY_TOLERANCE
from clearscale._transforms import (
    TransformSequence,
    IdentityTransform,
    ScaleTransform,
    TranslationTransform,
    TransformGraph,
    FileRef,
    NodeRef,
    CoordinateSystem,
    _UnresolvedRef,
    PRE_TRANSFORMS_VERSIONS,
    Transform,
)

if TYPE_CHECKING:
    from clearscale._multiscale import Multiscale

SUPPORTED_OME_ZARR_VERSIONS_READ = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6.rc0")
SUPPORTED_OME_ZARR_VERSIONS_WRITE = ("0.4", "0.5", "0.6.rc0")

####
# Reading
####


OME_ZARR_TRANSFORM = Mapping[str, Any]
"""
Single transform.
"type": str (Possible values vary by version. Inside OME_ZARR_DATASET only "scale" and "translation")
"scale": List[float] (if type=="scale")
"translation": List[float] (if type=="translation")
"input": 0.6+ only, inside OME_ZARR_DATASET, dict with key "path" (str)
"output": 0.6+ only, inside OME_ZARR_DATASET, dict with key "name" (str)
"""
OME_ZARR_TRANSFORMS = Union[OME_ZARR_TRANSFORM, List[OME_ZARR_TRANSFORM]]
OME_ZARR_DATASET = Mapping[str, Any]
"""
Single dataset (scale-level).
"path": str
"coordinateTransformations": List[OME_ZARR_TRANSFORM]
"""
OME_ZARR_MULTISCALE = Mapping[str, Any]
"""
Single entry of ["multiscales"] list.
"datasets": List[OME_ZARR_DATASET]
"axes": 0.3: List[str] (axis key strings), 0.4 and 0.5: List[Dict] (axis key under "name")
"coordinateSystems": 0.6+ only, List[Dict] (containing "axes" List[Dict] with "name" keys)
"coordinateTransformations": List[OME_ZARR_TRANSFORM]; 0.4 and 0.5: as inside "datasets"; 0.6: like in Scene
"""
GetShapeFunction = Callable[[str], Tuple[int, ...]]
"""
path: Relative path to a zarr array.
Returns: The `.shape` of the array at that path.
"""


@dataclass(slots=True)
class OmeMultiscaleProperties:
    """Additional properties of multiscale metadata that OME-Zarr loosely specifies.
    The standard describes what 'should' be written in them, but nothing about how the values
    should be interpreted. In practice, they are arbitrary fields for display purposes only.
    Since their semantics are not defined, clearscale does not use them for internal logic."""

    type: str = ""
    """The 'type' of downscaling method used to create the pyramid"""
    name: str = ""
    """Display name"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Scaling method description. This *should* specify keys 'method', 'version', 'args', 
    'kwargs', and 'description' (refer to OME-Zarr specification)"""

    @classmethod
    def from_ome_zarr(cls, multiscale_dict: OME_ZARR_MULTISCALE) -> "OmeMultiscaleProperties":
        typ = multiscale_dict.get("type", "")
        if not isinstance(typ, str):
            typ = ""
        name = multiscale_dict.get("name", "")
        if not isinstance(name, str):
            name = ""
        metadata = multiscale_dict.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return cls(type=typ, name=name, metadata=metadata)

    def __bool__(self):
        return bool(self.type or self.name or self.metadata)

    def to_ome_zarr(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.type:
            d["type"] = self.type
        if self.name:
            d["name"] = self.name
        if self.metadata and isinstance(self.metadata, dict):
            d["metadata"] = copy.deepcopy(self.metadata)
        elif self.metadata:
            raise ValueError(f"Must not replace Multiscale.ome.metadata. Found: {self.metadata}")
        return d


class HasShape(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...


ShapeValue = Union[Sequence[int], ShapeLike, HasShape]


class ShapeSourceMap(Protocol):
    def __getitem__(self, path: str, /) -> ShapeValue: ...


ShapeSource = Union[Literal["singletons"], Callable[[str], ShapeValue], ShapeSourceMap]
"""
Lets clearscale know how to obtain a zarr's array shape in this Python environment.
Options:
- "singletons": Skip obtaining shapes and use placeholder all-singleton shapes (like `Shape(x=1, y=1, z=1)`)
    This sets `Multiscale.has_shapes = False` as a convenience indicator.
- Callable: A function that takes a relative path that *should* point to an array, and retrieves its shape tuple
    Example: zarr.open_array
- Map: A dict-like that can be indexed to retrieve shapes like `{ <relative path> : <shape tuple or array> }`
    Example: zarr.Group or fsspec.FSMap
"""


def make_all_singleton_shapes(ndim_or_spec: Union[int, OME_ZARR_MULTISCALE]) -> GetShapeFunction:
    """
    Construct OME-Zarr Multiscale without accessing data arrays, avoiding potentially dozens of requests.
    `Multiscale.from_ome_zarr(ome_ms_dict, shape_source=make_all_singleton_shapes(3))`
    `Multiscale.from_ome_zarr(ome_ms_dict, shape_source=make_all_singleton_shapes(ome_ms_dict))`
    """
    if isinstance(ndim_or_spec, int):
        return lambda _: (1,) * ndim_or_spec
    else:
        ndim = len(ndim_or_spec.get("axes", [])) or 5  # OME-Zarr 0.1 and 0.2 without axes
        return lambda _: (1,) * ndim


def normalize_shape_source_to_callable(
    shape_source: ShapeSource, multiscale_dict: OME_ZARR_MULTISCALE
) -> GetShapeFunction:
    if shape_source == "singletons":
        return make_all_singleton_shapes(multiscale_dict)
    if isinstance(shape_source, (str, bytes)):
        raise TypeError(
            f"Cannot obtain array shape from plain path. Received: {shape_source!r}."
            "Provide the object you will use to read zarr data, e.g. from "
            "zarr.open_group or ts.open({...}).result(), "
            "or a custom `GetShapeFunction(relative_path_str) -> shape_tuple(int)`."
        )
    if callable(shape_source):
        return lambda path: _normalize_shape_value_to_tuple(shape_source(path))
    return lambda path: _normalize_shape_value_to_tuple(shape_source[path])


def _normalize_shape_value_to_tuple(value: ShapeValue) -> Tuple[int, ...]:
    raw_shape: Any = getattr(value, "shape", value)
    if isinstance(raw_shape, Mapping):
        raw_shape = raw_shape.values()
    try:
        return tuple(int(size) for size in raw_shape)
    except (TypeError, ValueError):
        raise TypeError(f"Expected shape or array-like with .shape. Received: {value!r}")


def _as_transform_list(ome_transformations: Optional[OME_ZARR_TRANSFORMS]) -> List[OME_ZARR_TRANSFORM]:
    if not ome_transformations:
        return []
    if isinstance(ome_transformations, dict):
        return [ome_transformations]
    if isinstance(ome_transformations, list):
        return ome_transformations
    return []


@dataclass(frozen=True, slots=True)
class MultiscaleTransforms(TransformSequence):
    def inverted(self) -> "InvertedMultiscaleTransforms":
        sup = super(MultiscaleTransforms, self).inverted()
        return InvertedMultiscaleTransforms(transforms=sup.transforms).bound(source=self.target, target=self.source)

    def __post_init__(self):
        if len(self.transforms) not in (1, 2):
            raise ValueError("MultiscaleTransforms requires one or two transforms.")
        if not isinstance(self.transforms[0], ScaleTransform):
            raise TypeError("First transform must be a ScaleTransform.")
        if len(self.transforms) == 2 and not isinstance(self.transforms[1], TranslationTransform):
            raise TypeError("Second transform must be a TranslationTransform.")

        TransformSequence.__post_init__(self)

    def to_legacy_ome_zarr(self) -> List[Dict[str, Any]]:
        """TransformSequence.to_ome_zarr is for 0.6.rc0 and upwards; this handles legacy 0.4/0.5"""
        version = "0.5"  # Same format for 0.4 and 0.5, so doesn't matter which
        return [t.to_ome_zarr(version) for t in self.transforms]

    @property
    def scale_transform(self) -> ScaleTransform:
        scale = self.transforms[0]
        assert isinstance(scale, ScaleTransform), f"Dev error: Expected scale, got {scale!r}"
        return scale

    @property
    def translation_transform(self) -> Optional[TranslationTransform]:
        if len(self.transforms) != 2:
            return None
        translation = self.transforms[1]
        assert isinstance(translation, TranslationTransform), f"Dev error: Expected translation, got {translation!r}"
        return translation

    @classmethod
    def from_list(cls, ome_transformations: Optional[OME_ZARR_TRANSFORMS]) -> Optional["MultiscaleTransforms"]:
        """
        Possibilities for ome_transformations:
        0.6.rc0 multiscale[datasets][n][coordinateTransformations]:
        - List of one ScaleTransform
        - List of one IdentityTransform
        - List of one TransformSequence containing one ScaleTransform and one TranslationTransform
        OME-Zarr v0.4 and 0.5:
        - multiscale[coordinateTransformations]:
          - absent or empty
          - List of one ScaleTransform
          - List of one ScaleTransform and one TranslationTransform
        - multiscale[datasets][][coordinateTransformations]:
          - List of one ScaleTransform
          - List of one ScaleTransform and one TranslationTransform
        """
        ome_transformations = _as_transform_list(ome_transformations)
        if not ome_transformations:
            return None
        scale: Optional[ScaleTransform] = None
        translation: Optional[TranslationTransform] = None
        for t_dict in ome_transformations:
            # Best effort: Find first valid combination,
            # and accept even a valid translation without valid scale,
            # but reject path-backed scale/translation (because nobody uses it in practice).
            # If it ever becomes necessary, we'd need an `array_source` param.
            try:
                t = Transform.from_ome_zarr(t_dict)
            except ValueError:
                continue
            if isinstance(t, TransformSequence):
                if len(t) == 1 and isinstance(t[0], ScaleTransform) and t[0].scale:
                    scale = t[0]
                    continue
                if len(t) == 2 and isinstance(t[0], ScaleTransform) and isinstance(t[1], TranslationTransform):
                    scale = t[0] if t[0].scale else None
                    translation = t[1] if t[1].translation else None
                    break
            if isinstance(t, ScaleTransform) and scale is None:
                scale = t if t.scale else None
                continue
            if isinstance(t, TranslationTransform) and translation is None:
                translation = t if t.translation else None
                if scale:
                    break
                continue
        if scale is None and translation is None:
            return None
        elif scale is None:
            assert isinstance(translation, TranslationTransform), "should've made sure by now"
            ndim = translation._ndim_by_payload()
            assert ndim.source is not None, "should've made sure translation values actually exist"
            scale = ScaleTransform(scale=tuple(1.0 for _ in range(ndim.source)))
        return cls(transforms=(scale,) if translation is None else (scale, translation))

    @classmethod
    def from_transforms(cls, transforms: Tuple["Transform", ...]) -> "MultiscaleTransforms":
        if len(transforms) == 1:
            t = transforms[0]
            if isinstance(t, MultiscaleTransforms):
                return t
            if isinstance(t, InvertedMultiscaleTransforms):
                return t.inverted()
            if isinstance(t, ScaleTransform):
                return MultiscaleTransforms((t,)).bound(source=t.source, target=t.target)
            if isinstance(t, TranslationTransform):
                identity_scale = ScaleTransform((1.0,) * len(t.translation))
                return MultiscaleTransforms((identity_scale, t)).bound(source=t.source, target=t.target)
            if isinstance(t, TransformSequence):
                try:
                    return cls.from_transforms(t.transforms).bound(source=t.source, target=t.target)
                except ValueError:
                    raise ValueError(f"Cannot represent as (scale[, translation]): {transforms}") from None
        if (
            len(transforms) == 2
            and isinstance(transforms[0], ScaleTransform)
            and isinstance(transforms[1], TranslationTransform)
        ):
            return MultiscaleTransforms(tuple(transforms))
        if (
            len(transforms) == 2
            and isinstance(transforms[1], ScaleTransform)
            and isinstance(transforms[0], TranslationTransform)
        ):
            return InvertedMultiscaleTransforms(tuple(transforms)).inverted()
        raise ValueError(f"Cannot represent as (scale[, translation]): {transforms}")

    def composed_with(self, earlier: "Transform") -> Optional["MultiscaleTransforms"]:
        if not isinstance(earlier, MultiscaleTransforms):
            return None
        if not self._endpoints_can_chain_after(earlier):
            return None
        # Could go via super().canonicalized() to skip the algebra, but then we would lose the information
        # whether self or earlier had an explicitly recorded identity translation.
        scale_product = self.scale_transform.composed_with(earlier.scale_transform)
        assert isinstance(scale_product, ScaleTransform), "scales can always compose"
        earlier_translation_rescaled = None
        if earlier.translation_transform is not None and not is_identity_scale(
            self.scale_transform.scale, tolerance=DETERMINANT_SINGULARITY_TOLERANCE
        ):
            earlier_translation_rescaled = TranslationTransform(
                tuple(t * s for t, s in zip(earlier.translation_transform.translation, self.scale_transform.scale))
            )
        later_translation_rescaled = None
        if self.translation_transform is not None and not is_identity_scale(
            earlier.scale_transform.scale, tolerance=DETERMINANT_SINGULARITY_TOLERANCE
        ):
            later_translation_rescaled = TranslationTransform(
                tuple(t * s for t, s in zip(self.translation_transform.translation, earlier.scale_transform.scale))
            )
        if earlier_translation_rescaled is not None and later_translation_rescaled is not None:
            translation_sum = later_translation_rescaled.composed_with(earlier_translation_rescaled)
            assert isinstance(translation_sum, TranslationTransform), "translations can always compose"
            transforms = (scale_product, translation_sum)
        elif earlier_translation_rescaled is not None:
            transforms = (scale_product, earlier_translation_rescaled)
        elif later_translation_rescaled is not None:
            transforms = (scale_product, later_translation_rescaled)
        else:
            transforms = (scale_product,)
        return replace(self, transforms=transforms).bound(
            source=self._composed_source(earlier), target=self._composed_target(earlier)
        )

    def simplified(self) -> "MultiscaleTransforms":
        return self  # Avoid dropping explicitly recorded global identity by simplification


class InvertedMultiscaleTransforms(TransformSequence):
    """Alias purely to roundtrip inversion while maintaining a recognisable legacy marker class."""

    def inverted(self) -> "MultiscaleTransforms":
        sup = super(InvertedMultiscaleTransforms, self).inverted()
        return MultiscaleTransforms(transforms=sup.transforms).bound(source=self.target, target=self.source)

    def composed_with(self, earlier: "Transform") -> Optional["InvertedMultiscaleTransforms"]:
        return None  # Avoid folding into other TransformSequences

    def simplified(self) -> "InvertedMultiscaleTransforms":
        return self  # Avoid dropping explicitly recorded global identity by simplification


def require_dataset_paths(raw: Mapping[str, Any]):
    """Light top-level checks. coordinateTransformations are validated later."""
    version = raw.get("version")
    if version and version not in SUPPORTED_OME_ZARR_VERSIONS_READ:
        warnings.warn(f"Attempting to parse unknown OME-Zarr version '{version}'. This might break...")

    if "datasets" not in raw or not raw["datasets"] or not isinstance(raw["datasets"], list):
        raise ValueError(f"Invalid OME-Zarr metadata: no datasets in: {raw!r}")

    seen_paths = set()
    for ds in raw["datasets"]:
        if not isinstance(ds, Mapping) or "path" not in ds or not ds["path"] or not isinstance(ds["path"], str):
            raise ValueError(f"Invalid OME-Zarr metadata: dataset missing path {ds!r} in: {raw!r}")
        if ds["path"] in seen_paths:
            raise ValueError(f"Invalid OME-Zarr metadata: multiple datasets reference path {ds['path']} in: {raw!r}")
        seen_paths.add(ds["path"])

    if version not in ("0.1", "0.2", "0.3") and any(
        (
            "coordinateTransformations" not in d
            or not d["coordinateTransformations"]
            or not isinstance(d["coordinateTransformations"], list)
        )
        for d in raw["datasets"]
    ):
        warnings.warn(
            f"Invalid OME-Zarr metadata: dataset(s) with no transformations. "
            f"Will infer pixel size as relative scaling factors. Received: {raw}"
        )


def _output_system_name_from_datasets(multiscale: OME_ZARR_MULTISCALE) -> Optional[str]:
    transforms = _as_transform_list(multiscale["datasets"][0].get("coordinateTransformations"))
    if not transforms:
        return None

    ref = transforms[0].get("output")
    if isinstance(ref, str):
        return ref or None
    if isinstance(ref, dict):
        return ref.get("name")
    return None


def extract_multiscale_graph(
    multiscale: OME_ZARR_MULTISCALE,
) -> Tuple[TransformGraph, NodeRef[CoordinateSystem]]:
    intrinsic_system_name: Optional[str] = None
    try:
        intrinsic_system_name = _output_system_name_from_datasets(multiscale)
        if not intrinsic_system_name:
            raise ValueError(f"Invalid OME-Zarr multiscale metadata (or version older than 0.6): {multiscale!r}")
        graph = TransformGraph.from_ome_zarr(
            multiscale.get("coordinateTransformations"), multiscale.get("coordinateSystems")
        )
        potential_intrinsics = [ref for ref in graph.all_system_refs if ref.name == intrinsic_system_name]
        if len(potential_intrinsics) != 1:
            raise ValueError(
                "Invalid OME-Zarr multiscale metadata: Expected exactly one coordinate system named "
                f"{intrinsic_system_name!r}. Received: {multiscale}"
            )
        intrinsic_system_ref = potential_intrinsics[0]
        return graph, intrinsic_system_ref
    except ValueError as e:
        try:
            # Best effort: Is there any coordinate system we can use at all?
            name_matches = [
                sys_d for sys_d in multiscale["coordinateSystems"] if sys_d["name"] == intrinsic_system_name
            ]
            if name_matches:
                intrinsic_sys = CoordinateSystem.from_ome_zarr(name_matches[0])
                intrinsic_system_name = name_matches[0]["name"]
            elif multiscale["coordinateSystems"]:
                intrinsic_sys = CoordinateSystem.from_ome_zarr(multiscale["coordinateSystems"][0])
                intrinsic_system_name = multiscale["coordinateSystems"][0]["name"]
            else:
                raise ValueError()
        except (KeyError, ValueError, TypeError):
            raise e
        warnings.warn(
            "Invalid coordinateTransformations and/or coordinateSystems metadata. Proceeding without. "
            f"Error: {str(e)}"
            f"Received: {multiscale}"
        )
        if not intrinsic_system_name:
            raise e
        intrinsic_system_ref = intrinsic_sys.as_ref(intrinsic_system_name)
        graph = TransformGraph.single_isolated_system(intrinsic_system_ref)
        return graph, intrinsic_system_ref


def global_t_scale_if_matches_legacy_convention(
    multiscale: OME_ZARR_MULTISCALE,
    global_transforms: MultiscaleTransforms,
    axes: Tuple[AxisKey, ...],
) -> Optional[float]:
    """
    Special case / ambiguity: OME-Zarr 0.4 and 0.5 define the dataset scale as the pixel size, but the spec's own
    primary example shows the pixel size along the time-axis only recorded in the global_transforms (multiscale-scale).

    nifti-zarr formalises this as a standard (https://github.com/neuroscales/nifti-zarr#24-nifti-header).
    Their own xyztc voxel size becomes [1,1,z,y,x] in dataset-scale, and [t,c,1,1,1] in multiscale-scale:
    nifti.zattrs["VoxelSize"][:3]  ==  zattrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2:5][::-1]
    nifti.zattrs["VoxelSize"][3]   ==  zattrs["multiscales"][0]["coordinateTransformations"][0]["scale"][0]
    nifti.zattrs["VoxelSize"][4]   ==  zattrs["multiscales"][0]["coordinateTransformations"][0]["scale"][1]

    At the same time, we have the opposite use, following the language of the spec rather than its example, at HHMI
    (https://github.com/AI-HHMI/miao/issues/25). Here, the dataset-scale is the voxel size, and the multiscale-scale
    is an instrument-specific magnification.
    This manifests as [z1, y1, x1] in dataset-scale and [z2, y2, x2] in multiscale-scale.

    -- So what do we make of it?
    Look strictly for the pattern "scale[t] is 1.0 in all datasets, and not 1.0 on global level".
    If this matches, interpret the *output* of the global transforms as the multiscale's intrinsic system.

    Return None if the metadata does not conform to the global-t-scale convention, otherwise the scale.
    """
    if "t" not in axes:
        return None
    t_index = axes.index("t")
    identity_values = (PixelSize._default(), 0)
    global_scale = global_transforms.scale_transform.scale
    global_scale_non_t = global_scale[:t_index] + global_scale[t_index + 1 :]
    if global_scale[t_index] in identity_values:
        # Violates convention rule 1: global/multiscale-scale[t] must be non-identity.
        # This means the convention can't express if t is exactly 1.0 units per pixel.
        # Doesn't matter: if that's the case, the metadata look the same with and without the convention anyway.
        return None
    if any(v not in identity_values for v in global_scale_non_t):
        # Violates convention rule 2: global/multiscale-scale must be *only* t, otherwise identity
        return None
    for ds in multiscale["datasets"]:
        try:
            pixel_size_values = ds["coordinateTransformations"][0]["scale"]
            if pixel_size_values[t_index] not in identity_values:
                # Violates convention rule 3: all dataset-scale must be identity for t
                return None
        except (KeyError, TypeError, ValueError):
            return None
    return global_scale[t_index]


def multiscale_graph_from_legacy(
    multiscale: OME_ZARR_MULTISCALE, *, name: str
) -> Tuple[TransformGraph, NodeRef[CoordinateSystem], Optional[Tuple[float, Optional[TranslationTransform]]]]:
    """
    Convert legacy metadata into modern coordinate graph interpretation.
    Returns graph, intrinsic ref, and None or a tuple of (pixel_size[t], global_transforms_to_fold).

    The third return tuple are the contents of multiscale[coordinateTransformations] if the entire multiscale
    conformed to a legacy convention where t-scale is written on multiscale level, which implies that the intrinsic
    coordinate system is the output of the global transforms rather than its input as the spec requires.
    """
    multiscale_tf_list = multiscale.get("coordinateTransformations")
    try:
        global_transforms = MultiscaleTransforms.from_list(multiscale_tf_list)
    except ValueError:
        global_transforms = None
    if multiscale_tf_list and global_transforms is None:
        warnings.warn(f"Pixel size metadata at multiscale-level was invalid. Received: {multiscale_tf_list!r}")

    intrinsic_system = CoordinateSystem.from_ome_zarr(multiscale)
    intrinsic_system_ref = intrinsic_system.as_ref(name)
    graph = TransformGraph.single_isolated_system(intrinsic_system_ref)
    if global_transforms is not None:
        global_t_scale = global_t_scale_if_matches_legacy_convention(
            multiscale, global_transforms, tuple(intrinsic_system.axes())
        )
        if global_t_scale:
            # Using the global transforms for pixel_size[t] as the convention does,
            # implies that the output of the global transforms *is* the intrinsic system
            # -> compose into dataset transforms (Scale props)
            # -> exclude from graph (the global transforms do not refer to some external coordinate system in this case)
            assert (
                sum(v == global_t_scale for v in global_transforms.scale_transform.scale) == 1
            ), f"dev error: {global_transforms.scale_transform.scale} doesn't actually use global-t convention"
            return graph, intrinsic_system_ref, (global_t_scale, global_transforms.translation_transform)
        own_axes = tuple(intrinsic_system.axes())
        synthetic_external = CoordinateSystem.without_semantics(own_axes).as_ref(f"external-{name}")
        try:
            bound_transform = global_transforms.bound(source=intrinsic_system_ref, target=synthetic_external)
            assert isinstance(bound_transform, MultiscaleTransforms), "should not change type"
            graph = TransformGraph([bound_transform])
        except ValueError:
            # E.g. mismatching number of axes between transforms and coordinate system
            warnings.warn(f"Invalid OME-Zarr metadata: Ignoring multiscale transforms: {multiscale_tf_list!r}.")
    return graph, intrinsic_system_ref, None


def scale_meta_from_dataset_transforms(
    axis_keys: Sequence[AxisKey],
    global_scale_meta: Optional[Tuple[float, Optional[TranslationTransform]]],
    relative_scale_pixel_size: PixelSize,
    transformations: Optional[OME_ZARR_TRANSFORMS],
) -> Tuple[PixelSize, Optional[Translation], Optional[Tuple[AxisKey, ...]]]:
    """
    Extract pixel size and translation according to this dataset's `coordinateTransformations`.
    Returns the scale's:
        - pixel size (defaulting to 1.0 or relative factor if invalid meta)
        - translation (defaulting to None if invalid meta)
        - a tuple of axis keys where the `scale` transform was 0.0 (normally None, purely for metadata round-trip)
    """
    if transformations is None:
        # Fine: OME-Zarr up to v0.3 didn't have coordinateTransformations
        return relative_scale_pixel_size, None, None

    try:
        dataset_transforms = MultiscaleTransforms.from_list(transformations)
    except ValueError:
        warnings.warn(
            f"Invalid OME-Zarr metadata: dataset scale and translation transform do not match each other. "
            "Continuing with relative scale factor as pixel size. Received: "
            f"{transformations!r}"
        )
        return relative_scale_pixel_size, None, None

    if dataset_transforms is None:
        # Meta existed and was valid but e.g. empty
        warnings.warn(
            f'Invalid OME-Zarr metadata: expected at least a "scale" transform, but found none. '
            "Continuing with relative scale factor as pixel size. Received: "
            f"{transformations!r}"
        )
        return relative_scale_pixel_size, None, None

    if len(axis_keys) != len(dataset_transforms.scale_transform.scale):
        warnings.warn(
            f"Invalid OME-Zarr metadata: dataset reports {len(dataset_transforms.scale_transform.scale)} "
            f"pixel size values, mismatching its axes {list(axis_keys)}. "
            "Continuing with relative scale factor as pixel size. Received: "
            f"{dataset_transforms.scale_transform.scale!r}"
        )
        return relative_scale_pixel_size, None, None

    if any(v < 0.0 for v in dataset_transforms.scale_transform.scale):
        warnings.warn(
            f"Invalid OME-Zarr metadata: dataset has negative pixel size values. "
            "Continuing with relative scale factor as pixel size. Received: "
            f"{dataset_transforms.scale_transform.scale!r}"
        )
        return relative_scale_pixel_size, None, None

    pixel_size_values = list(dataset_transforms.scale_transform.scale)
    scale_translation = (
        dataset_transforms.translation_transform.to_translation(axis_keys)
        if dataset_transforms.translation_transform
        else None
    )
    if global_scale_meta is not None:
        # Special case for the "t pixel size stored in global transforms" convention.
        # global_t_scale is the pixel size along t in this case.
        # global_tfs_to_compose has an identity scale and maybe some translation in this case.
        global_t_scale, global_translation_transform = global_scale_meta
        assert "t" in axis_keys, "dev error: multiscale_graph_from_legacy misidentified global_t_scale convention"
        pixel_size_values[list(axis_keys).index("t")] = global_t_scale
        if scale_translation is not None:
            scale_translation *= Factor(t=global_t_scale)
        if global_translation_transform is not None:
            if scale_translation is None:
                scale_translation = Translation.identity(axis_keys)
            scale_translation += global_translation_transform.to_translation(axis_keys)

    scale_pixel_size = scale_to_pixel_size_with_normalized_zeros(pixel_size_values, axis_keys)
    zeros = tuple(axis for axis, value in zip(axis_keys, pixel_size_values) if value == 0)
    return scale_pixel_size, scale_translation, zeros


def scale_to_pixel_size_with_normalized_zeros(scale: Sequence[float], axes: Sequence[AxisKey]) -> PixelSize:
    assert len(scale) == len(axes), "should make sure before calling"
    normalized_scale = (PixelSize._default() if value == 0 else value for value in scale)
    return PixelSize(zip(axes, normalized_scale))


def pixel_size_to_scale_with_reintroduced_zeros(
    pixel_size: PixelSize, serialized_zero_scale_axes: Iterable[AxisKey] = ()
) -> ScaleTransform:
    zero_axes = set(serialized_zero_scale_axes)
    scale = tuple(0.0 if axis in zero_axes else pixel_size[axis] for axis in pixel_size)
    return ScaleTransform(scale=scale)


####
# Writing
####


OME_ZARR_PATH_RE = re.compile(
    r"""
    ^                       # start of string
    [A-Za-z0-9._-]+         # first path segment: no empty, no special chars
    (?:                     # additional segments: (non-capturing)
        /                   #   forward slash as separator
        [A-Za-z0-9._-]+     #   another valid segment
    )*                      # zero or more additional segments
    $                       # end of string
    """,
    re.VERBOSE,
)


def validate_multiscale(multiscale: "Multiscale"):
    for scale_key in multiscale.keys():
        if not _is_valid_relative_path(str(scale_key)):
            raise ValueError(f"Scale key '{scale_key}' is not a valid relative filesystem path")

    axes = list(multiscale.axes())
    standard_axes_set = set("tczyx")

    if all(ax in standard_axes_set for ax in axes):
        expected_order = [ax for ax in "tczyx" if ax in axes]
        if axes != expected_order:
            warnings.warn(
                f"Axes {axes} are all standard (t,c,z,y,x) but not in OME-Zarr "
                f"canonical order. Expected: {expected_order}. "
                f"This may cause issues with some OME-Zarr readers."
            )


def _is_valid_relative_path(path: str) -> bool:
    if not OME_ZARR_PATH_RE.fullmatch(path):
        return False
    return all(seg not in {".", ".."} for seg in path.split("/"))


def build_dataset_dict(
    version,
    key,
    dataset_scale: PixelSize,
    dataset_translation: Translation,
    intrinsic_ref: Optional[NodeRef[CoordinateSystem]] = None,
    serialized_zero_scale_axes: Iterable[AxisKey] = (),
) -> Dict[str, Any]:
    scale = pixel_size_to_scale_with_reintroduced_zeros(dataset_scale, serialized_zero_scale_axes)
    components: Tuple[Transform, ...] = (scale,)
    if not dataset_translation.is_identity():
        translation = TranslationTransform.from_translation(dataset_translation)
        components = (scale, translation)
    if version in PRE_TRANSFORMS_VERSIONS:
        dataset_transforms = [t.to_ome_zarr(version) for t in components]
    else:
        dataset_path = _UnresolvedRef(name=None, file=FileRef.from_string(str(key)))
        # The "single transform inside a list" requirement of 0.6 actually makes this more awkward
        single_t = TransformSequence(components) if len(components) > 1 else components[0]
        single_dict = single_t.bound(source=dataset_path, target=intrinsic_ref).to_ome_zarr(version)
        dataset_transforms = [single_dict]
    dataset_dict = {"path": str(key), "coordinateTransformations": dataset_transforms}
    return dataset_dict
