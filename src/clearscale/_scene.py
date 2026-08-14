import functools
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Union, Tuple, Literal
from typing import Optional, List

from clearscale._errors import MismatchingMultiscaleError
from clearscale._axis_values import Translation
from clearscale._multiscale import Multiscale
from clearscale._transforms import (
    RelativePath,
    CoordinateSystem,
    CoordinateSystemName,
    NodeRef,
    _UnresolvedRef,
    AnyRef,
    Transform,
    TransformGraph,
    TransformGraphNode,
    TranslationTransform,
)

MultiscalesByPath = Mapping[RelativePath, Multiscale]
UserFacingCoordinateSystemKey = Union[
    CoordinateSystemName,
    Multiscale,
    Tuple[Multiscale, CoordinateSystemName],
    Dict[Literal["path", "name"], Union[RelativePath, CoordinateSystemName]],
]
Node = Union[Multiscale, NodeRef[CoordinateSystem]]


@dataclass(frozen=True)
class Scene:
    _internal_graph: TransformGraph
    _multiscale_paths: MultiscalesByPath
    """Helper property to round-trip paths: Scene.from_ome_zarr().with_resolved().to_ome_zarr()."""

    def __post_init__(self):
        if not isinstance(self._multiscale_paths, MappingABC):
            raise TypeError(
                f"_multiscale_paths must be a mapping like {{path: Multiscale}}. "
                f"Received: {self._multiscale_paths!r}"
            )
        paths = dict(self._multiscale_paths)
        object.__setattr__(self, "_multiscale_paths", paths)

    @property
    def is_fully_resolved(self) -> bool:
        return len(self._internal_graph.unresolved_transforms) == 0

    @functools.cached_property
    def unresolved_paths(self) -> List[RelativePath]:
        paths = []  # Order might matter to the consumer
        seen_paths = set()
        for t in self._internal_graph.unresolved_transforms:
            for endpoint in (t.source, t.target):
                if not isinstance(endpoint, _UnresolvedRef):
                    continue
                p = endpoint.path
                if not p or not isinstance(p, str) or p in seen_paths:
                    continue
                paths.append(p)
                seen_paths.add(p)
        return paths

    @functools.cached_property
    def _full_graph(self):
        all_transforms = list(self._internal_graph.transforms)
        for ms in self._multiscale_paths.values():
            all_transforms.append(ms._get_interface_transform())  # noqa: package-private, not class-private
            all_transforms.extend(ms._transform_graph.transforms)  # noqa: package-private, not class-private
        return TransformGraph(all_transforms)

    @classmethod
    def from_graph_edges(
        cls,
        source_transform_targets: Iterable[Tuple[Node, Transform, Node]],
    ) -> "Scene":
        """
        Basic low-level constructor by directly specifying graph edges (source->transform->target):
        Scene.from_graph_edges([
            (moving_ms1, AffineTransform(...), fixed_ms),
            (moving_ms2, AffineTransform(...), fixed_ms),
        ])
        """
        transforms = []
        for source_node, transform, target_node in source_transform_targets:
            source = cls._node_to_coord_sys_ref(source_node)
            target = cls._node_to_coord_sys_ref(target_node)
            bound = transform.bound(source=source, target=target)
            transforms.append(bound)
        return cls(_internal_graph=TransformGraph(transforms=transforms), _multiscale_paths={})

    @classmethod
    def from_star_graph(
        cls, multiscales: Iterable[Tuple[Multiscale, Transform]], *, center: Optional[Node] = None
    ) -> "Scene":
        """
        Low-level constructor for star-shaped graphs by specifying partial edges (source->transform).
        All transforms will target the node provided as `center`.
        If `center` is not provided, all transforms will target the first entry in `multiscales`.
        The first Multiscale should be paired with an IdentityTransform in that case.
        """
        multiscales = list(multiscales)
        if not multiscales and not center:
            return cls(_internal_graph=TransformGraph(transforms=()), _multiscale_paths={})
        elif not center:
            center = multiscales[0][0]

        central_system = cls._node_to_coord_sys_ref(center).owner.copy()
        central_ref = central_system.as_ref("world")

        return cls(
            _internal_graph=TransformGraph(
                transforms=(
                    transform.bound(
                        source=multiscale._intrinsic_ref,
                        target=central_ref,
                    )
                    for multiscale, transform in multiscales
                )
            ),
            _multiscale_paths={},
        )

    @classmethod
    def from_tiles_translations(cls, translations: Iterable[Tuple[Multiscale, Translation]]) -> "Scene":
        """
        Proof-of-concept constructor for tiling use-case. More generally useful might be something like
        from_tiles(tiles: Iterable[Tuple[Multiscale, AxisValues]]), or "from_clearscale"..?
        With an internal Transform.from_axis_values that can map clearscale types to accurate Transform representations.
        E.g. Translation -> TranslationTransform,
        Factor -> ScaleTransform,
        PixelOffset -> TranslationTransform (after multiplying the offset by the right PixelSize)
        """
        translations = list(translations)
        if not translations:
            return cls(_internal_graph=TransformGraph(transforms=()), _multiscale_paths={})

        first_ms = translations[0][0]
        for multiscale, translation in translations:
            # TODO: Validate units?
            if multiscale.axes() != first_ms.axes():
                raise ValueError(
                    "All Multiscales in a stitched Scene must have identical axis keys. "
                    f"Expected {first_ms.axes()!r}, received {multiscale.axes()!r}."
                )
            if translation.keys() != first_ms.axes():
                raise ValueError(
                    "All Translations for a stitched Scene must have identical axis keys. "
                    f"Expected {first_ms.axes()!r}, received {translation.keys()!r}."
                )

        return cls.from_star_graph(
            (ms, TranslationTransform.from_translation(translation)) for ms, translation in translations
        )

    @classmethod
    def from_ome_zarr(cls, scene_attrs: Mapping[str, Any]) -> "Scene":
        # TODO: accept an optional callable get_multiscale_meta;
        #  where the default provided implementation simply chooses
        #  the first entry in the multiscales-array at the path.
        #  Problem: will also need shape_source for Multiscale.from_ome_zarr :)
        transform_dicts = scene_attrs.get("coordinateTransformations", [])
        system_dicts = scene_attrs.get("coordinateSystems", [])
        graph = TransformGraph.from_ome_zarr(transform_dicts, system_dicts)
        return cls(graph, _multiscale_paths={})

    def with_resolved(
        self,
        multiscales_by_path: Optional[MultiscalesByPath] = None,
    ) -> "Scene":
        if not multiscales_by_path or not isinstance(multiscales_by_path, MappingABC):
            return self
        transforms = []
        resolved_paths = {}
        for t in self._internal_graph.transforms:
            try:
                maybe_resolved_t = t.with_resolved(multiscales_by_path)
            except MismatchingMultiscaleError as e:
                raise ValueError(
                    f"Invalid pairing: The Multiscale provide for path {e.path!r} does not seem to be the expected one "
                    f"(expected coordinate system named {e.name!r})."
                ) from None
            transforms.append(maybe_resolved_t)
            resolved_paths.update(self._resolved_multiscale_paths(t, maybe_resolved_t, multiscales_by_path))
        paths = dict(self._multiscale_paths)
        paths.update(resolved_paths)
        graph = replace(self._internal_graph, transforms=tuple(transforms))
        return replace(self, _internal_graph=graph, _multiscale_paths=paths)

    def to_ome_zarr(self, *, version: str = "0.6.rc0", multiscales_by_path: Optional[MultiscalesByPath] = None) -> Dict:
        # TODO: I think this is broken (would not dump all coord systems from the graph)
        # should probably delegate to self._internal_graph.to_ome_zarr, no?
        coordinate_system_dicts = []
        for ref in self._internal_graph.system_refs:
            assert ref.owner is not None, "Dev error: all refs to CoordinateSystems must be owned"
            coordinate_system_dicts.append(ref.owner.to_ome_zarr(name=ref.name, version=version))

        all_paths = dict(self._multiscale_paths)
        if multiscales_by_path is not None:
            if not isinstance(multiscales_by_path, MappingABC):
                raise TypeError(
                    f"multiscales_by_path must be a mapping like {{path: Multiscale}}. Received: {multiscales_by_path!r}"
                )
            cleaned = {k: v for k, v in multiscales_by_path.items() if k not in (None, "")}
            all_paths.update(cleaned)
        coordinate_transformations_dicts = [
            t.to_ome_zarr(version, nodes_by_path=all_paths) for t in self._internal_graph.transforms
        ]

        result: Dict = {"coordinateTransformations": coordinate_transformations_dicts}
        if coordinate_system_dicts:
            result["coordinateSystems"] = coordinate_system_dicts
        return result

    def transforms_between(
        self, source: UserFacingCoordinateSystemKey, target: UserFacingCoordinateSystemKey, include_children=False
    ) -> Optional[List[Transform]]:
        source_ref = self._get_ref_for_key(source, include_children)
        target_ref = self._get_ref_for_key(target, include_children)
        if source_ref is None or target_ref is None:
            return None
        if include_children:
            return self._full_graph.path_between(source_ref, target_ref)
        return self._internal_graph.path_between(source_ref, target_ref)

    def _get_ref_for_key(self, key: UserFacingCoordinateSystemKey, include_children: bool) -> Optional[AnyRef]:
        if isinstance(key, dict):  # Dict[Literal["path", "name"], Union[RelativePath, CoordinateSystemName]]
            if key["path"] in self._multiscale_paths:
                return self._multiscale_paths[key["path"]].as_ref(key["name"])
            return _UnresolvedRef(name=key["name"], path=key["path"])

        if isinstance(key, tuple):  # Tuple[Multiscale, CoordinateSystemName]
            if not isinstance(key[0], Multiscale) or not (isinstance(key[1], CoordinateSystemName)):
                raise TypeError(f"Coordinate system key must be tuple(multiscale, system_name). Received: {key}")
            return key[0].as_ref(key[1])

        if isinstance(key, Multiscale):
            # If there were more than 1 and user cared, they'd give us a tuple
            return key._intrinsic_ref  # noqa: package-private, not class-private

        if isinstance(key, CoordinateSystemName):
            # Purely matching by name could bring up refs to any TransformGraphNode
            # (Multiscale, CoordinateSystem, or placeholder _UnresolvedRef)
            own_systems = self._internal_graph.connected_system_refs
            for ref in own_systems:  # Expected: These CoordinateSystems can only be retrieved by name
                if ref.name == key:
                    return ref
            # Best effort: Maybe the name is still unique among unresolved refs or multiscales
            all_refs = self._full_graph.node_refs if include_children else self._internal_graph.node_refs
            name_matches = [ref for ref in all_refs if ref.name == key]
            if len(name_matches) > 1:
                raise ValueError(
                    f'Cannot retrieve transformations for name "{key}" because it is ambiguous. '
                    "Use a multiscale-name tuple to select multiscales, "
                    "or a {path, name} dict to select unresolved multiscales. "
                    f"Matches: {name_matches}."
                )
            return name_matches[0] if name_matches else None

        raise TypeError(f"Unsupported key type for coordinate system lookup: {key}")

    @classmethod
    def _node_to_coord_sys_ref(cls, node: Node) -> NodeRef[CoordinateSystem]:
        if isinstance(node, Multiscale):
            ref = node._intrinsic_ref
        elif isinstance(node, NodeRef) and isinstance(node.owner, CoordinateSystem):
            ref = node
        else:
            raise TypeError(
                f"Use CoordinateSystem.as_ref(name) to use a CoordinateSystem in a Scene. Received: {node!r}"
            )
        return ref

    @staticmethod
    def _resolved_multiscale_paths(
        before: Transform, after: Transform, multiscales_by_path: MultiscalesByPath
    ) -> Dict[RelativePath, Multiscale]:
        resolved_paths = {}
        for old_ref, new_ref in ((before.source, after.source), (before.target, after.target)):
            if new_ref is old_ref or not isinstance(old_ref, _UnresolvedRef) or not old_ref.path:
                continue
            multiscale = multiscales_by_path.get(old_ref.path)
            if multiscale is None:
                continue
            assert new_ref is not None, f"Dev error: previously not-None {old_ref!r} became None {new_ref!r}"
            if isinstance(new_ref, NodeRef) and isinstance(new_ref.owner, Multiscale) and new_ref.owner is multiscale:
                resolved_paths[old_ref.path] = multiscale
        return resolved_paths
