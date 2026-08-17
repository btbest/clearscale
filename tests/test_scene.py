import pytest

from clearscale import Multiscale, Scale, Scene, Shape, Translation
from clearscale._transforms import (
    CoordinateSystem,
    TranslationTransform,
    TransformGraph,
    _UnresolvedRef,
    ScaleTransform,
    AffineTransform,
    NodeRef,
    FileRef,
)


def _multiscale(**shape: int) -> Multiscale:
    return Multiscale({"s0": Scale(Shape(**shape))})


def test_from_graph_edges_binds_between_multiscales():
    source = _multiscale(z=2, y=3, x=4)
    target = _multiscale(z=2, y=3, x=4)

    scene = Scene.from_graph_edges([(source, ScaleTransform((2, 2, 2)), target)])

    path = scene.transforms_between(source, target)

    assert path
    assert len(path) == 1
    assert path[0] == ScaleTransform((2, 2, 2)).bound(
        source=source._intrinsic_ref,
        target=target._intrinsic_ref,
    )


def test_from_graph_edges_binds_coordinate_system_refs():
    world = CoordinateSystem.without_semantics("zyx")
    image = _multiscale(z=2, y=3, x=4)

    scene = Scene.from_graph_edges(
        [
            (
                image,
                ScaleTransform((2, 2, 2)),
                world.as_ref("world"),
            ),
        ]
    )

    path = scene.transforms_between(image, "world")

    assert path
    assert len(path) == 1
    assert path[0].target == world.as_ref("world")


def test_from_graph_edges_multiple_edges():
    moving = _multiscale(z=2, y=3, x=4)
    fixed = _multiscale(z=2, y=3, x=4)
    world = CoordinateSystem.without_semantics("zyx")

    scene = Scene.from_graph_edges(
        [
            (moving, ScaleTransform((2, 2, 2)), world.as_ref("world")),
            (world.as_ref("world"), AffineTransform(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))), fixed),
        ]
    )

    path = scene.transforms_between(moving, fixed)

    assert path
    assert len(path) == 2
    scale_unbound = path[0].bound(source=None, target=None)
    assert scale_unbound == ScaleTransform((2, 2, 2))
    affine_unbound = path[1].bound(source=None, target=None)
    assert affine_unbound == AffineTransform(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)))


def test_from_graph_edges_rejects_coordinate_system_and_other_types():
    world = CoordinateSystem.without_semantics("zyx")
    image = _multiscale(z=2, y=3, x=4)

    with pytest.raises(TypeError, match="Use CoordinateSystem.as_ref"):
        Scene.from_graph_edges([("some_string", ScaleTransform((2, 2, 2)), world)])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Use CoordinateSystem.as_ref"):
        Scene.from_graph_edges([(world, ScaleTransform((2, 2, 2)), image)])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Use CoordinateSystem.as_ref"):
        Scene.from_graph_edges([(image, ScaleTransform((2, 2, 2)), world)])  # type: ignore[arg-type]


def test_scene_from_tiles_translations_empty():
    scene = Scene.from_tiles_translations([])

    assert scene.transforms_between("world", "world") is None
    assert scene._internal_graph.transforms == ()


def test_scene_from_tiles_translations_multiple_multiscales():
    ms1 = _multiscale(z=2, y=3, x=4)
    ms2 = _multiscale(z=2, y=3, x=4)
    ms3 = _multiscale(z=2, y=3, x=4)

    t1 = Translation(z=0, y=0, x=0)
    t2 = Translation(z=0, y=512, x=0)
    t3 = Translation(z=0, y=0, x=512)

    scene = Scene.from_tiles_translations([(ms1, t1), (ms2, t2), (ms3, t3)])

    p1 = scene.transforms_between(ms1, "world")
    p2 = scene.transforms_between(ms2, "world")
    p3 = scene.transforms_between(ms3, "world")

    assert p1
    assert p2
    assert p3
    assert len(p1) == len(p2) == len(p3) == 1

    world_ref = p1[0].target
    assert world_ref != ms1._intrinsic_ref, "central reference system should be copy, not the exact object"
    assert isinstance(world_ref, NodeRef), "for pyright"
    assert world_ref.owner != ms1._intrinsic_ref.owner, "central reference system should be copy, not the exact object"
    assert p2[0].target == world_ref
    assert p3[0].target == world_ref

    assert p1[0] == TranslationTransform.from_translation(t1).bound(source=ms1._intrinsic_ref, target=world_ref)
    assert p2[0] == TranslationTransform.from_translation(t2).bound(source=ms2._intrinsic_ref, target=world_ref)
    assert p3[0] == TranslationTransform.from_translation(t3).bound(source=ms3._intrinsic_ref, target=world_ref)

    p13 = scene.transforms_between(ms1, ms3)
    assert p13
    assert len(p13) == 2


def test_scene_from_tiles_translations_rejects_mismatching_axes_across_multiscales():
    ms1 = _multiscale(z=2, y=3, x=4)
    ms2 = _multiscale(c=2, y=3, x=4)

    with pytest.raises(ValueError, match="identical axis keys"):
        Scene.from_tiles_translations(
            [
                (ms1, Translation.identity("zyx")),
                (ms2, Translation.identity("cyx")),
            ]
        )


def test_scene_from_tiles_translations_rejects_mismatching_axes_in_translations():
    ms1 = _multiscale(z=2, y=3, x=4)
    ms2 = _multiscale(z=2, y=3, x=4)

    with pytest.raises(ValueError, match="identical axis keys"):
        Scene.from_tiles_translations(
            [
                (ms1, Translation.identity("zyx")),
                (ms2, Translation.identity("cyx")),
            ]
        )


def test_transforms_between_accepts_path_addressed_unresolved_refs():
    world = CoordinateSystem.without_semantics("yx").as_ref("world")
    transform = TranslationTransform(
        translation=(1, 2),
        source=_UnresolvedRef(file=FileRef(path="tile_0"), name="physical"),
        target=world,
    )
    scene = Scene(TransformGraph([transform], system_refs=(world,)), _multiscale_paths={})

    result = scene.transforms_between({"path": "tile_0", "name": "physical"}, "world")

    assert result == [transform]


def test_transforms_between_can_include_child_multiscale_graphs():
    multiscale = _multiscale(y=2, x=3)
    world = CoordinateSystem.without_semantics("yx").as_ref("world")
    scene_transform = TranslationTransform(
        translation=(10, 20),
        source=multiscale.as_ref(multiscale._intrinsic_ref.name),
        target=world,
    )
    scene = Scene(TransformGraph([scene_transform], system_refs=(world,)), _multiscale_paths={"tile_0": multiscale})

    result = scene.transforms_between(multiscale, "world", include_children=True)

    assert result == [multiscale._get_interface_transform(), scene_transform]
