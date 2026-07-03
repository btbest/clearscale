from clearscale import Multiscale, Scale, Scene, Shape
from clearscale._transforms import CoordinateSystem, TranslationTransform, TransformGraph, _UnresolvedRef


def test_transforms_between_accepts_path_addressed_unresolved_refs():
    world = CoordinateSystem.without_semantics("yx").as_ref("world")
    transform = TranslationTransform(
        translation=(1, 2),
        source=_UnresolvedRef(path="tile_0", name="physical"),
        target=world,
    )
    scene = Scene(TransformGraph([transform], system_refs=(world,)), _multiscale_paths={})

    result = scene.transforms_between({"path": "tile_0", "name": "physical"}, "world")

    assert result == [transform]


def test_transforms_between_can_include_child_multiscale_graphs():
    multiscale = Multiscale({"s0": Scale(Shape(y=2, x=3))})
    world = CoordinateSystem.without_semantics("yx").as_ref("world")
    scene_transform = TranslationTransform(
        translation=(10, 20),
        source=multiscale.as_ref(multiscale._intrinsic_ref.name),
        target=world,
    )
    scene = Scene(TransformGraph([scene_transform], system_refs=(world,)), _multiscale_paths={"tile_0": multiscale})

    result = scene.transforms_between(multiscale, "world", include_children=True)

    assert result == [multiscale._get_interface_transform(), scene_transform]
