import pytest
from clearscale._scene import Scene
from clearscale._collections import OmeZarrGroup
from clearscale.ome_zarr import make_all_singleton_shapes

from tests.ome_zarr.multiscale_examples import (
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
    MultiscaleMetadataExample,
)
from tests.ome_zarr.scene_examples import (
    all_invalid_scene_examples,
    all_valid_scene_examples,
    scene_registration,
    scene_stitching,
    scene_to_group_attrs,
)


class MockZarrGroup:
    def __init__(self, attrs, shape_source=None):
        self.attrs = attrs
        self.shape_source = shape_source

    def __getitem__(self, path: str):
        if self.shape_source is None:
            raise ValueError("provide shape_source for tests that need group[] indexing")
        return self.shape_source(path)


@pytest.mark.parametrize("example", minimal_multiscale_examples_params())
def test_ome_zarr_group_parses_minimal_multiscale_examples(example: MultiscaleMetadataExample):
    zarr_group = MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
    ome_group = OmeZarrGroup(zarr_group)

    assert len(ome_group.multiscales) == 1
    assert tuple(ome_group.multiscales[0].keys()) == example.expected_paths
    if "version" in example.metadata:
        assert ome_group.version == example.id


@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_ome_zarr_group_parses_maximal_multiscale_examples(example: MultiscaleMetadataExample):
    zarr_group = MockZarrGroup(example.to_group_attrs(), make_all_singleton_shapes(example.ndim))
    ome_group = OmeZarrGroup(zarr_group)

    assert len(ome_group.multiscales) == 1
    assert tuple(ome_group.multiscales[0].keys()) == example.expected_paths
    assert ome_group.version == example.id


def test_ome_zarr_group_ignores_invalid_multiscale():
    invalid_meta = {"multiscales": [{"datasets": [{"malformed": "idk"}]}]}
    zarr_group = MockZarrGroup(invalid_meta, make_all_singleton_shapes(1))
    ome_group = OmeZarrGroup(zarr_group)

    assert len(ome_group.multiscales) == 0
    assert ome_group.version is None


def test_ome_zarr_group_parses_scene_stitching_example():
    zarr_group = MockZarrGroup(scene_to_group_attrs(scene_stitching()))
    ome_group = OmeZarrGroup(zarr_group)

    assert len(ome_group.scenes) == 1
    assert ome_group.scenes[0].unresolved_paths == ["tile_0", "tile_1", "tile_2", "tile_3"]
    assert ome_group.version == "0.6.rc0"


def test_ome_zarr_group_parses_scene_registration_example():
    zarr_group = MockZarrGroup(scene_to_group_attrs(scene_registration()))
    ome_group = OmeZarrGroup(zarr_group)

    assert len(ome_group.scenes) == 1
    assert ome_group.scenes[0].unresolved_paths == ["JRC2018F", "FCWB"]
    assert ome_group.version == "0.6.rc0"


@pytest.mark.parametrize("meta", all_valid_scene_examples())
def test_ome_zarr_group_parses_scene_public_examples(meta):
    zarr_group = MockZarrGroup(scene_to_group_attrs(meta))
    ome_group = OmeZarrGroup(zarr_group)

    assert len(ome_group.scenes) == 1
    assert isinstance(ome_group.scenes[0], Scene)
    assert ome_group.version == "0.6.rc0"


@pytest.mark.parametrize("meta", all_invalid_scene_examples())
def test_ome_zarr_group_ignores_scene_invalid_examples(meta):
    zarr_group = MockZarrGroup(scene_to_group_attrs(meta))
    ome_group = OmeZarrGroup(zarr_group)

    assert not ome_group.scenes
    assert ome_group.version == "0.6.rc0"
