from clearscale import Scene

from tests.ome_zarr.scene_examples import scene_stitching


def test_stitching_example_roundtrip(scene_stitching):
    scene = Scene.from_ome_zarr(scene_stitching)
    output_json = scene.to_ome_zarr(version="0.6.rc0")

    assert output_json == scene_stitching
