import copy
from typing import Dict, Any

from clearscale import Scene

from tests.ome_zarr.scene_examples import scene_stitching


def with_transform_names_removed(meta: Dict[str, Any]) -> Dict[str, Any]:
    removed = copy.deepcopy(meta)
    for t_dict in removed["coordinateTransformations"]:
        del t_dict["name"]
    return removed


def test_stitching_example_roundtrip(scene_stitching):
    scene = Scene.from_ome_zarr(scene_stitching)
    output_json = scene.to_ome_zarr(version="0.6.dev4")

    assert output_json == with_transform_names_removed(scene_stitching)
