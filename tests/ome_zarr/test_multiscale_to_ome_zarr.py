import copy
from typing import Any

import pytest
from clearscale import Multiscale
from clearscale.ome_zarr import make_all_singleton_shapes, SUPPORTED_OME_ZARR_VERSIONS_WRITE

from tests.ome_zarr.multiscale_examples import (
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
    MultiscaleMetadataExample,
    maximal_multiscale_example,
)

known_keys_that_should_roundtrip_but_todo = ("type", "labels", "omero", "metadata")
float_roundtrip_abs_tolerance = 2**-54


def with_written_version(metadata: dict[str, Any], version: str) -> dict[str, Any]:
    if "version" in metadata:
        assert metadata["version"] == version
    return metadata | {"version": version}


def with_approximate_floats(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: with_approximate_floats(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [with_approximate_floats(inner_value) for inner_value in value]
    if isinstance(value, float):
        return pytest.approx(value, rel=0, abs=float_roundtrip_abs_tolerance)
    return value


def without_known_feature_gaps(metadata: dict[str, Any]) -> dict[str, Any]:
    round_trippable_metadata = copy.deepcopy(metadata)
    for key in known_keys_that_should_roundtrip_but_todo:
        del round_trippable_metadata[key]
    return round_trippable_metadata


@pytest.fixture
def maximal_ome_zarr_0_6_dev4() -> MultiscaleMetadataExample:
    return maximal_multiscale_example("0.6.dev4")


@pytest.mark.filterwarnings("ignore:.*not in OME-Zarr canonical order.*:UserWarning")
@pytest.mark.parametrize("example", minimal_multiscale_examples_params())
def test_multiscale_roundtrips_minimal_ome_zarr(example: MultiscaleMetadataExample):
    if example.id not in SUPPORTED_OME_ZARR_VERSIONS_WRITE:
        pytest.skip(f"Writing version {example.id} not supported")
    multiscale = Multiscale.from_ome_zarr(example.metadata, shape_source=make_all_singleton_shapes(example.ndim))
    output_json = multiscale.to_ome_zarr(version=example.id)

    expected_output = with_written_version(example.metadata, example.id)
    assert output_json == expected_output


@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_multiscale_roundtrips_maximal_ome_zarr(example: MultiscaleMetadataExample):
    if example.id not in SUPPORTED_OME_ZARR_VERSIONS_WRITE:
        pytest.skip(f"Writing version {example.id} not supported")
    multiscale = Multiscale.from_ome_zarr(example.metadata, shape_source=make_all_singleton_shapes(example.ndim))
    output_json = multiscale.to_ome_zarr(version=example.id)

    for key in known_keys_that_should_roundtrip_but_todo:
        assert key not in output_json, "Update test when implementing round-trip for previously unsupported optionals"
    expected_output = with_written_version(without_known_feature_gaps(example.metadata), example.id)
    if example.id in ("0.4", "0.5"):
        # We only guarantee approximate roundtrip of
        # `multiscale[coordinateTransformations]` for legacy versions.
        # - We are not aware of any implementations that use this key.
        # - Its semantic meaning is undefined in these versions.
        # - The spec requires `multiscale[coordinateTransformations]`
        #   be composed with `dataset[coordinateTransformations]`.
        # - Which means we can only decompose and recover it to float precision when writing.
        assert output_json == with_approximate_floats(expected_output)
    else:
        assert output_json == expected_output


def test_multiscale_roundtrip_preserves_coordinate_system_order(
    maximal_ome_zarr_0_6_dev4: MultiscaleMetadataExample,
):
    metadata = maximal_ome_zarr_0_6_dev4.metadata
    metadata["coordinateSystems"] = list(reversed(metadata["coordinateSystems"]))

    multiscale = Multiscale.from_ome_zarr(
        metadata, shape_source=make_all_singleton_shapes(maximal_ome_zarr_0_6_dev4.ndim)
    )
    output_json = multiscale.to_ome_zarr(version="0.6.dev4")

    expected_output = with_written_version(without_known_feature_gaps(metadata), "0.6.dev4")
    assert output_json == with_approximate_floats(expected_output)


def test_multiscale_roundtrips_zero_scale_values_from_ome_zarr():
    metadata = {
        "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [0.0, 1.0, 1.0]}]},
        ],
    }
    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": (3, 100, 200)})

    output_json = multiscale.to_ome_zarr(version="0.5")

    assert output_json["datasets"][0]["coordinateTransformations"] == [{"type": "scale", "scale": [0.0, 1.0, 1.0]}]


def test_new_multiscale_writes_normalized_zero_scale_values():
    metadata = {
        "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [0.0, 1.0, 1.0]}]},
        ],
    }
    parsed = Multiscale.from_ome_zarr(metadata, shape_source={"s0": (3, 100, 200)})
    normalized = Multiscale({"s0": parsed["s0"]})

    output_json = normalized.to_ome_zarr(version="0.5")

    assert output_json["datasets"][0]["coordinateTransformations"] == [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]
