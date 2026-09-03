import copy
from typing import Any

import pytest
from clearscale import Multiscale, Translation
from clearscale.ome_zarr import make_all_singleton_shapes, SUPPORTED_OME_ZARR_VERSIONS_WRITE

from tests.ome_zarr.multiscale_examples import (
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
    MultiscaleMetadataExample,
    maximal_multiscale_example,
)

known_keys_that_should_roundtrip_but_todo = ("omero",)
float_roundtrip_abs_tolerance = 2**-54


def with_written_version(metadata: dict[str, Any], version: str) -> dict[str, Any]:
    if "version" in metadata:
        assert metadata["version"] == version
    return metadata | {"version": version}


def with_approximate_floats(value: Any) -> Any:
    """Recurse through `value`, replacing all floats with pytest.approx"""
    if isinstance(value, dict):
        return {key: with_approximate_floats(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [with_approximate_floats(inner_value) for inner_value in value]
    if isinstance(value, float):
        return pytest.approx(value, rel=0, abs=float_roundtrip_abs_tolerance)
    return value


def without_identity_translations_datasets(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove identity translations in
    meta['datasets'][]['coordinateTransformations']
    but not in
    meta['coordinateTransformations'].
    The former do not roundtrip because even in older OME-Zarr versions, their meaning is well-defined, and
    meta['datasets'][]['coordinateTransformations'] is always required to exist. The presence of identity translations
    is likely to just be laziness of the writer implementation (always writing translations rather than special-casing
    when translation is 0). Normalising by removing identity translations is acceptable here.
    The latter *do* roundtrip because in older OME-Zarr versions, the meaning of meta['coordinateTransformations'] was
    not defined, and was not required to be written. Presumably, if another writer made the specific effort to add
    this, and wrote a zero-translation, it had some special reason.

    If we want to round-trip identity translations in meta['datasets'][]['coordinateTransformations'],
    Multiscale.from_ome_zarr needs a special hidden list of which datasets had zero-translations, so that
    Multiscale.to_ome_zarr (or specifically the _services.ome_zarr.build_dataset_dict helper)
    can find out where to include them even if Multiscale[].translation.is_identity() is True.
    """
    metadata = copy.deepcopy(metadata)
    for dataset in metadata.get("datasets", []):
        if "coordinateTransformations" not in dataset or not isinstance(dataset["coordinateTransformations"], list):
            continue
        dataset["coordinateTransformations"] = [
            transform
            for transform in dataset["coordinateTransformations"]
            if not (transform.get("type") == "translation" and all(v == 0 for v in transform.get("translation", ())))
        ]
    return metadata


def without_known_feature_gaps(metadata: dict[str, Any]) -> dict[str, Any]:
    round_trippable_metadata = copy.deepcopy(metadata)
    for key in known_keys_that_should_roundtrip_but_todo:
        if key in round_trippable_metadata:
            del round_trippable_metadata[key]
    return round_trippable_metadata


@pytest.fixture
def maximal_ome_zarr_0_6_rc0() -> MultiscaleMetadataExample:
    return maximal_multiscale_example("0.6.rc0")


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
    expected_output = without_identity_translations_datasets(
        with_written_version(without_known_feature_gaps(example.metadata), example.id)
    )
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
    maximal_ome_zarr_0_6_rc0: MultiscaleMetadataExample,
):
    metadata = maximal_ome_zarr_0_6_rc0.metadata
    metadata["coordinateSystems"] = list(reversed(metadata["coordinateSystems"]))

    multiscale = Multiscale.from_ome_zarr(
        metadata, shape_source=make_all_singleton_shapes(maximal_ome_zarr_0_6_rc0.ndim)
    )
    output_json = multiscale.to_ome_zarr(version="0.6.rc0")

    expected_output = with_written_version(without_known_feature_gaps(metadata), "0.6.rc0")
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


def test_multiscale_roundtrip_folds_global_translation_into_scales_if_global_t_convention():
    """
    The convention implies that the multiscale's intrinsic system is the *output* of the global transforms:
    The axes would specify "time unit=seconds", but the dataset scales would have `1.0` t-scale. Only the
    global scale has the correct `12.0` t-scale. So "the coordinate system in which the scale meta is correct" is
    the system after applying *both* the dataset transforms and the global transforms.
    The only consistent interpretation of a global translation then is that it is part of the scale translation.
    This is an edge case: There are no known public cases that use both the "global t-scale" convention, and also
    put a global translation behind that global scale.

    We could identify the common portion of all scales' translation and extract
    that to put it on the global level. But: The normal case for scale translation is that the first downscale has
    `half pixel size along scaled axes`, and the subsequent ones have more. This means that if the raw data scale is missing,
    we would make it normal behaviour to extract a global translation next to the global t-scale. We would be
    practically guaranteed to create non-conventional metadata when deriving from a Multiscale that follows the convention,
    when omitting the raw scale.

    It's more likely to occur that someone is working with nifti-zarr (which uses this convention) and then might only
    continue working with downsampled data, needing to maintain nifti-zarr.

    Hence, accept the non-round-trip. If we see "global t-scale" convention, plus a global translation, compose it into
    every scale. But do not decompose it back out on write, to ensure written outputs follow the convention.
    """
    metadata = {
        "axes": [
            {"name": "t", "type": "time", "unit": "seconds"},
            {"name": "y", "type": "space", "unit": "nanometers"},
            {"name": "x", "type": "space", "unit": "nanometers"},
        ],
        "datasets": [
            # Raw scale missing (it would have zeros in its translation)
            {
                "path": "s1",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 40.0, 40.0]},
                    {"type": "translation", "translation": [0.0, 10.0, 10.0]},
                ],
            },
            {
                "path": "s2",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 80.0, 80.0]},
                    {"type": "translation", "translation": [0.0, 30.0, 30.0]},
                ],
            },
        ],
        "coordinateTransformations": [
            {"type": "scale", "scale": [3.4, 1.0, 1.0]},
            {
                "type": "translation",
                "translation": [1.0, 2.0, 3.0],
            },  # Unconventional - global translation meaning is undefined
        ],
    }

    parsed = Multiscale.from_ome_zarr(metadata, shape_source={"s1": (5, 100, 200), "s2": (5, 50, 100)})

    assert parsed["s1"].translation == Translation(t=1.0, y=12.0, x=13.0)  # dataset + global sum

    output_json = parsed.to_ome_zarr(version="0.4")

    assert output_json["coordinateTransformations"][0]["scale"] == [3.4, 1.0, 1.0]
    assert len(output_json["coordinateTransformations"]) == 1, "global translation not expected to roundtrip"
    assert output_json["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 40.0, 40.0]
    # The global t-translation needs to be divided by the global t-scale when folding it into dataset translation.
    # `dataset_scale * dataset_translation * global_scale + global_translation` must be eq before and after write.
    # In the input, that formula is:
    # 1.0 * 0.0 * 3.4 + 1.0 = 1.0
    # In the output, it becomes:
    # 1.0 / 3.4 * 1.0 * 3.4 + 0.0 = 1.0
    expected_s1_translation = [1.0 / 3.4, 12.0, 13.0]
    assert output_json["datasets"][0]["coordinateTransformations"][1]["translation"] == expected_s1_translation
