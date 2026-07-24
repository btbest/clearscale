import pytest

from clearscale import Multiscale, PixelSize, Shape, Scale
from clearscale.ome_zarr import make_all_singleton_shapes, make_proportional_shapes, SUPPORTED_OME_ZARR_VERSIONS_READ

from tests.ome_zarr.multiscale_examples import (
    MultiscaleMetadataExample,
    minimal_multiscale_examples_params,
    maximal_multiscale_examples_params,
)

IGNORE_INVALID = "ignore:.*invalid"


def test_all_versions_covered():
    example_params = minimal_multiscale_examples_params()
    versions = [params.id for params in example_params]
    assert set(versions) == set(
        SUPPORTED_OME_ZARR_VERSIONS_READ
    ), "Add at least a minimal test example when adding support for new OME-Zarr versions"


@pytest.mark.parametrize("example", minimal_multiscale_examples_params())
# The 0.1, 0.2 and 0.3 examples don't specify their version, making their validity ambiguous - prod code warns
@pytest.mark.filterwarnings(IGNORE_INVALID)
def test_from_ome_zarr_parses_minimal_multiscale_examples(example: MultiscaleMetadataExample):
    multiscale = Multiscale.from_ome_zarr(example.metadata, shape_source=make_all_singleton_shapes(example.ndim))

    assert tuple(multiscale.keys()) == example.expected_paths


@pytest.mark.parametrize("example", maximal_multiscale_examples_params())
def test_from_ome_zarr_parses_maximal_multiscale_examples(example: MultiscaleMetadataExample):
    multiscale = Multiscale.from_ome_zarr(example.metadata, shape_source=make_all_singleton_shapes(example.ndim))

    assert tuple(multiscale.keys()) == example.expected_paths


def _0_4_metadata_with(**updates):
    meta = {
        "version": "0.4",
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]},
        ],
    }
    meta.update(updates)
    return meta


def _0_4_metadata_without_axes():
    metadata = _0_4_metadata_with()
    del metadata["axes"]
    return metadata


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param({"bioformats2raw.layout": 3}, id="missing-datasets"),
        pytest.param({"datasets": 3}, id="datasets-not-list"),
        pytest.param({"datasets": []}, id="empty-datasets"),
        pytest.param({"datasets": [{"noop": 0}]}, id="dataset-missing-path"),
        pytest.param({"datasets": [{"path": ""}]}, id="dataset-empty-path"),
        pytest.param({"datasets": [{"path": 0}]}, id="dataset-non-string-path"),
        pytest.param({"version": "0.4", "datasets": [{"path": "s0"}]}, id="missing-axes-explicit-0_4"),
        pytest.param(
            {"datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0]}]}]},
            id="missing-axes-implicit-0_4-or-newer",
        ),
        pytest.param(_0_4_metadata_with(axes="yx"), id="axes-not-list"),
        pytest.param(_0_4_metadata_with(axes=[]), id="empty-axes"),
        pytest.param(_0_4_metadata_with(axes=[{}]), id="axis-missing-name"),
        pytest.param(_0_4_metadata_with(axes=[3, {"name": "x"}]), id="axis-not-mapping"),
        pytest.param(_0_4_metadata_with(axes=[{"name": "y"}, {"name": "y"}]), id="duplicate-axis-names"),
        pytest.param(_0_4_metadata_with(datasets=[3]), id="dataset-not-mapping"),
        pytest.param(
            _0_4_metadata_with(
                datasets=[
                    {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]},
                    {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0]}]},
                ]
            ),
            id="duplicate-dataset-paths",
        ),
    ],
)
@pytest.mark.filterwarnings(IGNORE_INVALID)
def test_from_ome_zarr_raises_when_axes_or_paths_unknown_version_0_4(metadata):
    with pytest.raises(ValueError):
        _ = Multiscale.from_ome_zarr(metadata, shape_source=lambda path: (1, 2))


def _0_4_metadata_with_s0_transforms(transformations):
    return _0_4_metadata_with(datasets=[{"path": "s0", "coordinateTransformations": transformations}])


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(_0_4_metadata_with(datasets=[{"path": "s0"}]), id="missing-dataset-transformations"),
        pytest.param(_0_4_metadata_with_s0_transforms([]), id="empty-dataset-transformations"),
        pytest.param(_0_4_metadata_with_s0_transforms(3), id="dataset-transformations-not-list"),
        pytest.param(
            _0_4_metadata_with_s0_transforms([{"type": "scale"}]),
            id="scale-transform-missing-scale",
        ),
        pytest.param(
            _0_4_metadata_with_s0_transforms([{"type": "scale", "scale": "abc"}]),
            id="scale-transform-scale-not-numeric",
        ),
        pytest.param(
            _0_4_metadata_with_s0_transforms([{"type": "scale", "scale": []}]),
            id="scale-transform-empty-scale",
        ),
        pytest.param(
            _0_4_metadata_with_s0_transforms([{"type": "scale", "scale": [1.0]}]),
            id="scale-transform-wrong-dimensionality",
        ),
        pytest.param(
            _0_4_metadata_with_s0_transforms([{"type": "scale", "scale": [1.0, -1.0]}]),
            id="scale-transform-negative-scale",
        ),
        pytest.param(
            _0_4_metadata_with_s0_transforms(
                [{"type": "scale", "scale": [1.0, 1.0]}, {"type": "translation", "translation": [0.0]}]
            ),
            id="translation-transform-wrong-dimensionality",
        ),
        pytest.param(
            _0_4_metadata_with_s0_transforms([{"type": "affine", "affine": [1.0, 0.0, 0.0, 1.0]}]),
            id="transform-type-invalid-for-dataset",
        ),
        pytest.param(_0_4_metadata_with(coordinateTransformations=3), id="multiscale-transforms-not-list"),
        pytest.param(_0_4_metadata_with(coordinateTransformations=[]), id="multiscale-transforms-empty"),
        pytest.param(_0_4_metadata_with(coordinateTransformations=[3]), id="multiscale-transform-not-list"),
        pytest.param(_0_4_metadata_with(coordinateTransformations=[{}]), id="multiscale-transform-empty"),
        pytest.param(
            _0_4_metadata_with(coordinateTransformations=[{"type": "scale"}]),
            id="multiscale-transform-scale-missing-values",
        ),
        pytest.param(
            _0_4_metadata_with(coordinateTransformations=[{"type": "scale", "scale": "abc"}]),
            id="multiscale-transform-scale-not-list",
        ),
        pytest.param(
            _0_4_metadata_with(coordinateTransformations=[{"type": "scale", "scale": []}]),
            id="multiscale-transform-scale-empty",
        ),
        pytest.param(
            _0_4_metadata_with(coordinateTransformations=[{"type": "scale", "scale": [1.0]}]),
            id="multiscale-transform-scale-wrong-dimensionality",
        ),
        pytest.param(
            _0_4_metadata_with(coordinateTransformations=[{"type": "scale", "scale": [1.0, -1.0]}]),
            id="multiscale-transform-negative-scale",
        ),
        pytest.param(
            _0_4_metadata_with(
                coordinateTransformations=[
                    {"type": "scale", "scale": [1.0, 1.0]},
                    {"type": "translation", "translation": [0.0]},
                ]
            ),
            id="multiscale-transform-translation-wrong-dimensionality",
        ),
        pytest.param(
            _0_4_metadata_with(coordinateTransformations=[{"type": "affine", "affine": [1.0, 0.0, 1.0]}]),
            id="multiscale-transform-type-invalid-for-0_4",
        ),
    ],
)
@pytest.mark.filterwarnings(IGNORE_INVALID)
def test_from_ome_zarr_ignores_invalid_transforms_metadata_version_0_4(metadata):
    read = Multiscale.from_ome_zarr(metadata, shape_source=lambda path: (1, 2))
    expected = Multiscale({"s0": Scale(shape=Shape(y=1, x=2), pixel_size=PixelSize(y=1.0, x=1.0))})
    assert read == expected


def _0_6_dev4_metadata_with(**updates):
    metadata = {
        "coordinateSystems": [
            {
                "name": "physical",
                "axes": [{"name": "y"}, {"name": "x"}],
            }
        ],
        "datasets": [
            {
                "path": "s0",
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0],
                        "input": {"path": "s0"},
                        "output": {"name": "physical"},
                    }
                ],
            },
        ],
    }
    metadata.update(updates)
    return metadata


def _0_6_dev4_metadata_with_axes(axes):
    return _0_6_dev4_metadata_with(coordinateSystems=[{"name": "physical", "axes": axes}])


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(_0_6_dev4_metadata_with(coordinateSystems="yx"), id="coord-sys-not-list"),
        pytest.param(_0_6_dev4_metadata_with(coordinateSystems=[]), id="empty-coord-sys"),
        pytest.param(_0_6_dev4_metadata_with(coordinateSystems=[3]), id="coord-sys-not-mapping"),
        pytest.param(_0_6_dev4_metadata_with(coordinateSystems=[{}]), id="empty-coord-sys-mapping"),
        pytest.param(_0_6_dev4_metadata_with(coordinateSystems=[{"name": "physical"}]), id="axes-missing"),
        pytest.param(_0_6_dev4_metadata_with_axes("yx"), id="axes-not-list"),
        pytest.param(_0_6_dev4_metadata_with_axes([]), id="empty-axes"),
        pytest.param(_0_6_dev4_metadata_with_axes([{}]), id="axis-missing-name"),
        pytest.param(_0_6_dev4_metadata_with_axes([3, {"name": "x"}]), id="axis-not-mapping"),
        pytest.param(_0_6_dev4_metadata_with_axes([{"name": "y"}, {"name": "y"}]), id="duplicate-axis-names"),
        pytest.param(_0_6_dev4_metadata_with(datasets=[3]), id="dataset-not-mapping"),
        pytest.param(
            _0_6_dev4_metadata_with(
                datasets=[
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 1.0],
                                "input": {"path": "s0"},
                                "output": {"name": "physical"},
                            }
                        ],
                    },
                    {
                        "path": "s0",
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [2.0, 2.0],
                                "input": {"path": "s0"},
                                "output": {"name": "physical"},
                            }
                        ],
                    },
                ]
            ),
            id="duplicate-dataset-paths",
        ),
    ],
)
@pytest.mark.filterwarnings(IGNORE_INVALID)
def test_from_ome_zarr_raises_when_axes_or_paths_unknown_version_0_6_dev4(metadata):
    with pytest.raises(ValueError):
        _ = Multiscale.from_ome_zarr(metadata, shape_source=lambda path: (1, 2))


def _0_6_dev4_metadata_with_s0_transforms(transformations):
    return _0_6_dev4_metadata_with(datasets=[{"path": "s0", "coordinateTransformations": transformations}])


def _0_6_dev4_metadata_with_s0_scale(scale):
    return _0_6_dev4_metadata_with_s0_transforms(
        [{"type": "scale", "scale": scale, "input": {"path": "s0"}, "output": {"name": "physical"}}]
    )


def _0_6_dev4_metadata_with_labels_transform_with(**updates):
    transform = {"input": {"name": "physical", "path": "labels/nuclei"}, "output": {"name": "physical"}}
    transform.update(updates)
    return _0_6_dev4_metadata_with(coordinateTransformations=[transform])


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(_0_6_dev4_metadata_with(datasets=[{"path": "s0"}]), id="missing-dataset-transformations"),
        pytest.param(_0_6_dev4_metadata_with_s0_transforms([]), id="empty-dataset-transformations"),
        pytest.param(_0_6_dev4_metadata_with_s0_transforms(3), id="dataset-transformations-not-list"),
        pytest.param(
            _0_6_dev4_metadata_with_s0_transforms(
                [{"type": "scale", "input": {"path": "s0"}, "output": {"name": "physical"}}]
            ),
            id="scale-transform-missing-scale",
        ),
        pytest.param(_0_6_dev4_metadata_with_s0_scale("abc"), id="scale-transform-scale-not-numeric"),
        pytest.param(_0_6_dev4_metadata_with_s0_scale([]), id="scale-transform-empty-scale"),
        pytest.param(_0_6_dev4_metadata_with_s0_scale([1.0]), id="scale-transform-wrong-dimensionality"),
        pytest.param(_0_6_dev4_metadata_with_s0_scale([1.0, -1.0]), id="scale-transform-negative-scale"),
        pytest.param(
            _0_6_dev4_metadata_with_s0_transforms(
                [
                    {
                        "type": "sequence",
                        "input": {"path": "s0"},
                        "output": {"name": "physical"},
                        "transformations": [
                            {"type": "scale", "scale": [1.0, 1.0]},
                            {"type": "translation", "translation": [0.0]},
                        ],
                    }
                ]
            ),
            id="translation-transform-wrong-dimensionality",
        ),
        pytest.param(
            _0_6_dev4_metadata_with_s0_transforms(
                [
                    {
                        "type": "affine",
                        "affine": [1.0, 0.0, 0.0, 1.0],
                        "input": {"path": "s0"},
                        "output": {"name": "physical"},
                    }
                ]
            ),
            id="transform-type-invalid-for-dataset",
        ),
        pytest.param(_0_6_dev4_metadata_with(coordinateTransformations=3), id="multiscale-transforms-not-list"),
        pytest.param(_0_6_dev4_metadata_with(coordinateTransformations=[]), id="multiscale-transforms-empty"),
        pytest.param(_0_6_dev4_metadata_with(coordinateTransformations=[3]), id="multiscale-transform-not-list"),
        pytest.param(_0_6_dev4_metadata_with(coordinateTransformations=[{}]), id="multiscale-transform-empty"),
        pytest.param(
            _0_6_dev4_metadata_with_labels_transform_with(type="scale"), id="multiscale-transform-scale-missing-values"
        ),
        pytest.param(
            _0_6_dev4_metadata_with_labels_transform_with(type="scale", scale="abc"),
            id="multiscale-transform-scale-not-list",
        ),
        pytest.param(
            _0_6_dev4_metadata_with_labels_transform_with(type="scale", scale=[]), id="multiscale-transform-scale-empty"
        ),
        pytest.param(
            _0_6_dev4_metadata_with_labels_transform_with(type="scale", scale=[1.0]),
            id="multiscale-transform-scale-wrong-dimensionality",
        ),
        pytest.param(
            _0_6_dev4_metadata_with_labels_transform_with(type="scale", scale=[1.0, -1.0]),
            id="multiscale-transform-negative-scale",
        ),
        pytest.param(
            _0_6_dev4_metadata_with_labels_transform_with(
                type="sequence",
                transformations=[{"type": "scale", "scale": [1.0, 1.0]}, {"type": "translation", "translation": [0.0]}],
            ),
            id="multiscale-transform-translation-wrong-dimensionality",
        ),
        pytest.param(
            _0_6_dev4_metadata_with_labels_transform_with(
                type="displacements", path="coordinateTransformations/displacements1"
            ),
            id="multiscale-transform-type-not-supported-by-clearscale",
        ),
    ],
)
@pytest.mark.filterwarnings(IGNORE_INVALID)
def test_from_ome_zarr_ignores_invalid_transforms_metadata_version_0_6_dev4(metadata):
    read = Multiscale.from_ome_zarr(metadata, shape_source=lambda path: (1, 2))
    expected = Multiscale({"s0": Scale(shape=Shape(y=1, x=2))})
    assert read == expected


def test_from_ome_zarr_accepts_shape_mapping():
    metadata = {
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]},
            {"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0]}]},
        ],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": (100, 200), "s1": (50, 100)})

    assert multiscale["s0"].shape == Shape(y=100, x=200)
    assert multiscale["s1"].shape == Shape(y=50, x=100)


def test_from_ome_zarr_normalizes_zero_scale_values():
    metadata = {
        "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [0.0, 1.0, 1.0]}]},
        ],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": (3, 100, 200)})

    assert multiscale["s0"].pixel_size == PixelSize(c=1.0, y=1.0, x=1.0)


def test_from_ome_zarr_normalizes_zero_scale_values_with_proportional_shape_source():
    metadata = {
        "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}],
        "datasets": [
            {"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [0.0, 1.0, 1.0]}]},
            {"path": "s1", "coordinateTransformations": [{"type": "scale", "scale": [0.0, 2.0, 2.0]}]},
        ],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source=make_proportional_shapes(metadata))

    assert tuple(multiscale.keys()) == ("s0", "s1")
    assert multiscale["s0"].pixel_size == PixelSize(c=1.0, y=1.0, x=1.0)
    assert multiscale["s1"].pixel_size == PixelSize(c=1.0, y=2.0, x=2.0)


def test_from_ome_zarr_accepts_array_mapping():
    class Array:
        def __init__(self, shape):
            self.shape = shape

    metadata = {
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": Array((100, 200))})

    assert multiscale["s0"].shape == Shape(y=100, x=200)


def test_from_ome_zarr_accepts_shape_values():
    metadata = {
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
    }

    multiscale = Multiscale.from_ome_zarr(metadata, shape_source={"s0": Shape(y=100, x=200)})

    assert multiscale["s0"].shape == Shape(y=100, x=200)


def test_from_ome_zarr_rejects_plain_string_shape_source():
    metadata = {
        "axes": [{"name": "y"}, {"name": "x"}],
        "datasets": [{"path": "s0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0]}]}],
    }

    with pytest.raises(TypeError, match="Cannot obtain array shape from plain path"):
        Multiscale.from_ome_zarr(metadata, shape_source="image.ome.zarr")  # type: ignore
