import copy
from dataclasses import dataclass
from typing import Any

import pytest
from clearscale.ome_zarr import SUPPORTED_OME_ZARR_VERSIONS_READ


@dataclass(frozen=True, slots=True)
class MultiscaleMetadataExample:
    id: str
    metadata: dict[str, Any]
    ndim: int

    def __post_init__(self):
        assert self.id in SUPPORTED_OME_ZARR_VERSIONS_READ, "Examples should use version of the metadata as ID"

    @property
    def expected_paths(self) -> tuple[str, ...]:
        return tuple(dataset["path"] for dataset in self.metadata["datasets"])


ALL_CANONICAL_AXES = [
    {"name": "t", "type": "time", "unit": "millisecond"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]

SCALING_METHOD_EXAMPLE = {
    "method": "skimage.transform.pyramid_gaussian",
    "version": "0.16.1",
    "args": "[True]",
    "kwargs": {"multichannel": True},
}

OMERO_EXAMPLE = {
    "id": 1,
    "name": "example.tif",
    "version": "0.3",
    "channels": [
        {
            "active": True,
            "coefficient": 1,
            "color": "0000FF",
            "family": "linear",
            "inverted": False,
            "label": "LaminB1",
            "window": {
                "end": 1500,
                "max": 65535,
                "min": 0,
                "start": 0,
            },
        }
    ],
    "rdefs": {
        "defaultT": 0,
        "defaultZ": 118,
        "model": "color",
    },
}

OME_ZARR_MIN_MS_0_1 = {"datasets": [{"path": "s0"}]}
OME_ZARR_MIN_MS_0_2 = {"datasets": [{"path": "s0"}]}
OME_ZARR_MIN_MS_0_3 = {"axes": ["x", "y"], "datasets": [{"path": "s0"}]}
OME_ZARR_MIN_MS_0_4 = {
    "axes": [{"name": "x"}, {"name": "y"}],
    "datasets": [
        {
            "path": "s0",
            "coordinateTransformations": [{"type": "scale", "scale": [0.2, 0.2]}],
        }
    ],
}
OME_ZARR_MIN_MS_0_5 = OME_ZARR_MIN_MS_0_4
OME_ZARR_MIN_MS_0_6_RC0 = {
    "coordinateSystems": [
        {
            "name": "physical",
            "axes": [
                {"name": "x", "type": "space"},
                {"name": "y", "type": "space"},
            ],
        }
    ],
    "datasets": [
        {
            "path": "s0",
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [0.2, 0.2],
                    "input": {"path": "s0"},
                    "output": {"name": "physical"},
                }
            ],
        }
    ],
}

OME_ZARR_MAX_MS_0_1 = {
    "version": "0.1",
    "datasets": [{"path": "s0"}, {"path": "s1"}],
    "labels": ["nuclei", "cells"],
}
OME_ZARR_MAX_MS_0_2 = {
    "version": "0.2",
    "name": "example",
    "datasets": [{"path": "s0"}, {"path": "s1"}],
    "labels": ["nuclei", "cells"],
    "type": "gaussian",
    "metadata": SCALING_METHOD_EXAMPLE,
    "omero": OMERO_EXAMPLE,
}
OME_ZARR_MAX_MS_0_3 = {
    "version": "0.3",
    "name": "example",
    "axes": ["t", "c", "z", "y", "x"],
    "datasets": [{"path": "s0"}, {"path": "s1"}],
    "labels": ["nuclei", "cells"],
    "type": "gaussian",
    "metadata": SCALING_METHOD_EXAMPLE,
    "omero": OMERO_EXAMPLE,
}
OME_ZARR_MAX_MS_0_4 = {
    "version": "0.4",
    "axes": ALL_CANONICAL_AXES,
    "datasets": [
        {
            "path": "s0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.2, 0.2]},
            ],
        },
        {
            "path": "s1",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.4, 0.4]},
                {"type": "translation", "translation": [0.0, 0.0, 0.0, 0.2, 0.2]},
            ],
        },
    ],
    "coordinateTransformations": [
        {"type": "scale", "scale": [120.0, 1.0, 1.0, 1.0, 1.0]},
        {"type": "translation", "translation": [0.0, 0.0, 0.0, 1.4, 1.4]},
    ],
    "labels": ["nuclei", "cells"],
    "type": "gaussian",
    "metadata": SCALING_METHOD_EXAMPLE,
    "omero": OMERO_EXAMPLE,
}
OME_ZARR_MAX_MS_0_5 = {key: value for key, value in OME_ZARR_MAX_MS_0_4.items() if key != "version"}
OME_ZARR_MAX_MS_0_6_RC0 = {
    "coordinateSystems": [
        {
            "name": "physical",
            "axes": ALL_CANONICAL_AXES,
        },
        {
            "name": "renamed",
            "axes": [
                {"name": "t", "type": "time", "unit": "millisecond"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "i", "type": "space", "unit": "micrometer"},
                {"name": "j", "type": "space", "unit": "micrometer"},
            ],
        },
    ],
    "datasets": [
        {
            "path": "s0",
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [120.0, 1.0, 1.0, 0.2, 0.2],
                    "input": {"path": "s0"},
                    "output": {"name": "physical"},
                }
            ],
        },
        {
            "path": "s1",
            "coordinateTransformations": [
                {
                    "type": "sequence",
                    "transformations": [
                        {
                            "type": "scale",
                            "scale": [120.0, 1.0, 1.0, 0.4, 0.4],
                        },
                        {
                            "type": "translation",
                            "translation": [0.0, 0.0, 0.0, 0.2, 0.2],
                        },
                    ],
                    "input": {"path": "s1"},
                    "output": {"name": "physical"},
                }
            ],
        },
    ],
    "coordinateTransformations": [
        {
            "type": "sequence",
            "transformations": [
                {
                    "type": "translation",
                    "translation": [0.0, 0.0, 0.0, 1.4, 1.4],
                },
            ],
            "input": {"name": "physical"},
            "output": {"name": "external"},
        },
        {
            "type": "identity",
            "input": {"name": "physical"},
            "output": {"name": "renamed"},
        },
    ],
    "labels": ["nuclei", "cells"],
    "type": "gaussian",
    "metadata": SCALING_METHOD_EXAMPLE,
    "omero": OMERO_EXAMPLE,
}

_MINIMAL_MULTISCALE_EXAMPLES = (
    MultiscaleMetadataExample("0.1", OME_ZARR_MIN_MS_0_1, ndim=5),
    MultiscaleMetadataExample("0.2", OME_ZARR_MIN_MS_0_2, ndim=5),
    MultiscaleMetadataExample("0.3", OME_ZARR_MIN_MS_0_3, ndim=2),
    MultiscaleMetadataExample("0.4", OME_ZARR_MIN_MS_0_4, ndim=2),
    MultiscaleMetadataExample("0.5", OME_ZARR_MIN_MS_0_5, ndim=2),
    MultiscaleMetadataExample("0.6.rc0", OME_ZARR_MIN_MS_0_6_RC0, ndim=2),
)

_MAXIMAL_MULTISCALE_EXAMPLES = (
    MultiscaleMetadataExample("0.1", OME_ZARR_MAX_MS_0_1, ndim=5),
    MultiscaleMetadataExample("0.2", OME_ZARR_MAX_MS_0_2, ndim=5),
    MultiscaleMetadataExample("0.3", OME_ZARR_MAX_MS_0_3, ndim=5),
    MultiscaleMetadataExample("0.4", OME_ZARR_MAX_MS_0_4, ndim=5),
    MultiscaleMetadataExample("0.5", OME_ZARR_MAX_MS_0_5, ndim=5),
    MultiscaleMetadataExample("0.6.rc0", OME_ZARR_MAX_MS_0_6_RC0, ndim=5),
)


def _copied_example(example: MultiscaleMetadataExample) -> MultiscaleMetadataExample:
    return MultiscaleMetadataExample(example.id, copy.deepcopy(example.metadata), ndim=example.ndim)


def minimal_multiscale_examples():
    return [_copied_example(example) for example in _MINIMAL_MULTISCALE_EXAMPLES]


def maximal_multiscale_examples():
    return [_copied_example(example) for example in _MAXIMAL_MULTISCALE_EXAMPLES]


def maximal_multiscale_example(version: str) -> MultiscaleMetadataExample:
    for example in _MAXIMAL_MULTISCALE_EXAMPLES:
        if example.id == version:
            return _copied_example(example)
    raise ValueError(f"No maximal multiscale example for OME-Zarr version {version!r}")


def minimal_multiscale_examples_params():
    return [pytest.param(example, id=example.id) for example in minimal_multiscale_examples()]


def maximal_multiscale_examples_params():
    return [pytest.param(example, id=example.id) for example in maximal_multiscale_examples()]
