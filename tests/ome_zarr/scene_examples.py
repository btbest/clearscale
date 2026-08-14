import copy
from typing import Mapping, Any, Dict

STITCHING = {
    "coordinateTransformations": [
        {
            "type": "translation",
            "output": {"name": "world"},
            "input": {"path": "tile_0", "name": "physical"},
            "translation": [0, 0],
            "name": "tile_0_mm to world",
        },
        {
            "type": "translation",
            "output": {"name": "world"},
            "input": {"path": "tile_1", "name": "physical"},
            "translation": [0, 348],
            "name": "tile_1_mm to world",
        },
        {
            "type": "translation",
            "output": {"name": "world"},
            "input": {"path": "tile_2", "name": "physical"},
            "translation": [276, 0],
            "name": "tile_2_mm to world",
        },
        {
            "type": "translation",
            "output": {"name": "world"},
            "input": {"path": "tile_3", "name": "physical"},
            "translation": [276, 348],
            "name": "tile_3_mm to world",
        },
    ],
    "coordinateSystems": [
        {
            "name": "world",
            "axes": [
                {"type": "space", "name": "x", "unit": "micrometer", "discrete": False},
                {"type": "space", "name": "y", "unit": "micrometer", "discrete": False},
            ],
        }
    ],
}

REGISTRATION = {
    "coordinateTransformations": [
        {
            "type": "bijection",
            "input": {"path": "JRC2018F", "name": "physical"},
            "output": {"path": "FCWB", "name": "physical"},
            "forward": {
                "type": "sequence",
                "name": "JRC2018F to FCWB",
                "transformations": [
                    {"type": "displacements", "path": "coordinateTransformations/dfield", "interpolation": "linear"},
                    {
                        "type": "affine",
                        "affine": [
                            [0.549687, -0.0138092, 0.000127526, 2.9986],
                            [0.0893289, 1.04339, -0.000121014, -6.39702],
                            [0.00779285, 0.00299018, 0.907875, -3.77146],
                        ],
                    },
                ],
            },
            "inverse": {
                "type": "sequence",
                "name": "FCWB to JRC2018F",
                "transformations": [
                    {
                        "type": "affine",
                        "affine": [
                            [1.8153162032371448, 0.024026315573955494, -0.00025178851007148946, -5.290659956068192],
                            [-0.1554184181171034, 0.9563570184920926, 0.00014930742384645888, 6.584435749976974],
                            [-0.015070089856986017, -0.003356093187801388, 1.1014748899286995, 4.177888664571422],
                        ],
                    },
                    {
                        "type": "displacements",
                        "path": "coordinateTransformations/invdfield",
                        "interpolation": "linear",
                        "name": "customfield",
                    },
                ],
            },
        }
    ]
}

SCALE_WITH_DISCRETE = {
    "coordinateSystems": [
        {
            "name": "in",
            "axes": [
                {"name": "k", "type": "channel", "discrete": True},
                {"name": "j", "type": "space", "discrete": False},
                {"name": "i", "type": "space", "discrete": False},
            ],
        },
        {
            "name": "out",
            "axes": [
                {"name": "c", "type": "channel", "discrete": True},
                {"name": "y", "type": "space", "discrete": False},
                {"name": "x", "type": "space", "discrete": False},
            ],
        },
    ],
    "coordinateTransformations": [
        {"type": "scale", "scale": [1, 3.12, 2], "input": {"name": "in"}, "output": {"name": "out"}}
    ],
}

ROTATION = {
    "coordinateSystems": [
        {"name": "ji", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "yx", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {"type": "rotation", "rotation": [[0, -1], [1, 0]], "input": {"name": "ji"}, "output": {"name": "yx"}}
    ],
}

AFFINE_WITH_CHANNEL = {
    "coordinateSystems": [
        {
            "name": "cji",
            "axes": [
                {"name": "k", "discrete": True, "type": "space"},
                {"name": "j", "discrete": False, "type": "space"},
                {"name": "i", "discrete": False, "type": "space"},
            ],
        },
        {
            "name": "cyx",
            "axes": [
                {"name": "c", "discrete": True, "type": "space"},
                {"name": "y", "discrete": False, "type": "space"},
                {"name": "x", "discrete": False, "type": "space"},
            ],
        },
    ],
    "coordinateTransformations": [
        {
            "type": "affine",
            "affine": [[1, 0, 0, 0], [0, 1, 2, 3], [0, 4, 5, 6]],
            "input": {"name": "cji"},
            "output": {"name": "cyx"},
        }
    ],
}

BY_DIMENSION_COORDINATES = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "i", "type": "space"}, {"name": "j", "type": "space"}]},
        {"name": "out", "axes": [{"name": "x", "type": "space"}, {"name": "y", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "type": "byDimension",
            "input": {"name": "in"},
            "output": {"name": "out"},
            "transformations": [
                {
                    "transformation": {
                        "type": "coordinates",
                        "path": "/coordinates",
                    },
                    "inputAxes": [0],
                    "outputAxes": [0],
                },
                {
                    "transformation": {
                        "type": "scale",
                        "scale": [2.0],
                    },
                    "inputAxes": [1],
                    "outputAxes": [1],
                },
            ],
        }
    ],
}

BY_DIMENSION_XARRAY = {
    "coordinateSystems": [
        {
            "name": "physical",
            "axes": [
                {"name": "x", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
            ],
        },
        {"name": "array", "axes": [{"name": "dim_0", "type": "space"}, {"name": "dim_1", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "type": "byDimension",
            "input": {"name": "array"},
            "output": {"name": "physical"},
            "transformations": [
                {
                    "transformation": {
                        "type": "coordinates",
                        "path": "xCoordinates",
                    },
                    "inputAxes": [0],
                    "outputAxes": [0],
                },
                {
                    "transformation": {
                        "type": "coordinates",
                        "path": "yCoordinates",
                    },
                    "inputAxes": [1],
                    "outputAxes": [1],
                },
            ],
        }
    ],
}

BIJECTION = {
    "coordinateSystems": [
        {"name": "src", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "tgt", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "type": "bijection",
            "forward": {"type": "coordinates", "path": "forward_coordinates"},
            "inverse": {"type": "coordinates", "path": "inverse_coordinates"},
            "input": {"name": "src"},
            "output": {"name": "tgt"},
        }
    ],
}

BIJECTION_VERBOSE = {
    "coordinateTransformations": [
        {  # semantically invalid due to systems referenced by name not existing
            "type": "bijection",
            "forward": {
                "type": "coordinates",
                "path": "forward_coordinates",
                "input": {"name": "src"},
                "output": {"name": "tgt"},
            },
            "inverse": {
                "type": "coordinates",
                "path": "inverse_coordinates",
                "input": {"name": "tgt"},
                "output": {"name": "src"},
            },
            "input": {"name": "src"},
            "output": {"name": "tgt"},
        }
    ]
}

MAP_AXIS_1 = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "out1", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
        {"name": "out2", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "name": "equivalent to identity",
            "type": "mapAxis",
            "mapAxis": [0, 1],
            "input": {"name": "in"},
            "output": {"name": "out1"},
        },
        {
            "name": "permutation",
            "type": "mapAxis",
            "mapAxis": [1, 0],
            "input": {"name": "in"},
            "output": {"name": "out2"},
        },
    ],
}

AFFINE_2D_3D = {
    "coordinateSystems": [
        {"name": "ij", "axes": [{"name": "i", "type": "space"}, {"name": "j", "type": "space"}]},
        {
            "name": "zyx",
            "axes": [{"name": "z", "type": "space"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        },
    ],
    "coordinateTransformations": [
        {
            "type": "affine",
            "affine": [[1, 0, 0], [2, 3, 4], [5, 6, 7]],
            "input": {"name": "ij"},
            "output": {"name": "zyx"},
        }
    ],
}

PROJECT_AXIS_2 = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "c"}, {"name": "i"}, {"name": "j"}]},
        {"name": "out", "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}]},
    ],
    "coordinateTransformations": [
        {
            "name": "up-project",
            "type": "projectAxis",
            "createdOutputs": [0],
            "droppedInputs": [0],
            "input": {"name": "in"},
            "output": {"name": "out"},
        }
    ],
}

BY_DIMENSION_2 = {
    "coordinateSystems": [
        {
            "name": "in",
            "axes": [
                {"name": "l", "type": "space"},
                {"name": "j", "type": "space"},
                {"name": "k", "type": "space"},
                {"name": "i", "type": "space"},
            ],
        },
        {
            "name": "out",
            "axes": [{"name": "z", "type": "space"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
        },
    ],
    "coordinateTransformations": [
        {
            "type": "byDimension",
            "input": {"name": "in"},
            "output": {"name": "out"},
            "transformations": [
                {
                    "transformation": {"type": "translation", "translation": [0.5, 1.5]},
                    "inputAxes": [3, 2],
                    "outputAxes": [1, 2],
                },
                {"transformation": {"type": "scale", "scale": [2]}, "inputAxes": [1], "outputAxes": [0]},
            ],
        }
    ],
}

AFFINE_2D_2D = {
    "coordinateSystems": [
        {"name": "ji", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "yx", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {"type": "affine", "affine": [[1, 2, 3], [4, 5, 6]], "input": {"name": "ji"}, "output": {"name": "yx"}}
    ],
}

SEQUENCE = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "out", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "type": "sequence",
            "input": {"name": "in"},
            "output": {"name": "out"},
            "transformations": [{"type": "translation", "translation": [0.1, 0.9]}, {"type": "scale", "scale": [2, 3]}],
        }
    ],
}

TRANSLATION = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "out", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {"type": "translation", "input": {"name": "in"}, "output": {"name": "out"}, "translation": [9, -1.42]}
    ],
}

IDENTITY = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "out", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [{"type": "identity", "input": {"name": "in"}, "output": {"name": "out"}}],
}

BY_DIMENSION_1 = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "out", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "type": "byDimension",
            "input": {"name": "in"},
            "output": {"name": "out"},
            "transformations": [
                {"transformation": {"type": "translation", "translation": [-1.0]}, "inputAxes": [1], "outputAxes": [1]},
                {"transformation": {"type": "scale", "scale": [2.0]}, "inputAxes": [0], "outputAxes": [0]},
            ],
        }
    ],
}

SCALE = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "out", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {"type": "scale", "scale": [2, 3.12], "input": {"name": "in"}, "output": {"name": "out"}}
    ],
}

PROJECT_AXIS = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "i"}, {"name": "j"}]},
        {"name": "out", "axes": [{"name": "c"}, {"name": "z"}, {"name": "y"}, {"name": "x"}]},
    ],
    "coordinateTransformations": [
        {
            "name": "up-project",
            "type": "projectAxis",
            "createdOutputs": [0, 1],
            "input": {"name": "in"},
            "output": {"name": "out"},
        }
    ],
}

XARRAY_LIKE = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "i", "type": "space"}, {"name": "j", "type": "space"}]},
        {"name": "out", "axes": [{"name": "x", "type": "space"}, {"name": "y", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "type": "byDimension",
            "input": {"name": "in"},
            "output": {"name": "out"},
            "transformations": [
                {
                    "transformation": {"type": "coordinates", "path": "/xCoordinates"},
                    "inputAxes": [0],
                    "outputAxes": [0],
                },
                {
                    "transformation": {"type": "coordinates", "path": "/yCoordinates"},
                    "inputAxes": [1],
                    "outputAxes": [1],
                },
            ],
        }
    ],
}

INVALID_BY_DIMENSION_OUTPUT_OUT_OF_BOUNDS = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "out", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "type": "byDimension",
            "input": {"name": "in"},
            "output": {"name": "out"},
            "transformations": [
                {"transformation": {"type": "translation", "translation": [-1.0]}, "inputAxes": [1], "outputAxes": [2]},
                {"transformation": {"type": "scale", "scale": [2.0]}, "inputAxes": [0], "outputAxes": [2]},
            ],
        }
    ],
}

INVALID_BY_DIMENSION_MISSING_OUTPUTS = {
    "coordinateSystems": [
        {"name": "in", "axes": [{"name": "j", "type": "space"}, {"name": "i", "type": "space"}]},
        {"name": "out", "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]},
    ],
    "coordinateTransformations": [
        {
            "type": "byDimension",
            "input": {"name": "in"},
            "output": {"name": "out"},
            "transformations": [
                {"transformation": {"type": "translation", "translation": [-1.0]}, "inputAxes": [1], "outputAxes": [1]},
                {"transformation": {"type": "scale", "scale": [2.0]}, "inputAxes": [1], "outputAxes": [1]},
            ],
        }
    ],
}


def scene_stitching():
    return copy.deepcopy(STITCHING)


def scene_registration():
    return copy.deepcopy(REGISTRATION)


def all_valid_scene_examples():
    return [
        copy.deepcopy(ex)
        for ex in [
            STITCHING,
            REGISTRATION,
            SCALE_WITH_DISCRETE,
            ROTATION,
            AFFINE_WITH_CHANNEL,
            BY_DIMENSION_COORDINATES,
            BY_DIMENSION_XARRAY,
            BIJECTION,
            BIJECTION_VERBOSE,
            MAP_AXIS_1,
            AFFINE_2D_3D,  #
            PROJECT_AXIS_2,
            BY_DIMENSION_2,
            AFFINE_2D_2D,
            SEQUENCE,
            TRANSLATION,
            IDENTITY,
            BY_DIMENSION_1,
            SCALE,
            PROJECT_AXIS,
            XARRAY_LIKE,
        ]
    ]


def all_invalid_scene_examples():
    return [
        copy.deepcopy(ex) for ex in [INVALID_BY_DIMENSION_OUTPUT_OUT_OF_BOUNDS, INVALID_BY_DIMENSION_MISSING_OUTPUTS]
    ]


def scene_to_group_attrs(scene_obj: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "ome": {
            "version": "0.6.rc0",
            "scene": scene_obj,
        }
    }
