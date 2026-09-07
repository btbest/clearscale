from unittest.mock import Mock, MagicMock

from clearscale import Shape, PixelSize, Unit, Scale, BlueprintShapes, Multiscale, OmeZarrGroup, GroupKind


def test_simple_dump_example():
    from clearscale import OmeZarrGroup, Multiscale, Scale

    image = Mock(shape=(128, 1024, 1024))

    # Write data
    array_path = "s0"

    # Write metadata
    # Make a Scale, expand it to a Multiscale, put that in an OME-Zarr group.
    scale = Scale(shape=dict(zip("zyx", image.shape)))  # Axis keys are the minimum you really must specify
    written = OmeZarrGroup.from_single(Multiscale.from_single(scale, scale_key=array_path)).to_attrs(version="0.5")

    assert "ome" in written
    assert "multiscales" in written["ome"]
    assert "version" in written["ome"] and written["ome"]["version"] == "0.5"
    assert len(written["ome"]["multiscales"]) == 1

    expected = {
        "ome": {
            "multiscales": [
                {
                    "axes": [{"name": "z"}, {"name": "y"}, {"name": "x"}],
                    "datasets": [
                        {
                            "coordinateTransformations": [{"scale": [1.0, 1.0, 1.0], "type": "scale"}],
                            "path": "s0",
                        }
                    ],
                    "version": "0.5",
                }
            ],
            "version": "0.5",
        }
    }

    assert written == expected


def test_downscale_2_example():
    my_data = Mock(shape=(3, 54, 1024, 1024))

    # 1. Annotate
    shape = Shape(zip("tzyx", my_data.shape))
    pixel_size = PixelSize(t=5.0, z=260.0, y=0.53, x=0.53)
    unit = Unit(t="s", z="micrometer", y="micrometer", x="micrometer")

    # 2. Define operation plan
    scaling_blueprint = BlueprintShapes.downscale_powers_of_2_xyz(
        base_shape=shape,
        rounding="ceil",
        shape_limit=Shape(z=8, y=128, x=128),
    )

    # 3. Scale data according to the blueprint
    # noop for this test

    # 4. Expand and write metadata
    base = Scale(shape, pixel_size, unit)
    ms = Multiscale.from_single(base, blueprint=scaling_blueprint)
    written = OmeZarrGroup.from_single(ms).to_attrs(version="0.6.rc0")

    assert "ome" in written
    assert "multiscales" in written["ome"]
    assert "version" in written["ome"] and written["ome"]["version"] == "0.6.rc0"
    assert len(written["ome"]["multiscales"]) == 1
    written_ms_dict = written["ome"]["multiscales"][0]
    assert len(written_ms_dict["coordinateSystems"]) == 1
    written_system_name = written_ms_dict["coordinateSystems"][0]["name"]  # Capture the randomly generated name

    expected = {
        "coordinateSystems": [
            {"axes": [{"name": "t"}, {"name": "z"}, {"name": "y"}, {"name": "x"}], "name": written_system_name}
        ],
        "datasets": [
            {
                "coordinateTransformations": [
                    {
                        "input": {"path": "s0"},
                        "output": {"name": written_system_name},
                        "scale": [5.0, 260.0, 0.53, 0.53],
                        "type": "scale",
                    }
                ],
                "path": "s0",
            },
            {
                "coordinateTransformations": [
                    {
                        "input": {"path": "s1"},
                        "output": {"name": written_system_name},
                        "scale": [5.0, 520.0, 1.06, 1.06],
                        "type": "scale",
                    }
                ],
                "path": "s1",
            },
            {
                "coordinateTransformations": [
                    {
                        "input": {"path": "s2"},
                        "output": {"name": written_system_name},
                        "scale": [5.0, 1002.8571428571429, 2.12, 2.12],
                        "type": "scale",
                    }
                ],
                "path": "s2",
            },
            {
                "coordinateTransformations": [
                    {
                        "input": {"path": "s3"},
                        "output": {"name": written_system_name},
                        "scale": [5.0, 2005.7142857142858, 4.24, 4.24],
                        "type": "scale",
                    }
                ],
                "path": "s3",
            },
        ],
        "version": "0.6.rc0",
    }

    assert written_ms_dict == expected


def test_skimage_pyramid_gaussian_example():
    # 1. Generate a pyramid
    image = Mock(shape=(128, 1024, 1024))
    pyramid = [
        Mock(shape=sh)
        for sh in [
            (128, 1024, 1024),
            (64, 512, 512),
            (32, 256, 256),
            (16, 128, 128),
            (8, 64, 64),
            (4, 32, 32),
            (2, 16, 16),
            (1, 8, 8),
            (1, 4, 4),
            (1, 2, 2),
            (1, 1, 1),
        ]
    ]

    # 2. Write arrays to zarr on disk and record scaled shapes
    group = MagicMock()  # Mock instead of open_group("example.ome.zarr", mode="w")
    scaled_shapes = []
    for i, level in enumerate(pyramid):
        scale_key = f"s{i}"
        group.create_array(scale_key, data=level)
        scaled_shapes.append((scale_key, Shape(zip("zyx", level.shape))))

    # 3. Describe the full-resolution image
    base = Scale(
        shape=Shape(zip("zyx", image.shape)),
        pixel_size=dict(z=25, y=240, x=240),
        unit=dict(z="micron", y="nanometer", x="nanometer"),
    )

    # 4. Use the recorded scale shapes as a blueprint to expand a Multiscale
    blueprint = BlueprintShapes(scaled_shapes)
    multiscale = Multiscale.from_single(base, blueprint=blueprint)

    # 5. Save OME-Zarr metadata
    group_meta = OmeZarrGroup.from_single(multiscale).to_attrs(version="0.5", axis_types="infer")
    group.attrs.update(group_meta)

    written = group_meta
    assert "ome" in written
    assert "multiscales" in written["ome"]
    assert "version" in written["ome"] and written["ome"]["version"] == "0.5"
    assert len(written["ome"]["multiscales"]) == 1
    written_ms_dict = written["ome"]["multiscales"][0]
    assert written_ms_dict["axes"] == [
        {"name": "z", "type": "space", "unit": "micron"},
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    assert tuple(multiscale.keys()) == ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10")
    assert written_ms_dict["datasets"][1]["coordinateTransformations"] == [
        {"scale": [50.0, 480.0, 480.0], "type": "scale"}
    ]


def test_extract_single_scale_example():
    remote_attrs = {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {"coordinateTransformations": [{"scale": [0.025, 0.01, 0.01], "type": "scale"}], "path": "s0"},
                    {"coordinateTransformations": [{"scale": [0.025, 0.02, 0.02], "type": "scale"}], "path": "s1"},
                    {"coordinateTransformations": [{"scale": [0.05, 0.04, 0.04], "type": "scale"}], "path": "s2"},
                    {"coordinateTransformations": [{"scale": [0.1, 0.08, 0.08], "type": "scale"}], "path": "s3"},
                    {"coordinateTransformations": [{"scale": [0.2, 0.16, 0.16], "type": "scale"}], "path": "s4"},
                    {"coordinateTransformations": [{"scale": [0.4, 0.32, 0.32], "type": "scale"}], "path": "s5"},
                    {"coordinateTransformations": [{"scale": [0.8, 0.64, 0.64], "type": "scale"}], "path": "s6"},
                    {"coordinateTransformations": [{"scale": [1.6, 1.28, 1.28], "type": "scale"}], "path": "s7"},
                    {"coordinateTransformations": [{"scale": [3.2, 2.56, 2.56], "type": "scale"}], "path": "s8"},
                    {"coordinateTransformations": [{"scale": [6.4, 5.12, 5.12], "type": "scale"}], "path": "s9"},
                ],
                "name": "platy-em",
                "version": "0.4",
            }
        ]
    }
    expected_written_meta = {
        "multiscales": [
            {
                "axes": [
                    {"name": "z", "unit": "micrometer"},
                    {"name": "y", "unit": "micrometer"},
                    {"name": "x", "unit": "micrometer"},
                ],
                "datasets": [
                    {"coordinateTransformations": [{"scale": [0.8, 0.64, 0.64], "type": "scale"}], "path": "s6"}
                ],
                "version": "0.4",
            }
        ]
    }

    array_mock = Mock()
    array_mock.shape = (357, 405, 430)
    remote_group_mock = MagicMock()
    remote_group_mock.attrs = remote_attrs
    remote_group_mock.return_value = array_mock
    local_group_mock = Mock()
    zarr = Mock()
    zarr.open_group = Mock(side_effect=[remote_group_mock, local_group_mock])

    URL = "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"
    SCALE_KEY = "s6"
    LOCAL_PATH = f"demo-output/platy-raw-{SCALE_KEY}.ome.zarr"

    # 1. Discover what's on the server
    remote_group = zarr.open_group(URL)
    remote_meta = OmeZarrGroup.from_group(remote_group)
    assert remote_meta.kind is GroupKind.MULTISCALE, "This tells us there is exactly one Multiscale in this group"
    assert remote_meta.version == "0.4"
    source_multiscale = remote_meta.multiscales[0]
    assert SCALE_KEY in source_multiscale, "Multiscale is dict-like, with the sub-paths to arrays as keys"
    assert source_multiscale[SCALE_KEY].shape == {"z": 357, "y": 405, "x": 430}

    # 2. Create the local target and download the data
    source_array = remote_group[SCALE_KEY]
    local_group = zarr.open_group(str(LOCAL_PATH), mode="w", zarr_format=2)
    local_array = local_group.create_array(SCALE_KEY, data=source_array, overwrite=True)

    # 3. Extract the correct scale metadata and upgrade it to valid independent metadata
    target_multiscale = source_multiscale.derive(SCALE_KEY)
    local_group_meta = OmeZarrGroup.from_single(target_multiscale)

    # 4. Write the new metadata to the downloaded store
    local_group.attrs.update(local_group_meta.to_attrs(version="0.4"))

    assert local_group_meta.to_attrs(version="0.4") == expected_written_meta
    local_group.attrs.update.assert_called_once_with(expected_written_meta)
