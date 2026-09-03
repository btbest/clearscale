from collections import OrderedDict
from typing import Optional, Mapping

import clearscale
import pytest
from clearscale import PixelSize, Unit, Shape, PixelOffset, Multiscale
from clearscale.types import AxisKey


def write_ome_zarr_like_ilastik(
    export_shape: clearscale.Shape,
    pixel_size: PixelSize,
    unit: Optional[Unit] = None,
    export_blueprint: Optional[clearscale.BlueprintShapes] = None,
    export_offset: Optional[clearscale.PixelOffset] = None,
    input_multiscale: Optional[clearscale.Multiscale] = None,
    input_scale_key: Optional[str] = None,
):
    # Couple of details from `lazyflow.utility.io_util.write_ome_zarr`
    ome_zarr_axes = "tczyx"
    export_pixel_size = pixel_size.with_axes(ome_zarr_axes)
    export_shape = export_shape.with_axes(ome_zarr_axes)
    if export_blueprint is None:
        SINGE_SCALE_DEFAULT_KEY = "s0"
        single_target_key = input_scale_key if input_scale_key else SINGE_SCALE_DEFAULT_KEY
        export_blueprint = clearscale.BlueprintShapes({single_target_key: export_shape})
    # And the rest is `lazyflow.utility.io_util._write_ome_zarr_and_ilastik_metadata`
    axes = list(export_pixel_size.keys())
    if unit:
        export_unit = unit.with_axes(axes)
    else:
        export_unit = clearscale.Unit.empty(axes)

    crop_translation = clearscale.Translation.identity(axes)
    if export_offset:
        crop_translation = export_offset.with_axes(axes).to_physical(export_pixel_size)

    input_scale = input_multiscale[input_scale_key] if input_multiscale and input_scale_key else None
    input_translation = input_scale.translation if input_scale else clearscale.Translation.identity(axes)
    export_scale = clearscale.Scale(export_shape, export_pixel_size, export_unit, input_translation.with_axes(axes))
    multiscale = export_blueprint.apply_to_scale(export_scale)
    if input_multiscale:
        derivation = []
        if axes != input_multiscale.axes():
            derivation.append(clearscale.AxisRearrangementTo(multiscale.axes()))
        if not crop_translation.is_identity():
            derivation.append(crop_translation)
        multiscale = multiscale.as_derived_from(input_multiscale, by=derivation)
    elif not crop_translation.is_identity():
        # Inverted: the export needs to be un-shifted to return to its original space
        multiscale = multiscale.with_coordinate_system("source_image_space", reached_by=crop_translation.inverted())
    multiscale.ome.metadata.update(
        {
            "description": "ilastik's lazyflow.operators.opResize.OpResize is a lazy implementation of skimage.transform.resize.",
            "method": "skimage.transform.resize",
            "version": "0.24.0",
            "kwargs": {"order": 1, "anti_aliasing": True, "preserve_range": True},
        }
        if export_blueprint.scaled_axes()
        else {}
    )

    return clearscale.OmeZarrGroup.from_single(multiscale).to_attrs(version="0.4", axis_types="infer")


def test_pixel_sizes_test_write_ome_zarr_single_scale():
    axes, shape, pixel_size, units = (
        ["t", "z", "y", "x", "c"],
        (6, 5, 4, 3, 2),
        [0.4, 5.0, 0.3, 6.4, 8.99991],
        ["sec", "um", "nm", "mm", "noodles"],
    )
    expected_axes = [
        {"name": "t", "type": "time", "unit": "sec"},
        {"name": "c", "type": "channel", "unit": "noodles"},  # obviously non-standard, but allowed
        {"name": "z", "type": "space", "unit": "um"},
        {"name": "y", "type": "space", "unit": "nm"},
        {"name": "x", "type": "space", "unit": "mm"},
    ]
    expected_dataset_transform = [{"type": "scale", "scale": [0.4, 8.99991, 5.0, 0.3, 6.4]}]

    result = write_ome_zarr_like_ilastik(
        Shape(zip(axes, shape)), PixelSize(zip(axes, pixel_size)), Unit(zip(axes, units))
    )

    assert "multiscales" in result
    m = result["multiscales"][0]
    assert "axes" in m
    assert m["axes"] == expected_axes
    assert "coordinateTransformations" not in m
    assert "datasets" in m and "path" in m["datasets"][0]
    assert len(m["datasets"]) == 1
    assert m["datasets"][0]["coordinateTransformations"] == expected_dataset_transform


@pytest.mark.parametrize(
    "shape, axes, target_scales",
    [
        ((1, 128, 127, 10, 1), "txyzc", None),  # ilastik default order
        ((1, 1, 3, 26, 25), "tczyx", None),  # OME-Zarr convention
        ((256, 255), "yx", None),
        ((10, 126, 125), "zyx", None),
        ((124, 123, 3), "yxc", None),
        (
            (21, 23, 3),
            "yxc",
            clearscale.BlueprintShapes(
                [
                    ("raw", Shape(zip("tczyx", (1, 3, 1, 21, 23)))),
                    ("scaled", Shape(zip("tczyx", (1, 3, 1, 10, 12)))),
                ]
            ),
        ),
        (
            (21, 23, 3),
            "yxc",
            clearscale.BlueprintShapes({"s0": Shape(zip("tczyx", (1, 3, 1, 10, 12)))}),
        ),
    ],
)
def test_write_ome_zarr_test_metadata_match_ome_zarr_spec(shape, axes, target_scales):
    result = write_ome_zarr_like_ilastik(
        Shape(zip(axes, shape)), PixelSize.identity(axes), export_blueprint=target_scales
    )

    expected_axiskeys = "tczyx"
    assert "multiscales" in result
    written_meta = result["multiscales"][0]
    required_keys = ("datasets", "axes", "version")  # version not required by spec but by us
    assert all([key in written_meta for key in required_keys])
    assert all(written_meta.values()), "Should not write empty values anywhere"
    assert written_meta["version"] == "0.4"
    assert [a["name"] for a in written_meta["axes"]] == list(expected_axiskeys)
    expected_len_datasets = 1 if target_scales is None else len(target_scales)
    assert len(written_meta["datasets"]) == expected_len_datasets, "not all specified scales written"

    discovered_keys = []
    for i, dataset in enumerate(written_meta["datasets"]):
        discovered_keys.append(dataset["path"])
        reported_scalings = [
            transform for transform in dataset["coordinateTransformations"] if transform["type"] == "scale"
        ]
        assert len(reported_scalings) == 1


def test_write_ome_zarr_test_unscaled_single_scale_export_writes_crop_to_multiscale_transforms():
    input_scale_key = "source_scale"
    units = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    resolution_t = 0.1
    resolution_xyz = 2.0
    expected_multiscale_transform = [
        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]},
        {"type": "translation", "translation": [0.0, 0.0, 8.0, 8.0, 8.0]},  # offset * resolution
    ]
    expected_source_scale_transform = [
        {"type": "scale", "scale": [resolution_t, 1.0, resolution_xyz, resolution_xyz, resolution_xyz]},
        {"type": "translation", "translation": [0.3, 0.0, 3.2, 1.0, 1.0]},  # source scale translation
    ]
    input_multiscale = clearscale.Multiscale.from_ome_zarr(
        {
            "name": "source_pyramid_without_global_transforms",
            "axes": [
                {"name": "t", "type": "time", "unit": units["t"]},
                {"name": "z", "type": "space", "unit": units["z"]},
                {"name": "y", "type": "space", "unit": units["y"]},
                {"name": "x", "type": "space", "unit": units["x"]},
            ],  # Input metadata tzyx, but e.g. Probabilities output would be tczyx
            "datasets": [
                {
                    "path": "raw_scale",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]},
                        {"type": "translation", "translation": [0.1, 5.0, 2.0, 1.0]},
                    ],
                },
                {
                    "path": "source_scale",
                    "coordinateTransformations": [
                        # Normally the "scale" transform is the source of the slot's pixel size.
                        # "scale" would have to be [0.1, 2.0, 2.0, 2.0] to match the slot meta.
                        # The export should use the slot meta.
                        {"type": "scale", "scale": [3.1, 1.3, 3.7, 6.9]},
                        {"type": "translation", "translation": [0.3, 3.2, 1.0, 1.0]},
                    ],
                },
            ],
        },
        shape_source={"raw_scale": (2, 17, 17, 17), "source_scale": (2, 9, 9, 9), "downscale": (2, 5, 5, 5)},
    )

    axes = "tczyx"
    shape = Shape(zip(axes, (2, 2, 5, 5, 5)))
    export_offset = PixelOffset(zip(axes, (0, 0, 4, 4, 4)))
    result = write_ome_zarr_like_ilastik(
        shape,
        PixelSize(zip(axes, [resolution_t, 1.0, resolution_xyz, resolution_xyz, resolution_xyz])),
        unit=Unit(units),
        export_offset=export_offset,
        input_multiscale=input_multiscale,
        input_scale_key=input_scale_key,
    )

    assert "multiscales" in result
    m = result["multiscales"][0]
    assert "datasets" in m and "path" in m["datasets"][0]
    assert len(m["datasets"]) == 1
    assert "name" not in m  # Input name should not be carried over - presumably it names the raw data
    assert m["axes"] == [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]  # Axis units should be carried over
    assert m["coordinateTransformations"] == expected_multiscale_transform
    assert m["datasets"][0]["path"] == "source_scale"
    assert m["datasets"][0]["coordinateTransformations"] == expected_source_scale_transform


def test_write_ome_zarr_test_unscaled_single_scale_export_round_trips_t_convention():
    """If the input used the convention that pixel_size[t] is written on global level, then that should be maintained."""
    input_scale_key = "source_scale"
    resolution_t = 0.1
    resolution_xyz = 2.0
    units = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    expected_multiscale_transform = [
        {"type": "scale", "scale": [resolution_t, 1.0, 1.0, 1.0, 1.0]},
    ]
    # When no actual scaling is done by ilastik, input scale should be carried over unmodified even if imprecise.
    expected_source_scale_transform = [
        {"type": "scale", "scale": [1.0, 1.0, resolution_xyz, resolution_xyz, resolution_xyz]},
        {"type": "translation", "translation": pytest.approx([0.5, 0.0, 3.2, 1.0, 1.0])},  # source scale translation
    ]
    input_multiscale = clearscale.Multiscale.from_ome_zarr(
        {
            "name": "pyramid_with_global_t_convention",
            "axes": [
                {"name": "t", "type": "time", "unit": units["t"]},
                {"name": "z", "type": "space", "unit": units["z"]},
                {"name": "y", "type": "space", "unit": units["y"]},
                {"name": "x", "type": "space", "unit": units["x"]},
            ],  # Input metadata tzyx, but e.g. Probabilities output would be tczyx
            "coordinateTransformations": [
                # "global t-scale" convention means first scale value here is not 1.0 and for the
                # dataset scales it is 1.0
                {"type": "scale", "scale": [0.1, 1.0, 1.0, 1.0]}
            ],
            "datasets": [
                {
                    "path": "raw_scale",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]},
                        {"type": "translation", "translation": [0.1, 5.0, 2.0, 1.0]},
                    ],
                },
                {
                    "path": "source_scale",
                    "coordinateTransformations": [
                        # Normally the "scale" transform is the source of the slot's pixel size.
                        # "scale" would have to be [0.1, 2.0, 2.0, 2.0] to match the slot meta
                        # The export should use the slot meta; inject nonsense here to enforce it's not reused.
                        # But keep t-scale 1.0 to be consistent with the "global t-scale" convention
                        {"type": "scale", "scale": [1.0, 1.3, 3.7, 6.9]},
                        {"type": "translation", "translation": [0.5, 3.2, 1.0, 1.0]},
                    ],
                },
                {
                    "path": "downscale",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 4.0, 4.0, 4.0]},
                        {"type": "translation", "translation": [5.1, 3.5, 5.4, 1.0]},
                    ],
                },
            ],
        },
        shape_source={"raw_scale": (2, 17, 17, 17), "source_scale": (2, 9, 9, 9), "downscale": (2, 5, 5, 5)},
    )

    axes = "tczyx"
    shape = Shape(zip(axes, (2, 2, 5, 5, 5)))
    export_offset = None
    result = write_ome_zarr_like_ilastik(
        shape,
        PixelSize(zip(axes, [resolution_t, 1.0, resolution_xyz, resolution_xyz, resolution_xyz])),
        unit=Unit(units),
        export_offset=export_offset,
        input_multiscale=input_multiscale,
        input_scale_key=input_scale_key,
    )

    assert "multiscales" in result
    m = result["multiscales"][0]
    assert "datasets" in m and "path" in m["datasets"][0]
    assert len(m["datasets"]) == 1
    assert "name" not in m  # Input name should not be carried over - presumably it names the raw data
    assert m["axes"] == [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]  # Axis units should be carried over
    assert m["coordinateTransformations"] == expected_multiscale_transform
    assert m["datasets"][0]["path"] == "source_scale"
    assert m["datasets"][0]["coordinateTransformations"] == expected_source_scale_transform


def test_write_ome_zarr_test_unscaled_single_scale_export_t_scale_convention_overrides_storing_crop():
    """
    When reading a zarr with global t-scale convention, export has a dilemma:
    Expressing a global offset (i.e. our crop) isn't consistent with the convention, so it's either
    maintain the convention, or express the crop.
    We figure if you're working with this convention, you need to stay in it to be compatible with
    downstream tools, so prioritise the convention.
    Specifically, nifti-zarr uses this convention; if you read nifti-zarr in, you probably need nifti-zarr out.
    Most likely the behaviour here doesn't matter much in practice: The nifti-zarr convention is probably niche,
    and expressing a crop as a global translation is also probably a niche use of the global transforms.
    """
    input_scale_key = "source_scale"
    resolution_t = 0.1
    resolution_xyz = 2.0
    units = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    expected_multiscale_transform = [
        {"type": "scale", "scale": [resolution_t, 1.0, 1.0, 1.0, 1.0]},
        # no translation: this would be inconsistent (the multiscale's "intrinsic" system would be
        # inbetween the scale and the translation; almost guaranteed to be misinterpreted by readers)
    ]
    expected_source_scale_transform = [
        {"type": "scale", "scale": [1.0, 1.0, resolution_xyz, resolution_xyz, resolution_xyz]},
        {
            "type": "translation",
            "translation": pytest.approx([0.1, 0.0, 3.2, 1.0, 1.0]),
        },  # source scale translation -- crop translation *not* added
    ]
    input_multiscale = clearscale.Multiscale.from_ome_zarr(
        {
            "name": "pyramid_with_global_t_convention",
            "axes": [
                {"name": "t", "type": "time", "unit": units["t"]},
                {"name": "z", "type": "space", "unit": units["z"]},
                {"name": "y", "type": "space", "unit": units["y"]},
                {"name": "x", "type": "space", "unit": units["x"]},
            ],  # Input metadata tzyx, but e.g. Probabilities output would be tczyx
            "coordinateTransformations": [
                # "global t-scale" convention means first scale value here is not 1.0 and for the
                # dataset scales it is 1.0. The value must actually match the export pixel_size[t].
                {"type": "scale", "scale": [0.1, 1.0, 1.0, 1.0]}
            ],
            "datasets": [
                {
                    "path": "raw_scale",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]},
                        {"type": "translation", "translation": [0.1, 5.0, 2.0, 1.0]},
                    ],
                },
                {
                    "path": "source_scale",
                    "coordinateTransformations": [
                        # Normally the "scale" transform is the source of the slot's pixel size.
                        # "scale" would have to be [0.1, 2.0, 2.0, 2.0] to match the slot meta
                        # The export should use the slot meta; inject nonsense here to enforce it's not reused.
                        # But keep t-scale 1.0 to be consistent with the "global t-scale" convention
                        {"type": "scale", "scale": [1.0, 1.3, 3.7, 6.9]},
                        {"type": "translation", "translation": [0.1, 3.2, 1.0, 1.0]},
                    ],
                },
                {
                    "path": "downscale",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 4.0, 4.0, 4.0]},
                        {"type": "translation", "translation": [5.1, 3.5, 5.4, 1.0]},
                    ],
                },
            ],
        },
        shape_source={"raw_scale": (2, 17, 17, 17), "source_scale": (2, 9, 9, 9), "downscale": (2, 5, 5, 5)},
    )

    axes = "tczyx"
    shape = Shape(zip(axes, (2, 2, 5, 5, 5)))
    export_offset = PixelOffset(zip(axes, (0, 0, 4, 4, 4)))
    result = write_ome_zarr_like_ilastik(
        shape,
        PixelSize(zip(axes, [resolution_t, 1.0, resolution_xyz, resolution_xyz, resolution_xyz])),
        unit=Unit(units),
        export_offset=export_offset,
        input_multiscale=input_multiscale,
        input_scale_key=input_scale_key,
    )

    assert "multiscales" in result
    m = result["multiscales"][0]
    assert "datasets" in m and "path" in m["datasets"][0]
    assert len(m["datasets"]) == 1
    assert "name" not in m  # Input name should not be carried over - presumably it names the raw data
    assert m["axes"] == [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]  # Axis units should be carried over
    assert m["coordinateTransformations"] == expected_multiscale_transform
    assert m["datasets"][0]["path"] == "source_scale"
    assert m["datasets"][0]["coordinateTransformations"] == expected_source_scale_transform


def test_write_ome_zarr_test_resized_single_scale_export():
    """If the export is a single resized scale, the scale key should be as specified target;
    `scale` should be source resolution * resizing factor.
    Input `.scales` meta is irrelevant for this now that pixel size goes via axistags."""
    target_scales = clearscale.BlueprintShapes({"resized_scale": Shape(zip("tczyx", (2, 2, 10, 10, 10)))})
    resolution_t = 1.0
    resolution_xyz = 3.0
    # Output is 10/5 upscaling of the (2, 2, 5, 5, 5) source array, so output scale is 3.0 / (10/5)
    expected_output_transform = [{"type": "scale", "scale": [1.0, 1.0, 1.5, 1.5, 1.5]}]

    axes = "tczyx"
    shape = Shape(zip(axes, (2, 2, 5, 5, 5)))
    result = write_ome_zarr_like_ilastik(
        shape,
        PixelSize(zip(axes, [resolution_t, 1.0, resolution_xyz, resolution_xyz, resolution_xyz])),
        export_blueprint=target_scales,
    )

    assert "multiscales" in result
    m = result["multiscales"][0]
    assert "coordinateTransformations" not in m
    assert "datasets" in m and "path" in m["datasets"][0]
    assert len(m["datasets"]) == 1
    assert m["datasets"][0]["path"] == "resized_scale"
    assert m["datasets"][0]["coordinateTransformations"] == expected_output_transform


def test_write_ome_zarr_test_transformations_multi_scale_export():
    """
    When the input is multiscale but not OME-Zarr, multiscale export should report
    resolution as scale transform and offset as multiscale translation (* scaling to get absolute values).
    """
    input_scale_key = "source_scale"
    s = 34 / 8  # source scaling (resolution) = base shape / uncropped source shape
    resolution_xyz = s
    target_scales = clearscale.BlueprintShapes(
        [
            ("weird_upscale", Shape(zip("tczyx", (2, 2, 13, 12, 12)))),
            ("downscale", Shape(zip("tczyx", (2, 2, 2, 2, 2)))),
        ]
    )
    # Expected output scaling: source scale * target scaling relative to source
    expected_upscale = pytest.approx(
        [1.0, 1.0, s * 5 / 13, s * 5 / 12, s * 5 / 12]
    )  # cropped source shape / target shape
    # 2px would be the result of scaling 5px by 2.0.
    # The scaling implementation in OpResize is precise though, so metadata should not be rounded.
    expected_downscale = pytest.approx([1.0, 1.0, s * 5 / 2, s * 5 / 2, s * 5 / 2])
    expected_multiscale_transforms = [
        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]},
        {"type": "translation", "translation": [0.0, 0.0, s * 3, s * 3, s * 3]},  # scaled offset
    ]

    axes = "tczyx"
    shape = Shape(zip(axes, (2, 2, 5, 5, 5)))
    input_multiscale = clearscale.Multiscale.from_shapes(clearscale.BlueprintShapes({input_scale_key: shape}))
    export_offset = PixelOffset(zip(axes, (0, 0, 3, 3, 3)))
    result = write_ome_zarr_like_ilastik(
        shape,
        PixelSize(zip(axes, [1.0, 1.0, resolution_xyz, resolution_xyz, resolution_xyz])),
        export_offset=export_offset,
        export_blueprint=target_scales,
        input_multiscale=input_multiscale,
        input_scale_key=input_scale_key,
    )

    assert "multiscales" in result
    m = result["multiscales"][0]
    assert "datasets" in m and "path" in m["datasets"][0]
    assert len(m["datasets"]) == 2
    assert m["datasets"][0]["path"] == "weird_upscale"
    assert m["coordinateTransformations"] == expected_multiscale_transforms
    upscale_transforms = m["datasets"][0]["coordinateTransformations"]
    assert upscale_transforms[0]["scale"] == expected_upscale
    assert m["datasets"][1]["path"] == "downscale"
    downscale_transforms = m["datasets"][1]["coordinateTransformations"]
    assert downscale_transforms[0]["scale"] == expected_downscale


def test_write_ome_zarr_test_port_ome_zarr_metadata_multi_scale_export():
    """
    See test above, but with OME-Zarr metadata on the input, translations from the source
    must be added to translation from the export offset.
    """
    input_scale_key = "source_scale"
    resolution_t = 0.1
    resolution_xyz = 2.0  # Writers might round scaling factors. We have to assume this is intentional and maintain it.
    units = {"t": "second", "z": "micrometer", "y": "micrometer", "x": "micrometer"}
    input_multiscale = clearscale.Multiscale.from_ome_zarr(
        {
            "name": "wonderful_pyramid",
            "axes": [
                {"name": "t", "type": "time", "unit": units["t"]},
                {"name": "z", "type": "space", "unit": units["z"]},
                {"name": "y", "type": "space", "unit": units["y"]},
                {"name": "x", "type": "space", "unit": units["x"]},
            ],  # Input metadata tzyx, but e.g. Probabilities output would be tczyx
            "coordinateTransformations": [{"type": "scale", "scale": "should not be accessed"}],
            "datasets": [
                {
                    "path": "upscale",  # The first scale is usually the raw data, but not necessarily
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 0.5, 0.5, 0.5]},
                    ],
                },
                {
                    "path": "raw_scale",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0]},
                        {"type": "translation", "translation": [0.1, 5.0, 2.0, 1.0]},
                    ],
                },
                {
                    "path": "source_scale",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": "should not be accessed"},
                        {"type": "translation", "translation": [3.1, 3.2, 2.1, 1.0]},
                    ],
                },
                {
                    "path": "downscale",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 4.0, 4.0, 4.0]},
                        {"type": "translation", "translation": [5.1, 3.5, 5.4, 1.0]},
                    ],
                },
            ],
        },
        shape_source=lambda path: (2, 5, 5, 5),
    )
    target_scales = clearscale.BlueprintShapes(
        [
            ("weird_upscale", Shape(zip("tczyx", (2, 2, 13, 12, 12)))),
            ("downscale", Shape(zip("tczyx", (2, 2, 2, 2, 2)))),
        ]
    )
    expected_multiscale_transform = [
        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]},
        {"type": "translation", "translation": [0.0, 0.0, 6.0, 6.0, 6.0]},  # crop translation in physical units
    ]
    s_abs = 2.0  # Even if OpResize scales precisely, output should be computed based on the input's metadata.
    upscale = [0.1, 1.0, s_abs * 5 / 13, s_abs * 5 / 12, s_abs * 5 / 12]
    downscale = [0.1, 1.0, s_abs * 5 / 2, s_abs * 5 / 2, s_abs * 5 / 2]
    expected_upscale_transform = [
        {"type": "scale", "scale": upscale},
        {"type": "translation", "translation": [3.1, 0.0, 3.2, 2.1, 1.0]},  # source scale translation
    ]
    expected_downscale_transform = [
        {"type": "scale", "scale": downscale},
        {"type": "translation", "translation": [3.1, 0.0, 3.2, 2.1, 1.0]},  # source scale translation
    ]

    axes = "tczyx"
    shape = Shape(zip(axes, (2, 2, 5, 5, 5)))
    export_offset = PixelOffset(zip(axes, (0, 0, 3, 3, 3)))
    result = write_ome_zarr_like_ilastik(
        shape,
        PixelSize(zip(axes, [resolution_t, 1.0, resolution_xyz, resolution_xyz, resolution_xyz])),
        unit=Unit(units),
        export_offset=export_offset,
        export_blueprint=target_scales,
        input_multiscale=input_multiscale,
        input_scale_key=input_scale_key,
    )

    assert "multiscales" in result
    m = result["multiscales"][0]
    assert "datasets" in m and "path" in m["datasets"][0]
    assert len(m["datasets"]) == 2
    assert "name" not in m  # Input name should not be carried over - presumably it names the raw data
    assert m["axes"] == [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]  # Axis units should be carried over
    assert m["coordinateTransformations"] == expected_multiscale_transform
    assert m["datasets"][0]["path"] == "weird_upscale"
    assert m["datasets"][0]["coordinateTransformations"] == expected_upscale_transform
    assert m["datasets"][1]["path"] == "downscale"
    assert m["datasets"][1]["coordinateTransformations"] == expected_downscale_transform


def testOpExportSlot_test_ome_zarr_single_scale():
    expected_dataset_transformations = [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}]
    # Crop offset is written as a global translation
    expected_multiscale_transformations = [
        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]},
        {"type": "translation", "translation": [0.0, 0.0, 0.0, 10.0, 20.0]},
    ]

    axes = "yx"
    shape = Shape(zip(axes, (90, 100)))
    export_offset = PixelOffset(zip(axes, (10, 20)))
    result = write_ome_zarr_like_ilastik(
        shape,
        PixelSize.identity(axes),
        export_offset=export_offset,
    )

    assert result["multiscales"][0]["datasets"][0]["coordinateTransformations"] == expected_dataset_transformations
    assert "coordinateTransformations" in result["multiscales"][0]
    assert result["multiscales"][0]["coordinateTransformations"] == expected_multiscale_transformations


def testOpExportSlot_test_ome_zarr_multi_scale():
    """Ensure multi-scale export generates one downscale for 550x510."""
    # Chunk size is 506x505 for square 2D (by BigRequestStreamer default),
    # so 550x510 is larger, and 275x255 is one chunk.

    def generate_default_target_scales(shape: Shape) -> clearscale.BlueprintShapes:
        chunk_shape = Shape(zip("yx", (506, 505))).with_axes("tczyx")
        shapes = clearscale.BlueprintShapes.downscale_powers_of_2_xyz(
            base_shape=shape.with_axes("tczyx"), shape_limit=chunk_shape.with_axes("tczyx"), rounding="floor"
        )
        return shapes

    expected_scales = OrderedDict(
        {
            "s0": OrderedDict(zip("tczyx", (1, 1, 1, 550, 510))),
            "s1": OrderedDict(zip("tczyx", (1, 1, 1, 275, 255))),
        }
    )
    axes = "yx"
    shape = Shape(zip(axes, (550, 510)))
    target_scales = generate_default_target_scales(shape)

    export_offset = PixelOffset(zip(axes, (10, 20)))
    result = write_ome_zarr_like_ilastik(
        shape,
        PixelSize.identity(axes),
        export_offset=export_offset,
        export_blueprint=target_scales,
    )

    assert generate_default_target_scales(shape) == expected_scales

    for i, scale in enumerate(["s0", "s1"]):
        expected_multiscale_transformations = [
            {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]},
            {"type": "translation", "translation": [0.0, 0.0, 0.0, 10.0, 20.0]},
        ]
        assert result["multiscales"][0]["coordinateTransformations"] == expected_multiscale_transformations
        expected_dataset_transformations = [{"type": "scale", "scale": [1.0, 1.0, 1.0, 2.0**i, 2.0**i]}]
        assert result["multiscales"][0]["datasets"][i]["coordinateTransformations"] == expected_dataset_transformations


def testOpExportSlot_test_ome_zarr_roundtrip():
    """Ensure that loading an OME-Zarr dataset and then re-exporting one of
    its scales produces the same data and metadata."""
    input_meta = Multiscale.from_ome_zarr(
        {
            "name": "input.zarr",
            "type": "sample",
            "version": "0.4",
            "axes": [
                {"type": "space", "name": "y", "unit": "nanometer"},
                {"type": "space", "name": "x", "unit": "nanometer"},
            ],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"scale": [0.2, 0.2], "type": "scale"},
                        {"translation": [0.0, 0.0], "type": "translation"},
                    ],
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [
                        {"scale": [1.4, 1.4], "type": "scale"},
                        {"translation": [7.62, 8.49], "type": "translation"},
                    ],
                },
            ],
            "coordinateTransformations": [
                {"scale": [1.0, 1.0], "type": "scale"},
                {"translation": [0.0, 0.0], "type": "translation"},
            ],
        },
        shape_source={"s0": (89, 99), "s1": (13, 15)},
    )
    # Expected written meta is the same as input, but tczyx, only with the respective scale,
    # and with no name
    expected_meta_s0 = [
        {
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"},
            ],
            "datasets": [
                {
                    "coordinateTransformations": [{"scale": [1.0, 1.0, 1.0, 0.2, 0.2], "type": "scale"}],
                    "path": "s0",
                }
            ],
            "coordinateTransformations": [
                {"scale": [1.0, 1.0, 1.0, 1.0, 1.0], "type": "scale"},
                {"translation": [0.0, 0.0, 0.0, 0.0, 0.0], "type": "translation"},
            ],
            "version": "0.4",
        }
    ]
    expected_meta_s1 = [
        {
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space", "unit": "nanometer"},
                {"name": "x", "type": "space", "unit": "nanometer"},
            ],
            "coordinateTransformations": [
                {"scale": [1.0, 1.0, 1.0, 1.0, 1.0], "type": "scale"},
                {"translation": [0.0, 0.0, 0.0, 0.0, 0.0], "type": "translation"},
            ],
            "datasets": [
                {
                    "coordinateTransformations": [
                        {"scale": [1.0, 1.0, 1.0, 1.4, 1.4], "type": "scale"},
                        {"translation": [0.0, 0.0, 0.0, 7.62, 8.49], "type": "translation"},
                    ],
                    "path": "s1",
                }
            ],
            "version": "0.4",
        }
    ]

    axes = "yx"

    # Raw scale first
    shape = Shape(zip(axes, (89, 99)))
    result = write_ome_zarr_like_ilastik(
        shape,
        input_meta["s0"].pixel_size,
        unit=input_meta["s0"].unit,
        input_multiscale=input_meta,
        input_scale_key="s0",
    )
    assert result["multiscales"] == expected_meta_s0

    # Same thing for the second scale
    shape = Shape(zip(axes, (13, 15)))
    result = write_ome_zarr_like_ilastik(
        shape,
        input_meta["s1"].pixel_size,
        unit=input_meta["s1"].unit,
        input_multiscale=input_meta,
        input_scale_key="s1",
    )
    assert result["multiscales"] == expected_meta_s1


def testOpExportSlot_test_ome_zarr_roundtrip_multiscale():
    """
    Ensure metadata roundtrips correctly when re-exporting a multiscale dataset as-is.
    Output metadata won't/shouldn't be *identical* to input. Even when just re-exporting
    the loaded data unprocessed, the export is actually a new multiscale. It's just
    scaled to the same scaling levels as the source pyramid. ilastik's scaling implementation
    (OpResize) will not reproduce the downscaled source data.
    The exported metadata must correctly describe the export, not carry over metadata
    from the source falsely. See detailed comment below.
    """

    def _match_target_scales_to_input(
        export_shape: Mapping[AxisKey, int], input_scales: clearscale.Multiscale, input_key: str
    ) -> clearscale.BlueprintShapes:
        SPATIAL_AXES = ["z", "y", "x"]
        source_scale_shape = input_scales[input_key].shape
        if source_scale_shape.matches(export_shape, only=SPATIAL_AXES):
            # Unmodified source shape - reproduce exact multiscale shapes
            shapes = clearscale.BlueprintShapes.from_multiscale(input_scales)
        else:

            def two_spatials_or_is_input(scale: str, shape: clearscale.Shape):
                remaining_spatial = len(shape.non_singleton_axes(SPATIAL_AXES))
                return remaining_spatial > 1 or scale == input_key

            shapes = clearscale.BlueprintShapes.from_multiscale_rescaled(
                input_scales,
                target_shape=export_shape,
                source_key=input_key,
                scaled_axes=SPATIAL_AXES,
                rounding="floor",
            ).filter_items(two_spatials_or_is_input)

        return shapes.with_axes("tczyx").with_sizes(export_shape, only_axes="tc")

    def match_target_scales_to_input_excluding_upscales(
        export_shape: Mapping[AxisKey, int], input_scales: clearscale.Multiscale, input_key: str
    ) -> clearscale.BlueprintShapes:
        """We assume people don't generally want to upscale lower-resolution segmentations to raw scale."""
        # Since input_scales is ordered largest-to-smallest, simply drop matching scales before input_key.
        all_matching_scales = _match_target_scales_to_input(export_shape, input_scales, input_key)
        return all_matching_scales.drop_before(input_key)

    input_meta = Multiscale.from_ome_zarr(
        {
            "name": "input.zarr",
            "type": "sample",
            "version": "0.4",
            "axes": [
                {"type": "space", "name": "z", "unit": "micrometer"},
                {"type": "space", "name": "y", "unit": "nanometer"},
                {"type": "space", "name": "x", "unit": "nanometer"},
            ],
            "datasets": [
                {
                    "path": "s0",
                    "coordinateTransformations": [
                        {"scale": [1.0, 0.2, 0.2], "type": "scale"},
                        {"translation": [0.0, 0.0, 0.0], "type": "translation"},
                    ],
                },
                {
                    "path": "s1",
                    "coordinateTransformations": [
                        {"scale": [1.0, 0.6, 0.6], "type": "scale"},
                        {"translation": [0.0, 7.62, 8.49], "type": "translation"},
                    ],
                },
            ],
            "coordinateTransformations": [
                {"scale": [0.3, 1.0, 1.0], "type": "scale"},
                {"translation": [2.0, 0.6, 0.0], "type": "translation"},
            ],
        },
        shape_source={"s0": (16, 15, 15), "s1": (16, 5, 5)},
    )
    # Expected written meta is the same as input, but:
    # - tczyx
    # - no name
    # - "s1" transformations are discarded. The exported "s1" is a newly generated downscale.
    expected_ms = {
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "nanometer"},
            {"name": "x", "type": "space", "unit": "nanometer"},
        ],
        "datasets": [
            {
                "coordinateTransformations": [
                    {"scale": [1.0, 1.0, 1.0, 0.2, 0.2], "type": "scale"},
                ],
                "path": "s0",
            },
            {
                "coordinateTransformations": [
                    {"scale": [1.0, 1.0, 1.0, 0.6, 0.6], "type": "scale"},
                ],
                "path": "s1",
            },
        ],
        "coordinateTransformations": [
            {"scale": [1.0, 1.0, 0.3, 1.0, 1.0], "type": "scale"},
            {"translation": [0.0, 0.0, 2.0, 0.6, 0.0], "type": "translation"},
        ],
        "version": "0.4",
        "metadata": {
            "description": "ilastik's lazyflow.operators.opResize.OpResize "
            "is a lazy implementation of skimage.transform.resize.",
            "kwargs": {"anti_aliasing": True, "order": 1, "preserve_range": True},
            "method": "skimage.transform.resize",
            "version": "0.24.0",
        },
    }

    result = write_ome_zarr_like_ilastik(
        input_meta["s0"].shape,
        input_meta["s0"].pixel_size,
        unit=input_meta["s0"].unit,
        export_blueprint=match_target_scales_to_input_excluding_upscales(input_meta["s0"].shape, input_meta, "s0"),
        input_multiscale=input_meta,
        input_scale_key="s0",
    )

    # Raw scale first
    assert len(result["multiscales"]) == 1
    written_ms = result["multiscales"][0]
    assert written_ms["axes"] == expected_ms["axes"]
    assert written_ms["metadata"] == expected_ms["metadata"]
    assert written_ms["version"] == expected_ms["version"]
    assert "coordinateTransformations" in written_ms
    _assert_transforms_eq(written_ms["coordinateTransformations"], expected_ms["coordinateTransformations"])
    assert len(written_ms["datasets"]) == len(expected_ms["datasets"])
    for written_ds, expected_ds in zip(written_ms["datasets"], expected_ms["datasets"]):
        assert written_ds["path"] == expected_ds["path"]
        _assert_transforms_eq(written_ds["coordinateTransformations"], expected_ds["coordinateTransformations"])


def _assert_transforms_eq(written_transforms, expected_transforms):
    assert len(written_transforms) == len(expected_transforms)
    if len(expected_transforms) == 0:
        return
    assert "scale" in written_transforms[0]
    written_scale = written_transforms[0]["scale"]
    expected_scale = expected_transforms[0]["scale"]
    assert len(written_scale) == len(expected_scale)
    assert written_scale == pytest.approx(expected_scale, abs=1e-15)
    if len(expected_transforms) == 1:
        return
    assert "translation" in written_transforms[1]
    written_transl = written_transforms[1]["translation"]
    expected_transl = expected_transforms[1]["translation"]
    assert len(written_transl) == len(expected_transl)
    assert written_transl == pytest.approx(expected_transl, abs=1e-15)
