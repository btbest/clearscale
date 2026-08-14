from unittest import mock
from unittest.mock import Mock

import pytest

from clearscale import Multiscale, Scene
from clearscale._collections import OmeZarrGroup


class MockZarrGroup:
    def __init__(self, attrs=None, **objects):
        self.attrs = {} if attrs is None else attrs
        self._objects = objects

    def __getitem__(self, path: str):
        return self._objects[path]


def test_ome_zarr_group_with_no_metadata():
    group = MockZarrGroup()
    ome_group = OmeZarrGroup(group)
    assert ome_group.version is None
    assert ome_group.multiscales == ()
    assert ome_group.scenes == ()
    assert ome_group.child_paths == ()


def test_ome_zarr_group_ignores_non_list_multiscales(monkeypatch):
    group = MockZarrGroup(attrs={"multiscales": {"not": "a list"}})
    with mock.patch("clearscale.Multiscale.from_ome_zarr") as multiscale_construct:
        result = OmeZarrGroup(group)

    assert not result.multiscales
    multiscale_construct.assert_not_called()


def test_ome_zarr_group_ignores_non_mapping_scene(monkeypatch):
    group = MockZarrGroup(attrs={"scene": ["not", "a", "mapping"]})
    with mock.patch("clearscale.Scene.from_ome_zarr") as scene_construct:
        result = OmeZarrGroup(group)

    assert result.scenes == ()
    scene_construct.assert_not_called()


def test_ome_zarr_group_ignores_empty_scene_mapping(monkeypatch):
    group = MockZarrGroup(attrs={"scene": {}})
    with mock.patch("clearscale.Scene.from_ome_zarr") as scene_construct:
        result = OmeZarrGroup(group)

    assert result.scenes == ()
    scene_construct.assert_not_called()


def test_ome_zarr_group_loads_multiscales_in_order(monkeypatch):
    multiscale_jsons = [
        {"name": "first"},
        {"name": "second"},
        {"name": "third"},
    ]
    group = MockZarrGroup(attrs={"multiscales": multiscale_jsons})
    multiscale_0 = object()
    multiscale_1 = object()
    multiscale_2 = object()
    with mock.patch(
        "clearscale.Multiscale.from_ome_zarr", side_effect=[multiscale_0, multiscale_1, multiscale_2]
    ) as multiscale_construct:
        result = OmeZarrGroup(group)

    assert multiscale_construct.call_args_list == [
        ((multiscale_jsons[0],), {"shape_source": group}),
        ((multiscale_jsons[1],), {"shape_source": group}),
        ((multiscale_jsons[2],), {"shape_source": group}),
    ]
    assert result.multiscales == (multiscale_0, multiscale_1, multiscale_2)


def test_ome_zarr_group_loads_scene(monkeypatch):
    scene_json = {"name": "test-scene"}
    group = MockZarrGroup(attrs={"scene": scene_json})
    scene = object()
    with mock.patch("clearscale.Scene.from_ome_zarr", return_value=scene) as scene_construct:
        result = OmeZarrGroup(group)

    assert len(result.scenes) == 1
    assert result.scenes[0] is scene
    scene_construct.assert_called_once_with(scene_json)


def test_ome_zarr_group_collects_labels_well_and_plate_paths_in_order():
    attrs = {
        "labels": {"labels": ["label-0", "label-1"]},
        "well": {
            "images": [
                {"path": "image-0"},
                {"path": "image-1"},
            ]
        },
        "plate": {
            "wells": [
                {"path": "well-0"},
                {"path": "well-1"},
            ]
        },
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.child_paths == (
        "label-0",
        "label-1",
        "image-0",
        "image-1",
        "well-0",
        "well-1",
    )


@pytest.mark.parametrize(
    "attrs",
    [
        {"labels": None},
        {"labels": "labels"},
        {"labels": []},
        {"well": None},
        {"well": "well"},
        {"plate": None},
        {"plate": "plate"},
        {"labels": {"labels": "not-a-list"}},
        {"well": {"images": "not-a-list"}},
        {"plate": {"wells": "not-a-list"}},
    ],
)
def test_ome_zarr_group_version_ignores_non_mapping_or_non_list_metadata(attrs):
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.version is None
    assert not ome_group.multiscales
    assert not ome_group.scenes
    assert not ome_group.child_paths


def test_ome_zarr_group_labels_list_only_accepts_strings():
    attrs = {
        "labels": {
            "labels": [
                "nuclei",
                1,
                None,
                {"path": "cells"},
                "",
                "cytoplasm",
            ]
        }
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.child_paths == ("nuclei", "cytoplasm")


def test_ome_zarr_group_well_images_only_accept_mapping_entries_with_string_path():
    attrs = {
        "well": {
            "images": [
                {"path": "image-0"},
                {"path": 1},
                {},
                None,
                "image-1",
                {"path": ""},
            ]
        }
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.child_paths == ("image-0",)


def test_ome_zarr_group_plate_wells_only_accept_mapping_entries_with_string_path():
    attrs = {
        "plate": {
            "wells": [
                {"path": "A/1"},
                {"path": 1},
                {},
                None,
                "B/2",
                {"path": ""},
            ]
        }
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.child_paths == ("A/1",)


def test_ome_zarr_group_prefers_top_level_version():
    group = MockZarrGroup(
        attrs={
            "version": "0.6",
            "scene": {"version": "0.5"},
            "labels": {"version": "0.4"},
            "well": {"version": "0.3"},
            "plate": {"version": "0.2"},
        }
    )
    result = OmeZarrGroup(group)
    assert result.version == "0.6"


@pytest.mark.parametrize(
    ("attrs", "expected_version"),
    [
        ({"labels": {"version": "0.6"}}, "0.6"),
        ({"well": {"version": "0.5"}}, "0.5"),
        ({"plate": {"version": "0.4"}}, "0.4"),
    ],
)
def test_ome_zarr_group_version_reads_from_labels_well_or_plate(attrs, expected_version):
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.version == expected_version


def test_ome_zarr_group_version_prefers_labels_over_well_and_plate():
    attrs = {
        "labels": {"version": "labels-version"},
        "well": {"version": "well-version"},
        "plate": {"version": "plate-version"},
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.version == "labels-version"


def test_ome_zarr_group_version_prefers_well_over_plate_when_labels_version_missing():
    attrs = {
        "well": {"version": "well-version"},
        "plate": {"version": "plate-version"},
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.version == "well-version"


@pytest.mark.parametrize(
    "attrs",
    [
        {"version": 0.5},
        {"version": None},
        {"version": []},
        {"version": {}},
    ],
)
def test_ome_zarr_group_ignores_non_string_versions(attrs):
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.version is None


def test_ome_zarr_group_version_uses_well_when_label_non_string():
    attrs = {
        "labels": {"version": 0.6},
        "well": {"version": "0.5"},
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.version == "0.5"


def test_ome_zarr_group_version_uses_well_when_label_empty_string():
    attrs = {
        "labels": {"version": ""},
        "well": {"version": "0.5"},
        "plate": {"version": "0.4"},
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup(group)
    assert ome_group.version == "0.5"


def test_ome_zarr_group_uses_multiscale_version_when_top_level_missing(monkeypatch):
    from_ome_zarr = Mock(return_value=object())
    monkeypatch.setattr(Multiscale, "from_ome_zarr", from_ome_zarr)

    group = MockZarrGroup(
        attrs={
            "multiscales": [
                {
                    "version": "0.5",
                }
            ]
        }
    )

    result = OmeZarrGroup(group)

    assert result.version == "0.5"


def test_ome_zarr_group_uses_first_string_multiscale_version(monkeypatch):
    from_ome_zarr = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(Multiscale, "from_ome_zarr", from_ome_zarr)

    group = MockZarrGroup(
        attrs={
            "multiscales": [
                {"version": 0.5},
                {"version": "0.4"},
            ]
        }
    )

    result = OmeZarrGroup(group)

    assert result.version == "0.4"


def test_ome_zarr_group_uses_scene_version_when_multiscale_version_missing(monkeypatch):
    monkeypatch.setattr(Multiscale, "from_ome_zarr", Mock())

    scene = object()
    scene_from_ome_zarr = Mock(return_value=scene)
    monkeypatch.setattr(Scene, "from_ome_zarr", scene_from_ome_zarr)

    group = MockZarrGroup(
        attrs={
            "scene": {
                "version": "0.5",
            }
        }
    )

    result = OmeZarrGroup(group)

    assert result.version == "0.5"
    assert len(result.scenes) == 1
    assert result.scenes[0] is scene
