from unittest import mock
from unittest.mock import Mock

import pytest

from clearscale import Multiscale, Scene
from clearscale._collections import OmeZarrGroup, GroupKind, ChildRef


@pytest.mark.parametrize(
    "kwargs, expected_kind",
    [
        pytest.param({}, None, id="empty"),
        pytest.param({"multiscales": (Mock(),)}, GroupKind.MULTISCALE, id="single-multiscale"),
        pytest.param({"multiscales": (Mock(), Mock())}, GroupKind.COLLECTION, id="multiple-multiscales"),
        pytest.param({"scenes": (Mock(),)}, GroupKind.SCENE, id="scene-without-children"),
        pytest.param(
            {"scenes": (Mock(),), "children": (ChildRef("multiscale", Mock()), ChildRef("multiscale", Mock()))},
            GroupKind.SCENE,
            id="scene-with-multiscale-children",
        ),
        pytest.param({"children": (ChildRef("well", Mock()),)}, GroupKind.PLATE, id="plate"),
        pytest.param(
            {"children": (ChildRef("well", Mock()), ChildRef("well", Mock()))},
            GroupKind.PLATE,
            id="plate-multiple-wells",
        ),
        pytest.param({"children": (ChildRef("multiscale", Mock()),)}, GroupKind.WELL, id="well"),
        pytest.param(
            {"children": (ChildRef("multiscale", Mock()), ChildRef("multiscale", Mock()))},
            GroupKind.WELL,
            id="well-multiple-multiscales",
        ),
        pytest.param({"children": (ChildRef("label", Mock()),)}, GroupKind.LABELS, id="labels"),
        pytest.param(
            {"children": (ChildRef("label", Mock()), ChildRef("label", Mock()))},
            GroupKind.LABELS,
            id="labels-multiple-labels",
        ),
        pytest.param({"multiscales": (Mock(),), "scenes": (Mock(),)}, GroupKind.COLLECTION, id="multiscale-and-scene"),
        pytest.param(
            {"multiscales": (Mock(), Mock()), "children": (ChildRef("multiscale", Mock()),)},
            GroupKind.COLLECTION,
            id="multiscales-and-children",
        ),
        pytest.param({"scenes": (Mock(), Mock())}, GroupKind.COLLECTION, id="multiple-scenes"),
        pytest.param(
            {"scenes": (Mock(),), "children": (ChildRef("well", Mock()),)},
            GroupKind.COLLECTION,
            id="scene-with-non-multiscale-child",
        ),
        pytest.param(
            {"children": (ChildRef("well", Mock()), ChildRef("multiscale", Mock()))},
            GroupKind.COLLECTION,
            id="mixed-child-types",
        ),
    ],
)
def test_ome_zarr_group_init_detects_group_kind(kwargs, expected_kind):
    group = OmeZarrGroup(**kwargs)
    assert group.kind is expected_kind


class MockZarrGroup:
    def __init__(self, attrs=None, **objects):
        self.attrs = {} if attrs is None else attrs
        self._objects = objects

    def __getitem__(self, path: str):
        return self._objects[path]


def test_ome_zarr_group_with_no_metadata():
    group = MockZarrGroup()
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is None
    assert ome_group.version is None
    assert ome_group.multiscales == ()
    assert ome_group.scenes == ()
    assert ome_group.children == ()


@pytest.mark.parametrize(
    "obj, expected_kind",
    [
        (Mock(spec=Multiscale), GroupKind.MULTISCALE),
        (Mock(spec=Scene), GroupKind.SCENE),
    ],
)
def test_ome_zarr_group_single(obj, expected_kind):
    group = OmeZarrGroup.from_single(obj)
    assert group.kind is expected_kind


def test_ome_zarr_group_single_raises_unknown():
    with pytest.raises(TypeError, match="Must be called with a Multiscale or Scene"):
        _ = OmeZarrGroup.from_single([Mock(spec=Multiscale), Mock(spec=Multiscale)])  # type: ignore[arg-type]


def test_ome_zarr_group_ignores_non_list_multiscales(monkeypatch):
    group = MockZarrGroup(attrs={"multiscales": {"not": "a list"}})
    with mock.patch("clearscale.Multiscale.from_ome_zarr") as multiscale_construct:
        result = OmeZarrGroup.from_group(group)

    assert result.kind is None
    assert not result.multiscales
    multiscale_construct.assert_not_called()


def test_ome_zarr_group_ignores_non_mapping_scene(monkeypatch):
    group = MockZarrGroup(attrs={"scene": ["not", "a", "mapping"]})
    with mock.patch("clearscale.Scene.from_ome_zarr") as scene_construct:
        result = OmeZarrGroup.from_group(group)

    assert result.kind is None
    assert result.scenes == ()
    scene_construct.assert_not_called()


def test_ome_zarr_group_ignores_empty_scene_mapping(monkeypatch):
    group = MockZarrGroup(attrs={"scene": {}})
    with mock.patch("clearscale.Scene.from_ome_zarr") as scene_construct:
        result = OmeZarrGroup.from_group(group)

    assert result.kind is None
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
        result = OmeZarrGroup.from_group(group)

    assert result.kind is GroupKind.COLLECTION
    assert multiscale_construct.call_args_list == [
        ((multiscale_jsons[0],), {"shape_source": group}),
        ((multiscale_jsons[1],), {"shape_source": group}),
        ((multiscale_jsons[2],), {"shape_source": group}),
    ]
    assert result.multiscales == (multiscale_0, multiscale_1, multiscale_2)
    assert result.maybe_subgroups == ("labels",)


def test_ome_zarr_group_ignores_legacy_keys_when_ome_present(monkeypatch):
    ome_ms = object()
    legacy_ms = object()
    ome_ms_json = {"name": "ome_ms"}
    legacy_ms_json = {"name": "legacy_ms"}
    group = MockZarrGroup(
        attrs={
            "ome": {"multiscales": [ome_ms_json]},
            "multiscales": [legacy_ms_json],
        }
    )
    with mock.patch("clearscale.Multiscale.from_ome_zarr", side_effect=[ome_ms, legacy_ms]) as multiscale_construct:
        result = OmeZarrGroup.from_group(group)

    multiscale_construct.assert_called_once_with(ome_ms_json, shape_source=group)
    assert result.kind is GroupKind.MULTISCALE
    assert result.multiscales == (ome_ms,)
    assert result.maybe_subgroups == ("labels",)


def test_ome_zarr_group_loads_scene(monkeypatch):
    scene_json = {"name": "test-scene"}
    group = MockZarrGroup(attrs={"ome": {"scene": scene_json}})
    scene = mock.Mock(unresolved_paths=["path1", "path2"])
    with mock.patch("clearscale.Scene.from_ome_zarr", return_value=scene) as scene_construct:
        result = OmeZarrGroup.from_group(group)

    assert result.kind is GroupKind.SCENE
    assert len(result.scenes) == 1
    assert result.scenes[0] is scene
    assert [child.file.path for child in result.children] == ["path1", "path2"]
    scene_construct.assert_called_once_with(scene_json)


@pytest.mark.parametrize("attrs", [{"bioformats2raw.layout": 3}, {"ome": {"bioformats2raw.layout": 3}}])
def test_ome_zarr_group_recognises_bf2raw_marker(attrs):
    group = MockZarrGroup(attrs=attrs)
    result = OmeZarrGroup.from_group(group)

    assert result.kind is GroupKind.BF2RAW
    assert result.version is None
    assert not result.multiscales
    assert not result.scenes
    assert not result.children
    assert result.maybe_subgroups == ("OME",)


def test_ome_zarr_group_collects_labels_series_well_and_plate_paths_in_order():
    attrs = {
        "labels": ["label-0", "label-1"],
        "series": ["series-0", "series-1"],
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
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.COLLECTION
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == (
        "label",
        "label",
        "multiscale",
        "multiscale",
        "multiscale",
        "multiscale",
        "well",
        "well",
    )
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == (
        "label-0",
        "label-1",
        "../series-0",  # <3 bf2raw
        "../series-1",
        "image-0",
        "image-1",
        "well-0",
        "well-1",
    )


@pytest.mark.parametrize(
    "attrs",
    [
        {"labels": None},
        {"labels": "not-a-list"},
        {"labels": []},
        {"series": None},
        {"series": "not-a-list"},
        {"series": []},
        {"well": None},
        {"well": "well"},
        {"plate": None},
        {"plate": "plate"},
        {"well": {"images": "not-a-list"}},
        {"plate": {"wells": "not-a-list"}},
        {"ome": {"labels": None}},
        {"ome": {"labels": "not-a-list"}},
        {"ome": {"labels": []}},
        {"ome": {"well": None}},
        {"ome": {"well": "well"}},
        {"ome": {"plate": None}},
        {"ome": {"plate": "plate"}},
        {"ome": {"well": {"images": "not-a-list"}}},
        {"ome": {"plate": {"wells": "not-a-list"}}},
    ],
)
def test_ome_zarr_group_version_ignores_non_mapping_or_non_list_metadata(attrs):
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is None
    assert ome_group.version is None
    assert not ome_group.multiscales
    assert not ome_group.scenes
    assert not ome_group.children


def test_ome_zarr_group_labels_list_only_accepts_strings():
    attrs = {
        "labels": [
            "nuclei",
            1,
            None,
            {"path": "cells"},
            "",
            "cytoplasm",
        ]
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.LABELS
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == ("label", "label")
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == ("nuclei", "cytoplasm")


def test_ome_zarr_group_labels_list_only_accepts_strings_under_ome():
    attrs = {
        "ome": {
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
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.LABELS
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == ("label", "label")
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == ("nuclei", "cytoplasm")


def test_ome_zarr_group_series_list_only_accepts_strings():
    attrs = {
        "series": [
            "0",
            1,
            None,
            {"path": "cells"},
            "",
            "1",
        ]
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.BF2RAW_OME
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == ("multiscale", "multiscale")
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == ("../0", "../1")  # <3 bf2raw


def test_ome_zarr_group_series_list_only_accepts_strings_under_ome():
    attrs = {
        "ome": {
            "series": [
                "0",
                1,
                None,
                {"path": "cells"},
                "",
                "1",
            ]
        }
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.BF2RAW_OME
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == ("multiscale", "multiscale")
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == ("../0", "../1")  # <3 bf2raw


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
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.WELL
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == ("multiscale",)
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == ("image-0",)


def test_ome_zarr_group_well_images_only_accept_mapping_entries_with_string_path_under_ome():
    attrs = {
        "ome": {
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
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.WELL
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == ("multiscale",)
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == ("image-0",)


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
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.PLATE
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == ("well",)
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == ("A/1",)


def test_ome_zarr_group_plate_wells_only_accept_mapping_entries_with_string_path_under_ome():
    attrs = {
        "ome": {
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
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is GroupKind.PLATE
    assert len(ome_group.children) > 0
    assert tuple(c.child_type for c in ome_group.children) == ("well",)
    assert all(c.file.path_type == "zarr" for c in ome_group.children)
    paths = tuple(c.file.path for c in ome_group.children)
    assert paths == ("A/1",)


@pytest.mark.parametrize(
    ("attrs", "expected_version"),
    [
        (
            {
                "ome": {
                    "version": "1",
                    "multiscales": [{"version": "2"}],
                    "scene": {"version": "3"},
                    "well": {"version": "4"},
                    "plate": {"version": "5"},
                },
                "version": "6",
                "multiscales": [{"version": "7"}],
                "scene": {"version": "8"},
                "well": {"version": "9"},
                "plate": {"version": "10"},
            },
            "1",
        ),
        (
            {
                "ome": {
                    "multiscales": [{"version": "2"}],
                    "scene": {"version": "3"},
                    "well": {"version": "4"},
                    "plate": {"version": "5"},
                },
                "version": "6",
                "multiscales": [{"version": "7"}],
                "scene": {"version": "8"},
                "well": {"version": "9"},
                "plate": {"version": "10"},
            },
            "2",
        ),
        (
            {
                "ome": {"scene": {"version": "3"}, "well": {"version": "4"}, "plate": {"version": "5"}},
                "version": "6",
                "multiscales": [{"version": "7"}],
                "scene": {"version": "8"},
                "well": {"version": "9"},
                "plate": {"version": "10"},
            },
            "3",
        ),
        (
            {
                "ome": {"well": {"version": "4"}, "plate": {"version": "5"}},
                "version": "6",
                "multiscales": [{"version": "7"}],
                "scene": {"version": "8"},
                "well": {"version": "9"},
                "plate": {"version": "10"},
            },
            "4",
        ),
        (
            {
                "ome": {"plate": {"version": "5"}},
                "version": "6",
                "multiscales": [{"version": "7"}],
                "scene": {"version": "8"},
                "well": {"version": "9"},
                "plate": {"version": "10"},
            },
            "5",
        ),
        (
            {
                "version": "6",
                "multiscales": [{"version": "7"}],
                "scene": {"version": "8"},
                "well": {"version": "9"},
                "plate": {"version": "10"},
            },
            "6",
        ),
        (
            {
                "multiscales": [{"version": "7"}],
                "scene": {"version": "8"},
                "well": {"version": "9"},
                "plate": {"version": "10"},
            },
            "7",
        ),
        ({"scene": {"version": "8"}, "well": {"version": "9"}, "plate": {"version": "10"}}, "8"),
        ({"well": {"version": "9"}, "plate": {"version": "10"}}, "9"),
        ({"plate": {"version": "10"}}, "10"),
    ],
)
def test_ome_zarr_group_version_reads_in_priority_order(monkeypatch, attrs, expected_version):
    monkeypatch.setattr(Multiscale, "from_ome_zarr", Mock(return_value=object()))
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.version == expected_version


def test_ome_zarr_group_ignores_non_ome_meta_when_ome_present_even_if_invalid():
    attrs = {
        "ome": None,
        "multiscales": {"version": "0.5"},
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.kind is None
    assert ome_group.version is None


@pytest.mark.parametrize("invalid", [0.6, "", None, {}])
def test_ome_zarr_group_version_uses_fallback_when_higher_prio_invalid(invalid):
    attrs = {
        "version": invalid,
        "multiscales": [{"version": invalid}],
        "well": {"version": "0.5"},
    }
    group = MockZarrGroup(attrs)
    ome_group = OmeZarrGroup.from_group(group)
    assert ome_group.version == "0.5"


def test_ome_zarr_group_uses_multiscale_version_when_top_level_missing(monkeypatch):
    from_ome_zarr = Mock(return_value=object())
    monkeypatch.setattr(Multiscale, "from_ome_zarr", from_ome_zarr)

    group = MockZarrGroup(attrs={"multiscales": [{"version": "0.5"}]})

    result = OmeZarrGroup.from_group(group)

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

    result = OmeZarrGroup.from_group(group)

    assert result.version == "0.4"
