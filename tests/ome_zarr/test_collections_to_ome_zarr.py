import re
import warnings
from unittest.mock import Mock

import pytest
from clearscale import Multiscale, OmeZarrGroup, ChildRef, FileRef, Scene


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6.rc0"])
def test_ome_zarr_group_to_attrs_empty(version):
    assert OmeZarrGroup().to_attrs(version) == {}


@pytest.mark.parametrize(
    "version, expected",
    [
        ("0.4", {"multiscales": [{"ms": "meta"}]}),
        ("0.5", {"ome": {"version": "0.5", "multiscales": [{"ms": "meta"}]}}),
    ],
)
def test_ome_zarr_group_to_attrs_multiscale(version, expected):
    multiscale = Mock(spec=Multiscale)
    multiscale.to_ome_zarr.return_value = {"ms": "meta"}
    group = OmeZarrGroup(multiscales=(multiscale,))
    result = group.to_attrs(version)
    assert result == expected
    multiscale.to_ome_zarr.assert_called_once_with(version=version, axis_types=None)


def test_ome_zarr_group_to_attrs_multiscale_0_6_rc0():
    version, expected = ("0.6.rc0", {"ome": {"version": "0.6.rc0", "multiscales": [{"ms": "meta"}]}})
    multiscale = Mock(spec=Multiscale)
    multiscale.to_ome_zarr.return_value = {"ms": "meta"}
    group = OmeZarrGroup(multiscales=(multiscale,))
    with pytest.warns(UserWarning, match="not a stable version"):
        result = group.to_attrs(version)
    assert result == expected
    multiscale.to_ome_zarr.assert_called_once_with(version=version, axis_types=None)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_ome_zarr_group_to_attrs_scene_rejects_unsupported_versions(version):
    scene = Mock(spec=Scene)
    group = OmeZarrGroup(scenes=(scene,))
    with pytest.raises(
        ValueError,
        match=rf"Cannot write this group in OME-Zarr version {re.escape(version)}: scene groups are not supported",
    ):
        group.to_attrs(version)
    scene.to_ome_zarr.assert_not_called()


def test_ome_zarr_group_to_attrs_scene():
    scene = Mock(spec=Scene)
    scene.to_ome_zarr.return_value = {"scene": "meta"}
    group = OmeZarrGroup(scenes=(scene,))
    with pytest.warns(UserWarning, match="not a stable version"):
        result = group.to_attrs("0.6.rc0")
    assert result == {"ome": {"version": "0.6.rc0", "scene": {"scene": "meta"}}}
    scene.to_ome_zarr.assert_called_once_with(version="0.6.rc0")


@pytest.mark.parametrize(
    "version, expected",
    [
        ("0.4", {"labels": ["labels/a", "labels/b"]}),
        ("0.5", {"ome": {"version": "0.5", "labels": ["labels/a", "labels/b"]}}),
        ("0.6.rc0", {"ome": {"version": "0.6.rc0", "labels": ["labels/a", "labels/b"]}}),
    ],
)
def test_ome_zarr_group_to_attrs_labels(version, expected):
    children = (ChildRef("label", FileRef("labels/a")), ChildRef("label", FileRef("labels/b")))
    group = OmeZarrGroup(children=children)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = group.to_attrs(version)
    assert result == expected


@pytest.mark.parametrize("child_type", ["well", "multiscale"])
@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6.rc0"])
def test_ome_zarr_group_to_attrs_plate_and_well_not_implemented(child_type, version):
    group = OmeZarrGroup(children=(ChildRef(child_type, FileRef("child")),))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError, match="Writing plate and well metadata is not supported yet"):
            group.to_attrs(version)


@pytest.mark.parametrize(
    "version, expected",
    [
        ("0.4", {"multiscales": [{"id": 0}, {"id": 1}]}),
        ("0.5", {"ome": {"version": "0.5", "multiscales": [{"id": 0}, {"id": 1}]}}),
    ],
)
def test_ome_zarr_group_to_attrs_multiple_multiscales_pre_transforms(version, expected):
    multiscale_0 = Mock(spec=Multiscale)
    multiscale_1 = Mock(spec=Multiscale)
    multiscale_0.to_ome_zarr.return_value = {"id": 0}
    multiscale_1.to_ome_zarr.return_value = {"id": 1}
    group = OmeZarrGroup(multiscales=(multiscale_0, multiscale_1))
    with pytest.warns(UserWarning, match="multiple multiscales"):
        result = group.to_attrs(version)
    assert result == expected
    multiscale_0.to_ome_zarr.assert_called_once_with(version=version, axis_types=None)
    multiscale_1.to_ome_zarr.assert_called_once_with(version=version, axis_types=None)


def test_ome_zarr_group_to_attrs_multiple_multiscales_0_6():
    version, expected = ("0.6.rc0", {"ome": {"version": "0.6.rc0", "multiscales": [{"id": 0}, {"id": 1}]}})
    multiscale_0 = Mock(spec=Multiscale)
    multiscale_1 = Mock(spec=Multiscale)
    multiscale_0.to_ome_zarr.return_value = {"id": 0}
    multiscale_1.to_ome_zarr.return_value = {"id": 1}
    group = OmeZarrGroup(multiscales=(multiscale_0, multiscale_1))
    with pytest.warns(UserWarning, match="multiple multiscales"):
        with pytest.warns(UserWarning, match="not a stable version"):
            result = group.to_attrs(version)
    assert result == expected
    multiscale_0.to_ome_zarr.assert_called_once_with(version=version, axis_types=None)
    multiscale_1.to_ome_zarr.assert_called_once_with(version=version, axis_types=None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"multiscales": (Mock(spec=Multiscale),), "scenes": (Mock(spec=Scene),)},
        {"scenes": (Mock(spec=Scene), Mock(spec=Scene))},
        {"children": (ChildRef("label", FileRef("label")), ChildRef("well", FileRef("well")))},
        {"multiscales": (Mock(spec=Multiscale),), "children": (ChildRef("well", FileRef("well")),)},
    ],
)
@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6.rc0"])
def test_ome_zarr_group_to_attrs_collection_rejects_unsupported_combinations(kwargs, version):
    group = OmeZarrGroup(**kwargs)
    with pytest.raises((ValueError, NotImplementedError)):
        group.to_attrs(version)


@pytest.mark.parametrize("version", ["0.3", "0.6", "", "1.0"])
def test_ome_zarr_group_to_attrs_rejects_unsupported_version(version):
    with pytest.raises(ValueError, match="Cannot write OME-Zarr"):
        OmeZarrGroup().to_attrs(version)


def test_ome_zarr_group_to_attrs_0_4_does_not_add_group_version():
    multiscale = Mock(spec=Multiscale)
    multiscale.to_ome_zarr.return_value = {"version": "0.4", "datasets": []}
    result = OmeZarrGroup(multiscales=(multiscale,)).to_attrs("0.4")
    assert result == {"multiscales": [{"version": "0.4", "datasets": []}]}
    assert "version" not in result


def test_ome_zarr_group_to_attrs_0_5_wraps_metadata_in_ome_and_adds_group_version():
    multiscale = Mock(spec=Multiscale)
    multiscale.to_ome_zarr.return_value = {"datasets": []}
    result = OmeZarrGroup(multiscales=(multiscale,)).to_attrs("0.5")
    assert result == {"ome": {"version": "0.5", "multiscales": [{"datasets": []}]}}


def test_ome_zarr_group_to_attrs_empty_0_6_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert OmeZarrGroup().to_attrs("0.6.rc0") == {}
