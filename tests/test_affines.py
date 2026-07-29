from collections import OrderedDict
import pytest
from typing import cast

from clearscale import Translation
from clearscale._affines import Coefficient, Linear, Affine


def test_coefficient_zeros():
    c = Coefficient.zeros(("z", "y", "x"))

    assert c == {"z": 0.0, "y": 0.0, "x": 0.0}


def test_linear_identity():
    linear = Linear.identity(("z", "y", "x"))

    assert linear == {
        "z": {"z": 1.0, "y": 0.0, "x": 0.0},
        "y": {"z": 0.0, "y": 1.0, "x": 0.0},
        "x": {"z": 0.0, "y": 0.0, "x": 1.0},
    }


def test_linear_fromkeys():
    linear = Linear.fromkeys(("z", "y", "x"))

    assert linear == {
        "z": {"z": 0.0, "y": 0.0, "x": 0.0},
        "y": {"z": 0.0, "y": 0.0, "x": 0.0},
        "x": {"z": 0.0, "y": 0.0, "x": 0.0},
    }


def test_linear_getitem_row():
    linear = Linear.identity(("x", "y"))

    assert linear["x"] == Coefficient({"x": 1.0, "y": 0.0})


def test_linear_at_element():
    linear = Linear.identity(("x", "y"))

    assert linear.at("x", "x") == 1.0
    assert linear.at("x", "y") == 0.0
    assert linear.at("y", "x") == 0.0
    assert linear.at("y", "y") == 1.0


def test_linear_init_accepts_nested_dict():
    linear = Linear({"x": {"x": 2.0}})

    assert isinstance(linear["x"], Coefficient)
    assert linear["x"] == Coefficient({"x": 2.0})


def test_linear_init_requires_identical_source_axes():
    with pytest.raises(ValueError, match="All source Coefficients must have identical axes"):
        Linear(
            {
                "x": Coefficient({"x": 1.0, "y": 0.0}),
                "y": Coefficient({"x": 0.0}),
            }
        )


def test_linear_init_requires_square():
    with pytest.raises(ValueError, match="Linear must be square"):
        Linear(
            {
                "x": {"x": 1.0, "y": 2.0},
            }
        )


def test_linear_init_requires_nested_mapping():
    with pytest.raises(TypeError, match="All values must be Coefficient or Mapping"):
        Linear({"x": 1.0})


@pytest.mark.parametrize(
    "array",
    [
        ((1, 2), (3, 4)),
        [[1, 2], [3, 4]],
    ],
)
def test_linear_from_array(array):
    linear = Linear.from_array(array, "xy")

    assert linear == {
        "x": {"x": 1.0, "y": 2.0},
        "y": {"x": 3.0, "y": 4.0},
    }


def test_linear_from_array_wrong_row_count():
    with pytest.raises(ValueError, match="Expected 2 rows"):
        Linear.from_array(((1, 0),), ("x", "y"))


def test_linear_from_array_wrong_column_count():
    with pytest.raises(ValueError, match="Expected 2 columns"):
        Linear.from_array(((1,), (0, 1)), ("x", "y"))


def test_linear_from_array_wrong_type():
    with pytest.raises(TypeError, match="All values must be float"):
        Linear.from_array((("oops", 0), (0, 1)), ("x", "y"))  # type: ignore[arg-type]


def test_linear_copy():
    linear = Linear(
        {
            "z": {"z": 1.0, "y": 3.2, "x": 0.0},
            "y": {"z": 2.4, "y": 1.0, "x": 4.5},
            "x": {"z": 5.7, "y": 0.0, "x": 1.0},
        }
    )
    copied = linear.copy()

    assert copied == linear
    assert copied is not linear
    assert copied["x"] is not linear["x"]
    assert copied["y"] is not linear["y"]


@pytest.mark.parametrize(
    ("given", "axes", "expected"),
    [
        pytest.param(
            {"x": {"x": 2, "y": 3}, "y": {"x": 4, "y": 5}},
            "xy",
            {"x": {"x": 2, "y": 3}, "y": {"x": 4, "y": 5}},
            id="same axes",
        ),
        pytest.param(
            {"x": {"x": 2, "y": 3}, "y": {"x": 4, "y": 5}},
            "yx",
            {"y": {"y": 5, "x": 4}, "x": {"y": 3, "x": 2}},
            id="reorder rows and columns",
        ),
        pytest.param(
            {"x": {"x": 2, "y": 3}, "y": {"x": 4, "y": 5}},
            "x",
            {"x": {"x": 2}},
            id="drop axis",
        ),
        pytest.param(
            {"x": {"x": 2, "y": 3}, "y": {"x": 4, "y": 5}},
            "xyz",
            {"x": {"x": 2, "y": 3, "z": 0}, "y": {"x": 4, "y": 5, "z": 0}, "z": {"x": 0, "y": 0, "z": 1}},
            id="append identity axis",
        ),
        pytest.param(
            {"x": {"x": 2, "y": 3}, "y": {"x": 4, "y": 5}},
            "zxy",
            {"z": {"z": 1, "x": 0, "y": 0}, "x": {"z": 0, "x": 2, "y": 3}, "y": {"z": 0, "x": 4, "y": 5}},
            id="prepend identity axis",
        ),
        pytest.param(
            {"x": {"x": 2, "y": 3}, "y": {"x": 4, "y": 5}},
            "xzy",
            {"x": {"x": 2, "z": 0, "y": 3}, "z": {"x": 0, "z": 1, "y": 0}, "y": {"x": 4, "z": 0, "y": 5}},
            id="insert identity axis",
        ),
        pytest.param(
            {"x": {"x": 2, "y": 3, "z": 4}, "y": {"x": 5, "y": 6, "z": 7}, "z": {"x": 8, "y": 9, "z": 10}},
            "zx",
            {"z": {"z": 10, "x": 8}, "x": {"z": 4, "x": 2}},
            id="drop and reorder",
        ),
    ],
)
def test_linear_with_axes(given, axes, expected):
    assert Linear(given).with_axes(axes) == expected


def test_linear_with_axes_empty():
    with pytest.raises(ValueError, match="Cannot create empty Linear"):
        Linear.identity("xy").with_axes(())


@pytest.mark.parametrize(
    ("linear", "axes", "expected"),
    [
        pytest.param(
            Linear.from_array(((1, 2), (3, 4)), axes="xy"),
            "yx",
            Linear.from_array(((4, 3), (2, 1)), axes="yx"),
            id="reverse",
        ),
        pytest.param(
            Linear.from_array(((1, 2, 3), (4, 5, 6), (7, 8, 9)), axes="xyz"),
            "zxy",
            Linear.from_array(((9, 7, 8), (3, 1, 2), (6, 4, 5)), axes="zxy"),
            id="rotate",
        ),
        pytest.param(
            Linear.identity("xyz"),
            "zxy",
            Linear.identity("zxy"),
            id="identity",
        ),
        pytest.param(
            Linear.from_array(((1, 2), (3, 4)), axes="xy"),
            "yxz",
            Linear.from_array(((4, 3), (2, 1)), axes="yx"),
            id="ignore missing",
        ),
    ],
)
def test_linear_with_axes_order(linear, axes, expected):
    assert cast(Linear, linear).with_axes_order(axes) == expected


def test_linear_with_axes_order_no_matching_axes():
    linear = Linear.identity("xy")

    with pytest.raises(ValueError, match="would drop"):
        linear.with_axes_order("ztc")


def test_linear_with_axes_order_rejects_subset():
    linear = Linear.identity("xyz")

    with pytest.raises(ValueError, match="would drop"):
        linear.with_axes_order("zx")


@pytest.mark.parametrize(
    ("linear", "axes", "expected"),
    [
        pytest.param(Linear.identity("xyz"), "xy", Linear.identity("xy"), id="drop_last"),
        pytest.param(Linear.identity("xyz"), "yz", Linear.identity("yz"), id="drop_first"),
        pytest.param(Linear.identity("xyz"), "z", Linear.identity("z"), id="single"),
        pytest.param(
            Linear.from_array(((1, 2, 3), (4, 5, 6), (7, 8, 9)), axes="xyz"),
            "zx",
            Linear.from_array(((1, 3), (7, 9)), axes="xz"),
            id="3d",
        ),
        pytest.param(Linear.identity("xyz"), "abxy", Linear.identity("xy"), id="ignore missing"),
        pytest.param(Linear.identity("xyz"), "zxy", Linear.identity("xyz"), id="no reordering"),
    ],
)
def test_linear_without_axes_except(linear, axes, expected):
    assert cast(Linear, linear).without_axes_except(axes) == expected


def test_linear_without_axes_except_no_matching_axes():
    with pytest.raises(ValueError, match="None of the specified axes"):
        Linear.identity("xyz").without_axes_except("ab")


@pytest.mark.parametrize(
    ("linear", "axes", "expected"),
    [
        pytest.param(
            Linear.identity("xy"),
            "x",
            Linear.identity("y"),
            id="identity",
        ),
        pytest.param(
            Linear.from_array(((1, 2), (3, 4)), axes="xy"),
            "x",
            Linear.from_array(((4,),), axes="y"),
            id="drop first",
        ),
        pytest.param(
            Linear.from_array(((1, 2, 3), (4, 5, 6), (7, 8, 9)), axes="xyz"),
            "y",
            Linear.from_array(((1, 3), (7, 9)), axes="xz"),
            id="drop middle",
        ),
        pytest.param(
            Linear.from_array(((1, 2, 3), (4, 5, 6), (7, 8, 9)), axes="xyz"),
            "xz",
            Linear.from_array(((5,),), axes="y"),
            id="drop multiple",
        ),
        pytest.param(
            Linear.identity("ntczyx"),
            "xtc",
            Linear.identity("nzy"),
            id="no reordering",
        ),
        pytest.param(
            Linear.identity("xy"),
            "z",
            Linear.identity("xy"),
            id="ignore missing",
        ),
    ],
)
def test_linear_without_axes(linear, axes, expected):
    assert cast(Linear, linear).without_axes(axes) == expected


def test_linear_without_axes_all():
    with pytest.raises(ValueError, match="would leave no axes"):
        Linear.identity("xy").without_axes("xy")


def test_linear_with_identity():
    linear = Linear(
        {
            "z": {"z": 10.0, "y": 3.0, "x": 0.0},
            "y": {"z": 2.0, "y": 12.0, "x": 4.0},
            "x": {"z": 5.0, "y": 0.0, "x": 13.0},
        }
    )
    assert linear.with_identity(()) is linear
    assert linear.with_identity_except(()) == Linear.identity("zyx")

    reset_y = linear.with_identity("y")
    reset_xz = linear.with_identity_except("y")
    assert reset_y == {
        "z": {"z": 10.0, "y": 0.0, "x": 0.0},
        "y": {"z": 0.0, "y": 1.0, "x": 0.0},
        "x": {"z": 5.0, "y": 0.0, "x": 13.0},
    }
    assert reset_xz == {
        "z": {"z": 1.0, "y": 0.0, "x": 0.0},
        "y": {"z": 0.0, "y": 12.0, "x": 0.0},
        "x": {"z": 0.0, "y": 0.0, "x": 1.0},
    }


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        pytest.param(
            {"x": {"x": 1, "y": 0}, "y": {"x": 0, "y": 2}},
            {
                "y": {"y": 2},
            },
            id="remove first identity",
        ),
        pytest.param(
            {"x": {"x": 2, "y": 0}, "y": {"x": 0, "y": 1}},
            {
                "x": {"x": 2},
            },
            id="remove second identity",
        ),
        pytest.param(
            {"x": {"x": 1, "y": 0, "z": 0}, "y": {"x": 0, "y": 3, "z": 0}, "z": {"x": 0, "y": 0, "z": 1}},
            {
                "y": {"y": 3},
            },
            id="keep middle",
        ),
        pytest.param(
            {"x": {"x": 1, "y": 0, "z": 0}, "y": {"x": 0, "y": 1, "z": 0}, "z": {"x": 0, "y": 0, "z": 3}},
            {
                "z": {"z": 3},
            },
            id="keep last",
        ),
        pytest.param(
            {"x": {"x": 1, "y": 1}, "y": {"x": 0, "y": 1}},
            {"x": {"x": 1, "y": 1}, "y": {"x": 0, "y": 1}},
            id="identity target is used as source in other targets",
        ),
    ],
)
def test_linear_without_identity(given, expected):
    assert Linear(given).without_identity() == expected


def test_linear_without_identity_all_identity():
    with pytest.raises(ValueError, match="Removing all identities would leave no axes"):
        Linear.identity("yx").without_identity()


@pytest.mark.parametrize(
    ("other", "only", "expected"),
    [
        pytest.param({}, None, Linear.identity("xy"), id="empty"),
        pytest.param({"x": {"x": 2}}, (), Linear.identity("xy"), id="empty only"),
        pytest.param({"x": {"x": 2}}, None, Linear.from_array(((2, 0), (0, 1)), axes="xy"), id="replace one value"),
        pytest.param({"x": {"x": 2, "y": 3}}, None, Linear.from_array(((2, 3), (0, 1)), axes="xy"), id="replace row"),
        pytest.param(
            {"x": {"x": 2}, "y": {"y": 4}},
            None,
            Linear.from_array(((2, 0), (0, 4)), axes="xy"),
            id="replace in multiple rows",
        ),
        pytest.param(
            {"x": {"x": 2, "y": 3}}, "x", Linear.from_array(((2, 0), (0, 1)), axes="xy"), id="only filters source axes"
        ),
        pytest.param(
            {"x": {"x": 2}, "y": {"y": 4}},
            "y",
            Linear.from_array(((1, 0), (0, 4)), axes="xy"),
            id="only filters target and source axes",
        ),
        pytest.param({"z": {"x": 2}}, None, Linear.identity("xy"), id="unknown row ignored"),
        pytest.param({"x": {"z": 2}}, None, Linear.identity("xy"), id="unknown column ignored"),
    ],
)
def test_linear_with_values(other, only, expected):
    linear = Linear.identity("xy")
    assert linear.with_values(other, only=only) == expected


def test_linear_with_values_requires_mapping():
    with pytest.raises(TypeError, match=r"Pass \{target_axis: \{source_axis: value\}\} mapping"):
        Linear.identity("xy").with_values(123)  # type: ignore[arg-type]


def test_linear_with_values_requires_nested_mapping():
    with pytest.raises(TypeError, match="Replacement for target axis 'x' must be a mapping"):
        Linear.identity("xy").with_values({"x": 2})  # type: ignore[arg-type]


def test_linear_to_tuples():
    linear = Linear.from_array([[1, 2], [3, 4]], "xy")
    tuples = linear.to_tuples()
    assert tuples == ((1, 2), (3, 4))
    assert isinstance(tuples, tuple)
    assert isinstance(tuples[0], tuple)
    assert all(isinstance(v, float) for row in tuples for v in row)


def test_linear_to_lists():
    linear = Linear.from_array(((1, 2), (3, 4)), "xy")
    lists = linear.to_lists()
    assert lists == [[1, 2], [3, 4]]
    assert isinstance(lists, list)
    assert isinstance(lists[0], list)
    assert all(isinstance(v, float) for row in lists for v in row)


def test_linear_to_dict():
    linear = Linear.from_array(((1, 2), (3, 4)), "xy")
    dicts = linear.to_dict()
    assert dicts == OrderedDict(
        (
            ("x", OrderedDict((("x", 1.0), ("y", 2.0)))),
            ("y", OrderedDict((("x", 3.0), ("y", 4.0)))),
        )
    )
    assert isinstance(dicts, OrderedDict)
    assert isinstance(dicts["x"], OrderedDict)


@pytest.mark.parametrize("linear", [Linear.identity("xyz"), Linear.from_array(((1, 0), (0, 1)), "yx")])
def test_linear_is_identity(linear):
    assert linear.is_identity()


@pytest.mark.parametrize(
    ("linear", "axes", "expected"),
    [
        pytest.param(Linear.identity("xyz"), "x", True, id="single axis"),
        pytest.param(Linear.identity("xyz"), "xy", True, id="multiple axes"),
        pytest.param(Linear.identity("xyz"), "xyz", True, id="all axes"),
        pytest.param(
            Linear({"x": {"x": 2, "y": 0, "z": 0}, "y": {"x": 0, "y": 1, "z": 0}, "z": {"x": 0, "y": 0, "z": 1}}),
            "yz",
            True,
            id="ignore non-identity x",
        ),
        pytest.param(
            Linear({"x": {"x": 2, "y": 0, "z": 0}, "y": {"x": 0, "y": 1, "z": 0}, "z": {"x": 0, "y": 0, "z": 1}}),
            "x",
            False,
            id="diagonal not one",
        ),
        pytest.param(
            Linear({"x": {"x": 1, "y": 3, "z": 0}, "y": {"x": 0, "y": 1, "z": 0}, "z": {"x": 0, "y": 0, "z": 1}}),
            "x",
            False,
            id="row not identity",
        ),
        pytest.param(
            Linear({"x": {"x": 1, "y": 0, "z": 0}, "y": {"x": 4, "y": 1, "z": 0}, "z": {"x": 0, "y": 0, "z": 1}}),
            "x",
            False,
            id="column not identity",
        ),
        pytest.param(
            Linear({"x": {"x": 1, "y": 0, "z": 0}, "y": {"x": 0, "y": 1, "z": 0}, "z": {"x": 0, "y": 7, "z": 1}}),
            "y",
            False,
            id="column below diagonal",
        ),
        pytest.param(
            Linear({"x": {"x": 1, "y": 0, "z": 0}, "y": {"x": 0, "y": 1, "z": 5}, "z": {"x": 0, "y": 0, "z": 1}}),
            "y",
            False,
            id="row right of diagonal",
        ),
    ],
)
def test_linear_is_identity_along(linear, axes, expected):
    assert cast(Linear, linear).is_identity_along(axes) == expected


@pytest.mark.parametrize(
    ("linear", "translation"),
    [
        (None, Translation(x=0.0)),
        (Linear.identity("x"), None),
    ],
)
def test_affine_init_accepts_partial_args(linear, translation):
    affine = Affine(linear=linear, translation=translation)
    assert affine == Affine.identity("x")
    assert tuple(affine.axes()) == ("x",)


def test_affine_from_linear_aliases_init():
    assert Affine.from_linear(Linear.identity("xy")) == Affine.identity("xy")


@pytest.mark.parametrize(
    ("linear", "translation", "match"),
    [
        pytest.param({"x": {"x": 1.0}}, Translation(x=0.0), "must be a Linear", id="linear dict"),
        pytest.param(Linear.identity("x"), {"x": 0.0}, "must be a Translation", id="translation dict"),
    ],
)
def test_affine_init_requires_clearscale_types(linear, translation, match):
    with pytest.raises(TypeError, match=str(match)):
        Affine(linear=linear, translation=translation)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("linear_axes", "translation_axes"),
    [
        ("xy", "yx"),
        ("xy", "y"),
        ("x", "xy"),
    ],
)
def test_affine_init_requires_identical_axes(linear_axes, translation_axes):
    with pytest.raises(ValueError, match="must have identical axes"):
        Affine(linear=Linear.identity(linear_axes), translation=Translation.identity(translation_axes))


@pytest.mark.parametrize(
    ("array", "axes", "expected"),
    [
        pytest.param(
            [[1, 2]],
            ("x",),
            Affine(linear=Linear({"x": {"x": 1}}), translation=Translation({"x": 2})),
            id="1d",
        ),
        pytest.param(
            [[1, 2], [0, 1]],
            ("x",),
            Affine(linear=Linear({"x": {"x": 1}}), translation=Translation({"x": 2})),
            id="1d homogeneous",
        ),
        pytest.param(
            [[1, 2, 3], [4, 5, 6]],
            ("y", "x"),
            Affine(
                linear=Linear({"y": {"y": 1, "x": 2}, "x": {"y": 4, "x": 5}}), translation=Translation({"y": 3, "x": 6})
            ),
            id="2d",
        ),
        pytest.param(
            [[1, 2, 3], [4, 5, 6], [0, 0, 1]],
            ("y", "x"),
            Affine(
                linear=Linear({"y": {"y": 1, "x": 2}, "x": {"y": 4, "x": 5}}), translation=Translation({"y": 3, "x": 6})
            ),
            id="2d homogeneous",
        ),
        pytest.param(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            ("z", "y", "x"),
            Affine(
                linear=Linear(
                    {"z": {"z": 1, "y": 2, "x": 3}, "y": {"z": 5, "y": 6, "x": 7}, "x": {"z": 9, "y": 10, "x": 11}}
                ),
                translation=Translation({"z": 4, "y": 8, "x": 12}),
            ),
            id="3d",
        ),
        pytest.param(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 1]],
            ("z", "y", "x"),
            Affine(
                linear=Linear(
                    {"z": {"z": 1, "y": 2, "x": 3}, "y": {"z": 5, "y": 6, "x": 7}, "x": {"z": 9, "y": 10, "x": 11}}
                ),
                translation=Translation({"z": 4, "y": 8, "x": 12}),
            ),
            id="3d homogeneous",
        ),
        pytest.param(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29, 30],
                [0, 0, 0, 0, 0, 1],
            ],
            "tczyx",
            Affine(
                linear=Linear.from_array(
                    [
                        [1, 2, 3, 4, 5],
                        [7, 8, 9, 10, 11],
                        [13, 14, 15, 16, 17],
                        [19, 20, 21, 22, 23],
                        [25, 26, 27, 28, 29],
                    ],
                    axes="tczyx",
                ),
                translation=Translation(zip("tczyx", [6, 12, 18, 24, 30])),
            ),
            id="5d",
        ),
    ],
)
def test_affine_from_array(array, axes, expected):
    assert Affine.from_array(array, axes) == expected


@pytest.mark.parametrize(
    ("array", "axes", "expected_error"),
    [
        pytest.param([], ("x",), "Expected either 1 or 2 rows"),
        pytest.param([[1, 2], [3, 4], [5, 6]], ("x",), "Expected either 1 or 2 rows"),
        pytest.param([[1], [0, 1]], ("x",), "Expected 2 columns for row 0"),
        pytest.param([[1, 2, 3]], ("x",), "Expected 2 columns for row 0"),
        pytest.param([[1, 2], [3, 4]], ("x",), "Homogeneous affine matrices must end with the perspective row"),
        pytest.param([[1, 2], [0, 0]], ("x",), "Homogeneous affine matrices must end with the perspective row"),
        pytest.param([[1, 2, 3], [4, 5, 6], [0, 1, 0]], ("y", "x"), "must end with the perspective row"),
    ],
)
def test_affine_from_array_errors(array, axes, expected_error):
    with pytest.raises(ValueError, match=str(expected_error)):
        Affine.from_array(array, axes)


def test_affine_with_axes():
    # These methods just forward. Test mainly for line coverage :)
    assert Affine.identity("y").with_axes("yzx") == Affine.identity("yzx")
    assert Affine.identity("xyt").with_axes_order("tzyxc") == Affine.identity("tyx")
    assert Affine.identity("xyz").without_axes("zx") == Affine.identity("y")
    assert Affine.identity("xyz").without_axes_except("zx") == Affine.identity("xz")


def test_affine_with_identity():
    affine = Affine(
        linear=Linear.from_array(((2, 0), (0, 3)), "xy"),
        translation=Translation({"x": 5, "y": 7}),
    )

    assert affine.with_identity(("x",)) == Affine(
        linear=Linear.from_array(((1, 0), (0, 3)), "xy"),
        translation=Translation({"x": 0, "y": 7}),
    )


def test_affine_with_identity_except():
    affine = Affine(
        linear=Linear.from_array(((2, 0), (0, 3)), "xy"),
        translation=Translation({"x": 5, "y": 7}),
    )

    assert affine.with_identity_except(("x",)) == Affine(
        linear=Linear.from_array(((2, 0), (0, 1)), "xy"),
        translation=Translation({"x": 5, "y": 0}),
    )


@pytest.mark.parametrize(
    ("affine", "expected"),
    [
        pytest.param(
            Affine(
                linear=Linear.identity("xy"),
                translation=Translation({"x": 5, "y": 0}),
            ),
            Affine(
                linear=Linear.identity("x"),
                translation=Translation({"x": 5}),
            ),
            id="drop identity axis",
        ),
        pytest.param(
            Affine(
                linear=Linear.identity("xy"),
                translation=Translation({"x": 5, "y": 1}),
            ),
            Affine(
                linear=Linear.identity("xy"),
                translation=Translation({"x": 5, "y": 1}),
            ),
            id="translation prevents drop",
        ),
        pytest.param(
            Affine(
                linear=Linear.from_array(((2, 0), (0, 3)), "xy"),
                translation=Translation.identity("xy"),
            ),
            Affine(
                linear=Linear.from_array(((2, 0), (0, 3)), "xy"),
                translation=Translation.identity("xy"),
            ),
            id="linear prevents drop",
        ),
    ],
)
def test_affine_without_identity(affine, expected):
    assert cast(Affine, affine).without_identity() == expected


def test_affine_without_identity_rejects_dropping_all():
    with pytest.raises(ValueError, match="Cannot create empty Affine"):
        _ = Affine.identity("zyx").without_identity()


def test_affine_with_linear():
    affine = Affine.identity(("y", "x"))
    linear = Linear.from_array(((2, 0), (0, 3)), ("y", "x"))

    result = affine.with_linear(linear)

    assert result.linear == linear
    assert result.translation == affine.translation


def test_affine_with_translation():
    affine = Affine.identity(("y", "x"))
    translation = Translation({"y": 5, "x": -2})

    result = affine.with_translation(translation)

    assert result.translation == translation
    assert result.linear == affine.linear


def test_affine_to_tuples_lists():
    affine = Affine(
        linear=Linear.from_array(((1, 2), (3, 4)), "xy"),
        translation=Translation((("x", 5), ("y", 6))),
    )
    assert affine.to_tuples() == ((1.0, 2.0, 5.0), (3.0, 4.0, 6.0))
    assert affine.to_tuples_homogenous() == ((1.0, 2.0, 5.0), (3.0, 4.0, 6.0), (0.0, 0.0, 1.0))
    assert affine.to_lists() == [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]
    assert affine.to_lists_homogenous() == [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [0.0, 0.0, 1.0]]


@pytest.mark.parametrize(
    ("affine", "expected"),
    [
        pytest.param(Affine.identity(("x", "y")), True, id="simple"),
        pytest.param(Affine.from_array(((1, 0, 0), (0, 1, 0)), axes="xy"), True, id="array"),
        pytest.param(Affine(translation=Translation(x=1)), False, id="translation not"),
        pytest.param(Affine(linear=Linear.from_array(((2, 0), (0, 1)), "xy")), False, id="linear not"),
    ],
)
def test_affine_is_identity(affine, expected):
    assert cast(Affine, affine).is_identity() is expected


@pytest.mark.parametrize(
    ("affine", "axes", "expected"),
    [
        pytest.param(Affine.from_array(((2, 0, 0), (0, 1, 0)), "xy"), "y", True, id="linear"),
        pytest.param(
            Affine(linear=Linear.identity("xy"), translation=Translation(x=3, y=0)), "y", True, id="translation"
        ),
        pytest.param(Affine.from_array(((2, 0, 4), (0, 1, 0)), "xy"), "y", True, id="both"),
    ],
)
def test_affine_is_identity_along(affine, axes, expected):
    assert cast(Affine, affine).is_identity_along(axes) is expected
