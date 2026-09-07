# clearscale basics

clearscale is designed to sit next to the code that already reads, writes, scales, crops, or reorders your image arrays.

The core idea is simple: always keep axis keys with the values they describe, in the order that applies to your data.

## At a glance

If you've been coding for a bit already, you most likely just need to know:
* Shape, Factor, PixelSize, Unit etc. are immutable, ordered `Mapping[AxisKey, <int/float/str respectively>]`
* AxisKey just aliases `Hashable`. Examples in these docs and common practice uses single lowercase letters `"tczyx"`, but you can use any str-convertible object (with caveat*)
* `.with_axes(new_axis_keys)` reorders, inserts and drops as needed. Each class has its respective default value for inserted axes (`1` for Shape, `""` for Unit etc.)
* Mathematical operations are axis-wise (PixelSize * Factor, etc.)
* Blueprints and Multiscale are immutable, ordered `Mapping[ScaleKey, <Shape/Factor/Scale respectively>]`
* Multiscale is the central metadata class

*Caveat: Axis keys (and orders) other than "tczyx" are generally badly supported across the OME-Zarr ecosystem

Potential gotchas:

* When describing relative scaling, `Factor` is a _multiplier_ for `PixelSize` and a _divisor_ for `Shape`. `Factor(x=2)` means "downscale by factor 2" - doubling pixel size and halving image shape. `Shape / Factor` needs to know how your scaling method rounds uneven divisions, so must be called as `shape.scaled_by(factor, rounding="ceil")`. Note that many scaling methods like `scipy.ndimage.zoom` expect *shape multiplier* factors, so you'd need to pass `factor.inverted().to_tuple()`.
* Scaling usually introduces a shift, or `Translation`, to the scaled image's origin in physical space. Methods that handle `Scale` objects (and hence `Scale.translation`) handle this via `translation_shift_func` (see [Which shift is the right one for my scaling method?](translation_shifts.md#which-shift-is-the-right-one-for-my-scaling-method))

## Axis values: Dicts are better than tuples

Most clearscale types are immutable ordered mappings:

```python
from clearscale import Shape

# Use whichever dict construction pattern you're familiar with
shape1 = Shape(c=3, x=1024, y=1024)
shape3 = Shape(zip("xcy", (1024, 3, 1024)))
shape2 = Shape({"y": 1024, "x": 1024, "c": 3})

assert shape1 != shape2, "The difference between plain dicts and Shape: Axis order matters"

# Iterate like a regular dict (Shape implements Mapping)
for axis, size in shape2.items():
    print(f"{size} pixels along {axis}")

# But immutable
# shape1["x"] = 5  # would raise TypeError: 'Shape' object does not support item assignment
```

Adapting existing code with clearscale is simple if you are already keeping track of axes and values using dicts:

```
scaling_limit = {"x": 1, "y": 1, "z": 120}
limit_shape = Shape(scaling_limit)
```

Same for the other primitives. clearscale calls these "axis values":

```python
from clearscale import PixelSize, PixelOffset, Translation, Factor, Unit

pixel_size  = PixelSize(zip("cyx", (1.0, 25.0, 25.0)))      # axis -> float
crop_offset = PixelOffset(t=20)                             # axis -> int
translation = Translation(y=0.5, x=0.5)                     # axis -> float
factor      = Factor(x=2.0, y=2.0, z=12.0)                  # axis -> float
unit        = Unit({"t": "seconds", "x": "nm", "y": "nm"})  # axis -> str
```

Axis keys don't have to be plain strings.
You can use any hashable object, such as frozen dataclass instances.
Or maybe you just want syntactic sugar like `x = "x"` to enable `my_shape[x]` instead of `my_shape["x"]`.

Axis keys do need to support conversion to string with `str(axis_key)` for OME-Zarr export.

### Tracking metadata across axis rearrangements

All axis values implement `.with_axes(target_axes)`.

Permutation / transposition:

```python
from clearscale import Shape
import numpy as np

image = np.zeros((3, 40, 512, 512))
source_axes = "czyx"
target_axes = "xyzc"

# Data permutation
permuted_indices = [list(source_axes).index(axis) for axis in target_axes]
transposed = image.transpose(permuted_indices)

# Metadata permutation
source_shape = Shape(zip(source_axes, image.shape))
target_shape = source_shape.with_axes(target_axes)

assert transposed.shape == target_shape.to_tuple()
```

Dropping and inserting an axis:

```python
from clearscale import Shape
import numpy as np

time_series = np.zeros((25, 128, 128))
source_axes = "tyx"
target_axes = "cyx"

# Data
with_channel_data = time_series[None, 0, :, :]

# Metadata
plane_shape = Shape(zip(source_axes, time_series.shape))
# with_axes removes t and detects c as a new axis key. It inserts the default Shape value for c (1)
with_channel_shape = plane_shape.with_axes(target_axes)

assert with_channel_data.shape == with_channel_shape.to_tuple()
```

Likewise for the other axis values:

```python
from clearscale import PixelSize, PixelOffset, Translation, Factor, Unit

pixel_size  = PixelSize(x=25.0, y=25.0).with_axes("xyc")  # pixel_size["c"]  == 1.0
crop_offset = PixelOffset(x=5, y=3).with_axes("xyc")      # crop_offset["c"] == 0
translation = Translation(x=4.0, y=3.0).with_axes("xyc")  # translation["c"] == 0.0
factor      = Factor(x=2.0, y=2.0).with_axes("xyc")       # factor["c"]      == 1.0
unit        = Unit(x="cm", y="cm").with_axes("xyc")       # unit["c"]        == ""
```

Most common manipulations should be supported with expressively named methods on clearscale objects.
Worst-case scenario, construction by dict comprehension is the escape hatch:

```python
from clearscale import Shape

shape1 = Shape(x=1024, y=1024, z=4)
shape2 = Shape(x=905, y=744)
shape3 = Shape(z=24)

custom_derived_shape = Shape({
    axis: (
        shape1[axis] + (
            shape2[axis] if axis in shape2 else shape3[axis]  # wild
        )
    )
    for axis in shape1.keys()
})
```

Though clearscale aims to support all manipulations.
If you find yourself needing to manually construct dicts, please consider opening an issue / feature request :)

### Axis-wise operations

```python
from clearscale import Factor, PixelOffset, PixelSize, Shape, Translation

pixel_size = PixelSize(z=0.5, y=0.25, x=0.25)
downsample_by_2_yx = Factor(y=2, x=2)

assert pixel_size * downsample_by_2_yx == PixelSize(z=0.5, y=0.5, x=0.5), "Factors are multipliers for pixel size"

offset = PixelOffset(y=32, x=48)

assert offset * pixel_size == Translation(y=8.0, x=12.0), "Pixel size converts pixel offsets to physical translations"

shape = Shape(z=40, y=125, x=125)

# scaled_shape = shape * downsample_xy  # Would raise TypeError. Factors are divisors for Shape

# scaled_shape = shape / downsample_xy  # Would still raise TypeError, because...

# you need to specify how your scaling implementation handles rounding when the Factor does not evenly divide the Shape:
assert shape.scaled_by(downsample_by_2_yx, rounding="ceil") == Shape(z=40, y=63, x=63)
```

Missing axes in a `Factor` default to identity (`1.0`), so `z` is unchanged in this `shape.scaled_by` example.
If the factor had extra axes, it would be rejected because they would describe metadata for data that is not there.

Same-kind arithmetic, such as `Translation(...) + Translation(...)`, requires identical axes in identical order:
```python
from clearscale import Factor, Shape, Translation

# Shapes with identical axes can be compared
shape = Shape(z=40, y=125, x=125)
downsampled_shape = Shape(z=20, y=25, x=25)

assert shape / downsampled_shape           == Factor(z=2.0, y=5.0, x=5.0)  # Can be hard to remember which way to divide :)
assert shape.scaling_to(downsampled_shape) == Factor(z=2.0, y=5.0, x=5.0)  # Explicit phrasing might be more intuitive

# Translations can be added or subtracted, but only with identical axes
crop1_transl = Translation(y=8.0, x=12.0)
crop2_transl = Translation(y=4.0)

# total_crop = crop1_transl + crop2_transl  # would raise ValueError

# If metadata are sourced from separate places with different axes, bringing them together needs to be explicit:
total_crop = crop1_transl.with_axes("cyx") + crop2_transl.with_axes("cyx")

assert total_crop == Translation(c=0.0, y=12.0, x=12.0)
```

## Immutability

The original objects used in any of these operations are never modified; a new object is returned:

```python
from clearscale import Shape

shape = Shape(x=5, y=5)
reordered = shape.with_axes("czyx")
assert shape == Shape(x=5, y=5), "original Shape is not modified"
assert reordered is not shape, "with_axes returns a new instance"
```

## Going multiscale: Dicts of dicts

`BlueprintShapes`, `BlueprintFactors` and `Multiscale` also mostly behave like immutable ordered mappings.
clearscale calls these "scale mappings".

```python
from clearscale import Shape, BlueprintShapes, Factor, BlueprintFactors, Scale, Multiscale

bps = BlueprintShapes({"s0": Shape(x=1, y=1)})          # scale_key -> Shape

bpf = BlueprintFactors({"s0": Factor(x=2.0, y=2.0)})    # scale_key -> Factor

ms  = Multiscale({"s0": Scale(shape=Shape(x=1, y=1))})  # scale_key -> Scale
```

Scale keys are relative paths that point to data arrays, so they are naturally plain strings.

Blueprints are simple but flexible containers that specify what scaling has been done or will be done.
Ideally, you would use their values as parameters for scaling, to ensure metadata and data operations stay synchronised.

```python
from clearscale import BlueprintShapes, PixelSize, Scale, Shape, Unit, Multiscale

base = Scale(
    shape=Shape(z=40, y=512, x=512),
    pixel_size=PixelSize(z=0.5, y=0.25, x=0.25),
    unit=Unit(z="micrometer", y="micrometer", x="micrometer"),
)

blueprint = BlueprintShapes.uniform_steps(
    step=2,
    scaled_axes="yx",
    base_shape=base.shape,
    rounding="ceil",
    shape_limit=Shape(y=128, x=128),
)

# Scale your data using this shape blueprint:
#for scale_key, target_shape in blueprint.items():
#    scaled_data = do_my_scaling(raw_data, target_shape.to_tuple())

multiscale = Multiscale.from_single(base, blueprint=blueprint)

assert list(multiscale.keys()) == ["s0", "s1", "s2"]
assert multiscale["s2"].shape == Shape(z=40, y=128, x=128)
assert multiscale["s2"].pixel_size == PixelSize(z=0.5, y=1.0, x=1.0)
```

By default, applying a blueprint to a Scale keeps the base Scale's translation for all derived Scales.
If your scaling operation changes where the first output pixel belongs in physical space (most commonly used scaling methods do), choose an explicit `translation_shift_func`.
See [Which shift is the right one for my scaling method?](translation_shifts.md#which-shift-is-the-right-one-for-my-scaling-method) for the built-in shift conventions and detector.

If your processing code naturally thinks in scaling factors instead of output shapes, use `BlueprintFactors`.
In this case, you need to specify how your data scaling handles rounding when factors unevenly divide shapes.
This is necessary for accurate metadata calculation.

You can directly build the blueprint and apply to the same `base` Scale:

```python
from clearscale import BlueprintFactors, Factor, Multiscale

factors = BlueprintFactors(
    {
        "s0": Factor.identity("zyx"),
        "s1": Factor(z=1, y=2, x=2),
        "s2": Factor(z=1, y=4, x=4),
    }
)

multiscale = Multiscale.from_single(base, blueprint=factors, rounding="ceil")
```

The number of scales you want to output will probably depend on the input image shape though.
In practice, you might want to go through `BlueprintShapes` anyway.
Taking the same setup as in the example above:

```python
base = Scale(...)
shapes = BlueprintShapes.uniform_steps(step=2, base_shape=base.shape, ...)

# You can simply convert this:

factors = shapes.to_factors()

# Scale your data using this factor blueprint.
# This only works if your scaling method can handle arbitrary fractional factors.

#for scale_key, scale_factor in factors.items():
#    scaled_data = do_my_scaling_by_factor(raw_data, scale_factor.to_tuple())

# Whether you use the `shapes` or `factors` blueprint now makes no difference.
# But with `shapes` you don't have to provide `rounding` :)
multiscale = Multiscale.from_single(base, blueprint=shapes)

assert tuple(multiscale.keys()) == ("s0", "s1", "s2")
assert multiscale["s2"].shape == Shape(z=40, y=128, x=128)
assert multiscale["s2"].pixel_size == PixelSize(z=0.5, y=1.0, x=1.0)
```

Take note though that Factors in clearscale are *divisors for shape*:
`1024 pixels downscaled by factor 2 = 1024 / 2 = 512 pixels`. Scaling functions that accept factors as parameters may expect the inverse.
For example, to downscale by 2, you could use `skimage.transform.rescale(image, 0.5)` or `scipy.ndimage.zoom(image, 0.5)`.
In this case you would use `scale_factor.inverted().to_tuple()`.
