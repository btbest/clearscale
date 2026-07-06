# clearscale basics

clearscale is designed to sit next to the code that already reads, writes, scales, crops, or reorders your image arrays.

The core idea is simple: describe array axes by name, keep the values ordered, and let clearscale derive the matching metadata when your data changes shape.

## Axis values: Dicts are better than tuples

Most clearscale types are immutable ordered mappings:

```python
from clearscale import Shape

shape1 = Shape(c=3, x=1024, y=1024)
shape2 = Shape({"y": 1024, "x": 1024, "c": 3})
assert shape1 != shape2, "Axis order matters!"

for axis, size in shape2.items():
    print(f"{size} pixels along {axis}")

# shape1["x"] = 5  # would raise TypeError: 'Shape' object does not support item assignment
```

Adapting existing code to clearscale is simple if you are already keeping track of axes and values using dicts:

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

Axes don't have to be plain strings.
You can use any hashable object, such as frozen dataclass instances.
Or maybe you just want syntactic sugar like `x = "x"` to enable `my_shape[x]` instead of `my_shape["x"]`.

Axes do need to support conversion to string with `str(axis)` for OME-Zarr export.

### Metadata manipulation mirrors data manipulation

Transposition:

```python
from clearscale import Shape
import numpy as np

image = np.zeros((3, 40, 512, 512))
shape = Shape(zip("czyx", image.shape))

target_shape = shape.with_axes("zyxc")
transposed_indices = [list(shape).index(axis) for axis in target_shape]

transposed = image.transpose(transposed_indices)

assert transposed.shape == target_shape.to_tuple()
```

Adding a singleton axis:

```python
from clearscale import Shape
import numpy as np

# data
plane = np.zeros((512, 512))
with_channel = plane[None, :, :]

# metadata
plane_shape = Shape(zip("yx", plane.shape))
with_channel_shape = plane_shape.with_axes("cyx")

assert with_channel.shape == with_channel_shape.to_tuple()
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
        shape1[axis] + (shape2[axis] if axis in shape2 else shape3[axis])  # wild
    )
    for axis in shape1.keys()
})
```

Though clearscale aims to support all manipulations.
If you find yourself needing to manually construct dicts, please consider opening an issue / feature request :)

### Axis-wise operations are simple

```python
from clearscale import Factor, PixelOffset, PixelSize, Shape, Translation

pixel_size = PixelSize(z=0.5, y=0.25, x=0.25)
downsample_by_2_yx = Factor(y=2, x=2)

assert pixel_size * downsample_by_2_yx == PixelSize(z=0.5, y=0.5, x=0.5)

offset = PixelOffset(y=32, x=48)

assert offset * pixel_size == Translation(y=8.0, x=12.0)

shape = Shape(z=40, y=125, x=125)

# scaled_shape = shape / downsample_xy  # would raise TypeError

# Dividing Shape/Factor is not supported. You need to specify how your scaling implementation handles rounding:
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

# If metadata are sourced from separate places with different axis conventions, 
# bringing them together needs to be explicit:
total_crop = crop1_transl.with_axes("cyx") + crop2_transl.with_axes("cyx")

assert total_crop == Translation(c=0.0, y=12.0, x=12.0)
```

`with_axes` is available on all "axis values". This
* reorders existing values to the specified order,
* removes axes not present in the target set,
* inserts the type's default value for new axes (Shape: `1`, PixelOffset: `0`, PixelSize and Factor: `1.0`, Translation: `1.0`)

Except of course, these are all immutable objects. The original object is not modified; a new object is returned:

```python
from clearscale import Shape

shape = Shape(x=5, y=5)
reordered = shape.with_axes("czyx")
assert reordered is not shape
assert shape == Shape(x=5, y=5)
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

Blueprints are simple but flexible containers that specify precisely what scaling has been done or will be done.
Ideally, you would use their values as parameters for scaling, to ensure metadata and data operations stay synchronised.

```python
from clearscale import BlueprintShapes, PixelSize, Scale, Shape, Unit

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

multiscale = blueprint.apply_to_scale(base)

assert tuple(multiscale.keys()) == ("s0", "s1", "s2")
assert multiscale["s2"].shape == Shape(z=40, y=128, x=128)
assert multiscale["s2"].pixel_size == PixelSize(z=0.5, y=1.0, x=1.0)
```

By default, `apply_to_scale` keeps the base scale's translation for all derived scales.
If your scaling operation changes where the first output pixel belongs in physical space, choose an explicit `translation_shift_func`.
(Most commonly used scaling methods do.)
See [Which shift is the right one for my scaling method?](translation_shifts.md#which-shift-is-the-right-one-for-my-scaling-method) for the built-in shift conventions and detector.

If your processing code naturally thinks in scaling factors instead of output shapes, use `BlueprintFactors`.
In this case, you need to specify how your data scaling handles rounding when factors unevenly divide shapes.
This is necessary for accurate metadata calculation.

You can directly build the blueprint and apply to the same `base` Scale:

```python
from clearscale import BlueprintFactors, Factor

factors = BlueprintFactors(
    {
        "s0": Factor.identity("zyx"),
        "s1": Factor(z=1, y=2, x=2),
        "s2": Factor(z=1, y=4, x=4),
    }
)

multiscale = factors.apply_to_scale(base, rounding="ceil")
```

The number of scales you want to output will probably depend on the input image shape though.
In practice, you might want to go through `BlueprintShapes` anyway.
Taking the same setup as in the example above:

```python
base = Scale(...)
blueprint = BlueprintShapes.uniform_steps(step=2, base_shape=base.shape, ...)

# You can simply convert this:

factors = blueprint.to_factors()

# Scale your data using this factor blueprint:

#for scale_key, scale_factor in factors.items():
#    scaled_data = do_my_scaling_by_factor(raw_data, scale_factor.to_tuple())

multiscale = blueprint.apply_to_scale(base)

assert tuple(multiscale.keys()) == ("s0", "s1", "s2")
assert multiscale["s2"].shape == Shape(z=40, y=128, x=128)
assert multiscale["s2"].pixel_size == PixelSize(z=0.5, y=1.0, x=1.0)
```

Take note though that Factors in clearscale are *divisors for shape*:
`1024 pixels scaled by factor 2 = 1024 / 2 = 512 pixels`. Scaling functions that accept factors as parameters may expect the inverse.
For example, to downscale by 2, you could use `skimage.transform.rescale(image, 0.5)` or `scipy.ndimage.zoom(image, 0.5)`.
In this case you would use `scale_factor.inverted().to_tuple()`.

### OME-Zarr stays at the edge

Most code should not need to know the OME-Zarr metadata schema.
Keep clearscale objects in your own logic, then convert at the read/write boundary:

```python
from clearscale import Multiscale

# zarr_group is whatever object your zarr library uses for group access.
ome_multiscale = zarr_group.attrs["multiscales"][0]
multiscale = Multiscale.from_ome_zarr(ome_multiscale, shape_source=zarr_group)

for scale_key, scale in multiscale.items():
    print(scale.to_display_string(scale_key))

zarr_group.attrs["ome"]["multiscales"] = [
    multiscale.to_ome_zarr(version="0.5", axis_types="infer")
]
```

### What you still need to know about OME-Zarr

The clearscale `Multiscale` reflects a single multiscale object.
Most OME-Zarr versions store a *list* of multiscale definitions in the zarr metadata:

```python
# OME-Zarr v0.1 to v0.4
ome = zarr_group.attrs["multiscales"]
assert isinstance(ome, list)

# OME-Zarr v0.5 and newer
ome = zarr_group.attrs["ome"]["multiscales"]
assert isinstance(ome, list)
```

So to read OME-Zarr from some zarr group, your code needs to do something like:

```python
from clearscale import Multiscale
try:
    ome = zarr_group.attrs["ome"]["multiscales"]
except KeyError:
    try:
        ome = zarr_group.attrs["multiscales"]
    except KeyError as e:
        raise ValueError("No multiscale metadata found in zarr group")

valid_multiscales = []
for ome_ms in ome:
    try:
        valid_multiscales.append(Multiscale.from_ome_zarr(ome_ms, shape_source=zarr_group))
    except ValueError as e:
        continue  # Invalid multiscale - maybe warn, or just skip
        
if not valid_multiscales:
    raise ValueError("Multiscale metadata in this zarr group was all invalid")

if len(valid_multiscales) > 1:
    final_ms = let_user_choose_multiscale(valid_multiscales)
else:
    final_ms = valid_multiscales[0]
```

Likewise, when writing OME-Zarr metadata, you need to make sure that

1. you use the correct *zarr* format version (zarr-v2 for OME-Zarr 0.4, and zarr-v3 for newer versions)
1. the multiscale metadata is put under the version-appropriate key and within a list
1. for versions newer than 0.4, the `version` needs to be written *next to* `multiscales`

```python
ome_ms = ms.to_ome_zarr(version="0.4")
zarr_group.attrs = {"multiscales": [ome_ms]}
# Note: Actual data has to be written in zarr format v2 for OME-Zarr 0.4

ome_ms = ms.to_ome_zarr(version="0.5")
zarr_group.attrs = {"ome": {"version": "0.5", "multiscales": [ome_ms]}}

ome_ms = ms.to_ome_zarr(version="0.6.dev4")
zarr_group.attrs = {"ome": {"version": "0.6.dev4", "multiscales": [ome_ms]}}
```
