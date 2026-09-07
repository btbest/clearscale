# clearscale examples

These examples are meant to show where clearscale fits in existing image code: You keep using your preferred array and zarr libraries for data, and clearscale keeps the multiscale metadata aligned with the shapes you produce.

## Add metadata to existing arrays

```python
from clearscale import BlueprintShapes, PixelSize, Scale, Shape, Unit, Multiscale, OmeZarrGroup

# zarr_group can be zarr-python, TensorStore-backed code, or your own wrapper.
# The only thing clearscale needs here is each array's shape.
scale_keys = ("s0", "s1", "s2")
recorded_shapes = [
    (scale_key, Shape(zip("zyx", zarr_group[scale_key].shape)))
    for scale_key in scale_keys
]

base_scale = Scale(
    shape=recorded_shapes[0][1],
    pixel_size=PixelSize(z=0.5, y=0.25, x=0.25),
    unit=Unit(z="micrometer", y="micrometer", x="micrometer"),
)

blueprint = BlueprintShapes(recorded_shapes)
multiscale = Multiscale.from_single(base_scale, blueprint=blueprint)

zarr_group.attrs.update(
    OmeZarrGroup.from_single(multiscale).to_attrs(version="0.5", axis_types="infer")
)
```

## Reuse a multiscale's scaling pattern for a new image

```python
from clearscale import BlueprintShapes, Multiscale, PixelSize, Scale, Shape, Unit

# Let's assume you have some existing Multiscale (maybe loaded by Multiscale.from_ome_zarr)
template = Multiscale(
    {
        "s0": Scale(
            shape=Shape(c=1, z=64, y=1024, x=1024),
            pixel_size=PixelSize(c=1, z=0.5, y=0.25, x=0.25),
            unit=Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
        "s1": Scale(
            shape=Shape(c=1, z=32, y=512, x=512),
            pixel_size=PixelSize(c=1, z=1.0, y=0.5, x=0.5),
            unit=Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
        "s2": Scale(
            shape=Shape(c=1, z=16, y=256, x=256),
            pixel_size=PixelSize(c=1, z=2.0, y=1.0, x=1.0),
            unit=Unit(c="", z="micrometer", y="micrometer", x="micrometer"),
        ),
    }
)

target_base = Scale(
    shape=Shape(t=65, c=3, z=40, y=2048, x=2048),
    pixel_size=PixelSize(t=0.5, c=1.0, z=0.7, y=0.1, x=0.1),
    unit=Unit(t="seconds", c="", z="micrometer", y="micrometer", x="micrometer"),
)

blueprint = BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=target_base.shape,
    rounding="ceil",
).with_axes("tczyx")

target_multiscale = blueprint.apply_to_scale(target_base)

assert target_multiscale["s0"].shape == Shape(t=65, c=3, z=40, y=2048, x=2048)
assert target_multiscale["s1"].shape == Shape(t=65, c=3, z=20, y=1024, x=1024)
assert target_multiscale["s2"].shape == Shape(t=65, c=3, z=10, y=512, x=512)
```

You can also restrict the inherited scaling pattern to selected axes:

```python
xy_only = BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=target_base.shape,
    rounding="ceil",
    scaled_axes="yx",
)

assert xy_only["s2"] == Shape(t=65, c=3, z=40, y=512, x=512)  # z not scaled
```

Or rebase from a non-root scale when your input data is already downsampled:

```python
rebased = BlueprintShapes.from_multiscale_rescaled(
    template,
    target_shape=Shape(c=3, z=20, y=1024, x=1024),
    source_key="s1",
    rounding="ceil",
)

assert rebased["s0"] == Shape(c=3, z=40, y=2048, x=2048)
assert rebased["s1"] == Shape(c=3, z=20, y=1024, x=1024)
assert rebased["s2"] == Shape(c=3, z=10, y=512, x=512)
```
