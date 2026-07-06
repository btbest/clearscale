# Translation shifts

`Scale.translation` is the physical coordinate of the first pixel center.
When clearscale derives a scaled `Scale`, the shape and pixel size can be calculated from the output shape, but the translation depends on how the scaled pixels were sampled from the original data.
This cannot be inferred purely from existing metadata or the scaling blueprint.

Pass a `translation_shift_func` to `blueprint.apply_to_scale()` when your scaling operation changes where the first scaled pixel belongs in physical space:

```python
multiscale = blueprint.apply_to_scale(
    base,
    translation_shift_func=half_pixel_space_preservation,
)
```

The right shift is the one that matches the data operation you actually ran.

Most common scaling methods do apply one of the provided shift functions.
Implement your own and pass it if your scaling method applies a different shift.

## `half_pixel_space_preservation`: extent-preserving resize

Use `half_pixel_space_preservation` for interpolation or resampling methods that preserve the full image extent under a pixel-center convention.
This is the convention where the data samples are considered to be positioned at the center of a box ranging from `-0.5 * pixel_size` to `0.5 * pixel_size`.

Common examples include output-shape based interpolation such as `skimage.transform.resize(..., anti_aliasing=True)`, or in deep learning contexts, `nn.Upsample(..., align_corners=False)`.

```python
from clearscale import (
    BlueprintShapes,
    PixelSize,
    Scale,
    Shape,
    Translation,
    Unit,
    half_pixel_space_preservation,
)

base = Scale(
    shape=Shape(y=1024, x=1536),
    pixel_size=PixelSize(y=0.3, x=0.3),
    unit=Unit(y="micrometer", x="micrometer"),
    translation=Translation(y=0.0, x=0.0),
)

blueprint = BlueprintShapes.uniform_steps(
    step=2,
    base_shape=base.shape,
    rounding="round",
    shape_limit=Shape(y=256, x=384),
)

# Data scaling, for example with scikit-image:
# from skimage.transform import resize
#
# scaled_arrays = {}
# for scale_key, target_shape in blueprint.items():
#     scaled_arrays[scale_key] = resize(
#         raw_yx,
#         output_shape=target_shape.to_tuple(),
#         anti_aliasing=True,
#         preserve_range=True,
#     )

multiscale = blueprint.apply_to_scale(
    base,
    translation_shift_func=half_pixel_space_preservation,
)

assert multiscale["s1"].shape == Shape(y=512, x=768)
assert multiscale["s1"].pixel_size == PixelSize(y=0.6, x=0.6)
assert multiscale["s1"].translation == Translation(y=0.15, x=0.15)
```

The first output pixel center comes out `0.15 micrometer` translated.

Under the pixel-center convention, the first *raw* pixel at coordinate `0.0`, with pixel size `0.3 micrometer`, represents the space from `-0.15 micrometer` to `0.15 micrometer` (half a pixel before and after its center coordinate).
Therefore, the data space represented by the image begins at `-0.15`.

For `s1`, the output pixel size is `0.6 micrometer`.
The first `s1` pixel's coordinate (i.e. its center) is therefore exactly half of that (`0.3 micrometer`) shifted from the beginning of the space it represents.
If the scaling method preserves the full data space, then the space still begins at `-0.15`.
The first `s1` pixel is therefore at `-0.15 + 0.3 = 0.15`.

## `discrete_bin_center`: bin-averaging

Use `discrete_bin_center` for block or bin downsampling where each output value represents the center of the source pixels that contributed to that bin.
This fits local mean, max, min, median, or similar pooling methods.

For exact integer factors, the numeric shift can be the same as `half_pixel_space_preservation`.
The distinction still matters: choosing the appropriate function documents the pooling convention.

```python
from clearscale import (
    BlueprintFactors,
    Factor,
    PixelSize,
    Scale,
    Shape,
    Translation,
    Unit,
    discrete_bin_center,
)

base = Scale(
    shape=Shape(y=2048, x=2048),
    pixel_size=PixelSize(y=0.25, x=0.25),
    unit=Unit(y="micrometer", x="micrometer"),
    translation=Translation(y=0.0, x=0.0),
)

block_factors = BlueprintFactors(
    {
        "s0": Factor(y=1, x=1),
        "s1": Factor(y=2, x=2),
        "s2": Factor(y=4, x=4),
    }
)
blueprint = block_factors.to_shapes(base.shape, rounding="ceil")

# Data scaling, for example with scikit-image:
# import numpy as np
# from skimage.measure import block_reduce
#
# binned_arrays = {}
# for scale_key, factor in block_factors.items():
#     block_size = tuple(int(factor[axis]) for axis in base.shape)
#     binned_arrays[scale_key] = block_reduce(
#         raw_yx,
#         block_size=block_size,
#         func=np.mean,
#     )

multiscale = blueprint.apply_to_scale(
    base,
    translation_shift_func=discrete_bin_center,
)

assert multiscale["s2"].shape == Shape(y=512, x=512)
assert multiscale["s2"].pixel_size == PixelSize(y=1.0, x=1.0)
assert multiscale["s2"].translation == Translation(y=0.375, x=0.375)
```

For `s2`, each output pixel summarizes a `4 x 4` block.
The first output pixel represents input pixel centers at `0.0`, `0.25`, `0.5`, and `0.75 micrometer` along each axis, so its center is `0.375 micrometer`.

## `first_value_decimation`: stride from the first sample

Use `first_value_decimation` when the scaled array keeps the first source value and then takes every `n`-th value.
This is common for quick preview pyramids or for label images where the chosen policy is explicitly first-sample decimation, for example `labels[::4, ::4]`.

If your nearest-neighbor implementation samples block centers instead, use a shift function that matches that center-sampling behavior instead.

```python
from clearscale import (
    BlueprintFactors,
    Factor,
    PixelSize,
    Scale,
    Shape,
    Translation,
    Unit,
    first_value_decimation,
)

base = Scale(
    shape=Shape(y=2048, x=2048),
    pixel_size=PixelSize(y=0.25, x=0.25),
    unit=Unit(y="micrometer", x="micrometer"),
    translation=Translation(y=12.0, x=-3.0),
)

stride_factors = BlueprintFactors(
    {
        "s0": Factor(y=1, x=1),
        "s1": Factor(y=2, x=2),
        "s2": Factor(y=4, x=4),
    }
)
blueprint = stride_factors.to_shapes(base.shape, rounding="ceil")

# Data scaling with first-value stride slicing:
# decimated_arrays = {}
# for scale_key, factor in stride_factors.items():
#     stride_y = int(factor["y"])
#     stride_x = int(factor["x"])
#     decimated_arrays[scale_key] = labels_yx[::stride_y, ::stride_x]

multiscale = blueprint.apply_to_scale(
    base,
    translation_shift_func=first_value_decimation,
)

assert multiscale["s2"].shape == Shape(y=512, x=512)
assert multiscale["s2"].pixel_size == PixelSize(y=1.0, x=1.0)
assert multiscale["s2"].translation == Translation(y=12.0, x=-3.0)
```

The first output value is the first input value, so the translation does not move.
Only the pixel size changes to describe the larger spacing between retained samples.

## Which shift is the right one for my scaling method?

`clearscale` offers a `detect_translation_shift` that can be used to infer the appropriate translation shift function.

To use it, you need to wrap the scaling method you actually use into a function that takes a 1D sequence of floats, scales it, and returns the scaled values.
It doesn't matter how much you scale.
If your scaling method requires a NumPy array, convert the received sequence inside your wrapper (as in the examples below).

The function returns the closest-matching prebuilt shift function, and its absolute error from the actual shift the scaling method introduced.

```python
from clearscale import detect_translation_shift, half_pixel_space_preservation, discrete_bin_center, first_value_decimation

import numpy as np
from skimage.transform import resize

matching_function, error = detect_translation_shift(
    lambda x:
    resize(
        np.asarray(x),
        (327,),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
)

assert matching_function is half_pixel_space_preservation
assert error < 1e-13

from skimage.transform import downscale_local_mean

matching_function, error = detect_translation_shift(
    lambda x:
    downscale_local_mean(
        np.asarray(x),
        (3,),
    )
)

assert matching_function is discrete_bin_center
assert error < 1e-13

from scipy.ndimage import zoom

matching_function, error = detect_translation_shift(
    lambda x:
    zoom(
        np.asarray(x),
        0.27,
        order=1,
        grid_mode=False,
    )
)

assert matching_function is first_value_decimation
assert error < 1e-13
```
