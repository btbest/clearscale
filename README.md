# clearscale

`clearscale` is a small pure-Python package for clear multi-scale image metadata manipulation.

Fits in any Python environment. Works with existing code. Handles all OME-Zarr versions.

With `clearscale`, metadata runs alongside data processing.

Ideally, to ensure data and metadata are in sync, the pipeline first specifies the operation plan, and then uses the metadata itself as parameters:

```python
from clearscale import Shape, PixelSize, Unit, BlueprintShapes, Scale, Multiscale, OmeZarrGroup

# 1. Annotate
shape      = Shape(zip("tzyx", my_data.shape))
pixel_size = PixelSize(t=5.0, z=260.0, y=0.53, x=0.53)
unit       = Unit(t="s", z="micrometer", y="micrometer", x="micrometer")

# 2. Define operation plan
scaling_blueprint = BlueprintShapes.downscale_powers_of_2_xyz(
    base_shape=shape,
    rounding="ceil",
    shape_limit=Shape(z=8, y=128, x=128),
)

# 3. Scale data according to the blueprint
for scale_key, target_shape in scaling_blueprint.items():
    scaled_data = do_my_scaling(my_data, target_shape.to_tuple())
    zarr_group.create_array(scale_key, data=scaled_data)

# 4. Expand and write metadata
base = Scale(shape, pixel_size, unit)
ms = Multiscale.from_single(base, blueprint=scaling_blueprint)
group_meta = OmeZarrGroup.from_single(ms).to_attrs(version="0.6.rc0")
zarr_group.attrs.update(group_meta)
```

It may sometimes be simpler to instead describe the data processing post-hoc, and derive matching metadata (as in the numpy example further down).
Proceed whichever way best fits your project.

`clearscale` is independent of the actual data-handling packages in your environment.
For example, if you use `numpy`, `scikit-image` and `zarr-python`, the placeholders above could look like:

```
import os, numpy, skimage, zarr

my_data = numpy.random.rand(3, 12, 512, 512)
do_my_scaling = skimage.transform.resize
zarr_group = zarr.open_group(os.path.expanduser("~/cltest.ome.zarr"), mode="w")

# Now you can actually run the example snippet above and produce a valid `cltest.ome.zarr`
```

## Features

* Zero dependencies, runs with Python 3.10+
* Reads Neuroglancer Precomputed and OME-Zarr metadata (all versions)
* Writes OME-Zarr versions 0.4, 0.5 and 0.6.rc0
* Saves you learning about the metadata format(s)
* Helps you write expressive code
* Metadata manipulation lives alongside data manipulation

## Install

`clearscale` is still being actively developed.
We would love to hear your feedback if you try it out!

This also means the API is subject to change, particularly regarding names of classes, methods, etc., and regarding required parameters.
We want the 1.0 API to be as intuitive as possible.
If you encounter anything unclear, please feel free to suggest alternatives.

Until the first package release you can install directly from GitHub:

```bash
pip install git+https://github.com/ilastik/clearscale.git
```

Or add to your conda `env.yaml`:

```yaml
dependencies:
  - pip
  - pip:
      - git+https://github.com/ilastik/clearscale.git
```

## End-to-end examples

### Dump a numpy array to OME-Zarr

Even a single array needs to become a "multiscale" for OME-Zarr.

```python
import numpy as np
import zarr
from clearscale import OmeZarrGroup, Multiscale, Scale

# You have some array
image = np.random.random((128, 1024, 1024)).astype(np.float32)

# 1. Create a zarr group to write it to
# (zarr-format v3 because we specify OME-Zarr version 0.5 below, which must be written in zarr-format v3)
group = zarr.open_group("demo-output/example1.ome.zarr", mode="w", zarr_format=3)

# 2. Write data
array_path = "s0"
group.create_array(array_path, data=image)

# 3. Write metadata
# (Make a Scale, expand it to a Multiscale, put that in an OME-Zarr group)
scale = Scale(shape=dict(zip("zyx", image.shape)))  # Axis keys are the minimum you really must specify
group.attrs.update(
    OmeZarrGroup
    .from_single(Multiscale.from_single(scale, scale_key=array_path))
    .to_attrs(version="0.5")
)
```

### Downsample a numpy array and save it as OME-Zarr

The first example at the top of this Readme used a "metadata-first" approach (first compute metadata blueprint, then scale data according to it).
This example works the other way round (first scale data, then record what was done in blueprint).

```python
# This example assumes numpy, scikit-image and zarr-python 3.* are installed
import numpy as np
import zarr
from skimage.transform import pyramid_gaussian

from clearscale import BlueprintShapes, Scale, Shape, OmeZarrGroup, Multiscale

# 1. Generate a pyramid
image = np.random.random((128, 1024, 1024)).astype(np.float32)
pyramid = [level.astype(np.float32) for level in pyramid_gaussian(image, downscale=2)]

# 2. Write arrays to zarr on disk and record scaled shapes
group = zarr.open_group("demo-output/example2.ome.zarr", mode="w")
scaled_shapes = []
for i, level in enumerate(pyramid):
    scale_key = f"s{i}"
    group.create_array(scale_key, data=level)
    scaled_shapes.append(
        (
            scale_key,
            Shape(zip("zyx", level.shape))
        )
    )

# 3. Describe the full-resolution image
base = Scale(
    shape=Shape(zip("zyx", image.shape)),
    pixel_size=dict(z=25, y=240, x=240),
    unit=dict(z="micrometer", y="nanometer", x="nanometer"),
)

# 4. Use the recorded scale shapes as a blueprint to expand a Multiscale
blueprint = BlueprintShapes(scaled_shapes)
multiscale = Multiscale.from_single(base, blueprint=blueprint)

# 5. Save OME-Zarr metadata
group_meta = OmeZarrGroup.from_single(multiscale).to_attrs(version="0.5", axis_types="infer")
group.attrs.update(group_meta)
```

### Download a single scale of a public OME-Zarr as a valid local OME-Zarr

```python
from pathlib import Path
import zarr  # This example assumes zarr-python 3.* is installed (plus fsspec, requests and aiohttp for https reading)
import clearscale as clear

URL = "https://s3.embl.de/i2k-2020/platy-raw.ome.zarr"
SCALE_KEY = "s6"
LOCAL_PATH = Path(f"demo-output/platy-raw-{SCALE_KEY}.ome.zarr")

# 1. Discover what's on the server
remote_group = zarr.open_group(URL)

remote_meta = clear.OmeZarrGroup.from_group(remote_group)
assert remote_meta.kind is clear.GroupKind.MULTISCALE, "This tells us there is exactly one Multiscale in this group"
assert remote_meta.version == "0.4", "OME-Zarr version is a property of the group, not of individual Multiscales"

source_multiscale = remote_meta.multiscales[0]
assert SCALE_KEY in source_multiscale, "Multiscale is dict-like, with the sub-paths to arrays as keys"
assert source_multiscale[SCALE_KEY].shape == {"z": 357, "y": 405, "x": 430}

# 2. Create the local target and download the data
source_array = remote_group[SCALE_KEY]
local_group = zarr.open_group(str(LOCAL_PATH), mode="w", zarr_format=2)  # OME-Zarr v0.4 must use zarr-format v2
print(f"Downloading {SCALE_KEY} data...")
local_array = local_group.create_array(SCALE_KEY, data=source_array)

# 3. Derive a new Multiscale from the chosen Scale entry in the existing Multiscale
target_multiscale = source_multiscale.derive(SCALE_KEY)

# 4. Write the new metadata to the downloaded store
local_group_meta = clear.OmeZarrGroup.from_single(target_multiscale)
local_group.attrs.update(local_group_meta.to_attrs(version="0.4"))
```

## Documentation

Docs for clearscale's concepts are at [`docs/index.md`](docs/index.md).
The OME-Zarr How-To is at [`docs/ome_zarr.md`](docs/ome_zarr.md).

Generally though, we hope the API is sufficiently self-documenting and discoverable through tab-completion and type annotations ;)

## Why clearscale?

The motivation behind clearscale is to make upgrading from

> My tool/script does **zarr**

to

> My tool/script does **OME-Zarr**

as easy as possible.

### Why not use one of the existing libraries?

You *should* probably use one of the existing libraries, if they work for you :)

Check out the [NGFF Tools](https://ngff.openmicroscopy.org/resources/tools/index.html) page for some examples.
Python libraries include e.g.
[bioio](https://github.com/bioio-devs/bioio),
[ngff-zarr](https://github.com/thewtex/ngff-zarr),
[ngio](https://biovisioncenter.github.io/ngio/stable/),
[ome-zarr-py](https://github.com/ome/ome-zarr-py), and
[yaozarrs](https://github.com/tlambert03/yaozarrs).
Several of these handle not only the metadata, but also data manipulation all in one.

### Dependencies are hard

But maybe you can't, or don't want to, use any of the above.

* Technical reasons (dependency conflicts)
* Legal reasons (licensing conflicts, commercial use)
* You don't want to reimplement existing data processing with another library. 
* You just want to minimise bloat.

clearscale is tiny, dependency-free, permissively licensed, and works around your existing data processing.

### Zarr is complex enough

There
[are](https://github.com/zarr-developers/zarr-python)
[several](https://github.com/canpute/simplezarr)
[backend](https://github.com/google/tensorstore)
[packages](https://github.com/constantinpape/z5)
your application or script might use for reading and writing zarr data.
Implementing data processing in an efficient, chunk-wise (shard-wise) manner is complex on its own.
You shouldn't need to learn the OME metadata specification on top to make your datasets accessible to others, or interoperable with other tools.

With clearscale, no matter how you handle zarr *data*, handling multiscale *metadata* like OME-Zarr looks the same.

### Custom applications

There are thousands of ways to scale or otherwise transform an image.
Maybe none of the existing libraries that can write OME-Zarr implements quite what you need, so you have to roll your own.

clearscale isn't tied to data processing.
Whatever you're doing, clearscale can simplify metadata manipulation.

### Should I use `clearscale` in my project right now?

If you're willing to adapt to future changes in the API: Yes, please do try it!

If you are looking for a stable API, please wait until the first formal release to PyPI.
After that point, API changes will follow semver and deprecations will be avoided or give appropriate notice through DeprecationWarnings.

## License

Licensed under either the [MIT license](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.