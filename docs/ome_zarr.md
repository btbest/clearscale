In a nutshell, for simple cases:

```python
from clearscale import OmeZarrGroup

# Read

# zarr_group is whatever object your zarr library uses for group access.
# e.g. zarr-python: zarr_group = zarr.open(path)
#      z5py:        zarr_group = z5py.File(path)
multiscale = OmeZarrGroup.from_group(zarr_group).multiscales[0]

# Do stuff

output_multiscale = make_multiscale_according_to_actual_data_processing()

# Write

output_group_meta = OmeZarrGroup.from_single(output_multiscale).to_attrs(version="0.5")
output_zarr_group.attrs.update(output_group_meta)
```

# Micro-intro to plain zarr metadata

OME-Zarr "datasets" are always described in the metadata (attributes) of a plain zarr "group".
A zarr group is just a folder that contains:
```
Folder with Zarr-format version 2:

  .zgroup  # Tiny indicator file that means "this folder is a zarr group"
  .zattrs  # Metadata file
  subfolder/  # Can be sub-groups, or zarr arrays (or just plain zarr-unaware folders)

Folder with Zarr-format version 3:

  zarr.json  # Combined file containing both the definition "this folder is a zarr group" and the metadata
  subfolder/
```

In the context of zarr, “the attrs” refers to a JSON object containing a zarr group’s user-defined attributes.
The location of "the attrs" again differs by zarr-format version:
* Zarr-format version 2: The entire contents of `.zattrs` are "the attrs" (a single JSON object)
* Zarr-format version 3: `zarr.json` contains a JSON object, which contains the `"attributes"` key.
  So "the attrs" are the JSON object found at `zarr.json["attributes"]`

If you use the `zarr-python` or `z5py` packages as your zarr backend, **you don't need to think about this**.
Both have a `Group` class that exposes `Group.attrs`, which reads from, and writes to, the correct location.

If you use `tensorstore`, you do have to implement a small adapter to read and write "the attrs" using the logic above :)

# Reading OME-Zarr

Typically, you're given a path to some dataset:

```python
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=Path, required=True)
```
```
python script.py --path /data/might/be/zarr
python script.py --path https://someserver.org/public-data/00121
```

Your zarr backend will tell you whether that path points to a zarr object at all.
Assuming it does, the next thing you will want to know is, "Is this *zarr* object an *OME-Zarr* object?"

clearscale recognises all valid OME-Zarr objects of all OME-Zarr versions, up to and including 0.6.rc0.

## "Is this zarr thing an OME-Zarr thing?"
`clearscale.OmeZarrGroup` answers this question.

```python
from clearscale import OmeZarrGroup

# Option 1
ome_meta = OmeZarrGroup.from_group(zarr_group)

# Option 2, if you want to avoid requests for array metadata (e.g. when reading remote Zarr)
ome_meta = OmeZarrGroup.from_attrs(zarr_group.attrs, shape_source="singletons")
```

You can either pass the group to `.from_group`, or the attrs to `.from_attrs`.

`from_attrs` needs you to specify a `shape_source`, because it needs to obtain array shapes, and the only way to obtain array shapes is to read them from the group...
So if you don't provide the group, there needs to be another way to obtain them.
`"singletons"` is a magic word that will tell clearscale to skip obtaining shapes entirely and just put `1` along all axes.

You can now find out whether the metadata describes an OME-Zarr dataset, and if so, what kind, through its `.kind`:

```python
if ome_meta.kind is None:
  print("Looks like this isn't an OME-Zarr group at all")
else:
  print(ome_meta.version)  # "0.4", "0.5", ... (could still be None if not actually written, even if the metadata was valid)
```

```python
if ome_meta.kind is clearscale.GroupKind.MULTISCALE:
    print("Nice - it's a multiscale")
    multiscale = ome_meta.multiscales[0]

if ome_meta.kind is clearscale.GroupKind.COLLECTION:
    print("Hm, there's more stuff here.")
    print("This zarr group describes:")
    print(f"- {len(ome_meta.multiscales)} multiscales")
    print(f"- {len(ome_meta.scenes)} scenes (OME-Zarr 0.6)")
    print(f"- {len(ome_meta.children)} references to OME-Zarr objects at other paths")
```

## Obtaining actual image data at a particular scale

In the `MULTISCALE` case, you now know that there is exactly one multiscale you can find here:
```python
if ome_meta.kind is GroupKind.MULTISCALE:
    assert (
        len(ome_meta.multiscales) == 1
        and not ome_meta.scenes 
        and not ome_meta.children
    ), "The MULTISCALE kind guarantees there is exactly one multiscale"
```
This is probably the most common case.

`ome_meta.multiscales[0]` is a `clearscale.Multiscale` instance.
In clearscale, Multiscales work like a dict:

```python
print(multiscale)
# Multiscale({
#   "s0": Scale(shape=Shape({'z': 11416, 'y': 25916, 'x': 27499}), pixel_size=...),
#   "s1": Scale(shape=Shape({'z': 11416, 'y': 12958, 'x': 13750}), pixel_size=...),
#   "s2": Scale(shape=Shape({'z': 5708,  'y': 6479,  'x': 6875}), pixel_size=...),
#   ...
# })
```

The keys (s0, s1, s2...) are relative paths to the data arrays.
Reading the data depends on your zarr backend, but usually looks like:

```python
# The Multiscale's keys can be used as keys to index your zarr Group in zarr-python and z5py
scale_data = zarr_group["s2"]

print(scale_data.shape)  # (5708, 6479, 6875)

# Do whatever you do with your data arrays
first_z_slice_data = numpy.take(scale_data, 0, axis=list(multiscale.axes()).index("z"))
```

And that's it!

Other important information you can get from a Multiscale includes:

* The axes: `list(ms.axes())`, usually `["t", "c", "z", "y", "x"]` or a subset of them, in that order.
* Each Scale `ms[scale_key]` provides:

```python
multiscale["s2"] == Scale(
  shape       = Shape({'z': 5708, 'y': 6479, 'x': 6875}),  # Number of pixels along each axis
  pixel_size  = PixelSize({'z': 0.05, 'y': 0.04, 'x': 0.04}),  # Physical size of each pixel
  unit        = Unit({'z': 'micrometer', 'y': 'micrometer', 'x': 'micrometer'}),  # Physical units of the pixel size
  translation = Translation({'z': 0.0, 'y': 0.0, 'x': 0.0})  # Shift of this scale's first pixel center from the multiscale's common origin
)
```

More detail on `.translation` in  [`translation_shifts.md`](translation_shifts.md) (depends on scaling method, relevant for correct positioning when displaying data from different scales on top of each other).

## Handling the non-OME case

Users don't always provide paths to a zarr group that actually contains OME-Zarr metadata, even though the path is part of an OME-Zarr dataset:

* They might provide a path directly to a zarr *array* (an individual scale level of a multiscale)
* They might provide a path within a nested OME-Zarr structure like one column group of an OME-Zarr "Plate"

The common practice is "bubbling up" the provided path to check whether any of the parent folders contain OME-Zarr metadata:

```python
path = "/drive/project/raw_data/multiscale_em.zarr/raw/scale3"
zarr_object = zarr.open(path)  # Let's say this is actually a zarr array. OmeZarrGroup ignores that as long as zarr_object has `.attrs`
ome_meta = OmeZarrGroup.from_group(zarr_object)
assert ome_meta.kind is None, "scale3 is a zarr *array* and does not itself contain OME-Zarr group metadata"

parent = "/drive/project/raw_data/multiscale_em.zarr/raw"
parent_group = zarr.open(parent)
parent_meta = OmeZarrGroup.from_group(parent_group)
assert parent_meta.kind is None, "And let's say `raw` is some intermediate group with no metadata. This is allowed in OME-Zarr."

parent2 = "/drive/project/raw_data/multiscale_em.zarr"
parent2_group = zarr.open(parent2)
parent2_meta = OmeZarrGroup.from_group(parent2_group)
assert parent2_meta.kind is clearscale.GroupKind.MULTISCALE, "Found it"
assert "raw/scale3" in parent2_meta.multiscales[0], "Scale key is the full relative path to the originally received array path"
```

In practice, the only way to be sure the provided path really is not OME-Zarr, is to iterate all the way up all parents.

## Retrieving paths for multiple multiscales or collections

The other `GroupKind` are all some form of collection that point to other OME-Zarr groups.

The simplest thing to do is to choose (or let the user choose) one of the referenced paths and try to load it.
You can access all paths recorded in the group metadata itself through `OmeZarrGroup.children`.

`.children` contains `clearscale.ChildRef` instances.
You can learn what kind of children (well, multiscale, label) from each child's `.child_type`, or you can just collect and display each child's `.file.path` to choose from.
Up to OME-Zarr version 0.6, these paths are always relative to the group you loaded the metadata from.

Groups of kind `MULTISCALE`, `PLATE` or `BF2RAW` can have `OmeZarrGroup.maybe_subgroups`.
The OME-Zarr standard under some circumstances specifies that certain specially-named subgroups may exist, and if they do exist, you will find metadata there that points you to more multiscales.
This concerns "labels" and the "bioformats2raw.layout".
You can either implement special handling for these cases (see advanced section), or iterate the `.maybe_subgroups` to retrieve any `.children` they reference.
Note that if present, `subgroup_meta.children` will contain paths to more multiscales *relative to the subgroup*.

More detail in the "Advanced OME-Zarr features" section.

# Writing OME-Zarr

You need two things:

* The `clearscale.Multiscale` that describes the actual zarr group and arrays you wrote, and
* the OME-Zarr version you want to write.

clearscale supports writing OME-Zarr metadata in the stable OME-Zarr versions `0.4` and `0.5`, and the "in-development" version `0.6.rc0`.

## Making new Multiscales
The two most important methods here are: `Multiscale.from_single` and `Multiscale.derive`.

### Creating a Multiscale from scratch
`from_single` is for creating a new Multiscale from scratch, from a single `Scale` object you manually create.

The absolute minimum path when you really have no metadata at all:

```python
# If you don't even know axes, you can't use OME-Zarr
scale = Scale(
  shape=Shape.fromkeys("zyx")
)

multiscale = Multiscale.from_single(
  scale, 
  scale_key="data_subpath"
)
```

If more metadata is available (pixel size, units, etc.), pass it to the `Scale`.

If you're scaling the data, make a matching blueprint and add the `blueprint=your_blueprint` parameter to expand the single Scale to multiple scales matching your blueprint.
Don't forget the appropriate `translation_shift_func`.

(Also cf. the other examples in [the top-level readme](../README.md))

### Deriving a new Multiscale from an existing one
`derive` is for creating new Multiscales from one of an existing Multiscale's entries.

This is intended to match workflows that load a multiscale dataset and select one of its scale levels for further processing.

`derive` easily creates a new single-scale-Multiscale that lets you carry over the metadata with no extra effort:

```python
assert "s3" in old_multiscale, "Deriving from an existing scale at 's3'"

new_multiscale = old_multiscale.derive("s3")
# Multiscale({"s3": Scale(...)})
```

If you also scale the processing output derived from that scale, provide the matching blueprint and the translation shift as for `from_single`.

`derive` accepts an additional parameter `derived_by: Union[SpatialRelation, Sequence[SpatialRelation]]`.
You can use it to specify *how* the new Multiscale was derived from the source Scale.
Note that when it comes to writing output metadata, only OME-Zarr version 0.6 can fully express all `SpatialRelations`.
When generating OME-Zarr versions 0.5 or 0.4, clearscale will express the relation in the output metadata only if possible (e.g. if it only consists of `Factor`, `Translation` and/or `AxisRearrangementTo`).

## Making new OmeZarrGroups

All OME-Zarr datasets are zarr *groups*.
Your own code needs to create the output group, and write the array(s) to the correct paths inside it.
clearscale helps you generate the zarr attrs for the group.

Your processing should have created a new Multiscale as described in the previous section:

```python
from clearscale import Multiscale

new_multiscale: Multiscale = see_section_above()
```

As on the reading side, the Multiscale's keys must match the paths within the output zarr group where the corresponding data of that scale level is written.
In principle, the layout on disk needs to correspond to:

```python
new_zarr_group = zarr.open_group(output_path, "w")
for scale_key in new_multiscale.keys():
    new_zarr_group.create_array(
      scale_key, 
      data=get_data_for_scale(scale_key)
    )
```

If you did not scale the data, you would of course only have one scale, and you would know its key (the array path within the group).
If you did scale the data, it might be simpler to write the data according to the scaling blueprint you used as shown in the [`README`](../README.md) examples.

Generating the new zarr attrs then simply becomes:

```python
from clearscale import OmeZarrGroup

new_ome_meta = OmeZarrGroup.from_single(new_multiscale)

new_attrs = new_ome_meta.to_attrs(version="0.5")

new_zarr_group.attrs.update(new_attrs)
```

Or more compact:

```python
from clearscale import OmeZarrGroup

new_zarr_group.attrs.update(
    OmeZarrGroup
    .from_single(new_multiscale)
    .to_attrs(version="0.5")
)
```

That's all.

## OME-Zarr version specifics

There is only one important detail about OME-Zarr versions that you need to know:

1. OME-Zarr version 0.4 *MUST* write the data in zarr-format version 2
2. OME-Zarr version 0.5 and newer *MUST* write the data in zarr-format version 3

For example, in `zarr-python`, this means you need to provide `zarr_format=2` or `zarr_format=3` when creating the groups and arrays.

clearscale doesn't write data, so you need to make sure you use the correct zarr-format version.

## Which OME-Zarr version should I write?

In general, the current "most stable" version is 0.5.

Note though that many tools only support one version of OME-Zarr.
Support for 0.4 is probably about as common as support for 0.5.
If you have a specific downstream use-case, it is important to check which version(s) work downstream.

Very few tools support 0.6.

# Advanced OME-Zarr features

This covers multiple multiscales, HCS plates, labels, bioformats2raw, scenes.

## Path discovery

Special cases for further path discovery that you may want to handle for full support:

* OME-Zarr multiscale groups *usually* define only a single multiscale (in which case they would be `GroupKind.MULTISCALE`), but they can define multiple (in which case they are `GroupKind.COLLECTION`). In this case, you could obtain the first scale key from each Multiscale, form the full paths like `f"{group_path}/{first_scale_key}"`, and display them to choose from. Then on the second round, pick the Multiscale that contains the chosen scale key.
* In the `GroupKind.BF2RAW` case, there may or may not be a subgroup called `"OME"` that you can try inspecting like `bf2raw_ome_meta = OmeZarrGroup.from_group(bf2raw_group["OME"])`.
  If it exists, it may or may not have `bf2raw_ome_meta.children` with `child.file.path` like `"../ms_0"`.
  The paths will all begin with `../` because they reference multiscales in `bf2raw_group`, which is the parent of `bf2raw_group["OME"]` (where the `bf2raw_ome_meta.children` are defined).
  If the `bf2raw_group["OME"]` does *not* exist or if there are no `bf2raw_ome_meta.children`, this means there are an unknown number of multiscales in the parent `bf2raw_group` and the only way to find out how many is to try progressively loading `bf2raw_group["0"]`, `bf2raw_group["1"]`, etc., until a load fails because that number is absent.
* Any OME-Zarr group that has `.multiscales` may also have one `"labels"` subgroup per multiscale.
  This subgroup will be a sibling to the multiscale's individual scale arrays.
  If a multiscale's first key is `ms0/scale0`, there may or may not be a group at `ms0/labels` that points to label multiscales (usually segmentation overlays).
  If you construct an `OmeZarrGroup` from this subgroup, its kind will be `GroupKind.LABELS` and it will contain some `.children` whose paths point to label multiscales.
* Scenes can contain paths to multiscales that are part of the scene. The `OmeZarrGroup` folds these into `.children`, so you don't have to retrieve them separately.

### Special treatment for advanced OME-Zarr kinds

If a group is `GroupKind.SCENE` (single scene) or `GroupKind.COLLECTION`, the group may contain one or more `.scenes`.
These are `clearscale.Scene` instances and handle the scene as defined in OME-Zarr version 0.6.
The specifics of the `Scene` class in clearscale are still in development, so expect changes to methods and parameters in the future.

If you want to specially handle HCS plates or labels, you would currently need to implement custom parsing.
clearscale does not offer a convenient API to access plate layout or label display metadata yet, though the contained Multiscales can be found via `group_meta.children`.
You can detect these cases using `GroupKind.PLATE`, `.WELL`, `.LABELS`.
