# Repo structure

`src/clearscale/`, in order from lowest to highest level:

- `types.py` defines common types (AxisKey)
- `_spatial_relations.py` defines the SpatialRelation base and simple non-mapping implementations (PermutationTo, ProjectionTo)
- `_axis_values.py` defines primitives (Shape, Unit, PixelSize, ...)
- `_affines.py` defines nested AxisValues (Linear matrices, their Coefficient building block, and Affine container)
- `_transforms` contains the new concepts from OME-Zarr 0.6. `_base.py` defines the transformation graph, transform base class and TransformSequence (which still interacts with the graph). `_transform_types.py` implements the various Transform subclasses according to the OME-Zarr 0.6 spec. The public API for users to interact with Transforms is the SpatialRelation.
- `_multiscale.py` defines the primary user-value objects (Scale, Multiscale, BlueprintShapes and BlueprintFactors)
- `services` breaks out static helpers, mostly for `_multiscale.py`. `matrices.py` is for `_transforms` (Rotation, Affine)
- `_translation_shifts.py` defines public helpers for diagnosing and calculating coordinate offsets introduced by scaling functions (related to `_multiscale.py`)
- `_scene.py` defines the Scene concept of OME-Zarr 0.6
- `_collections.py` serves as the primary entrypoint to discover OME-Zarr contents of arbitrary zarr attrs 
- `ome_zarr.py` lives outside of this hierarchy; it provides user-facing helpers for `Multiscale`

Dependencies should flow only downward in this list.
Lower-level modules must not import higher-level modules.

# Philosophy and formatting guidelines

- Strictly zero dependencies. "Fits in any Python environment" is a promise.
- Use the strongest practical typing.
- When strict typing conflicts with API ergonomics, favour ergonomics, but document the tradeoff.
- Immutable by default.
- Prioritise correct by default. Loosen when convenience strongly outweighs correctness.
- User should require no knowledge of the supported metadata formats (e.g. OME-Zarr). Metadata format details should remain implementation details.
- It should be impossible to produce invalid output metadata, and as hard as possible to produce (semantically) incorrect but valid output metadata.
- When reading metadata, be as permissive as possible. Error only if necessary information is missing or ambiguous.
- Avoid modelling concepts that cannot be represented in standardised metadata formats (i.e. OME-Zarr).
- Similarly, methods should not by design create states or constellations that cannot be serialised.

## Public APIs

clearscale aims to provide a hyper-intuitive API with optimal IDE support like tab-completion and greppability.

- Optimise public APIs for ergonomics, discoverability, and easy adoption in existing code bases. "Works with existing code."
- Internal implementation complexity is acceptable when it significantly simplifies the public API.
- If possible, public methods should accept both clearscale types and equivalent native Python types (e.g. `Union[Shape, Mapping[AxisKey, int]]`) (but always return clearscale types).

## Collection-like APIs

Many clearscale types intentionally behave like mappings or collections for intuitive use on the API consumer's side.

- Users should rarely or never need to iterate them directly, or construct them from iteration.
- If a common operation would require iteration by the caller, prefer adding a dedicated method.

## Class-internal method ordering

Class methods should be ordered from top to bottom like:
1. constructors (`__init__` last)
2. properties
3. base overrides
4. homotypic manipulators (return instances of their class)
5. converters, utilities, and methods that delegate to contained values
6. internal helpers

## Naming

- Use immutable-style descriptive naming ("with_axes") rather than mutable-style imperative ("reorder")
- Methods on base classes need to be generally named, so their names may be more technical ("with_default"). Subclasses should provide semantically informative and intuitive names for the API consumer that are more specific to the type of value the respective subclass handles ("with_singleton").
- Use familiar names from core python or popular packages when their functionality is equivalent, but not when it is only similar and could cause unexpected or confusing behaviour

## Commit hygiene

- Strictly separate refactoring changes, and functional changes
- Strictly separate moving large pieces of code from any modifications inside them
- Agents must not insert optional white space. Do not add line breaks or indents that are not required for valid Python syntax. Formatting is done by the `black` pre-commit hook.
