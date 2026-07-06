# Repo structure

`src/clearscale/`, in order from lowest to highest level:

- `_axis_values.py` defines primitives (Shape, Unit, PixelSize, ...)
- `_transforms.py` defines the transformation graph primitives for OME-Zarr 0.6 concepts
- `_multiscale.py` defines the primary user-value objects (Scale, Multiscale, BlueprintShapes and BlueprintFactors)
- `services` currently only breaks out helpers for `_multiscale.py`.
- `_translation_shifts.py` defines public helpers for diagnosing and calculating coordinate offsets introduced by scaling functions (related to `_multiscale.py`)
- `_scene.py` defines the Scene concept of OME-Zarr 0.6
- `ome_zarr.py` lives outside of this hierarchy; it currently provides some user-facing helpers for `Multiscale`

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

## Public APIs

- Optimise public APIs for ergonomics, discoverability, and easy adoption in existing code bases. "Works with existing code."
- Internal implementation complexity is acceptable when it significantly simplifies the public API.
- If possible, public methods should accept both clearscale types and equivalent native Python types (e.g. `Union[Shape, Mapping[ScaleKey, int]]`) (but always return clearscale types).

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

- Use immutable-style adverbial and adjectival naming ("with_axes") rather than verbal/action-oriented ("reorder")
- Methods on base classes need to be generally named, so their names may be more technical ("with_default"). Subclasses should provide semantically informative and intuitive names for the API consumer that are more specific to the type of value the respective subclass handles ("with_singleton").
- Use familiar names from core python or popular packages when their functionality is equivalent, but not when it is only similar and could cause unexpected or confusing behaviour

## Commit hygiene

- Strictly separate refactoring changes, and functional changes
- Strictly separate moving large pieces of code from any modifications inside them
