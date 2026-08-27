import pytest

from clearscale._spatial_relations import PermutationTo, ProjectionTo
from clearscale._transforms import (
    TransformSequence,
    relation_chain_target_axes,
    relations_to_transform,
)


def test_relations_to_transform_rejects_empty_list():
    with pytest.raises(ValueError, match="at least one"):
        relations_to_transform([], source_axes=("y", "x"))


def test_relations_to_transform_returns_bare_transform_for_single_relation():
    result = relations_to_transform([ProjectionTo(("t", "y", "x"))], source_axes=("y", "x"))
    assert not isinstance(result, TransformSequence)


def test_relations_to_transform_wraps_multiple_relations_in_sequence():
    relations = [ProjectionTo(("t", "z", "y", "x")), PermutationTo(("t", "y", "x", "z"))]
    result = relations_to_transform(relations, source_axes=("z", "y", "x"))
    assert isinstance(result, TransformSequence)
    assert len(result.transforms) == 2


def test_relation_chain_target_axes_threads_each_hop():
    relations = [ProjectionTo(("t", "z", "y", "x")), PermutationTo(("t", "y", "x", "z"))]
    assert relation_chain_target_axes(relations, source_axes=("z", "y", "x")) == ("t", "y", "x", "z")
