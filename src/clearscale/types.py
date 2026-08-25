from typing import Hashable, Protocol, Iterator, TypeVar

AxisKey = Hashable
AxisKeyT_co = TypeVar("AxisKeyT_co", bound=AxisKey, covariant=True)


class _Ordered(Protocol[AxisKeyT_co]):
    """Defined order (unlike Iterable, Collection), but not necessarily indexable (unlike Sequence).
    To match e.g. strings, but also odict_keys and of course _AxisMapping"""

    def __iter__(self) -> Iterator[AxisKeyT_co]: ...
    def __len__(self) -> int: ...


OrderedAxes = _Ordered[AxisKey]
