#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from collections.abc import Collection


def intersection[T](*args: Collection[T]) -> frozenset[T]:
    """Computes the intersection of several collections."""
    assert len(args) > 0
    return frozenset(args[0]).intersection(*args[1:])


def union[T](*args: Collection[T]) -> frozenset[T]:
    """Computes the union of several collections."""
    assert len(args) > 0
    return frozenset(args[0]).union(*args[1:])
