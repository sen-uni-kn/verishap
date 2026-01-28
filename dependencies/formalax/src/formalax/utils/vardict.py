# Copyright (c) 2025. The Formalax Authors.
# Licensed under the MIT license.
from collections.abc import Iterator, Mapping, Sequence

from jax.extend.core import Var
from jax.tree_util import register_pytree_node_class

from .zip import strict_zip


@register_pytree_node_class
class VarDict[T](Mapping[Var, T]):
    """An immutable dictionary-like container with Jaxpr variables as keys.

    Since ``jax.tree`` sorts dictionary keys but jaxpr ``Var``s are not
    comparable, it is not possible pass ``dict[Var, ...]`` instances to many
    jax functions.
    This class provides a workaround by using the id of the ``Var`` as the key.

    Args:
        d: The dictionary to wrap. This dictionary is copied.
    """

    def __init__(self, d: dict[Var, T]):
        self.__keys: tuple[Var, ...] = tuple(d.keys())
        self.__dict: dict[int, T] = {id(key): val for key, val in d.items()}

    def __getitem__(self, key: Var) -> T:
        return self.__dict[id(key)]

    def __contains__(self, key: object) -> bool:
        return id(key) in self.__dict

    def __len__(self) -> int:
        return len(self.__dict)

    def __iter__(self) -> Iterator[Var]:
        return iter(self.__keys)

    # --------------------------------------------------------------------------
    # MARK: PyTree Compatibility
    # --------------------------------------------------------------------------

    def tree_flatten(self):
        if len(self) == 0:
            return (), ()
        d = {key: self[key] for key in self.__keys}
        aux_data, children = strict_zip(
            *sorted(d.items(), key=lambda kv: (id(kv[0]), kv[1]))
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[Var, ...], children: Sequence[T]):
        return cls({key: val for key, val in strict_zip(aux_data, children)})
