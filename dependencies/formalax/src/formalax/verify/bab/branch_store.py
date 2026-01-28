#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
"""Branch store for branch and bound."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import copy
from enum import Enum, auto, unique
from typing import Protocol, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree

from ...utils.zip import strict_zip

__all__ = [
    "BranchStore",
    "ArrayRef",
    "BranchSelection",
    "SimpleBranchStore",
    "MaskBranchSelection",
    "SimpleArrayRef",
]


class BranchSelection(Protocol):
    """A selection of branches in a ``BranchStore``."""

    def __and__[S](self: S, other: S) -> S: ...

    def __or__[S](self: S, other: S) -> S: ...

    @classmethod
    def all(cls, store: "BranchStore") -> "BranchSelection":
        """Returns the selection of all branches."""
        ...

    @classmethod
    def empty(cls, store: "BranchStore") -> "BranchSelection":
        """Returns an empty selection of branches."""
        ...

    def __len__(self) -> int:
        """The number of branches in this selection."""
        ...


class ArrayRef[Store: "BranchStore", Selection: BranchSelection](Protocol):
    """Provides an interface to an array stored in a ``BranchStore``.

    Args:
        store: The ``BranchStore`` that stores the array referenced by this ``ArrayRef``.
        array_index: The index of the array in the ``BranchStore``.
        selection: The selection of branches to reference.
    """

    def __init__(self, store: Store, array_index: int, selection: Selection): ...

    def __len__(self) -> int:
        """The size of the batch axis of the array referenced by this ``ArrayRef``."""
        ...

    def min(self) -> Array:
        """Computes the element-wise minimum across the batch axis."""
        ...

    def max(self) -> Array:
        """Computes the element-wise maximum across the batch axis."""
        ...

    def argmax_k(self, k: int, approx: bool = True) -> Selection:
        """Returns the selection of the branches with the ``k`` largest elements.

        This operation is only permitted for arrays storing a scalar for each branch.
        The array referenced by this ``ArrayRef`` must either be a vector, or a matrix
        with one column.

        Args:
            k: The number of largest elements to return.
            approx: Whether it is acceptable to use an approximation algorithm.
        Returns:
            A ``BranchSelection`` of the ``k`` branches with the largest elements.
        """
        ...

    def argmin_k(self, k: int, approx: bool = True) -> Selection:
        """Returns the selection of the branches with the ``k`` smallest elements.

        This operation is only permitted for arrays storing a scalar for each branch.
        The array referenced by this ``ArrayRef`` must either be a vector, or a matrix
        with one column.

        Args:
            k: The number of smallest elements to return.
            approx: Whether it is acceptable to use an approximation algorithm.
        Returns:
            A ``BranchSelection`` of the ``k`` branches with the smallest elements.
        """
        ...

    def __eq__[Self](self: Self, other: Self | Array) -> Selection: ...

    def __ne__[Self](self: Self, other: Self | Array) -> Selection: ...

    def __gt__[Self](self: Self, other: Self | Array) -> Selection: ...

    def __ge__[Self](self: Self, other: Self | Array) -> Selection: ...

    def __lt__[Self](self: Self, other: Self | Array) -> Selection: ...

    def __le__[Self](self: Self, other: Self | Array) -> Selection: ...

    def __add__[Self](
        self: Self, other: Self | Array
    ) -> "ArrayRef[Store, Selection]": ...

    def __sub__[Self](
        self: Self, other: Self | Array
    ) -> "ArrayRef[Store, Selection]": ...

    def __mul__[Self](
        self: Self, other: Self | Array
    ) -> "ArrayRef[Store, Selection]": ...

    def __truediv__[Self](
        self: Self, other: Self | Array
    ) -> "ArrayRef[Store, Selection]": ...

    __hash__ = None


class BranchStore[D: PyTree, S: BranchSelection, R: ArrayRef](ABC):
    """
    A data structure for storing branches in branch and bound.

    The ``BranchStore`` stores a pytree of fixed structure.
    Each array in the pytree is equipped with a leading additional
    axis that corresponds to the branches in the branch store.

    Selections
    --------------
    Subsets of the branches stored in a ``BranchStore`` are addressed
    using a ``BranchSelection``.
    All ``BranchSelections`` are *invalidated* when branches added or removed
    from the branch store.

    In particular, the ``add``, ``remove``, and ``pop`` methods
    invalidate existing ``BranchSelections``.
    Addressing branches using invalidated ``BranchSelections`` will result in
    undefined behavior.

    Implementation
    --------------

    The ``__init__`` implementation of ``BranchStore`` initializes
    the ``self.pytree`` field using the ``root_entry`` argument.
    It calls ``_init_array_storage`` to initialize the storage of this
    branch store and stores the root entry in the branch store
    using ``_add_arrays``.

    Args:
        root_entry: The root node of the branch and bound tree.
            This the first data values stored in this branch store.
            It determines the structure of the data stored in this branch store.
            Each array in the root entry must have a leading batch axis of size 1.
        selection_cls: The class of the ``BranchSelection`` to use to select branches.
        array_ref_cls: The class of the ``ArrayRef`` to use to access the data stored
            in this branch store.

    Attributes:
        pytree (PyTreeDef): The structure of the data stored in this branch store.
        num_leaves (int): The number of arrays stored in this branch store.
        leaf_shapes (tuple[tuple[int, ...], ...]): The shapes of the arrays in the store, without the leading batch axis.
        data (PyTree[ArrayRef, "..."]): An accessor to the branch data stored in the branch store.
            This only provides ``ArrayRef`` instances, which only allow limited access to the
            arrays stored by the branch store.
            Use the ``pop`` method to retrieve branch data as a pytree of arrays.
    """

    def __init__[Self](
        self: Self,
        root_entry: D,
        selection_cls: type[S],
        array_ref_cls: type[R],
    ):
        self._selection_cls = selection_cls
        self._array_ref_cls = array_ref_cls

        root_data, pytree = jax.tree.flatten(root_entry)
        self.num_leaves = len(root_data)
        self.pytree = pytree

        for data in root_data:
            if data.ndim == 0 or data.shape[0] != 1:
                raise ValueError(
                    f"Each array in the root entry must have a leading batch axis of size 1. Got shape {data.shape}."
                )

        self.leaf_shapes = tuple(data.shape[1:] for data in root_data)
        self._init_array_storage()
        self._add(*root_data)

        all_ = selection_cls.all(self)
        array_refs = [array_ref_cls(self, i, all_) for i in range(self.num_leaves)]
        self.data = jax.tree.unflatten(pytree, array_refs)

    @abstractmethod
    def _init_array_storage(self):
        """Initializes the storage of this branch store.

        The ``_init_array_storage`` method is called by the ``__init__``
        implementation of ``BranchStore`` to initialize the storage of this
        branch store.

        Use the ``self.pytree`` and ``self.leaf_shapes`` fields to initialize
        the array storage.
        """
        raise NotImplementedError()

    def add(self, branches: D) -> None:
        """Adds a batch of branches to this branch store.

        Args:
            branches: The branches to add.
                Each array in the pytree needs to have a leading batch axis.
        """
        data, pytree = jax.tree.flatten(branches)
        assert pytree == self.pytree
        for array, shape in strict_zip(data, self.leaf_shapes):
            assert array.shape[1:] == shape
        self._add(*data)

    @abstractmethod
    def _add(self, *arrays: Array) -> None:
        """Adds branches to the array storage.

        Args:
            arrays: The data of the branches to add.
                The arrays are in the same order as the ``self.leaf_shapes`` tuple.
        """
        raise NotImplementedError()

    @abstractmethod
    def remove(self, branches: S) -> None:
        """Remove branches from the branch store."""
        raise NotImplementedError()

    def pop(self, branches: S) -> D:
        """Remove branches from the branch store and return their data.

        Returns an in-memory pytree of arrays

        Args:
            branches: The selection of branches to pop.
        Returns:
            A pytree of arrays.
        """
        arrays = self._pop(branches)
        return jax.tree.unflatten(self.pytree, arrays)

    @abstractmethod
    def _pop(self, branches: S) -> Iterable[Array, ...]:
        """Removes and returns the array entries of the ``branches`` selection.

        Args:
            branches: The selection of branches to pop.
        Returns:
            An iterable of arrays.
        """
        raise NotImplementedError()

    @property
    def all(self) -> S:
        """A selection referring to all branches in the branch store."""
        return self._selection_cls.all(self)

    @property
    def empty_selection(self) -> S:
        """Returns an empty selection of branches."""
        return self._selection_cls.empty(self)

    def subset(self, branches: S) -> PyTree[ArrayRef, "..."]:
        """Returns a subset of the branches in the branch store."""
        array_refs = [
            self._array_ref_cls(self, i, branches) for i in range(self.num_leaves)
        ]
        return jax.tree.unflatten(self.pytree, array_refs)

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of branches in the branch store."""
        raise NotImplementedError()


# ==================================================================================
# MARK: Concatenating In-Memory Branch Store
# ==================================================================================


class MaskBranchSelection(BranchSelection):
    """A selection of branches in a ``SimpleBranchStore``."""

    @unique
    class SpecialMask(Enum):
        all = auto()
        """All values in a data store."""
        none = auto()
        """No values in a data store. The empty selection."""

        def __and__(self, other):
            if self is SpecialMask.none:
                return SpecialMask.none
            elif self is SpecialMask.all:
                return other
            else:
                raise NotImplementedError()

        def __or__(self, other):
            if self is SpecialMask.none:
                return other
            elif self is SpecialMask.all:
                return SpecialMask.all
            else:
                raise NotImplementedError()

    def __init__(
        self,
        store: BranchStore,
        mask: Union[Bool[Array, "b"], "MaskBranchSelection.SpecialMask"],
    ):
        self.store = store
        self.mask = mask

    def __and__[S](self: S, other: S) -> S:
        if not isinstance(other, MaskBranchSelection) or self.store is not other.store:
            raise ValueError(
                "Cannot intersect MaskBranchSelections from different stores."
            )
        return MaskBranchSelection(self.store, self.mask & other.mask)

    def __or__[S](self: S, other: S) -> S:
        if not isinstance(other, MaskBranchSelection) or self.store is not other.store:
            raise ValueError("Cannot union MaskBranchSelections from different stores.")
        return MaskBranchSelection(self.store, self.mask | other.mask)

    @classmethod
    def all(cls, store: "BranchStore") -> "MaskBranchSelection":
        return MaskBranchSelection(store, MaskBranchSelection.SpecialMask.all)

    @classmethod
    def empty(cls, store: "BranchStore") -> "MaskBranchSelection":
        return MaskBranchSelection(store, MaskBranchSelection.SpecialMask.none)

    def __len__(self) -> int:
        match self.mask:
            case MaskBranchSelection.SpecialMask.all:
                return len(self.store)
            case MaskBranchSelection.SpecialMask.none:
                return 0
            case _:
                return jnp.sum(self.mask)


class SimpleArrayRef[Store](ArrayRef[Store, MaskBranchSelection]):
    """An array reference in a ``SimpleBranchStore``."""

    def __init__(
        self,
        store: "SimpleBranchStore",
        array_index: int,
        selection: MaskBranchSelection,
    ):
        self.store = store
        self.array_index = array_index
        self.selection = selection
        self.__data = None

    @property
    def data(self) -> Array:
        if self.__data is not None:
            return self.__data

        match self.selection.mask:
            case MaskBranchSelection.SpecialMask.all:
                return self.store._arrays[self.array_index]
            case MaskBranchSelection.SpecialMask.none:
                return jnp.empty((0, *self.store.leaf_shapes[self.array_index]))
            case _:
                return self.store._arrays[self.array_index][self.selection.mask]

    def __len__(self) -> int:
        return len(self.selection)

    def min(self) -> Array:
        return self.data.min(axis=0)

    def max(self) -> Array:
        return self.data.max(axis=0)

    def argmax_k(self, k: int, approx: bool = True) -> MaskBranchSelection:
        if self.data.ndim > 2 or (self.data.ndim == 2 and self.data.shape[1] != 1):
            raise ValueError(
                f"argmax_k is unsupported for data with shape {self.data.shape}."
            )

        max_k = jax.lax.approx_max_k if approx else jax.lax.top_k
        indices = max_k(self.data, k=k)[1]
        indices = indices[:k]  # approx_max_k may return more than k indices

        mask = jnp.zeros(len(self.store), dtype=bool).at[indices].set(True)
        return MaskBranchSelection(self.store, mask)

    def argmin_k(self, k: int, approx: bool = True) -> MaskBranchSelection:
        if self.data.ndim > 2 or (self.data.ndim == 2 and self.data.shape[1] != 1):
            raise ValueError(
                f"argmin_k is unsupported for data with shape {self.data.shape}."
            )

        if approx:
            indices = jax.lax.approx_min_k(self.data, k=k)[1]
            indices = indices[:k]  # approx_min_k may return more than k indices
        else:
            indices = jax.lax.top_k(-self.data, k=k)[1]

        mask = jnp.zeros(len(self.store), dtype=bool).at[indices].set(True)
        return MaskBranchSelection(self.store, mask)

    def __eq__[Self](self: Self, other: Self | Array) -> MaskBranchSelection:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        return MaskBranchSelection(self.store, self.data == other)

    def __ne__[Self](self: Self, other: Self | Array) -> MaskBranchSelection:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        return MaskBranchSelection(self.store, self.data != other)

    def __gt__[Self](self: Self, other: Self | Array) -> MaskBranchSelection:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        return MaskBranchSelection(self.store, self.data > other)

    def __ge__[Self](self: Self, other: Self | Array) -> MaskBranchSelection:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        return MaskBranchSelection(self.store, self.data >= other)

    def __lt__[Self](self: Self, other: Self | Array) -> MaskBranchSelection:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        return MaskBranchSelection(self.store, self.data < other)

    def __le__[Self](self: Self, other: Self | Array) -> MaskBranchSelection:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        return MaskBranchSelection(self.store, self.data <= other)

    def __add__[Self](self: Self, other: Self | Array) -> Self:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        new = copy(self)
        new.__data = self.data + other
        return new

    def __sub__[Self](self: Self, other: Self | Array) -> Self:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        new = copy(self)
        new.__data = self.data - other
        return new

    def __mul__[Self](self: Self, other: Self | Array) -> Self:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        new = copy(self)
        new.__data = self.data * other
        return new

    def __truediv__[Self](self: Self, other: Self | Array) -> Self:
        if isinstance(other, SimpleArrayRef):
            other = other.data
        new = copy(self)
        new.__data = self.data / other
        return new

    __hash__ = None


class SimpleBranchStore[D: PyTree](BranchStore[D, MaskBranchSelection, SimpleArrayRef]):
    """A simple branch store implementation.

    This branch store stores all branches in memory.
    The data of new branches is concatenated to the existing data.
    """

    def __init__(self, root_entry: D):
        self._arrays = []
        super().__init__(root_entry, MaskBranchSelection, SimpleArrayRef)

    def _init_array_storage(self):
        self._clear()

    def _clear(self):
        self._arrays = [jnp.empty((0, *shape)) for shape in self.leaf_shapes]

    def _add(self, *arrays: Array) -> None:
        self._arrays = [
            jnp.concatenate([old, new], axis=0)
            for old, new in strict_zip(self._arrays, arrays)
        ]

    def remove(self, branches: MaskBranchSelection):
        if branches.mask is MaskBranchSelection.SpecialMask.all:
            self._clear()
        elif branches.mask is not MaskBranchSelection.SpecialMask.none:
            self._arrays = [array[~branches.mask] for array in self._arrays]

    def _pop(self, branches: BranchSelection) -> Iterable[Array, ...]:
        if branches.mask is MaskBranchSelection.SpecialMask.all:
            arrays = self._arrays
        elif branches.mask is MaskBranchSelection.SpecialMask.none:
            arrays = [jnp.empty(0, *shape) for shape in self.leaf_shapes]
        else:
            arrays = [array[branches.mask] for array in self._arrays]
        self.remove(branches)
        return arrays

    def __len__(self) -> int:
        return self._arrays[0].shape[0]
