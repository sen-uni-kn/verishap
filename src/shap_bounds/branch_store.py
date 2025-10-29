#  Copyright (c) 2025. David Boetius.
#  Licensed under the MIT license.
import jax
import jax.numpy as jnp
from formalax.utils.zip import strict_zip
from jaxtyping import Array, Bool, PyTree

__all__ = [
    "BranchStore",
]


class BranchStore[D: PyTree]:
    """
    A data structure for storing branches in branch and bound.

    The ``BranchStore`` stores a pytree of fixed structure.
    Each array in the pytree is equipped with a leading additional
    axis that corresponds to the branches in the branch store.

    Args:
        root_entry: The root node of the branch and bound tree.
            This the first data values stored in this branch store.
            It determines the structure of the data stored in this branch store.
            Each array in the root entry must have a leading batch axis of size 1.

    Attributes:
        pytree (PyTreeDef): The structure of the data stored in this branch store.
        num_leaves (int): The number of arrays stored in this branch store.
        leaf_shapes (tuple[tuple[int, ...], ...]): The shapes of the arrays in the store, without the leading batch axis.
        data (PyTree[Array, "..."]): The branch data in the branch store.
    """

    def __init__[Self](
        self: Self,
        root_entry: D,
    ):
        root_data, pytree = jax.tree.flatten(root_entry)
        self.num_leaves = len(root_data)
        self.pytree = pytree

        for data in root_data:
            if data.ndim == 0 or data.shape[0] != 1:
                raise ValueError(
                    f"Each array in the root entry must have a leading batch axis of size 1. Got shape {data.shape}."
                )

        self.leaf_shapes = tuple(data.shape[1:] for data in root_data)
        self.arrays = tuple(root_data)

    @property
    def data(self) -> D:
        return jax.tree.unflatten(self.pytree, self.arrays)

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

        self.arrays = [
            jnp.concatenate([old, new], axis=0)
            for old, new in strict_zip(self.arrays, data)
        ]

    def remove(self, mask: Bool[Array, " b"]) -> None:
        """Remove branches from the branch store."""
        self.arrays = [array[~mask] for array in self.arrays]

    def pop(self, mask: Bool[Array, " b"]) -> D:
        """Remove branches from the branch store and return their data.

        Returns an in-memory pytree of arrays

        Args:
            branches: The selection of branches to pop.
        Returns:
            A pytree of arrays.
        """
        arrays = [array[mask] for array in self.arrays]
        self.remove(mask)
        return jax.tree.unflatten(self.pytree, arrays)

    def __len__(self) -> int:
        """Returns the number of branches in the branch store."""
        return self.arrays[0].shape[0]
