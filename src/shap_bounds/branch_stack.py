#  Copyright (c) 2025. David Boetius.
#  Licensed under the MIT license.

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real

__all__ = [
    "BranchStack",
]


class BranchStack[D: PyTree]:
    """A batched stack for storing branches in branch and bound.

    The ``BranchStack`` stores a pytree of fixed structure.
    Each array in the pytree is equipped with a leading additional
    axis that corresponds to the branches in the branch stack.

    Args:
        root_entry: The root node of the branch and bound tree.
            This the first data values stored in this branch store.
            It determines the structure of the data stored in this branch store.
            Each array in the root entry must have a leading batch axis of size 1.

    Attributes:
        batch_size (int): The batch size of the branch stack.
        pytree (PyTreeDef): The structure of the data stored in this branch store.
        num_leaves (int): The number of arrays stored in this branch store.
        leaf_shapes (tuple[tuple[int, ...], ...]): The shapes of the arrays in the
            store, without the leading batch axis.
    """

    def __init__(self, root_entry: D, batch_size: int):
        root_data, pytree = jax.tree.flatten(root_entry)
        self.num_leaves = len(root_data)
        self.pytree = pytree

        for data in root_data:
            if data.ndim == 0:
                raise ValueError(
                    "Each array in the root entry must have a leading batch axis. "
                    f"Got shape {data.shape}."
                )

        self.leaf_shapes = tuple(data.shape[1:] for data in root_data)
        self.batch_size = batch_size

        self.__stack = [root_data]
        self.__size = root_data[0].shape[0]

    def __len__(self) -> int:
        assert self.__size >= 0
        return self.__size

    def insert(self, branches: D):
        """Inserts a batch of data into the priority queue.

        Insert accepts up to ``2 * batch_size`` branches at a time.

        Args:
            branches: The data to insert.
                Each array in the pytree needs to have a leading batch axis.
        """
        branches, _ = jax.tree.flatten(branches)
        for branch, shape in zip(branches, self.leaf_shapes, strict=True):
            assert branch.shape[1:] == shape

        self.__size += branches[0].shape[0]
        self.__stack.append(branches)

    def pop(self, return_size: bool = False) -> tuple[D, int] | D:
        """Returns the last batch from the stack.

        If there are fewer than ``batch_size`` branches in the stack,
        ``pop`` returns fewer than ``batch_size`` branches.
        """
        data = self.__stack.pop(-1)
        while data[0].shape[0] < self.batch_size and len(self.__stack) > 0:
            next_data = self.__stack.pop(-1)
            data, remaining = self._split(data, next_data)
            if remaining[0].shape[0] > 0:
                self.__stack.append(remaining)
        if data[0].shape[0] > self.batch_size:
            data = [d[: self.batch_size] for d in data]
            remaining = [d[self.batch_size :] for d in data]
            self.__stack.append(remaining)

        removed_size = data[0].shape[0]
        self.__size -= removed_size
        out = jax.tree.unflatten(self.pytree, data)
        if return_size:
            return out, removed_size
        else:
            return out

    def _split(
        self, left: list[Array], right: list[Array]
    ) -> tuple[list[Array], list[Array]]:
        """Splits data into two batches.

        Args:
            left: The left batch of data.
            right: The right batch of data.

        Returns:
            A tuple of two new batches of data.
            The first batch is filled up to the batch size.
            The second batch contains any remaining data.
        """
        data = [jnp.concat([ld, rd]) for ld, rd in zip(left, right, strict=True)]
        left = [d[: self.batch_size] for d in data]
        right = [d[self.batch_size :] for d in data]
        return left, right
