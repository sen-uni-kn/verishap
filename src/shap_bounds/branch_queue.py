#  Copyright (c) 2025. David Boetius.
#  Licensed under the MIT license.
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real

__all__ = [
    "BranchQueue",
]

class BranchQueue[D: PyTree]:
    """A batched queue for storing branches in branch and bound.

    The ``BranchQueue`` stores a pytree of fixed structure.
    Each array in the pytree is equipped with a leading additional
    axis that corresponds to the branches in the branch queue.

    Args:
        root_entry: The root node of the branch and bound tree.
            This the first data values stored in this branch store.
            It determines the structure of the data stored in this branch store.
            Each array in the root entry must have a leading batch axis of size 1.

    Attributes:
        batch_size (int): The batch size of the branch queue.
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
            if data.ndim == 0 or data.shape[0] != 1:
                raise ValueError(
                    "Each array in the root entry must have a leading batch "
                    f"axis of size 1. Got shape {data.shape}."
                )

        self.leaf_shapes = tuple(data.shape[1:] for data in root_data)
        self.batch_size = batch_size

        self.__queue = []
        if batch_size > 1:
            # Holds inserted data until the data makes up a full batch.
            self.__buffer = root_data
        else:
            self.__queue.append(root_data)
            self.__buffer = self._empty_node()
        self.__size = 1

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
        branches, pytree = jax.tree.flatten(branches)
        # assert pytree == self.pytree
        for branch, shape in zip(branches, self.leaf_shapes, strict=True):
            assert branch.shape[1:] == shape

        self.__size += branches[0].shape[0]

        buf1, buf2 = self._split(self.__buffer, branches)
        buf2, buf3 = buf2[: self.batch_size], buf2[self.batch_size :]

        if len(buf1) < self.batch_size:
            self.__buffer = buf1
        elif len(buf2) < self.batch_size:
            self.__buffer = buf2
            self.__queue.append(buf1)
        elif len(buf3) < self.batch_size:
            self.__buffer = buf3
            self.__queue.append(buf1)
            self.__nodes.append(buf2)
        else:
            self.__buffer = self._empty_node()
            self.__queue.append(buf1)
            self.__queue.append(buf2)
            self.__queue.append(buf3)

    def pop(self, return_size: bool = False) -> tuple[D, int]:
        """Returns the first batch from the queue.

        If there are fewer than ``batch_size`` branches in the queue,
        ``pop`` returns fewer than ``batch_size`` branches.
        """
        if len(self.__queue) == 0:
            data = self.__buffer
            self.__buffer = self._empty_node()
        elif len(self.__queue) >= 1:
            data = self.__queue.pop(0)

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
        data = [
            jnp.concat([ld, rd]) for ld, rd in zip(left, right, strict=True)
        ]
        left = [d[:self.batch_size] for d in data]
        right = [d[self.batch_size:] for d in data]
        return left, right

    def _empty_node(self) -> list[Array]:
        return [jnp.empty((0, *shape)) for shape in self.leaf_shapes]
