#  Copyright (c) 2025. David Boetius.
#  Licensed under the MIT license.
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree, Real

__all__ = [
    "PriorityBranchStore",
]


class _HeapNode:
    """A node in the heap."""

    __slots__ = ("priority", "data")

    def __init__(self, priority: Real[Array, " b"], data: Sequence[Array]):
        self.priority = jnp.atleast_1d(priority)
        self.data = [jnp.atleast_1d(d) for d in data]

    @property
    def highest_priority(self) -> Real[Array, ""] | None:
        if len(self) == 0:
            return None
        return self.priority[0]

    @property
    def lowest_priority(self) -> Real[Array, ""] | None:
        if len(self) == 0:
            return None
        return self.priority[-1]

    def __len__(self) -> int:
        return self.priority.shape[0]

    def __getitem__(self, idx: int) -> Array:
        return _HeapNode(self.priority[idx], [d[idx] for d in self.data])


class PriorityBranchStore[D: PyTree]:
    """A batched priority queue for storing branches in branch and bound.

    This is a max-priority queue that returns the elements with the highest
    priority in ``extract_top``.

    The ``PriorityBranchStore`` stores a pytree of fixed structure.
    Each array in the pytree is equipped with a leading additional
    axis that corresponds to the branches in the branch store.

    This implementation is based on [Chen et al., 2021] with some adaptations.
    This implementation is not thread-safe.

    Args:
        root_entry: The root node of the branch and bound tree.
            This the first data values stored in this branch store.
            It determines the structure of the data stored in this branch store.
            Each array in the root entry must have a leading batch axis of size 1.

    Attributes:
        batch_size (int): The batch size of the priority queue.
        pytree (PyTreeDef): The structure of the data stored in this branch store.
        num_leaves (int): The number of arrays stored in this branch store.
        leaf_shapes (tuple[tuple[int, ...], ...]): The shapes of the arrays in the
            store, without the leading batch axis.

    [Chen et al., 2021] Yan-Hao Chen, Fei Hua, Yuwei Jin, Eddy Z. Zhang:
        BGPQ: A Heap-Based Priority Queue Design for GPUs. ICPP 2021: 9:1-9:10
    """

    def __init__(self, priority: Real[Array, " 1"], root_entry: D, batch_size: int):
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

        # We index the __nodes list like the array in an array-based heap.
        self.__nodes = []
        if batch_size > 1:
            # Holds inserted data until the data makes up a full batch.
            self.__buffer = _HeapNode(priority, root_data)
        else:
            self.__nodes.append(_HeapNode(priority, root_data))
            self.__buffer = self._empty_node()
        self.__size = 1

    def __len__(self) -> int:
        assert self.__size >= 0
        return self.__size

    def highest_priority(self) -> Real[Array, ""] | None:
        """Returns the highest priority value in the queue.

        Returns:
            The highest priority value in the queue.
            Returns ``None`` if the queue is empty.
        """
        if self.__size == 0:
            return None
        if len(self.__nodes) == 0:
            return self.__buffer.priority[0]
        else:
            return self.__nodes[0].priority[0]

    def insert(self, priority: Real[Array, " d"], branches: D):
        """Inserts a batch of data into the priority queue.

        Insert accepts up to ``2 * batch_size`` branches at a time.

        Args:
            priority: The priority of the data.
            branches: The data to insert.
                Each array in the pytree needs to have a leading batch axis.
        """
        assert priority.shape[0] <= 2 * self.batch_size
        branches, pytree = jax.tree.flatten(branches)
        assert pytree == self.pytree
        for branch, shape in zip(branches, self.leaf_shapes, strict=True):
            assert branch.shape[1:] == shape

        self.__size += priority.shape[0]

        buf1, buf2 = self._split_sort(self.__buffer, _HeapNode(priority, branches))
        buf2, buf3 = buf2[: self.batch_size], buf2[self.batch_size :]

        heapify = lambda _: None  # noqa: E731
        if len(self.__nodes) > 0:
            # Ensure the top priority nodes are in root
            root, buf1 = self._split_sort(self.__nodes[0], buf1)
            self.__nodes[0] = root
            heapify = self._heapify_up

        if len(buf1) < self.batch_size:
            self.__buffer = buf1
        elif len(buf2) < self.batch_size:
            self.__buffer = buf2
            self.__nodes.append(buf1)
            heapify(len(self.__nodes) - 1)
        elif len(buf3) < self.batch_size:
            self.__buffer = buf3
            self.__nodes.append(buf1)
            heapify(len(self.__nodes) - 1)
            self.__nodes.append(buf2)
            heapify(len(self.__nodes) - 1)
        else:
            self.__buffer = self._empty_node()
            self.__nodes.append(buf1)
            heapify(len(self.__nodes) - 1)
            self.__nodes.append(buf2)
            heapify(len(self.__nodes) - 1)
            self.__nodes.append(buf3)
            heapify(len(self.__nodes) - 1)

    def extract_max(self, return_size: bool = False) -> tuple[D, int]:
        """Returns the data with the highest priority as a batch of size ``batch_size``.

        If there are fewer than ``batch_size`` branches in the queue,
        ``exact_max`` returns fewer than ``batch_size`` branches.
        """
        if len(self.__nodes) == 0:
            data = self.__buffer.data
            self.__buffer = self._empty_node()
        elif len(self.__nodes) == 1:
            data = self.__nodes.pop().data
        else:
            data = self.__nodes[0].data
            tmp_root = self.__nodes.pop()
            self.__nodes[0], self.__buffer = self._split_sort(tmp_root, self.__buffer)
            self._heapify_down(0)

        removed_size = data[0].shape[0]
        self.__size -= removed_size
        out = jax.tree.unflatten(self.pytree, data)
        if return_size:
            return out, removed_size
        else:
            return out

    def _split_sort(
        self, left: _HeapNode, right: _HeapNode
    ) -> tuple[_HeapNode, _HeapNode]:
        """Sorts the data in two nodes and returns the sorted data as two new nodes.

        Args:
            left: The left node to sort.
            right: The right node to sort.

        Returns:
            A tuple of two new nodes with the sorted data.
            The first node contains the data with the higher priority.
            The first node is filled up to the batch size.
            The second node contains any remaining data.
        """
        prios = jnp.concat([left.priority, right.priority])
        sorted_idx = jnp.argsort(prios, descending=True)
        top_idx = sorted_idx[: self.batch_size]
        bottom_idx = sorted_idx[self.batch_size :]

        data = [
            jnp.concat([ld, rd]) for ld, rd in zip(left.data, right.data, strict=True)
        ]
        top = _HeapNode(prios[top_idx], [d[top_idx] for d in data])
        bottom = _HeapNode(prios[bottom_idx], [d[bottom_idx] for d in data])
        return top, bottom

    def _heapify_up(self, i: int):
        if i == 0:
            return

        parent_i = (i - 1) // 2
        parent, child = self.__nodes[parent_i], self.__nodes[i]
        self.__nodes[parent_i], self.__nodes[i] = self._split_sort(parent, child)
        self._heapify_up(parent_i)

    def _heapify_down(self, i: int):
        li = 2 * i + 1
        ri = 2 * i + 2
        if li >= len(self.__nodes):
            return
        elif ri >= len(self.__nodes):
            this, left = self._split_sort(self.__nodes[i], self.__nodes[li])
            self.__nodes[i] = this
            self.__nodes[li] = left
            return

        this, left, right = self.__nodes[i], self.__nodes[li], self.__nodes[ri]
        # Sort the children, so that the heap property is maintained on one
        # side of the tree (the side with the smaller lowest priority).
        xi, yi = (li, ri) if left.lowest_priority < right.lowest_priority else (ri, li)
        self.__nodes[yi], self.__nodes[xi] = self._split_sort(left, right)
        self.__nodes[i], self.__nodes[yi] = self._split_sort(this, self.__nodes[yi])
        self._heapify_down(yi)

    def _empty_node(self) -> _HeapNode:
        return _HeapNode(
            jnp.empty((0,)), [jnp.empty((0, *shape)) for shape in self.leaf_shapes]
        )
