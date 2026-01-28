#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import itertools
from collections import defaultdict
from collections.abc import Mapping, Set
from types import EllipsisType
from typing import Collection, Iterable, Iterator, Protocol, Sequence
from warnings import warn

import jax
from frozendict import frozendict
from jax import lax
from jax.core import Atom
from jax.extend.core import ClosedJaxpr, Literal, Primitive, Var, primitives

from formalax.utils.jaxpr import HashableLiteral
from formalax.utils.sets import intersection, union
from formalax.utils.zip import strict_zip

__all__ = (
    "BatchAxisMapping",
    "BatchAxisGraph",
    "infer_batch_axes",
    "BatchAxisMappingRule",
    "batch_axis_mapping_rules",
    "identity_batch_axis_mapping_rule",
)


class BatchAxisMapping:
    """Describes the mapping of batch axes between two variables in a Jaxpr.
    Also tracks broadcasting relationships between batch axes.

    Mappings between batch axes can either be one to one or "``None`` to one"
    if an output batch axis is created by broadcasting.
    ``BatchAxisMapping`` also tracks the batch axes that are created by broadcasting
    a size one axis to a larger size.
    In this case, the size one axis is not treated as a batch axis, but the larger
    axis can be.

    Args:
        mapping: A sequence of pairs of batch axes.
            Each pair represents a mapping from an input batch axis to an output batch axis.
            Use elements like ``(None, 0)`` for batch axes created by broadcasting.
            Each element of ``mapping`` can may also contain a third boolean
            entry that indicates whether the output batch axis originates from
            a size one input axis that is broadcast.
            By default, the third entry is ``False`` (no broadcasting).
    """

    __slots__ = ("__mapping", "__inverse", "__broadcast", "__in_axes_set")

    def __init__(self, *mapping: tuple[int | None, int] | tuple[int | None, int, bool]):
        proper_in_axes = [in_ax for in_ax, *_ in mapping if in_ax is not None]
        if len(set(proper_in_axes)) != len(proper_in_axes):
            raise ValueError(f"Duplicate input batch axes. Got {proper_in_axes}.")
        out_axes = [out_ax for _, out_ax, *_ in mapping]
        if len(set(out_axes)) != len(out_axes):
            raise ValueError(f"Duplicate output batch axes. Got {out_axes}.")

        mapping = [
            (in_ax, out_ax, (in_ax is None) if len(broadcast) == 0 else broadcast[0])
            for in_ax, out_ax, *broadcast in mapping
        ]

        self.__mapping = frozendict(
            {in_ax: out_ax for in_ax, out_ax, broadcast in mapping if in_ax is not None}
        )
        self.__inverse = frozendict({out_ax: in_ax for in_ax, out_ax, _ in mapping})
        self.__broadcast = frozenset(
            out_ax for _, out_ax, broadcast in mapping if broadcast
        )
        self.__in_axes_set = frozenset(proper_in_axes)

    @classmethod
    def identity(cls, axes: Iterable[int]) -> "BatchAxisMapping":
        """Returns a ``BatchAxisMapping`` that maps ``axes`` to themselves."""
        return cls(*((ax, ax) for ax in axes))

    @classmethod
    def empty(cls) -> "BatchAxisMapping":
        """Returns an empty ``BatchAxisMapping``."""
        return cls()

    @property
    def in_axes(self) -> tuple[int, ...]:
        """The input batch axes of this ``BatchAxisMapping`` as a sorted tuple."""
        return tuple(sorted(self.in_axes_set))

    @property
    def in_axes_set(self) -> Set[int]:
        """The input batch axes of this ``BatchAxisMapping`` as a set.

        Accessing this property is cheaper than ``in_axes``.
        """
        return self.__in_axes_set

    @property
    def out_axes(self) -> tuple[int, ...]:
        """The output batch axes of this ``BatchAxisMapping`` as a sorted tuple."""
        return tuple(sorted(self.out_axes_set))

    @property
    def out_axes_set(self) -> Set[int]:
        """The output batch axes of this ``BatchAxisMapping`` as a set.

        Accessing this property is cheaper than ``out_axes``.
        """
        return self.__inverse.keys()

    @property
    def broadcast_axes(self) -> Set[int]:
        """Returns the output batch axes that are created by broadcasting."""
        return self.__broadcast

    def source(self, out_batch_axis: int) -> int | None:
        """The input batch axis that ``out_batch_axis`` originates from.

        Args:
            out_batch_axis: The output batch axis.

        Returns:
            The input batch axis that ``out_batch_axis`` originates from.
            If ``out_batch_axis`` is created by broadcasting, the return value
            may not be a batch axis.
            When broadcasting, the return value can either be ``None``, or
            the index of a size one axis that is not contained in ``self.in_axes``.

        Throws:
            ValueError: If ``out_batch_axis`` is not a batch axis in this mapping.
        """
        try:
            return self.__inverse[out_batch_axis]
        except KeyError:
            raise ValueError(
                f"Output axis {out_batch_axis} not in batch axis mapping."
            ) from None

    def sink(self, in_batch_axis: int) -> int:
        """The output batch axis that ``in_batch_axis`` is mapped to.

        Args:
            in_batch_axis: The input batch axis.

        Returns:
            The output batch axis that ``in_batch_axis`` is mapped to.

        Throws:
            ValueError: If ``in_batch_axis`` is not a batch axis in this mapping.
        """
        try:
            return self.__mapping[in_batch_axis]
        except KeyError:
            raise ValueError(
                f"Input axis {in_batch_axis} not in batch axis mapping."
            ) from None

    def __iter__(self) -> Iterator[tuple[int | None, int]]:
        """Iterates over the pairs of batch axes in ``self``.

        Yields:
            Pairs of input batch axes and output batch axes.
            The input batch axes may be ``None`` if the output batch axis
            is created by broadcasting.
        """
        return ((i, j) for j, i in self.__inverse.items())

    def __len__(self) -> int:
        return len(self.__inverse)

    def __eq__(self, other):
        return (
            isinstance(other, BatchAxisMapping)
            and self.__inverse == other.__inverse
            and self.__broadcast == other.__broadcast
        )

    def __hash__(self):
        return hash((self.__inverse, self.__broadcast))

    def __str__(self):
        def pair_str(i, j):
            if j in self.__broadcast:
                return f"{i} ~~> {j}"
            else:
                return f"{i} <-> {j}"

        return "{" + ", ".join(pair_str(i, j) for i, j in self) + "}"

    def __repr__(self):
        return f"BatchAxisMapping({self})"

    def filter_in_axis(self, remove: Collection[int]) -> "BatchAxisMapping":
        """Removes mapping entries where the input axis is in ``remove``."""
        keep = ((i, j, j in self.__broadcast) for i, j in self if i not in remove)
        return BatchAxisMapping(*keep)

    def filter_out_axis(self, remove: Collection[int]) -> "BatchAxisMapping":
        """Removes mapping entries where the output axis is in ``remove``."""
        keep = ((i, j, j in self.__broadcast) for i, j in self if j not in remove)
        return BatchAxisMapping(*keep)

    def chain(self, other: "BatchAxisMapping") -> "BatchAxisMapping":
        """Chain two ``BatchAxisMapping``.

        If ``(0, 1)`` is in ``self`` and ``(1, 2)``, in ``other`` the returned
        mapping contains ``(0, 2)``.
        """
        from_self = (
            (self.source(j), k, k in self.__broadcast or j in other.__broadcast)
            for j, k in other
            if j in self.out_axes_set
        )
        from_broadcasting = ((None, k) for j, k in other if j is None)
        return BatchAxisMapping(*from_self, *from_broadcasting)


SOURCE_TYPE = Var | HashableLiteral


class BatchAxisGraph:
    """A graph capturing the batch axes mappings in a Jaxpr."""

    __slots__ = ("__edges", "__predecessors", "__successors", "__sinks")

    def __init__(
        self,
        edges: Mapping[tuple[Var | HashableLiteral, Var], BatchAxisMapping],
        sinks: Mapping[Var | HashableLiteral, Collection[int]],
    ):
        predecessors = defaultdict(set)
        successors = defaultdict(set)
        for start, end in edges.keys():
            predecessors[end].add(start)
            successors[start].add(end)

        self.__edges: dict[tuple[SOURCE_TYPE, Var], BatchAxisMapping] = dict(edges)
        self.__predecessors: dict[Var, set[SOURCE_TYPE]] = predecessors
        self.__successors: dict[SOURCE_TYPE, set[Var]] = successors
        self.__sinks: dict[SOURCE_TYPE, frozenset[int]] = {
            v: frozenset(s) for v, s in sinks.items()
        }

    @property
    def atoms(self) -> Set[Atom]:
        """All variables in this graph."""
        return union(self.__successors.keys(), self.__sinks.keys())

    def __getitem__(
        self, item: Atom | tuple[Atom, Atom]
    ) -> tuple[int, ...] | BatchAxisMapping:
        if isinstance(item, tuple):
            return self.__edges[item]
        elif isinstance(item, Atom):
            return self.batch_axes(item)
        else:
            raise TypeError(f"Expected Atom or tuple[Atom, Atom], got {type(item)}.")

    def batch_axes(self, var: Atom) -> tuple[int, ...]:
        """Returns the batch axes of variable ``var`` recorded in this graph.

        Args:
            var: The variable whose batch axes to retrieve.

        Returns:
            The batch axes of ``var`` that appear in this graph as a
            sorted tuple of unique integers.
        """
        return tuple(sorted(self.batch_axes_set(var)))

    def batch_axes_set(self, var: Var) -> Set[int]:
        """Returns the batch axes of variable ``var`` recorded in this graph as a set.

        Args:
            var: The variable whose batch axes to retrieve.

        Returns:
            The batch axes of ``var`` that appear in this graph.
        """
        if isinstance(var, Literal):
            var = HashableLiteral(var)
        if var in self.__sinks:
            return self.__sinks[var]
        successor = next(iter(self.__successors[var]))
        return self.__edges[(var, successor)].in_axes_set

    def mapping[D](
        self, start: Var, end: Var, default: D = ...
    ) -> BatchAxisMapping | D:
        """Returns the batch axis mapping between ``start`` and ``end``.

        If there is no immediate edge between ``start`` and ``end``, searches
        for a path from ``start`` to ``end`` and returns the batch axis mapping
        along that path.
        If no path exists, returns ``default``.

        Args:
            start: The start of the edge.
            end: The end of the edge.
            default: The default value to return if no mapping exists.
                If not specified, the default value is an empty batch axis mapping.

        Returns:
            The batch axis mapping between ``start`` and ``end``.
            If no path between ``start`` and ``end`` exists, returns ``default``.
        """
        if default is ...:
            default = BatchAxisMapping.empty()

        if isinstance(start, Literal):
            start = HashableLiteral(start)
        if isinstance(end, Literal):
            end = HashableLiteral(end)

        if (start, end) in self.__edges:
            return self.__edges[(start, end)]
        elif start == end:
            return BatchAxisMapping.identity(self.batch_axes(start))

        for successor in self.__successors[start]:
            if successor == start:  # we sometimes use self-loops
                continue
            sub_mapping = self.mapping(successor, end, default=None)
            if sub_mapping is not None:
                break
        else:
            return default

        new_map = self.__edges[(start, successor)].chain(sub_mapping)
        self.__edges[(start, end)] = new_map
        return new_map

    def _cache_mapping(self, start: Var, end: Var, mapping: BatchAxisMapping):
        """Stores a batch axis mapping between ``start`` and ``end``."""
        assert (start, end) not in self.__edges
        self.__edges[(start, end)] = mapping
        self.__predecessors[end].add(start)
        self.__successors[start].add(end)

    def mappings_to(
        self, target: Var, sources: Collection[Var] | None = None
    ) -> dict[Var, BatchAxisMapping]:
        """Computes the batch axis mappings from the variable in ``sources`` to ``target``.

        Args:
            target: The target variable.
            sources: From where to compute the mappings.
                If ``None``, computes the mappings from all variables in this graph.

        Returns:
            Returns a dictionary mapping each variable in ``source`` to the
            ``BatchAxisMapping`` between that variable and ``target``.
        """
        if sources is None:
            sources = self.atoms
        return {var: self.mapping(var, target) for var in sources}


class _BatchAxisGraphBuilder:
    """Iteratively build a ``BatchAxisGraph``."""

    def __init__(self):
        self.edges: dict[tuple[SOURCE_TYPE, Var], BatchAxisMapping] = {}
        self.predecessors: defaultdict[Var, set[SOURCE_TYPE]] = defaultdict(set)
        self.successors: defaultdict[SOURCE_TYPE, set[Var]] = defaultdict(set)
        self.sinks: dict[SOURCE_TYPE, set[int]] = {}  # variables without successors

    @property
    def atoms(self) -> Set[Atom]:
        return self.successors.keys() | self.sinks.keys()

    def add_edge(self, start: Atom, end: Var, mapping: BatchAxisMapping):
        if isinstance(start, Literal):
            start = HashableLiteral(start)
        assert not isinstance(end, Literal)
        assert (start, end) not in self.edges, "Edge already exists"
        assert start not in self.sinks, "Edge from a sink"
        assert start != end, "No self loops allowed"
        self.update_edge(start, end, mapping)

    def update_edge(self, start: Atom, end: Var, mapping: BatchAxisMapping):
        if isinstance(start, Literal):
            start = HashableLiteral(start)
        self.edges[(start, end)] = mapping
        self.predecessors[end].add(start)
        self.successors[start].add(end)

    def add_sink(self, var: Var, batch_axes: Collection[int]):
        if isinstance(var, Literal):
            var = HashableLiteral(var)
        assert var not in self.sinks, "Sink already exists"
        assert len(self.successors[var]) == 0, "Sink has successors"
        self.update_sink(var, batch_axes)

    def update_sink(self, var: Var, batch_axes: Collection[int]):
        if isinstance(var, Literal):
            var = HashableLiteral(var)
        self.sinks[var] = set(batch_axes)

    def not_batch(self, var: Var, axis: int | None):
        """Updates the graph to remove ``axis`` as a batch axis of ``var``.

        Recursively updates the edges in this graph to propagate the effect
        of removing ``axis`` as a batch axis.
        If ``axis`` is ``None`` or already not a batch axis of ``var``,
        this function does nothing.
        """
        if axis is None:
            return

        if isinstance(var, Literal):
            var = HashableLiteral(var)

        if var in self.sinks:
            self.sinks[var].remove(axis)

        # propagate forward
        for successor in self.successors[var]:
            mapping = self.edges[var, successor]
            if axis in mapping.in_axes_set:
                self.edges[var, successor] = mapping.filter_in_axis([axis])
                self.not_batch(successor, mapping.sink(axis))

        # propagate backwards
        for predecessor in self.predecessors[var]:
            mapping = self.edges[predecessor, var]
            if axis in mapping.out_axes_set:
                self.edges[predecessor, var] = mapping.filter_out_axis([axis])
                self.not_batch(predecessor, mapping.source(axis))

    def filter_batch_axes(self, var: Atom, keep: Collection[int]):
        """Removes batch axes of ``var`` that are not in ``keep``.

        Recursively updates the edges in this graph using ``self.not_batch``
        to remove batch axes that are not in ``keep``.
        """
        if isinstance(var, Literal):
            var = HashableLiteral(var)

        keep = frozenset(keep)
        current_batch_axes = set(
            itertools.chain(
                *(self.edges[var, s].in_axes_set for s in self.successors[var]),
                *(self.edges[p, var].out_axes_set for p in self.predecessors[var]),
                self.sinks.get(var, ()),
            )
        )
        for axis in current_batch_axes:
            if axis not in keep:
                self.not_batch(var, axis)

    def build(self) -> BatchAxisGraph:
        """Check the current graph for consistency and return it as a BatchAxisGraph."""
        for var in self.successors:  # all nodes with a successor
            if len(self.successors[var]) == 0:
                continue
            successor = next(iter(self.successors[var]))
            batch_axes = self.edges[var, successor].in_axes_set

            for successor in self.successors[var]:
                assert self.edges[var, successor].in_axes_set == batch_axes

            if len(self.predecessors[var]) > 0:
                incoming = union(
                    *(self.edges[p, var].out_axes_set for p in self.predecessors[var])
                )
                assert incoming == batch_axes

        for var in self.sinks:
            assert len(self.successors[var]) == 0
            batch_axes = self.sinks[var]
            if len(self.predecessors[var]) > 0:
                incoming = union(
                    *(self.edges[p, var].out_axes_set for p in self.predecessors[var])
                )
                assert incoming == batch_axes

        # all nodes without successors should be sinks
        no_successors = self.predecessors.keys() - self.successors.keys()
        assert no_successors.issubset(self.sinks.keys())

        return BatchAxisGraph(self.edges, self.sinks)


def infer_batch_axes(
    jaxpr: ClosedJaxpr,
    in_batch_axes: (
        EllipsisType | Collection[int] | Sequence[Collection[int] | EllipsisType]
    ) = ...,
    out_batch_axes: (
        EllipsisType | Collection[int] | Sequence[Collection[int] | EllipsisType]
    ) = ...,
) -> BatchAxisGraph:
    """Analyzes a Jaxpr to find batch axes.

    Batch axes are axes whose elements do not interact with each other.
    To simplify the inference, batch axes must remain their shape throughout the
    computation.
    Information about batch axes is used, for example, by LiRPA to determine the shape
    of the LiRPA weights.

    Size-one axes that are added for broadcasting or by reshaping are generally
    considered batch axes.

    The arguments ``in_batch_axes`` and ``out_batch_axes`` allow to place constraints
    on the batch axes of the input and output of ``jaxpr``.
    This allows for marking axes as non-batch axes that would normally be detected
    as batch axes.
    The constraints on the input and output are taken into account when inferring the
    batch axes of intermediate variables as well.

    Args:
        jaxpr: The Jaxpr to find the batch axes of.
        in_batch_axes: A constraint on the batch axes of the input of ``jaxpr``.
            While an ellipsis (``...``) means no constraints, a collection of integers
            is used to express that only the axes in the collection are batch axes.
            An empty collection means that an input has no batch axes.
            The input constraint can either be given as a single ellipsis of collection
            of integers that refers to all inputs or one ellipsis/collection of integers
            per input.
        out_batch_axes: A constraint on the batch axes of the output of ``jaxpr``.
            The format is as for ``in_batch_axes``.

    Returns:
        A ``BatchAxisGraph`` that describes the mappings between batch axes
        in ``jaxpr``.
    """
    if (
        in_batch_axes is ...
        or isinstance(in_batch_axes, Collection)
        and isinstance(in_batch_axes[0], int)
    ):
        in_batch_axes = (in_batch_axes,) * len(jaxpr.in_avals)
    if (
        out_batch_axes is ...
        or isinstance(out_batch_axes, Collection)
        and isinstance(out_batch_axes[0], int)
    ):
        out_batch_axes = (out_batch_axes,) * len(jaxpr.out_avals)

    # Algorithm:
    # 1. Collect batch axis mappings.
    #   Use the batch axis rules to determine the fundamental mappings between
    #   batch axes for each equation in the jaxpr.
    #   These mappings contain information on which axes can not be batch axes.
    #   For example:
    #    - dot_general => axes that are contracted can not be batch axes
    #    - reduce_mean => axes that are reduced can not be batch axes
    #    - elementwise operation => all axes can be batch axes.
    #   Store the mappings in a batch axis graph builder.
    # 2. Enforce constraints.
    #   Remove batch axes in the mappings that are not in ``in_batch_axes``
    #   and ``out_batch_axes``.
    #   Recursively propagate the removal in the graph.
    # 3. Resolve inconsistencies.
    #   Check the mappings for cases where
    #   - mappings from the same variable have different batch axes
    #     (batch axis disappears on one path),
    #   - incoming edges contain batch axes not present in the outgoing edges,
    #   - outgoing edges contain batch axes not present in any incoming edge.
    #   Resolve these inconsistencies by removing the conflicting batch axes
    #   and propagating the removal in the graph.

    # Step 1: Collect batch axes mappings

    graph: _BatchAxisGraphBuilder = _BatchAxisGraphBuilder()

    for eqn in jaxpr.eqns:
        if eqn.primitive in batch_axis_mapping_rules:
            rule = batch_axis_mapping_rules[eqn.primitive]

            in_shapes = tuple(v.aval.shape for v in eqn.invars)
            out_shapes = tuple(v.aval.shape for v in eqn.outvars)
            mappings = rule(in_shapes, out_shapes, **eqn.params)
            for out_var, maps in strict_zip(eqn.outvars, mappings):
                for in_var, map_ in strict_zip(eqn.invars, maps):
                    graph.add_edge(in_var, out_var, map_)
        else:
            warn(
                f"No batch axis mapping rule for {eqn.primitive}. "
                f"Assuming that input and output of {eqn.primitive} have no batch axes. "
                f"Register a batch axis mapping rule for {eqn.primitive} to enable "
                f"better batch axis inference.",
                stacklevel=1,
            )
            for out_var in eqn.outvars:
                for in_var in eqn.invars:
                    graph.add_edge(in_var, out_var, BatchAxisMapping())

    # Declaring sinks allows handling empty Jaxprs, as well as jax.core.DropVars
    effective_sinks = graph.predecessors.keys() - graph.successors.keys()
    for var in set(jaxpr.jaxpr.outvars) | effective_sinks:
        graph.add_sink(var, range(len(var.aval.shape)))

    # Jaxprs may also contain unused input variables. Also declare those as sinks.
    for var in jaxpr.jaxpr.invars:
        if var not in graph.atoms:
            graph.add_sink(var, range(len(var.aval.shape)))

    # Step 2: Apply constraints

    for i, var in enumerate(jaxpr.jaxpr.invars):
        if in_batch_axes[i] is not Ellipsis:
            graph.filter_batch_axes(var, in_batch_axes[i])
    for i, var in enumerate(jaxpr.jaxpr.outvars):
        if out_batch_axes[i] is not Ellipsis:
            graph.filter_batch_axes(var, out_batch_axes[i])

    # Step 3: Resolve inconsistencies

    for var in graph.atoms:
        # since batch axes may not disappear on any path, all mappings from
        # var need to have the same in_batch_axes
        if var in graph.sinks:
            batch_axes = graph.sinks[var]
        else:
            batch_axes = intersection(
                *(graph.edges[var, s].in_axes_set for s in graph.successors[var]),
            )
        if len(graph.predecessors[var]) > 0:
            # all batch axes also need to originate from somewhere
            in_batch_axes = union(
                *(graph.edges[p, var].out_axes_set for p in graph.predecessors[var])
            )
            batch_axes = batch_axes.intersection(in_batch_axes)
        graph.filter_batch_axes(var, batch_axes)

    return graph.build()


class BatchAxisMappingRule(Protocol):
    """A rule for inferring batch axes mappings of a JAX primitive.

    A batch axis is an axis whose elements do not interact with each other during
    a computation.
    A batch axis mapping describes the relationship between the batch axes of an input
    variable and the batch axes of an output variable.

    For example, in ``a = b - c``, the value of ``b[0]`` does not interact with ``b[1]``
    to determine the value of ``a``.
    Therefore, the first axis of ``b`` is a batch axis.
    The batch axis mapping is ``((0, 0), (1, 1), ...)`` for ``b -> a``,
    depending on the number of axes of ``b``.
    The mapping for ``c -> a`` similarly is ``((0, 0), (1, 1), ...)``.

    Considering ``a = mean(b, axis=0)``, the values of ``b[0]`` and ``b[1]`` interact
    with each other to determine the value of ``a``.
    Therefore, the first axis of ``b`` is not a batch axis.
    The mapping for ``b -> a`` similarly is ``((1, 1), ...)``.

    If we consider ``a = matmul(b, c)``, where ``b`` and ``c`` are matrices, the values of
    ``b[:, 0]`` and ``b[:, 1]`` interact with each other to determine the value of ``a``.
    Similarly, the values of ``c[0, :]`` and ``c[1, :]`` interact with each other, while
    the values of ``c[:, 0]`` and ``c[:, 1]`` do not.
    The batch axis mapping is ``((0, 0),)`` for ``b -> a`` and ``((1, 1),)`` for ``c -> a``.
    """

    def __call__(
        self,
        in_shapes: tuple[tuple[int, ...], ...],
        out_shapes: tuple[tuple[int, ...], ...],
        **kwargs,
    ) -> tuple[tuple[BatchAxisMapping, ...], ...]:
        """Declare the batch axes mapping of the primitive.

        Args:
            in_shapes: The shapes of the inputs.
            out_shapes: The shapes of the outputs.
            **kwargs: Any additional parameters of the Jaxpr equation.

        Returns:
            For each output variable, returns a mapping between the batch axes of each
            input variable and the batch axes of the output variable.
            If the jaxpr primitive has ``m`` outputs and ``n`` inputs, the return value
            is a m-tuple of n-tuples.
            The mappings are from the input variable (first variable) to the
            output variable (second variable).
        """
        ...


batch_axis_mapping_rules: dict[Primitive, BatchAxisMappingRule] = {}


def identity_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
):
    """A batch axis mapping rule for element-wise primitives.

    This rule is suitable for all primitives that do not change the shape of
    their arguments, except for potentially broadcasting individual arguments
    in their leading axes or in size one axes.
    """

    def mapping(in_shape):
        out_shape = out_shapes[0]
        ref_size = len(out_shape)
        in_size = len(in_shape)
        n_broadcast = ref_size - in_size
        broadcast_size_one = (
            n == 1 and m > 1
            for n, m in zip(in_shape, out_shape[n_broadcast:], strict=False)
        )
        return BatchAxisMapping(
            *((None, i) for i in range(n_broadcast)),
            *((i, n_broadcast + i, b) for i, b in enumerate(broadcast_size_one)),
        )

    return (tuple(mapping(in_shape) for in_shape in in_shapes),) * len(out_shapes)


# MARK: Elementwise Primitives
for primitive in (
    # Unary:
    lax.abs_p,
    lax.acos_p,
    lax.acosh_p,
    lax.asin_p,
    lax.asinh_p,
    lax.atan_p,
    lax.atan2_p,
    lax.atanh_p,
    lax.bessel_i0e_p,
    lax.bessel_i1e_p,
    lax.cbrt_p,
    lax.ceil_p,
    lax.clz_p,
    lax.conj_p,
    lax.convert_element_type_p,
    lax.cos_p,
    lax.cosh_p,
    lax.device_put_p,
    lax.digamma_p,
    lax.erf_p,
    lax.erfc_p,
    lax.erf_inv_p,
    lax.exp_p,
    lax.expm1_p,
    lax.floor_p,
    lax.imag_p,
    lax.integer_pow_p,
    lax.is_finite_p,
    lax.lgamma_p,
    lax.log_p,
    lax.log1p_p,
    lax.logistic_p,
    lax.neg_p,
    lax.not_p,
    lax.population_count_p,
    lax.real_p,
    lax.reduce_precision_p,
    lax.round_p,
    lax.rsqrt_p,
    lax.sign_p,
    lax.sin_p,
    lax.sinh_p,
    lax.square_p,
    lax.sqrt_p,
    lax.tan_p,
    lax.tanh_p,
    # Binary:
    lax.add_p,
    lax.and_p,
    lax.complex_p,
    lax.div_p,
    lax.eq_p,
    lax.ge_p,
    lax.gt_p,
    lax.igamma_p,
    lax.igammac_p,
    lax.le_p,
    lax.lt_p,
    lax.max_p,
    lax.min_p,
    lax.mul_p,
    lax.ne_p,
    lax.nextafter_p,
    lax.polygamma_p,
    lax.pow_p,
    lax.rem_p,
    lax.rev_p,
    lax.shift_left_p,
    lax.shift_right_arithmetic_p,
    lax.shift_right_logical_p,
    lax.sub_p,
    lax.or_p,
    lax.xor_p,
    lax.zeta_p,
    # Ternary:
    lax.regularized_incomplete_beta_p,
    lax.clamp_p,
    # Variadic:
    lax.select_n_p,
):
    batch_axis_mapping_rules[primitive] = identity_batch_axis_mapping_rule


def _dot_general_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(out_shapes) == 1
    lhs_shape, rhs_shape = in_shapes

    dimension_numbers = kwargs["dimension_numbers"]
    # The axes of ``lhs`` and ``rhs`` not in ``dimension_numbers``.
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_specified = set(lhs_batch + lhs_contracting)
    rhs_specified = set(rhs_batch + rhs_contracting)
    lhs_remainder = [i for i in range(len(lhs_shape)) if i not in lhs_specified]
    rhs_remainder = [i for i in range(len(rhs_shape)) if i not in rhs_specified]

    # shape rule for dot_general:
    # batch axes first, lhs non-batch non-contracting, rhs non-batch non-contracting
    batch_axes_lhs = BatchAxisMapping(
        *((lhs_i, out_i) for out_i, lhs_i in enumerate(lhs_batch)),
        *((lhs_i, len(lhs_batch) + i) for i, lhs_i in enumerate(lhs_remainder)),
    )
    batch_axes_rhs = BatchAxisMapping(
        *((rhs_i, out_i) for out_i, rhs_i in enumerate(rhs_batch)),
        *(
            (rhs_i, len(lhs_batch) + len(lhs_remainder) + i)
            for i, rhs_i in enumerate(rhs_remainder)
        ),
    )
    return ((batch_axes_lhs, batch_axes_rhs),)


batch_axis_mapping_rules[lax.dot_general_p] = _dot_general_batch_axis_mapping_rule


def _conv_general_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(out_shapes) == 1
    assert len(in_shapes) == 2
    dimension_numbers = kwargs["dimension_numbers"]
    assert isinstance(dimension_numbers, lax.ConvDimensionNumbers)

    lhs_batch, _, *_ = dimension_numbers.lhs_spec
    rhs_feature_out, _, *_ = dimension_numbers.rhs_spec
    out_batch, out_feature, *_ = dimension_numbers.out_spec

    # The lhs_batch axis and the rhs_feature_out axis can be batch axes,
    # since their elements do not interact (in this convolution).
    # Later convolutions will typically contract the rhs_feature_out axis.

    batch_axes_lhs = BatchAxisMapping((lhs_batch, out_batch))
    batch_axes_rhs = BatchAxisMapping((rhs_feature_out, out_feature))
    return ((batch_axes_lhs, batch_axes_rhs),)


batch_axis_mapping_rules[lax.conv_general_dilated_p] = (
    _conv_general_batch_axis_mapping_rule
)


def _broadcast_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    """Batch axis mapping rule for ``lax.broadcast_in_dim``."""
    assert len(in_shapes) == len(out_shapes) == 1
    in_shape = in_shapes[0]
    out_shape = out_shapes[0]

    broadcast_dimensions = kwargs["broadcast_dimensions"]
    # Broadcasting dimensions are considered batch axes by default
    batch_axes = BatchAxisMapping(
        *(  # remapped input batch axes
            (in_ax, out_ax, in_shape[in_ax] == 1 and out_shape[out_ax] > 1)
            for in_ax, out_ax in enumerate(broadcast_dimensions)
        ),
        *(  # added output batch axes
            (None, out_ax)
            for out_ax in range(len(out_shape))
            if out_ax not in broadcast_dimensions
        ),
    )
    # there may be additional dynamic shape arguments
    return ((batch_axes, *(BatchAxisMapping() for _ in in_shapes[1:])),)


batch_axis_mapping_rules[lax.broadcast_in_dim_p] = _broadcast_batch_axis_mapping_rule


def _reshape_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(in_shapes) == 1
    assert len(out_shapes) == 1
    in_shape, out_shape = in_shapes[0], out_shapes[0]
    dimensions = kwargs["dimensions"]
    if dimensions is None:
        dimensions = tuple(range(len(in_shape)))

    mapping = []
    # match output axes to input axes
    in_axes = set()
    out_axes = set()
    out_axes_size = in_axes_size = 1

    in_axes_remaining = reversed(range(len(in_shape)))
    for out_ax in reversed(range(len(out_shape))):
        if out_shape[out_ax] == 1:
            # size one axes can always be added or removed
            mapping.append((None, out_ax))
            continue

        out_axes.add(out_ax)
        out_axes_size *= out_shape[out_ax]

        if out_axes_size < in_axes_size:
            continue
        elif out_axes_size > in_axes_size:
            for in_i in in_axes_remaining:
                in_ax = dimensions[in_i]
                if in_shape[in_ax] == 1:
                    # size one axes can always be removed
                    continue

                in_axes.add(in_ax)
                in_axes_size *= in_shape[in_ax]
                if in_axes_size >= out_axes_size:
                    break

        if out_axes_size == in_axes_size:
            if len(out_axes) == 1 and len(in_axes) == 1:
                in_ax, out_ax = next(iter(in_axes)), next(iter(out_axes))
                mapping.append((in_ax, out_ax))

            in_axes.clear()
            out_axes.clear()
            out_axes_size = in_axes_size = 1

    return ((BatchAxisMapping(*mapping),),)


batch_axis_mapping_rules[lax.reshape_p] = _reshape_batch_axis_mapping_rule


def _pad_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(in_shapes[1]) == 0, "padding_value must be scalar"
    # second input is the padding value
    return ((BatchAxisMapping.identity(range(len(in_shapes[0]))), BatchAxisMapping()),)


batch_axis_mapping_rules[lax.pad_p] = _pad_batch_axis_mapping_rule


def _transpose_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(in_shapes) == 1
    assert len(out_shapes) == 1
    permutation = kwargs["permutation"]
    batch_axes = BatchAxisMapping(
        *((in_i, out_i) for out_i, in_i in enumerate(permutation))
    )
    return ((batch_axes,),)


batch_axis_mapping_rules[lax.transpose_p] = _transpose_batch_axis_mapping_rule


def _reduce_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(in_shapes) <= 2
    assert len(out_shapes) == 1
    in_shape = in_shapes[0]
    # reduce_sum, reduce_prod, ... use "axes" but reduce and squeeze use "dimensions"
    axes = kwargs["axes"] if "axes" in kwargs else kwargs["dimensions"]

    # all dimensions that are not reduced can be batch axes
    not_reduced = [ax for ax in range(len(in_shape)) if ax not in axes]
    mapping = BatchAxisMapping(
        *((in_i, out_i) for out_i, in_i in enumerate(not_reduced))
    )
    if len(in_shapes) == 2:  # lax.reduce_p
        # second argument is the init_value, which can not be batched.
        return ((mapping, BatchAxisMapping.empty()),)
    else:
        return ((mapping,),)


batch_axis_mapping_rules[lax.reduce_p] = _reduce_batch_axis_mapping_rule
batch_axis_mapping_rules[lax.reduce_sum_p] = _reduce_batch_axis_mapping_rule
batch_axis_mapping_rules[lax.reduce_prod_p] = _reduce_batch_axis_mapping_rule
batch_axis_mapping_rules[lax.reduce_min_p] = _reduce_batch_axis_mapping_rule
batch_axis_mapping_rules[lax.reduce_max_p] = _reduce_batch_axis_mapping_rule
batch_axis_mapping_rules[lax.reduce_or_p] = _reduce_batch_axis_mapping_rule
batch_axis_mapping_rules[lax.reduce_and_p] = _reduce_batch_axis_mapping_rule
batch_axis_mapping_rules[lax.reduce_xor_p] = _reduce_batch_axis_mapping_rule
batch_axis_mapping_rules[lax.squeeze_p] = _reduce_batch_axis_mapping_rule


def _reduce_window_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(out_shapes) == 1
    in_shape = in_shapes[0]  # there may be an additional input for init_values.
    window_dimensions = kwargs["window_dimensions"]

    # If the window size in a dimension is one, the elements of that
    # dimension do not interact during the window reduction.
    batch_axes = {i for i in range(len(in_shape)) if window_dimensions[i] == 1}
    mapping = BatchAxisMapping.identity(batch_axes)
    return ((mapping, *(BatchAxisMapping(),) * len(in_shapes[1:])),)


for primitive in (
    lax.reduce_window_p,
    lax.reduce_window_max_p,
    lax.reduce_window_min_p,
    lax.reduce_window_sum_p,
):
    batch_axis_mapping_rules[primitive] = _reduce_window_batch_axis_mapping_rule


def _iota_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(in_shapes) == 0
    assert len(out_shapes) == 1
    mapping = BatchAxisMapping(*((None, i) for i in range(len(out_shapes[0]))))
    return ((mapping,),)


batch_axis_mapping_rules[lax.iota_p] = _iota_batch_axis_mapping_rule


class _SlicingBatchAxisMappingRule(BatchAxisMappingRule):
    """A batch axis inference rule for ``lax.dynamic_slice``,
    ``lax.dynamic_update_slice`` and ``lax.splice``.

    Input axes should not be sliced (start index = 0 and size = axis size)
    for the corresponding output axes to be batch axes.
    """

    def __init__(self, num_batched_args: int):
        self.num_batched_args = num_batched_args

    def __call__(
        self,
        in_shapes: tuple[tuple[int, ...], ...],
        out_shapes: tuple[tuple[int, ...], ...],
        **kwargs,
    ) -> tuple[tuple[BatchAxisMapping, ...], ...]:
        in_shape = in_shapes[0]
        for other_in_shape in in_shapes[1 : self.num_batched_args]:
            assert other_in_shape == in_shape

        start_indices = kwargs["start_indices"]
        if "limit_indices" in kwargs:  # lax.slice
            limit_indices = kwargs["limit_indices"]
            slice_sizes = tuple(
                limit - start
                for start, limit in strict_zip(start_indices, limit_indices)
            )
        else:  # lax.dynamic_slice and lax.dynamic_update_slice
            slice_sizes = kwargs["slice_sizes"]

        batch_axes = filter(
            lambda i: start_indices[i] == 0 and slice_sizes[i] == in_shape[i],
            range(len(in_shape)),
        )
        mapping = BatchAxisMapping.identity(batch_axes)
        # there may be additional inputs for start_indices and slice_sizes.
        num_other_args = len(in_shapes) - self.num_batched_args
        return (
            (mapping,) * self.num_batched_args + (BatchAxisMapping(),) * num_other_args,
        )


batch_axis_mapping_rules[lax.dynamic_slice_p] = _SlicingBatchAxisMappingRule(1)
batch_axis_mapping_rules[lax.dynamic_update_slice_p] = _SlicingBatchAxisMappingRule(2)
batch_axis_mapping_rules[lax.slice_p] = _SlicingBatchAxisMappingRule(1)


def _split_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(in_shapes) == 1
    in_shape = in_shapes[0]
    axis = kwargs["axis"]

    batch_axes = (i for i in range(len(in_shape)) if i != axis)
    return ((BatchAxisMapping.identity(batch_axes),),) * len(out_shapes)


if hasattr(lax, "split_p"):
    batch_axis_mapping_rules[lax.split_p] = _split_batch_axis_mapping_rule


def _top_k_batch_axis_mapping_rule(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    **kwargs,
) -> tuple[tuple[BatchAxisMapping, ...], ...]:
    assert len(in_shapes) == 1
    assert len(out_shapes) == 1
    in_shape = in_shapes[0]
    # all but the last axis can be batch axes
    return ((BatchAxisMapping.identity(range(len(in_shape) - 1)),),)


batch_axis_mapping_rules[lax.top_k_p] = _top_k_batch_axis_mapping_rule


class _JaxprBatchAxisMappingRule(BatchAxisMappingRule):
    """A batch axis inference rule for a nested jaxpr (e.g. ``jax.jit``)."""

    def __init__(self, jaxpr_key: str):
        self.jaxpr_key = jaxpr_key

    def __call__(
        self,
        in_shapes: tuple[tuple[int, ...], ...],
        out_shapes: tuple[tuple[int, ...], ...],
        **kwargs,
    ) -> tuple[tuple[BatchAxisMapping, ...], ...]:
        jaxpr = kwargs[self.jaxpr_key]
        graph = infer_batch_axes(jaxpr)
        return tuple(
            tuple(graph.mapping(i, o) for i in jaxpr.jaxpr.invars)
            for o in jaxpr.jaxpr.outvars
        )


batch_axis_mapping_rules[primitives.jit_p] = _JaxprBatchAxisMappingRule("jaxpr")
batch_axis_mapping_rules[jax.custom_derivatives.custom_jvp_call_p] = (
    _JaxprBatchAxisMappingRule("call_jaxpr")
)
batch_axis_mapping_rules[jax.custom_derivatives.custom_vjp_call_p] = (
    _JaxprBatchAxisMappingRule("fun_jaxpr")
)


# TODO:
#  - lax.gather_p
#  - lax.scatter_p and other scatters
#  - lax.sort_p
