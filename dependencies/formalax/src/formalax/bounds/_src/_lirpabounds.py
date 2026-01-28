#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
"""Linear Relaxation Perturbation Analysis (LiRPA) bounds classes."""

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import jax.tree_util
from jaxtyping import Array, Real

from ...core.batch_axes import BatchAxisMapping
from ...core.zero import Zero, is_zero, zero
from ...utils.batch_axes import (
    batch_shape,
    full_shape,
    non_batch_shape,
    split_shape,
)
from ...utils.expand_as_diagonal import expand_as_diagonal
from ...utils.sets import union
from ...utils.zip import strict_zip
from ._affinebounds import AffineBoundsABC
from ._bounds import Bounds, example_args

__all__ = (
    "LiRPAWeights",
    "LiRPABounds",
    "incorporate_batch_axes",
    "pull_batch_axes",
    "LiRPAWeightsInfo",
)


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False, frozen=True, slots=True, kw_only=True)
class LiRPABounds[T: Real[Array, "..."] | Zero](AffineBoundsABC[T]):
    """Affine bounds created by Linear Relaxation Perturbation Analysis (LiRPA)
    bound propagation.

    ``LiRPABounds`` implements a batched affine transformation, tracking batch
    shapes and positions to simplify evaluating the bounds.
    See ``LiRPAWeights`` for more details on the shape format of the weights.
    The biases of a ``LiRPABounds`` have the shape ``full_out_shape``.
    In other words, the biases have the shape of the actual output with
    batch axes placed at their actual positions.

    ``LiRPABounds`` can be unfolded using ``*bounds``
    which yields
    ``*(bounds.lb_weights, bounds.ub_weights, bounds.lb_bias, bounds.ub_bias)``
    and can also be split like a tuple:

        lb_weights, ub_weights, lb_bias, ub_bias = bounds
    """

    full_in_shapes: tuple[tuple[int, ...], ...]
    full_out_shape: tuple[int, ...]
    batch_axis_mappings: tuple[BatchAxisMapping, ...]
    in_batch_shapes: tuple[tuple[int, ...], ...]
    in_shapes: tuple[tuple[int, ...], ...]
    out_shape: tuple[int, ...]
    out_batch_axes: tuple[int, ...]

    # --------------------------------------------------------------------------
    # Create LiRPABounds
    # --------------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        lb_weights: Sequence[T],
        ub_weights: Sequence[T],
        lb_bias: T,
        ub_bias: T,
        domain: Sequence[Bounds[T]],
        full_in_shapes: Sequence[tuple[int, ...]],
        full_out_shape: tuple[int, ...],
        batch_axis_mappings: Sequence[BatchAxisMapping],
    ):
        """Create ``LiRPABounds`` from weight and bias arrays.

        Args:
            lb_weights: The lower bound weights.
                Each element of ``lb_weights`` has the shape
                ``(*batch_shape, *out_shape, *in_shape)`` where ``in_shape`` is the
                matching shape in ``in_shapes``.
            ub_weights: The upper bound weights.
                The weights need to have the same shape as ``lb_weights``.
            lb_bias: The lower bound bias.
                Needs to have the shape ``*full_out_shape``.
            ub_bias: The upper bound bias.
                Needs to have the same shape as ``lb_bias``.
            domain: The domain of each input for which this ``LiRPABounds`` instance
                is valid.
            full_in_shapes: The full shapes (batch + non-batch axes) of the inputs.
            full_out_shape: The full shape (batch + non-batch axes) of the output,
                that is, the result of multiplying the weights with an input.
            batch_axis_mappings: The batch axis mappings from each of the inputs
                to the output.
        """
        lb_weights = tuple(lb_weights)
        ub_weights = tuple(ub_weights)
        domain = tuple(domain)
        full_in_shapes = tuple(full_in_shapes)
        batch_axis_mappings = tuple(batch_axis_mappings)

        # Validate shapes
        cls._validate_bias_shape(lb_bias, full_out_shape)
        cls._validate_bias_shape(ub_bias, full_out_shape)

        in_batch_shapes, in_shapes = zip(
            *(
                split_shape(full_in_shape, ba_map.in_axes)
                for full_in_shape, ba_map in strict_zip(
                    full_in_shapes, batch_axis_mappings
                )
            ),
            strict=False,
        )

        for lb_w, ub_w, in_batch, in_shape, ba_map in strict_zip(
            lb_weights, ub_weights, in_batch_shapes, in_shapes, batch_axis_mappings
        ):
            out_shape_ = non_batch_shape(full_out_shape, ba_map.out_axes)
            cls._validate_weight_shape(lb_w, in_batch, in_shape, out_shape_)
            cls._validate_weight_shape(ub_w, in_batch, in_shape, out_shape_)

        out_batch_axes = union(*(map_.out_axes_set for map_ in batch_axis_mappings))
        out_batch_axes = tuple(sorted(out_batch_axes))
        out_shape = non_batch_shape(full_out_shape, out_batch_axes)
        return cls(
            lb_weights=lb_weights,
            ub_weights=ub_weights,
            lb_bias=lb_bias,
            ub_bias=ub_bias,
            domain=domain,
            full_in_shapes=full_in_shapes,
            full_out_shape=full_out_shape,
            batch_axis_mappings=batch_axis_mappings,
            in_batch_shapes=in_batch_shapes,
            in_shapes=in_shapes,
            out_shape=out_shape,
            out_batch_axes=out_batch_axes,
        )

    @classmethod
    def from_weights(
        cls,
        in_weights: Sequence["LiRPAWeights[T]"],
        lb_bias: T,
        ub_bias: T,
        domain: Sequence[Bounds[T]],
        full_out_shape: tuple[int, ...] = None,
    ):
        """Create a ``LiRPABounds`` instance from a sequence of ``LiRPAWeights`` and biases.

        Args:
            in_weights: The sequence of ``LiRPAWeights`` to use.
            lb_bias: The lower bound bias.
            ub_bias: The upper bound bias.
            domain: The domains of each input.
            full_out_shape: The full output shape (batch and non-batch axes)
                of the weights.
                By default, the output shape is inferred from the first weight in
                ``in_weights``.
                If ``in_weights`` is empty, ``out_shape`` must be specified.
        """
        if len(in_weights) == 0 and full_out_shape is None:
            raise ValueError("Cannot infer output shape from empty weights sequence.")

        lb_weights, ub_weights = strict_zip(*in_weights)
        domain = tuple(domain)

        in_batch_shapes = tuple(w.batch_shape for w in in_weights)
        in_shapes = tuple(w.in_shape for w in in_weights)
        batch_axis_mappings = tuple(w.batch_axis_mapping for w in in_weights)

        for lb_w, ub_w, in_batch, in_shape, ba_map in strict_zip(
            lb_weights, ub_weights, in_batch_shapes, in_shapes, batch_axis_mappings
        ):
            out_shape_ = non_batch_shape(full_out_shape, ba_map.out_axes)
            cls._validate_weight_shape(lb_w, in_batch, in_shape, out_shape_)
            cls._validate_weight_shape(ub_w, in_batch, in_shape, out_shape_)

        full_in_shapes = tuple(w.full_in_shape for w in in_weights)
        out_batch_axes = union(*(map_.out_axes_set for map_ in batch_axis_mappings))
        out_batch_axes = tuple(sorted(out_batch_axes))
        if full_out_shape is None:
            out_shape = lb_weights[0].out_shape
            ref_w = in_weights[0]
            batch_shape, ba_map = ref_w.batch_shape, ref_w.batch_axis_mapping
            full_out_shape = full_shape(batch_shape, out_shape, ba_map.out_axes_set)
        else:
            out_shape = non_batch_shape(full_out_shape, out_batch_axes)

        cls._validate_bias_shape(lb_bias, full_out_shape)
        cls._validate_bias_shape(ub_bias, full_out_shape)

        return cls(
            lb_weights=lb_weights,
            ub_weights=ub_weights,
            lb_bias=lb_bias,
            ub_bias=ub_bias,
            domain=domain,
            full_in_shapes=full_in_shapes,
            full_out_shape=full_out_shape,
            batch_axis_mappings=batch_axis_mappings,
            in_batch_shapes=in_batch_shapes,
            in_shapes=in_shapes,
            out_shape=out_shape,
            out_batch_axes=out_batch_axes,
        )

    @classmethod
    def zero_bounds(
        cls,
        domain: Sequence[Bounds[T]],
        full_in_shapes: Sequence[tuple[int, ...]],
        full_out_shape: tuple[int, ...],
        batch_axis_mappings: Sequence[BatchAxisMapping],
    ):
        """ """
        assert len(batch_axis_mappings) == len(full_in_shapes)
        n_args = len(full_in_shapes)
        return cls.from_arrays(
            (Zero(),) * n_args,
            (Zero(),) * n_args,
            Zero(),
            Zero(),
            domain,
            full_in_shapes,
            full_out_shape,
            batch_axis_mappings,
        )

    @classmethod
    def _validate_bias_shape(cls, bias, full_out_shape):
        if not is_zero(bias):
            assert len(bias.shape) == len(full_out_shape)

    @classmethod
    def _validate_weight_shape(cls, weight, in_batch_shape, in_shape, out_shape):
        n_prefix = len(in_batch_shape) + len(out_shape)
        if not is_zero(weight):
            assert len(weight.shape) == n_prefix + len(in_shape)
            assert weight.shape[len(in_batch_shape) : n_prefix] == out_shape
            assert weight.shape[n_prefix:] == in_shape

    # --------------------------------------------------------------------------
    # Jax PyTree Compatibility
    # --------------------------------------------------------------------------

    def tree_flatten(self):
        children = (
            self.lb_weights,
            self.ub_weights,
            self.lb_bias,
            self.ub_bias,
            self.domain,
        )
        aux_data = (
            self.full_in_shapes,
            self.full_out_shape,
            self.batch_axis_mappings,
            self.in_batch_shapes,
            self.in_shapes,
            self.out_shape,
            self.out_batch_axes,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        lb_weights, ub_weights, lb_bias, ub_bias, domain = children
        (
            full_in_shapes,
            full_out_shape,
            batch_axis_mappings,
            in_batch_shapes,
            in_shapes,
            out_shape,
            out_batch_axes,
        ) = aux_data
        return cls(
            lb_weights=lb_weights,
            ub_weights=ub_weights,
            lb_bias=lb_bias,
            ub_bias=ub_bias,
            domain=domain,
            full_in_shapes=full_in_shapes,
            full_out_shape=full_out_shape,
            batch_axis_mappings=batch_axis_mappings,
            in_batch_shapes=in_batch_shapes,
            in_shapes=in_shapes,
            out_shape=out_shape,
            out_batch_axes=out_batch_axes,
        )

    # --------------------------------------------------------------------------
    # Accessors
    # --------------------------------------------------------------------------

    def weights_iter(self) -> Iterator["LiRPAWeights[T]"]:
        """
        Iterates over pairs of weights from ``lb_weights``, ``ub_weights``.
        """
        for lb_w, ub_w, in_batch_shape, in_shape, ba_map in strict_zip(
            self.lb_weights,
            self.ub_weights,
            self.in_batch_shapes,
            self.in_shapes,
            self.batch_axis_mappings,
        ):
            out_shape = non_batch_shape(self.full_out_shape, ba_map.out_axes_set)
            yield LiRPAWeights(lb_w, ub_w, in_batch_shape, out_shape, in_shape, ba_map)

    def __iter__(self) -> Iterator[T]:
        yield self.lb_weights
        yield self.ub_weights
        yield self.lb_bias
        yield self.ub_bias

    # --------------------------------------------------------------------------
    # AffineBounds implementation
    # --------------------------------------------------------------------------

    def affine(
        self,
        xs: tuple[Real[Array, "..."]],
        weights: tuple[Real[Array, "..."], ...],
        bias: T | Zero = zero,
    ) -> T:
        """Computes a batch affine transformation.

        Args:
            xs: The inputs to the affine transformation.
                The shape of `xs[i]` is ``full_in_shapes[i]``.
                The batch axes `xs[i]` need to be
                ``self.batch_axis_mappings[i].in_axes``.
            weights: The weights of the affine transformation.
                The weights need to have leading batch axes, followed by
                output axes, and then input axes.
            bias: The bias of the affine transformation.
                Can be ``Zero``, which stands for a scalar zero.
                The shape of ``bias`` is ``full_out_shape``.

        Returns:
            An affine transformation like ``\\sum_i xs_i @ weights_i + bias``.
            The shape of the return value is ``full_out_shape``.
        """
        out = bias

        for x, w, batch_shape_, in_shape, ba_map in strict_zip(
            xs, weights, self.in_batch_shapes, self.in_shapes, self.batch_axis_mappings
        ):
            out_shape = non_batch_shape(self.full_out_shape, ba_map.out_axes_set)
            val = _lirpa_dot(x, w, batch_shape_, out_shape, in_shape, ba_map)
            out = out + val
        return out


@jax.tree_util.register_pytree_node_class
@dataclass(eq=False, frozen=True, slots=True)
class LiRPAWeights[T: Real[Array, "..."] | Zero]:
    """The weights of a LiRPA bound.

    ``LiRPAWeights`` are propagated in backwards LiRPA.
    They serve as the input to backwards LiRPA rules.

    ``LIRPAWeights`` also track information about output and input shapes and
    batch axes.
    The weights of a ``LiRPAWeights`` generally have the shape
    ``(*batch_shape, *out_shape, *in_shape)``, where ``out_shape`` and ``in_shape``
    are stripped of the batch axes.
    Note that batch axes are assumed not to change shape during a computation.

    ``LiRPAWeights`` can be unfolded using ``*weights``
    which yields ``*(weights.lb_weight, weights.ub_weight)``
    and can also be split like a tuple:

        lb_weight, ub_weight = weights
    """

    lb_weight: T
    ub_weight: T
    batch_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    in_shape: tuple[int, ...]
    batch_axis_mapping: BatchAxisMapping

    # --------------------------------------------------------------------------
    # Jaxpr Compatibility
    # --------------------------------------------------------------------------

    def tree_flatten(self):
        children = (self.lb_weight, self.ub_weight)
        aux_data = (
            self.batch_shape,
            self.out_shape,
            self.in_shape,
            self.batch_axis_mapping,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    # --------------------------------------------------------------------------
    # Create LiRPAWeights
    # --------------------------------------------------------------------------

    @classmethod
    def identity_for[T](
        cls, array: T, batch_axes: tuple[int, ...]
    ) -> "LiRPAWeights[T]":
        """Construct a pair of left identity arrays for use with ``dot_general``.

        The returned ``LiRPAWeights`` consists of two identical identity tensors.
        If ``array`` is a vector and ``batch_axes`` is ``()``, the identity matrix
        created by ``identity_for`` satisfies

            identity @ array = array

        If ``batch_axes`` are not ``()`` the identity array returned by
        ``identity_for`` satisfies something like

            array = lax.dot_general(
                identity,
                array,
                dimension_numbers=(
                    (
                        ...rear axes of identity except the batch axis...,
                        ...all axes of array except the batch axis...,
                    ),
                    ((0, ..., len(batch_axes)-1), batch_axes),
                )
            )

        """
        batch_axes = tuple(len(array.shape) + ba if ba < 0 else ba for ba in batch_axes)
        batch_shape, data_shape = split_shape(array.shape, batch_axes)

        flat_size = math.prod(data_shape)
        flat_identity = jnp.identity(flat_size, array.dtype)

        batch_size = math.prod(batch_shape)
        flat_identity = jnp.broadcast_to(
            flat_identity, (batch_size, flat_size, flat_size)
        )
        identity = flat_identity.reshape(batch_shape + 2 * data_shape)
        id_mapping = BatchAxisMapping.identity(batch_axes)
        return cls(identity, identity, batch_shape, data_shape, data_shape, id_mapping)

    @classmethod
    def zeros(
        cls,
        batch_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
        in_shape: tuple[int, ...],
        batch_axis_mapping: BatchAxisMapping,
    ):
        return cls(
            Zero(),
            Zero(),
            batch_shape,
            out_shape,
            in_shape,
            batch_axis_mapping,
        )

    def backwards_zeros(
        self, new_in_shape: tuple[int, ...], batch_axis_mapping: BatchAxisMapping
    ) -> "LiRPAWeights[T]":
        """Creates Zero LiRPA with analogous shape information to a backwards
        propagation.

        Args:
            new_in_shape: The shape of the new input.
            batch_axis_mapping: The batch axis mapping between the new input
                and the current input of ``self``.

        Returns:
            A ``LiRPAWeights`` instance with ``Zero`` weights, the output shape
            of ``self``, ``new_in_shape`` as the new input shape, and the batch
            shape and batch axis mapping transformed according to
            ``batch_axis_mapping``.
        """
        new_mapping = batch_axis_mapping.chain(self.batch_axis_mapping)
        # Move batch axes that do not originate from the new input into the
        # output shape.
        # Also do this for batch axes that are created by broadcasting.
        new_mapping = new_mapping.filter_out_axis(new_mapping.broadcast_axes)
        new_batch, new_out = split_shape(self.full_out_shape, new_mapping.out_axes)
        return type(self).zeros(new_batch, new_out, new_in_shape, new_mapping)

    # --------------------------------------------------------------------------
    # Accessors
    # --------------------------------------------------------------------------

    @property
    def is_zero_weights(self):
        """Whether both weights are zero."""
        return is_zero(self.lb_weight) and is_zero(self.ub_weight)

    @property
    def full_in_shape(self) -> tuple[int, ...]:
        """The full shape of the input of this ``LiRPAWeights``."""
        return full_shape(
            self.batch_shape, self.in_shape, self.batch_axis_mapping.in_axes_set
        )

    @property
    def full_out_shape(self) -> tuple[int, ...]:
        """The full shape of the output of this ``LiRPAWeights``."""
        full_in_shape = self.full_in_shape
        out_batch_shape = tuple(
            full_in_shape[self.batch_axis_mapping.source(i)]
            for i in self.batch_axis_mapping.out_axes
        )
        return full_shape(
            out_batch_shape, self.out_shape, self.batch_axis_mapping.out_axes_set
        )

    @property
    def shape_info(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """The shape information stored by this ``LiRPAWeights``.

        Returns:
            The batch shape, output shape, and input shape.
        """
        return self.batch_shape, self.out_shape, self.in_shape

    def __iter__(self) -> Iterator[T]:
        yield self.lb_weight
        yield self.ub_weight

    # --------------------------------------------------------------------------
    # Calculating with LiRPAWeights
    # --------------------------------------------------------------------------

    def positive_parts(self) -> tuple[T, T]:
        """The positive entries of the weights. Negative entries are zeroed out."""
        return jnp.clip(self.lb_weight, min=0), jnp.clip(self.ub_weight, min=0)

    def negative_parts(self) -> tuple[T, T]:
        """The negative entries of the weights. Positive entries are zeroed out."""
        return jnp.clip(self.lb_weight, max=0), jnp.clip(self.ub_weight, max=0)

    def pos_neg_parts(self) -> tuple[T, T, T, T]:
        """The positive and negative entries of the weights.

        Returns:
            max(lb_weight, 0), max(ub_weight, 0), min(lb_weight, 0), min(ub_weight, 0)
        """
        return self.positive_parts() + self.negative_parts()

    def expand_input(self, x: T) -> T:
        """Expands ``x`` to have the same number of axes as ``self.lb_weight``.

        Adds axes of size 1 for each axis of ``out_shape`` after the batch axes.

        Args:
            x: The input to expand.
            Needs to have the shape ``(*in_batch_shape, *in_shape)``.
        """
        n_batch, n_out = len(self.batch_shape), len(self.out_shape)
        out_axes = tuple(range(n_batch, n_batch + n_out))
        return jnp.expand_dims(x, axis=out_axes)

    def __add__(self, other: "LiRPAWeights[T]") -> "LiRPAWeights[T]":
        if self.shape_info != other.shape_info:
            raise ValueError(
                f"Cannot add LiRPAWeights with incompatible shapes: "
                f"{self.shape_info=} and {other.shape_info=}."
            )
        if self.batch_axis_mapping != other.batch_axis_mapping:
            raise ValueError(
                f"Cannot add LiRPAWeights with incompatible batch axis mappings: "
                f"{self.batch_axis_mapping=} and {other.batch_axis_mapping=}."
            )
        return LiRPAWeights(
            self.lb_weight + other.lb_weight,
            self.ub_weight + other.ub_weight,
            self.batch_shape,
            self.out_shape,
            self.in_shape,
            self.batch_axis_mapping,
        )

    def __radd__(self, other: "LiRPAWeights[T]") -> "LiRPAWeights[T]":
        return self + other

    def dot(self, w: T, x: T, batch_axis_mapping: BatchAxisMapping | None = None) -> T:
        """Batched dot product of ``w`` with ``x``.

        Computes a batched tensor product ``w @ x``.
        The tensor product is batched over the leading batch axes of ``w`` and
        the batch axes of ``x``.
        It contracts the ``in_shape`` dimensions of ``w``.

        Args:
            w: The left hand side of the dot product.
                Needs to have the shape ``(*batch_shape, *out_shape, *in_shape)``.
                For example, ``self.lb_weight`` or ``self.ub_weight``.
            x: The right hand side of the dot product.
                The shape of ``x`` needs to be compatible with the batch axes
                (``batch_shape``) and input axes (``in_shape``) of ``w``,
                where the batch axes of ``x`` are specified by ``batch_axis_mapping``.
            batch_axis_mapping: The mapping between the batch axes of ``x`` and the output.
                If ``None``, ``self.batch_axis_mapping`` is used as the batch axis mapping.
        Returns:
            The dot product ``w @ x``.
            The return value may be ``Zero`` if ``w`` or ``x`` is ``Zero``.
            The return value has the batch axes placed according to ``batch_axis_mapping``.
        """
        if batch_axis_mapping is None:
            batch_axis_mapping = self.batch_axis_mapping
        return _lirpa_dot(x, w, *self.shape_info, batch_axis_mapping)

    def dot_weights(
        self, x: T, batch_axis_mapping: BatchAxisMapping | None = None
    ) -> tuple[T, T]:
        """Batched dot product of ``self.lb_weight`` and ``self.ub_weight`` with ``x``.

        Computes a batched tensor product ``self.lb_weight @ x`` and
        ``self.ub_weight @ x``.
        See ``LiRPAWeights.dot`` for more details on the tensor product.

        Args:
            x: The right hand side of the dot product.
                The shape of ``x`` needs to be compatible with the batch axes
                (``batch_shape``) and input axes (``in_shape``) of ``w``,
                where the batch axes of ``x`` are specified by ``batch_axes``.
            batch_axis_mapping: The mapping between the batch axes of ``x`` and the output.
                If ``None``, ``self.batch_axis_mapping`` is used as the batch axis mapping.
        Returns:
            The tuple of ``self.lb_weight @ x`` and ``self.ub_weight @ x``.
            Both values may be ``Zero`` if ``self.lb_weight`` or ``self.ub_weight``
            is ``Zero``.
        """
        lb_res = self.dot(self.lb_weight, x, batch_axis_mapping)
        ub_res = self.dot(self.ub_weight, x, batch_axis_mapping)
        return lb_res, ub_res

    # --------------------------------------------------------------------------
    # Working with Batch Axes
    # --------------------------------------------------------------------------

    def unbatch_axes(self, w: T, backwards_step_mapping: BatchAxisMapping) -> T:
        """Expands and moves certain batch axes of ``w`` making then non-batch axes.

        This method expands and moves batch axes
         1. that do not appear as output batch axes in
           ``backwards_step_mapping`` or
         2. are broadcasting batch axes in ``backwards_step_mapping``.

        For the first case, each of the relevant batch axes is expanded
        into two new axes of the same size as the expanded batch axis.
        Of the two newly created axes for each expanded batch axes,
        one is in the output shape and one is in the input shape.
        This turns a batch axis into a regular data axis.
        The position of the new axis in the output shape is the position indicated
        by ``self.batch_axis_mapping`` for the expanded batch axis.
        The position of the new axis in the input shape is determined similarly
        through ``self.batch_axis_mapping`.

        For the second case (batch axes created by broadcasting), ``unbatch_axes``
        moves the relevant batch axes into the output shape.

        Args:
            w: The array to transform.
                Needs to have the same shape as ``self.lb_weight``.
            backwards_step_mapping: A ``BatchAxisMapping`` describing how the batch axes
                in ``self.batch_axis_mapping`` originate.
                This mapping is from a new input to the current input of ``self``.

        Returns:
            The ``w`` array with batch axes expanded according to ``backwards_step_mapping``.
        """
        if is_zero(w):
            return Zero()

        broadcast_axes = backwards_step_mapping.broadcast_axes
        expand_axes = self.batch_axis_mapping.in_axes_set - (
            backwards_step_mapping.out_axes_set - broadcast_axes
        )

        # track the batch axes not (yet) expanded
        batch_axes = list(self.batch_axis_mapping.in_axes)
        # separately track which batch axes were not (yet) added to the in_shape
        in_batch_axes = set(self.batch_axis_mapping.in_axes_set)
        out_batch_axes = set(self.batch_axis_mapping.out_axes_set)
        n_out = len(self.out_shape)

        for in_axis in expand_axes:
            out_axis = self.batch_axis_mapping.sink(in_axis)

            # The position corresponding to in_axis in the batch shape
            batch_axis_pos = batch_axes.index(in_axis)
            out_pos = out_axis - sum(ba < out_axis for ba in out_batch_axes)
            in_pos = in_axis - sum(ba < in_axis for ba in in_batch_axes)
            out_pos += len(batch_axes) - 1  # account for the removed batch axis
            in_pos += len(batch_axes) + n_out  # here one axis removed, one added

            if in_axis in broadcast_axes:
                w = jnp.moveaxis(w, batch_axis_pos, out_pos)
                if backwards_step_mapping.source(in_axis) is not None:
                    # expanded size one axis
                    w = jnp.expand_dims(w, in_pos)
                    in_batch_axes.remove(in_axis)
                # else: batch axis not added to the in_shape, so do not remove
                # from in_batch_axes
            else:
                w = expand_as_diagonal(w, batch_axis_pos, out_pos, in_pos)
                in_batch_axes.remove(in_axis)

            out_batch_axes.remove(out_axis)
            batch_axes.remove(in_axis)
            n_out += 1

        return w

    def unbatch_weights(
        self, backwards_step_mapping: BatchAxisMapping
    ) -> "LiRPAWeights[T]":
        """Applies ``unbatch_axes`` to ``self.lb_weights`` and ``self.ub_weights``.

        See ``unbatch_axes`` for more details.

        Args:
            backwards_step_mapping: A ``BatchAxisMapping`` describing how the batch axes
                in ``self.batch_axis_mapping`` originate.
                This mapping is from a new input to the current input of ``self``.

        Returns:
            A new ``LiRPAWeights`` instance with the same weights but with batch axes
            expanded according to ``backwards_step_mapping``.
            The batch axis mapping of the returned instance is
            ``self.batch_axis_mapping`` with the expanded batch axes removed.
        """
        lb_w = self.unbatch_axes(self.lb_weight, backwards_step_mapping)
        ub_w = self.unbatch_axes(self.ub_weight, backwards_step_mapping)

        broadcast_axes = backwards_step_mapping.broadcast_axes
        expand_axes = self.batch_axis_mapping.in_axes_set - (
            backwards_step_mapping.out_axes_set - broadcast_axes
        )
        new_mapping = self.batch_axis_mapping.filter_in_axis(expand_axes)
        new_batch = batch_shape(self.full_in_shape, new_mapping.in_axes)
        new_batch1, new_out = split_shape(self.full_out_shape, new_mapping.out_axes)
        assert new_batch == new_batch1
        new_in = tuple(
            1 if i in broadcast_axes else s
            for i, s in enumerate(self.full_in_shape)
            if i not in new_mapping.in_axes_set
            and (
                i not in broadcast_axes or backwards_step_mapping.source(i) is not None
            )
        )
        return type(self)(lb_w, ub_w, new_batch, new_out, new_in, new_mapping)

    # --------------------------------------------------------------------------
    # Create LiRPABounds
    # --------------------------------------------------------------------------

    def backwards_step[U: Real[Array, "..."] | Zero](
        self,
        new_lb_weights: Sequence[U],
        new_ub_weights: Sequence[U],
        new_lb_bias: U,
        new_ub_bias: U,
        domain: Sequence[Bounds[T]],
        batch_axis_mappings: Sequence[BatchAxisMapping],
    ) -> LiRPABounds[U]:
        """Creates a ``LiRPABounds`` instance for a result of propagating ``self``
        backwards using the shape information of ``self``.

        Depending on the ``batch_axis_mappings``, the new weights may have
        fewer leading batch axes than the current weights.
        This is the case for batch axes created by broadcasting and batch axes
        originating from other variables than the new input.
        The ``new_batch_shape`` is determined from the current batch shape
        and the ``batch_axis_mappings``.

        Args:
            new_lb_weights: The new lower bound weights.
                Each element of ``new_lb_weights`` has the shape
                ``(*new_batch_shape, *out_shape, *new_in_shape)``.
            new_ub_weights: The new upper bound weights.
                The weights need to have the same shape as ``new_lb_weights``.
            new_lb_bias: The new lower bound bias.
                Needs to have the shape ``full_out_shape``.
            new_ub_bias: The new upper bound bias.
                Needs to have the same shape as ``new_lb_bias``.
            domain: The domains of the inputs for the new weights.
            batch_axis_mappings: Batch axis mappings from each of the new inputs
                to the current input.
                This batch axis mapping determines the ``new_batch_shape``.
        """
        full_out_shape = self.full_out_shape
        new_full_in_shapes = tuple(x.shape for x in example_args(domain))
        new_batch_axis_mappings = tuple(
            batch_axis_mapping.chain(self.batch_axis_mapping)
            for batch_axis_mapping in batch_axis_mappings
        )
        return LiRPABounds.from_arrays(
            new_lb_weights,
            new_ub_weights,
            new_lb_bias,
            new_ub_bias,
            domain,
            new_full_in_shapes,
            full_out_shape,
            new_batch_axis_mappings,
        )


def _lirpa_dot(
    x: jax.Array | Zero,
    weight: jax.Array | Zero,
    batch_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    in_shape: tuple[int, ...],
    batch_axis_mapping: BatchAxisMapping,
):
    """Computes a batched affine transformation with LiRPA weights.

    Simplifying the batch handling a little bit, this function computes

        jnp.einsum("ij,ijk->ik", x, weight) + bias,

    where ``i`` are batch axes, ``j`` signifies the input shape that is contracted
    and ``k`` signifies the output shape.
    In general, only ``weight`` needs to have learning batch axes.
    Both ``x`` and the output can have batch axes at any position.
    These positions are specified by ``batch_axis_mapping``.

    Args:
        x: The input to the affine transformation.
            The shape of ``x`` is ``full_in_shape``,
            that is ``batch_shape`` combined with ``in_shape``
            according to ``batch_axis_mapping``.
        weight: The weight of the affine transformation.
            The shape of ``weight`` is ``(*batch_shape, *out_shape, *in_shape)``.
        batch_shape: The shape of the leading batch axes of ``weight``.
        out_shape: The shape of the "output", that is, the result of multiplying
            the weights with an input, without batch axes.
        in_shape: The shape of the input without batch axes and the
            "``other_shape``" part.
        batch_axis_mapping: The batch axis mapping between the input ``x`` and
            output of this function.

    Returns:
        The return value has the shape ``full_out_shape``, that is,
        ``batch_shape`` combined with ``out_shape`` according to
        ``batch_axis_mapping``.
    """
    if is_zero(weight) or is_zero(x):
        return Zero()

    batch_size, out_size, in_size = len(batch_shape), len(out_shape), len(in_shape)
    assert len(x.shape) == batch_size + in_size
    assert len(weight.shape) == batch_size + out_size + in_size

    # The position of an axis in weight is the reference index for einsum
    w_axes = list(range(len(weight.shape)))

    # pick a batch axis index or an in_shape index for x
    ba_iter = iter(range(len(batch_shape)))
    in_iter = iter(range(batch_size + out_size, len(weight.shape)))
    x_batch_axes = batch_axis_mapping.in_axes
    x_axes = [
        next(ba_iter) if i in x_batch_axes else next(in_iter)
        for i in range(batch_size + in_size)
    ]

    # for the output, the batch axes may be reordered
    out_iter = iter(range(batch_size, batch_size + out_size))
    out_batch_axes = batch_axis_mapping.out_axes_set
    out_axes = [
        (
            x_batch_axes.index(batch_axis_mapping.source(j))
            if j in out_batch_axes
            else next(out_iter)
        )
        for j in range(batch_size + out_size)
    ]

    return jnp.einsum(x, x_axes, weight, w_axes, out_axes)


def incorporate_batch_axes[T](
    a: T, shape_parts: tuple[int, int, int], batch_axes: Sequence[int]
) -> T:
    """Place leading batch axes at intended actual positions.

    For example, moves the leading batch axes of a LiRPA weight array into
    the ``in_shape`` part of the weight shape.

    If ``a`` has the shape ``(*batch_axes, *additional_shape, *in_shape)``,
    where additional_shape is arbitrary and ``batch_axes`` has the same
    length as the ``batch_axes`` argument, this function returns an array of shape
    ``(*additional_shape, *full_in_shape)``, where ``full_in_shape`` includes
    both ``in_shape`` and ``batch_axes``, with the batch axes placed at
    the positions specified by the ``batch_axes`` argument.

    Args:
        a: The array whose batch axes to rearrange.
           Needs to have leading batch axes.
        shape_parts: A three tuple of the number of leading batch axes,
            a number of additional shape elements, and, finally,
            the number of input axes in which to place the batch axes.
        batch_axes: The axis indices of the batch axes in the full
            input shape (including the batch axes).
            The leading batch axes of ``a`` are placed in the return value
            in the order specified by ``batch_axes``.
            For example, if batch_axes is ``(3, 1)`` first leading batch
            axis is placed at position 3 in the return value and the
            second leading batch axis is placed at position 1.

    Returns:
        The array ``a`` with rearranged batch axes.
        If ``a`` is ``Zero``, returns ``Zero``.
    """
    if is_zero(a):
        return a

    n_batch_axes, n_additional, n_in_axes = shape_parts
    n_prefix = n_batch_axes + n_additional
    assert len(a.shape) == n_prefix + n_in_axes

    batch_idx = tuple(range(n_batch_axes))
    in_iter = iter(range(n_prefix, n_prefix + n_in_axes))
    additional_axes = tuple(range(n_batch_axes, n_prefix))
    perm = additional_axes + tuple(
        batch_idx[batch_axes.index(i)] if i in batch_axes else next(in_iter)
        for i in range(n_batch_axes + n_in_axes)
    )
    return jnp.permute_dims(a, perm)


def pull_batch_axes[T](
    a: T, shape_parts: tuple[int, int, int], batch_axes: Sequence[int]
) -> T:
    """Move batch axes to the front.

    For example, restore leading batch axes after calling ``incorporate_batch_axes``
    on a LiRPA weight array.

    If ``a`` has the shape ``(*additional_shape, *full_in_shape)``,
    where additional_shape is arbitrary and ``full_batch_axes`` is the shape
    of the input including batch axes, this function returns an array of shape
    ``(*batch_axes, *additional_shape, *in_shape)``, where ``batch_axes``
    and ``in_shape`` are the ``full_in_shape`` when combined according to
    the ``batch_axes`` argument.

    Args:
        a: The array whose batch axes to rearrange.
            The shape of ``a`` is ``(*additional_shape, *full_in_shape)``,
            where ``full_in_shape`` contains the batch axes.
            If ``shape_parts`` is ``(n, m, k)``, the size of ``additional_shape``
            is ``m`` and the size of ``full_in_shape`` is ``n + k``.
        shape_parts: The shape parts of the return value of this function,
            as for ``incorporate_batch_axes``.
        batch_axes: The axis indices of the batch axes in the ``full_in_shape``
            part of ``a``.

    Returns:
        The array ``a`` with leading batch axes.
        If ``a`` is ``Zero``, returns ``Zero``.
    """
    if is_zero(a):
        return a

    n_batch_axes, n_out_axes, n_in_axes = shape_parts
    n_suffix = n_batch_axes + n_in_axes
    assert len(a.shape) == n_out_axes + n_suffix

    perm = (
        tuple(n_out_axes + i for i in batch_axes)
        + tuple(range(n_out_axes))
        + tuple(n_out_axes + i for i in range(n_suffix) if i not in batch_axes)
    )
    return jnp.permute_dims(a, perm)


# ======================================================================================
# Utils
# ======================================================================================


LiRPAWeightsInfo = tuple[
    bool,
    bool,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    BatchAxisMapping,
]
