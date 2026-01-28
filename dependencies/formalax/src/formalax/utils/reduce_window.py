#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from collections.abc import Sequence

import jax
import jax.core
import jax.numpy as jnp
import jax.tree
from jax import lax
from jax.interpreters import ad

__all__ = (
    "select_and_scatter_add2",
    "reduce_window_conv_transpose",
    "reduce_window_patches",
)


def select_and_scatter_add2(
    operand: jax.Array,
    source: jax.Array,
    *,
    window_dimensions: tuple[int, ...],
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    base_dilation: Sequence[int],
    window_dilation: Sequence[int],
) -> jax.Array:
    """
    The _select_and_gather_add_transpose from
    https://github.com/google/jax/blob/5e418f5ab2692d4791816e85ed82eb0834a579cb/jax/_src/lax/windowed_reductions.py#L962
    This is a select_and_scatter_add operation, adapted to work with
    a base_dilation.

    See also https://openxla.org/xla/operation_semantics#selectandscatter

    Used by _backwards_crown_reduce_window_max_rule.

    Example:

        >>> import jax.numpy as jnp
        >>> from jax import lax
        >>> from formalax.utils.reduce_window import select_and_scatter_add2
        >>> x = jnp.arange(100).reshape((10, 10))
        >>> x
        Array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
               [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
               [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
               [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
               [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
               [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
               [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
               [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]], dtype=int32)
        >>> kws = dict(
        ...     window_dimensions=(3, 3),
        ...     window_strides=(2, 2),
        ...     padding=((0, 0), (0, 0)),
        ...     base_dilation=(1, 1),
        ...     window_dilation=(1, 1),
        ... )
        >>> lax.reduce_window_max_p.bind(x, **kws)
        Array([[22, 24, 26, 28],
               [42, 44, 46, 48],
               [62, 64, 66, 68],
               [82, 84, 86, 88]], dtype=int32)
        >>> y = jnp.arange(1, 17).reshape((4, 4))
        >>> y
        Array([[ 1,  2,  3,  4],
               [ 5,  6,  7,  8],
               [ 9, 10, 11, 12],
               [13, 14, 15, 16]], dtype=int32)
        >>> select_and_scatter_add2(x, **kws)
        Array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  1,  0,  2,  0,  3,  0,  4,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  5,  0,  6,  0,  7,  0,  8,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  9,  0, 10,  0, 11,  0, 12,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0, 13,  0, 14,  0, 15,  0, 16,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]], dtype=int32)
    """
    select_prim = lax.ge_p
    tangents = ad.UndefinedPrimal(jax.core.get_aval(operand))
    try:
        # the init value for the output array is 0 implicitly
        return ad.primitive_transposes[lax.select_and_gather_add_p](
            source,
            tangents,
            operand,
            select_prim=select_prim,
            window_dimensions=window_dimensions,
            window_strides=window_strides,
            padding=padding,
            base_dilation=base_dilation,
            window_dilation=window_dilation,
        )[0]
    except NotImplementedError as ex:
        raise NotImplementedError(
            f"Backwards CROWN not implemented for reduce_window_max (MaxPool) "
            f"with window dilation. Got {window_dilation=}."
        ) from ex


def reduce_window_conv_transpose(
    operand: jax.Array,
    kernel: jax.Array,
    source: jax.Array,
    *,
    window_dimensions: tuple[int, ...],
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    base_dilation: Sequence[int],
    window_dilation: Sequence[int],
) -> jax.Array:
    """
    This function is a transposed convolution that allows distributing values
    in the output shape of a reduce_window operation back to the input while giving
    different weights for different input positions.
    It uses a kernel of the size of the window that contains these weights.

    Concretely, this function provides the transpose of a convolution with a
    batch size of 1 and input and output feature sizes of 1.

    Used by _backwards_crown_reduce_window_max_rule.

    Example:
        >>> import jax.numpy as jnp
        >>> from formalax.utils.reduce_window import reduce_window_conv_transpose
        >>> x = jnp.empty((4, 4), dtype=jnp.int32)  # can have any value
        >>> k1 = y = jnp.arange(4).reshape((2, 2))
        >>> reduce_window_conv_transpose(
        ...     x,
        ...     k1,
        ...     y,
        ...     window_dimensions=(2, 2),
        ...     window_strides=(2, 2),
        ...     padding=((0, 0), (0, 0)),
        ...     base_dilation=(1, 1),
        ...     window_dilation=(1, 1),
        ... )
            Array([[[[0, 0, 0, 1],
                     [0, 0, 2, 3],
                     [0, 2, 0, 3],
                     [4, 6, 6, 9]]]], dtype=int32)
        >>> # one kernel for each output element/window position
        >>> k2 = jnp.arange(16).reshape((2, 2, 2, 2))
        >>> reduce_window_conv_transpose(
        ...     x,
        ...     k2,
        ...     y,
        ...     window_dimensions=(2, 2),
        ...     window_strides=(2, 2),
        ...     padding=((0, 0), (0, 0)),
        ...     base_dilation=(1, 1),
        ...     window_dilation=(1, 1),
        ... )
            Array([[[[ 0,  0,  4,  5],
                     [ 0,  0,  6,  7],
                     [16, 18, 36, 39],
                     [20, 22, 42, 45]]]], dtype=int32)

    Args:
        operand: The input to the reduce_window operation.
            This argument is only used to construct an abstract value.
            The data in this argument is not used.
        kernel: The weights for distributing the ``source`` values to the input.
        source: The values to distribute to the input.
        window_dimensions: See ``lax.reduce_window``.
        window_strides: See ``lax.reduce_window``.
        padding: See ``lax.reduce_window``.
        base_dilation: See ``lax.reduce_window``.
        window_dilation: See ``lax.reduce_window``.
    """
    if len(kernel.shape) == len(window_dimensions):
        # one kernel for all windows
        # conv arguments always has a batch dimension and a feature dimension
        kernel = kernel.reshape((1, 1, *kernel.shape))

        def conv(x):
            dim_idx = tuple(range(len(window_dimensions) + 2))
            return lax.conv_general_dilated(
                x,
                kernel,
                window_strides=window_strides,
                padding=padding,
                dimension_numbers=lax.ConvDimensionNumbers(
                    lhs_spec=dim_idx, rhs_spec=dim_idx, out_spec=dim_idx
                ),
                batch_group_count=1,
                feature_group_count=1,
                precision=None,
                preferred_element_type=None,
            )

    elif len(kernel.shape) == len(window_dimensions) + len(source.shape):
        # one kernel for each window position
        # => use conv_general_dilated_local
        # Kernel shape: (1, prod(window_dimensions), *source.shape)
        kernel = kernel.reshape((1, *source.shape, -1))
        kernel = jnp.moveaxis(kernel, -1, 1)

        def conv(x):
            return lax.conv_general_dilated_local(
                x,
                kernel,
                filter_shape=window_dimensions,
                window_strides=window_strides,
                padding=padding,
                lhs_dilation=base_dilation,
                rhs_dilation=window_dilation,
            )
    else:
        raise ValueError(
            f"The reduce_window_max parameter has invalid shape: {kernel.shape}. "
            f"Required is either {window_dimensions} (external-shared) or "
            f"{source.shape + window_dimensions} (external-full)."
        )

    def conv_(x):
        x = x.reshape((1, 1, *x.shape))
        y = conv(x)
        return y.reshape(*source.shape)

    return jax.linear_transpose(conv_, operand)(source)[0]


def reduce_window_patches(
    operand: jax.Array,
    *,
    window_dimensions: tuple[int, ...],
    window_strides: Sequence[int],
    padding: Sequence[tuple[int, int]],
    pad_value: jax.Array,
    base_dilation: Sequence[int],
    window_dilation: Sequence[int],
) -> jax.Array:
    """
    Constructs a matrix containing the contents of the moving window (patches)
    for each output position that are aggregated by pooling.
    The shape of the output is `(*out_shape, prod(window_shape))`.

    This function is a convenice wrapper around ``jax.conv_general_dilated_patches``.
    Used by _backwards_crown_reduce_window_max_rule.

    Example:
        >>> import jax.numpy as jnp
        >>> from formalax.utils.reduce_window import reduce_window_patches
        >>> x = jnp.arange(16).reshape((4, 4))
        >>> x
        Array([[[[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11],
                 [12, 13, 14, 15]]]], dtype=int32)
        >>> reduce_window_patches(
        ...     x,
        ...     window_dimensions=(2, 2),
        ...     window_strides=(1, 1),
        ...     padding=((0, 0), (0, 0)),
        ...     pad_value=-jnp.inf,
        ...     base_dilation=(1, 1),
        ...     window_dilation=(1, 1),
        ... )
        Array([[[[ 0,  1,  4,  5],
                [ 1,  2,  5,  6],
                [ 2,  3,  6,  7]],

               [[ 4,  5,  8,  9],
                [ 5,  6,  9, 10],
                [ 6,  7, 10, 11]],

               [[ 8,  9, 12, 13],
                [ 9, 10, 13, 14],
                [10, 11, 14, 15]]]], dtype=int32)

    Args:
        operand: The input to the reduce_window operation.
        window_dimensions: See ``lax.reduce_window``.
        window_strides: See ``lax.reduce_window``.
        padding: See ``lax.reduce_window``.
        pad_value: The value to pad with.
            Important: this function will produce `nan` values if `pad_value`
            is infinite.
        base_dilation: See ``lax.reduce_window``.
        window_dilation: See ``lax.reduce_window``.
    """
    operand = lax.pad(operand, pad_value, [(lo, hi, 0) for lo, hi in padding])

    operand = jnp.expand_dims(operand, axis=(0, 1))
    out = lax.conv_general_dilated_patches(
        operand,
        filter_shape=window_dimensions,
        window_strides=window_strides,
        padding=[(0, 0)] * len(padding),
        lhs_dilation=base_dilation,
        rhs_dilation=window_dilation,
    )
    return jnp.moveaxis(out, 1, -1).squeeze(0)
