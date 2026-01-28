#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import typing
from collections.abc import Sequence
from functools import partial
from types import EllipsisType
from typing import Callable, Protocol

import jax
import jax._src.api_util as jax_api_util
import jax._src.linear_util as lu
import jax._src.sharding_impls as jax_sharding_impls
import jax.core
import jax.interpreters.partial_eval as pe
import jax.numpy as jnp
import jax.tree
from frozendict import frozendict
from jax import lax
from jax.core import Atom
from jax.extend.core import (
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Literal,
    Primitive,
    Var,
    primitives,
)
from jax.interpreters import ad
from jaxtyping import Array, PyTree, PyTreeDef, Real

from ...core.batch_axes import BatchAxisGraph, BatchAxisMapping, infer_batch_axes
from ...core.markers import Marker, markup_primitive
from ...core.zero import Zero, is_zero
from ...utils.batch_axes import (
    canonicalize_batch_axes,
    non_batch_shape,
    split_shape,
)
from ...utils.caching import HashablePartial
from ...utils.fun_to_jaxpr import fun_to_jaxpr_for_bounding
from ...utils.linear_util import args_to_kwargs
from ...utils.name_stack import eqn_name_stack_ctx, transform_name_stack_ctx
from ...utils.zip import strict_zip
from ._affinebounds import AffineBounds
from ._bounds import (
    Bounds,
    all_as_bounds,
    example_args,
    is_bounds,
)
from ._ibp import ibp_rule_compare_greater, ibp_rule_compare_less
from ._lirpabounds import (
    LiRPABounds,
    LiRPAWeights,
    incorporate_batch_axes,
    pull_batch_axes,
)

__all__ = (
    "backwards_lirpa",
    "BackwardsLiRPARule",
    "LiRPACallable",
    "ComputeBoundsCallable",
    "COMPUTE_BOUNDS",
    "register_backwards_lirpa_rule",
    "nonlinear_backwards_lirpa_rule",
    "transpose_as_backwards_lirpa_rule_unary",
    "transpose_as_backwards_lirpa_rule_bilinear",
    "transpose_as_backwards_lirpa_rule_binary_left_only",
    "constant_bounds_backwards_lirpa_rule",
    "backwards_lirpa_rule_jaxpr",
    "fun_to_jaxpr_for_lirpa",
    "_single_output_asserts",
    "_reshape_as_output",
    "_restore_weights_shape",
)


# ==============================================================================
# Interface Utils
# ==============================================================================


def fun_to_jaxpr_for_lirpa(lirpa_name: str, with_params_args: bool = False):
    """A decorator that converts functions to jaxprs for LiRPA functions.

    The function after decoration accepts both jaxprs and functions as argument.

    The decorated function needs to accept a jaxpr as first argument.
    This decorator will check whether a passed first argument is a function
    or a jaxpr.
    If it is a jaxpr, it passes it on directly,
    If it is a function, the function is traced and the resulting jaxpr is given
    to the decorated function.

    Args:
        lirpa_name: The name of the LiRPA procedure that this decorator decorates.
            For example ``"backwards LiRPA"`` or ``CROWN-IBP``.
        with_params_args: Whether the return value of the wrapped LiRPA function
            has a leading ``params`` argument that should be disregared when converting
            functions to jaxprs.
    """
    # jax_util.wraps docstring argument:
    # {{fun}} placeholder is the function name; {{doc}} is the original docstring.
    docstring = (
        f"Computes output lower and upper bounds of {{fun}} using {lirpa_name}.\n\n\n"
        f"Original documentation of {{fun}}\n"
        f"--------------------------------------------------------------------------------\n"
        f"\n {{doc}}"
    )
    return fun_to_jaxpr_for_bounding(lirpa_name, docstring, with_params_args)


# ==============================================================================
# Interface
# ==============================================================================


class LiRPACallable(Protocol):
    """Type signature of a backwards LiRPA callable.

    This is the type signature of the functions created by
    ``backwards_lirpa``.
    Each ``LiRPACallable`` is connected to a ``jaxpr``,
    on which it performs Linear Relaxation Perturbation Analysis
    (LiRPA)
    See ``backwards_lirpa`` for more details.
    """

    def __call__(
        self, params: PyTree[Real[Array, "..."]], *args: Bounds | Real[Array, "..."]
    ) -> tuple[AffineBounds[Real[Array, "..."] | Zero], ...]:
        """
        Performs Linear Relaxation Perturbation Analysis (LiRPA) on the
        ``jaxpr`` encapsulated by this ``LiRPACallable`` using the input
        bounds and concrete arguments provided as ``args``.

        Args:
            params: Any parameters of the LiRPA procedure.
            *args: Constant input bounds and concrete arguments for which to perform
                backwards bound propagation.
                Each element of ``args`` corresponds to one input variable of ``jaxpr``.

        Returns:
            Returns a tuple of affine bounds, one for each output variable of ``jaxpr``.
            Each affine bound is a pair of an affine lower bound and an
            affine upper bound.
            Each affine bound is an affine function from the bounded elements
            of ``*args`` (the elements of ``*args`` that are ``Bounds`` instances)
            to the value of the corresponding output variable.
        """
        ...


class ComputeBoundsCallable(Protocol):
    """Type signature of a function that computes intermediate bounds for LiRPA.

    This is the type signature of the ``compute_bounds`` argument of
    ``backwards_lirpa``.
    """

    def __call__(
        self,
        jaxpr: ClosedJaxpr,
        params: PyTree[Real[Array, "..."]],
        *args: Bounds | Real[Array, "..."],
        in_batch_axes: tuple[tuple[int, ...], ...] | EllipsisType,
        out_batch_axes: tuple[tuple[int, ...], ...] | EllipsisType,
        batch_axis_graph: BatchAxisGraph,
    ) -> dict[Var, Bounds | Real[Array, "..."]]:
        """
        Computes bounds on the intermediate values of ``jaxpr``.

        Args:
            jaxpr: The ``jaxpr`` to compute bounds on.
            params: Any parameters of the LiRPA procedure.
            *args: Constant input bounds and concrete arguments
                Each element of ``args`` corresponds to one input variable of ``jaxpr``.
            in_batch_axes: The batch axes of the inputs of ``jaxpr``.
            out_batch_axes: The batch axes of the outputs of ``jaxpr``.
            batch_axis_graph: The batch axis graph of ``jaxpr``.

        Returns:
            Returns a dictionary mapping each input and intermediate variable of
            ``jaxpr`` to its bounds.
            The bounds are either ``Bounds`` instances or ``jax.Arrays``.
        """
        ...


COMPUTE_BOUNDS = ComputeBoundsCallable | typing.Literal["bootstrap"]


class BackwardsLiRPARule(Protocol):
    """Type signature of a backwards LiRPA rule.

    Any callable (including a function) that has the type signature of
    ``BackwardsLiRPARule`` can be used as a primitive rule for backwards LiRPA.
    Use ``register_backwards_lirpa_rule`` to add a new backwards LiRPA rule for
    a primitive.
    """

    def __call__(
        self,
        params: PyTree[Real[Array, "..."]],
        out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        *,
        full_in_shapes: Sequence[tuple[int, ...]],
        full_out_shapes: Sequence[tuple[int, ...]],
        in_batch_axes: Sequence[tuple[int, ...]],
        out_batch_axes: Sequence[tuple[int, ...]],
        batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
        backwards_lirpa: Callable[
            [
                ClosedJaxpr,
                Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
                Bounds | jax.Array,
                ...,
            ],
            tuple[LiRPABounds[Real[Array, "..."] | Zero], ...],
        ],
        **kwargs,
    ) -> (
        tuple[LiRPABounds[Real[Array, "..."] | Zero], ...]
        | LiRPABounds[Real[Array, "..."] | Zero]
    ):
        """Back-propagates LiRPA output weights through a primitive.

        Args:
            params: Parameters of this LiRPA rule.
                If there are no parameters for this rule, ``params`` is ``None``.
            out_weights: LiRPA weights on the output of the primitive to propagate.
            in_bounds: Constant lower and upper bounds on the inputs of the primitive.
                Individual input bounds may also be ``jax.Arrays``,
                as for ``out_bounds``.
            full_in_shapes: The input shapes the primitive, including batch axes.
            full_out_shapes: The output shapes the primitive, including batch axes.
            in_batch_axes: The batch axes of the inputs.
            out_batch_axes: The batch axes of the outputs.
            batch_axis_mappings: How the batch axes of each input are mapped to the
                batch axes of the outputs.
                If the primitive has ``n`` inputs and ``m`` outputs,
                ``batch_axes`` is a ``m``-sequence of ``n``-sequences of
                batch axis mappings.
                The ``(j, i)``-th batch axis mapping is the mapping from the ``i``-th
                input to the ``j``-th output.
            backwards_lirpa: A callable for performing backwards LiRPA on a Jaxpr.
                The signature is as for ``formalax.bounds.backwards_lirpa``,
                except that  ``compute_bounds``, ``get_params``, and ``nonlinear_rules``
                were already specified.
                Additionally, ``backwards_lirpa`` also requires LiRPA weights
                for the output of the Jaxpr.
            **kwargs: Any keyword arguments of the primitive.

        Returns:
            LiRPA weights and biases for each input of the primitive.
            If there is just a single input, may also return a single ``LiRPABounds``
            instance instead of a length one tuple.
        """
        ...


def _all_params_none(eqn, params):
    return None


@fun_to_jaxpr_for_lirpa(lirpa_name="backwards LiRPA", with_params_args=True)
def backwards_lirpa(
    jaxpr: ClosedJaxpr,
    nonlinear_rules: frozendict[Primitive, BackwardsLiRPARule],
    compute_bounds: COMPUTE_BOUNDS,
    get_params: Callable[
        [JaxprEqn, PyTree[Real[Array, "..."]]], PyTree[Real[Array, "..."]]
    ] = _all_params_none,
    in_batch_axes: int | None | tuple[int | None | tuple[int, ...], ...] = ...,
    out_batch_axes: int | None | tuple[int | None | tuple[int, ...], ...] = ...,
) -> LiRPACallable:
    """Creates a function that perform backwards LiRPA on ``jaxpr``.

    The returned function performs Linear Relaxation Perturbation Analysis
    (LiRPA) on ``jaxpr``.
    LiRPA [Xu et al., 2020] propagates linear functions backwards through a
    computational graph (here given as a ``jaxpr``) to compute a linear lower
    bound and a linear upper bounds on the computational graph output.

    The returned function takes bounds on the input variables of ``jaxpr``
    and returns affine bounds for the output variables of ``jaxpr``.
    See ``LiRPACallable`` for more details.

    [Xu et al., 2020]: Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang,
        Minlie Huang, Bhavya Kailkhura, Xue Lin, Cho-Jui Hsieh: Automatic Perturbation
        Analysis for Scalable Certified Robustness and Beyond. NeurIPS 2020

    Args:
        jaxpr: The ``jaxpr`` to propagate bounds through.
            If ``jaxpr`` is batched, also specify ``out_batch_axes``.
        nonlinear_rules: The rules used for bounding non-linear functions.
            This dictionary determines, for example, the rule used for propagating
            linear functions through ReLU.
        compute_bounds: A callable that computes preliminary bounds on each
            intermediate variable of a Jaxpr.
            This callable is also used as a key for caching.
            For this to work reliably, you should define this function
            at the top level, so that it has a consistent hash.
        get_params: Obtains the parameters of a Jaxpr equation from the ``params``
            argument of the `LiRPACallable` this function returns.
            The default value discards all parameters.
        in_batch_axes: An integer, ``None``, or tuple specifying which axes of
            each input of ``jaxpr`` is a batch axis.
            If no ``in_batch_axes`` are specified, the batch axes are inferred using
            ``infer_batch_axes``.

            If ``in_batch_axes`` is specified as a tuple, it specifies a separate batch
            dimension for each element of ``args``.
            Here, a value of ``None`` means that an element of ``args`` does not have
            a batch dimension, while an integer ``i`` means that the element of ``args``
            has batch dimension ``i``.
            An element of the ``in_batch_axes`` tuple may also be a tuple of integers,
            which means that the corresponding element of ``args`` has multiple batch
            axes.
            Typically, elements of ``args`` that are not ``Bounds`` should have no
            batch dimension (``None``).

            For example, if ``args`` has three elements,
            ``in_batch_axes=(None, 1, (2, 3))`` means that the first element of
            ``args`` has no batch axis, while axis one is the batch axis for the
            second element and axes two and three are the batch axis of the
            third element.

            Specifying ``in_batch_axes=None`` signifies that no element of ``args``
            is batched.
            If ``in_batch_axes`` is a single integer ``i``, ``backwards_lirpa``
            assumes that all ``Bounds`` elements of ``*args`` have ``i`` as
            their batch dimension, while all other elements have
            no batch dimension (``None``).
        out_batch_axes: An integer, ``None``, or tuple specifying which axes of
            each output of ``jaxpr`` is a batch axis.
            If no ``out_batch_axes`` are specified, the batch axes are inferred using
            ``infer_batch_axes``.
            The format of ``out_batch_axes`` is as for ``in_batch_axes``.

    Returns:
        A function that performs backwards LiRPA on ``jaxpr``.
    """

    def fun_backwards_lirpa(
        params: PyTree[Real[Array, "..."]],
        *args: Bounds | jax.Array,
    ) -> tuple[LiRPABounds[Real[Array, "..."] | Zero], ...]:
        in_batch_axes_ = canonicalize_batch_axes(
            in_batch_axes, [is_bounds(a) for a in args]
        )
        out_batch_axes_ = canonicalize_batch_axes(
            out_batch_axes, [True] * len(jaxpr.out_avals)
        )
        batch_axis_graph = infer_batch_axes(jaxpr, in_batch_axes_, out_batch_axes_)

        in_batch_axes_ = tuple(batch_axis_graph[var] for var in jaxpr.jaxpr.invars)
        out_batch_axes_ = tuple(batch_axis_graph[var] for var in jaxpr.jaxpr.outvars)

        return _backwards_lirpa(
            jaxpr,
            params,
            [
                LiRPAWeights.identity_for(aval, batch_ax)
                for aval, batch_ax in strict_zip(jaxpr.out_avals, out_batch_axes_)
            ],
            *args,
            in_batch_axes=in_batch_axes_,
            out_batch_axes=out_batch_axes_,
            nonlinear_rules=nonlinear_rules,
            compute_bounds=compute_bounds,
            get_params=get_params,
            batch_axis_graph=batch_axis_graph,
        )

    fun_backwards_lirpa.__doc__ = LiRPACallable.__doc__
    return fun_backwards_lirpa


# ==============================================================================
# Implementation
# ==============================================================================


def _backwards_lirpa(
    jaxpr: ClosedJaxpr,
    params: PyTree[Real[Array, "..."]],
    out_weights: Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
    *args: Bounds | jax.Array,
    in_batch_axes: int | None | tuple[int | None | tuple[int, ...], ...] = ...,
    out_batch_axes: int | None | tuple[int | None | tuple[int, ...], ...] = ...,
    nonlinear_rules: frozendict[Primitive, BackwardsLiRPARule],
    compute_bounds: COMPUTE_BOUNDS,
    get_params: Callable[
        [JaxprEqn, PyTree[Real[Array, "..."]]], PyTree[Real[Array, "..."]]
    ],
    batch_axis_graph: BatchAxisGraph | None = None,
    transform_stack: bool = True,
) -> tuple[LiRPABounds[Real[Array, "..."] | Zero], ...]:
    """Backwards LiRPA implementation entry point.

    Computes initial bounds and runs ``_backwards_lirpa_one_output` for each output
    variable.

    Args:
        jaxpr: The jaxpr on which to compute bounds.
        params: Parameters of the LiRPA procedure.
        outvar_weights: The weights of ``outvar`` to start propagation with.
        *args: ``Bounds`` and concrete input arrays for the input variables of ``jaxpr``.
        in_batch_axes: See ``backwards_lirpa``.
        out_batch_axes: See ``backwards_lirpa``.
        nonlinear_rules: See ``backwards_lirpa``.
        compute_bounds: How to compute preliminary bounds on the intermediate variables of
            ``jaxpr``.
            If ``compute_bounds`` is ``"bootstrap"``, `_backwards_lirpa` is called
            recursively to produce the intermediate bounds.
            Otherwise, `compute_bounds` is a function that takes a Jaxpr and input bounds
            and returns bounds on the intermediate variables of ``jaxpr``.
        get_params: See ``backwards_lirpa``.
        batch_ais_graph: An optional ``BatchAxisGraph`` of ``jaxpr``.
            If omitted, this graph is inferred using ``infer_batch_axes``
            using ``in_batch_axes`` and ``out_batch_axes``.
        transform_stack: Whether to transform the jax name stack.
    """
    in_batch_axes = canonicalize_batch_axes(in_batch_axes, [is_bounds(a) for a in args])
    out_batch_axes = canonicalize_batch_axes(
        out_batch_axes, [True] * len(jaxpr.out_avals)
    )
    # We also require batch axes for intermediate values.
    if batch_axis_graph is None:
        batch_axis_graph = infer_batch_axes(jaxpr, in_batch_axes, out_batch_axes)

    # intermediate initial bounds are the same for all outputs
    match compute_bounds:
        case "bootstrap":
            init_bounds = _bootstrap_intermediate_bounds(
                jaxpr,
                params,
                *args,
                in_batch_axes=in_batch_axes,
                out_batch_axes=out_batch_axes,
                nonlinear_rules=nonlinear_rules,
                get_params=get_params,
                batch_axis_graph=batch_axis_graph,
                transform_stack=transform_stack,
            )
        case _:
            init_bounds = compute_bounds(
                jaxpr,
                params,
                *args,
                in_batch_axes=in_batch_axes,
                out_batch_axes=out_batch_axes,
                batch_axis_graph=batch_axis_graph,
            )

    return tuple(
        (
            _backwards_lirpa_one_output(
                jaxpr,
                params,
                out_var,
                ow,
                init_bounds,
                nonlinear_rules,
                compute_bounds,
                get_params,
                batch_axis_graph,
                transform_stack,
            )
            if not ow.is_zero_weights
            else LiRPABounds.zero_bounds(
                all_as_bounds(*args),
                [x.shape for x in example_args(args)],
                ow.full_out_shape,
                [batch_axis_graph.mapping(v, out_var) for v in jaxpr.jaxpr.invars],
            )
        )
        for out_var, ow in strict_zip(jaxpr.jaxpr.outvars, out_weights)
    )


# global function for hashing/caching purposes
def _get_bounds_bootstrap(
    already_computed: dict[Var, Bounds | jax.Array],
    bootstrap_fun: Callable[
        [ClosedJaxpr, Bounds | jax.Array, ...], dict[Var, Bounds | jax.Array]
    ],
    jaxpr: ClosedJaxpr,
    params: PyTree[Real[Array, "..."]],
    *args: Bounds | jax.Array,
    **kwargs,
):
    if jaxpr.jaxpr.invars[0] not in already_computed:
        # _get_bounds_bootstrap is called on an embedded Jaxpr
        # => recursive call to _bootstrap_intermediate_bounds
        return bootstrap_fun(jaxpr, params, *args, **kwargs)

    return already_computed


class _HashByKeysDict(frozendict):
    def __hash__(self):
        return hash(tuple(self.keys()))


def _bootstrap_intermediate_bounds(
    jaxpr: ClosedJaxpr,
    params: PyTree[Real[Array, "..."]],
    *args: Bounds | jax.Array,
    in_batch_axes: tuple[tuple[int, ...], ...] | EllipsisType,
    out_batch_axes: tuple[tuple[int, ...], ...] | EllipsisType,
    nonlinear_rules: frozendict[Primitive, BackwardsLiRPARule],
    get_params: Callable[
        [JaxprEqn, PyTree[Real[Array, "..."]]], PyTree[Real[Array, "..."]]
    ],
    batch_axis_graph: BatchAxisGraph,
    transform_stack: bool = True,
) -> dict[Var, Bounds | jax.Array]:
    full_jaxpr = jaxpr.jaxpr
    consts = jaxpr.consts

    bootstrap_fun = HashablePartial(
        _bootstrap_intermediate_bounds,
        nonlinear_rules=nonlinear_rules,
        get_params=get_params,
    )

    bounds = {var: bound for var, bound in strict_zip(full_jaxpr.invars, args)}
    bounds |= {var: val for var, val in strict_zip(full_jaxpr.constvars, consts)}
    for i, eqn in enumerate(full_jaxpr.eqns):
        if all(isinstance(v, Literal) or not is_bounds(bounds[v]) for v in eqn.invars):
            # outvars of equation have concrete values
            invals = [
                v.val if isinstance(v, Literal) else bounds[v] for v in eqn.invars
            ]
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = (outvals,)
            for outvar, outval in strict_zip(eqn.outvars, outvals):
                bounds[outvar] = outval
            continue

        sub_eqns = full_jaxpr.eqns[: i + 1]
        last_outs = eqn.outvars

        sub_jaxpr = Jaxpr(
            constvars=full_jaxpr.constvars,
            invars=full_jaxpr.invars,
            outvars=last_outs,
            eqns=sub_eqns,
        )
        sub_jaxpr = ClosedJaxpr(sub_jaxpr, consts)

        eqn_out_batch_axes = tuple(batch_axis_graph[var] for var in last_outs)
        eqn_out_weights = [
            LiRPAWeights.identity_for(aval, batch_ax)
            for aval, batch_ax in strict_zip(sub_jaxpr.out_avals, eqn_out_batch_axes)
        ]
        # get_bounds needs to be hashable for caching in _backwards_lirpa
        # hashing bounds by keys is sufficient for differentiating the different
        # calls to _backwards_lirpa in this function.
        get_bounds = HashablePartial(
            _get_bounds_bootstrap, _HashByKeysDict(bounds), bootstrap_fun
        )
        eqn_bounds = _backwards_lirpa(
            sub_jaxpr,
            params,
            eqn_out_weights,
            *args,
            in_batch_axes=in_batch_axes,
            out_batch_axes=eqn_out_batch_axes,
            nonlinear_rules=nonlinear_rules,
            compute_bounds=get_bounds,
            get_params=get_params,
            batch_axis_graph=batch_axis_graph,
            transform_stack=transform_stack,
        )
        for outvar, bound in strict_zip(last_outs, eqn_bounds):
            bounds[outvar] = bound.concrete

    return bounds


def _backwards_lirpa_one_output(
    jaxpr: ClosedJaxpr,
    params: PyTree[Real[Array, "..."]],
    outvar: Var,
    outvar_weights: LiRPAWeights[Real[Array, "..."]],
    bounds: dict[Var, Bounds | Real[Array, "..."]],
    nonlinear_rules: frozendict[Primitive, BackwardsLiRPARule],
    compute_bounds: COMPUTE_BOUNDS,
    get_params: Callable[
        [JaxprEqn, PyTree[Real[Array, "..."]]], PyTree[Real[Array, "..."]]
    ],
    batch_axes: BatchAxisGraph,
    transform_stack: bool = True,
) -> LiRPABounds[Real[Array, "..."] | Zero]:
    """Performs backwards LiRPA for one output variable.

    Args:
        jaxpr: See ``_backwards_lirpa``.
        params: See ``_backwards_lirpa``.
        outvar: The output variable of ``jaxpr`` for which to compute bounds.
        outvar_weights: The weights of ``outvar`` to start propagation with.
        bounds: Preliminary bounds on the variables in ``jaxpr`` (e.g. from IBP).
        nonlinear_rules: See ``_backwards_lirpa``.
        compute_bounds: See ``_backwards_lirpa``.
        get_params: See ``_backwards_lirpa``.
        batch_axes: See ``_backwards_lirpa``.
        transform_stack: See ``_backwards_lirpa``.
    """
    jaxpr = jaxpr.jaxpr
    full_target_shape = outvar_weights.full_out_shape

    has_bounds = {var for var, val in bounds.items() if is_bounds(val)}
    requires_weights = has_bounds | {outvar}
    if any(v not in bounds for v in jaxpr.invars):
        raise ValueError("All input variables must have bounds.")
    in_domains = all_as_bounds(*(bounds[v] for v in jaxpr.invars))
    weights_env: dict[Var, LiRPAWeights[Real[Array, "..."] | Zero]] = {}

    def get_bounds(v: Atom) -> Bounds[jax.Array] | jax.Array:
        if isinstance(v, Literal):
            return v.val
        else:
            return bounds[v]

    def pop(v: Atom) -> LiRPAWeights[Real[Array, "..."] | Zero]:
        v_in_shape = non_batch_shape(v.aval.shape, batch_axes[v])
        # use for, for example, equations that are disconnected from outvar
        default_weight = outvar_weights.backwards_zeros(
            v_in_shape, batch_axes.mapping(v, outvar)
        )
        return weights_env.pop(v, default_weight)

    def update(v: Atom, weights: LiRPAWeights[Real[Array, "..."]]):
        if isinstance(v, Literal) or v not in requires_weights:
            return
        weights_env[v] = weights_env[v] + weights if v in weights_env else weights

    # Passed on to primitive rules for recursive calls
    bw_lirpa = HashablePartial(
        _backwards_lirpa,
        nonlinear_rules=nonlinear_rules,
        compute_bounds=compute_bounds,
        get_params=get_params,
    )

    with transform_name_stack_ctx("backwards_lirpa", transform_stack):
        update(outvar, outvar_weights)  # all other outvars get (Zero, Zero) weights
        lb_bias = ub_bias = Zero()  # these have full_out_shape. No leading batch axes!

        for eqn in reversed(jaxpr.eqns):
            if all(
                isinstance(var, Literal) or var not in has_bounds for var in eqn.invars
            ):
                # Equation does not depend on bounded arguments.
                # All preceding equations also don't have bounds
                # and outputs of equation have concrete values.
                if outvar in eqn.outvars:
                    # If we continue running, we would return zero bounds for
                    # all input variables, which is not correct.
                    # Instead, return here with the concrete value for outvar.
                    invals = [bounds[v] for v in eqn.invars]  # all jax.Arrays
                    outval = eqn.primitive.bind(*invals, **eqn.params)
                    if eqn.primitive.multiple_results:
                        outval = outval[eqn.outvars.index(outvar)]

                    in_weights = [pop(v) for v in jaxpr.invars]  # all zero
                    assert all(w.is_zero_weights for w in in_weights)
                    return LiRPABounds.from_weights(
                        in_weights, outval, outval, in_domains, full_target_shape
                    )
                else:
                    continue

            out_weights = [pop(v) for v in eqn.outvars]
            if all(w.is_zero_weights for w in out_weights):
                # Can skip, the lirpa bounds would be zero anyways.
                continue

            eqn_params = get_params(eqn, params)
            in_bounds = [get_bounds(v) for v in eqn.invars]
            full_in_shapes = [v.aval.shape for v in eqn.invars]
            full_out_shapes = [v.aval.shape for v in eqn.outvars]
            in_batch_axes = [batch_axes[v] for v in eqn.invars]
            out_batch_axes = [batch_axes[v] for v in eqn.outvars]
            batch_axis_maps = [
                [batch_axes.mapping(iv, ov) for iv in eqn.invars] for ov in eqn.outvars
            ]

            with eqn_name_stack_ctx(eqn), eqn.ctx.manager:
                prim = markup_primitive(eqn)
                if prim in nonlinear_rules:
                    rule = nonlinear_rules[prim]
                elif prim in _primitive_backwards_lirpa_rules:
                    rule = _primitive_backwards_lirpa_rules[prim]
                else:
                    raise NotImplementedError(
                        f"No backwards LiRPA rule implemented for {prim}."
                    )
                in_lirpa_bounds = rule(
                    eqn_params,
                    out_weights,
                    in_bounds,
                    full_in_shapes=full_in_shapes,
                    full_out_shapes=full_out_shapes,
                    in_batch_axes=in_batch_axes,
                    out_batch_axes=out_batch_axes,
                    batch_axis_mappings=batch_axis_maps,
                    backwards_lirpa=bw_lirpa,
                    **eqn.params,
                )

            if not prim.multiple_results and isinstance(in_lirpa_bounds, LiRPABounds):
                in_lirpa_bounds = (in_lirpa_bounds,)

            for in_bnds in in_lirpa_bounds:
                for v, in_w in strict_zip(eqn.invars, in_bnds.weights_iter()):
                    if not isinstance(v, Literal) and v in requires_weights:
                        update(v, in_w)
                    elif not in_w.is_zero_weights:
                        # Apply the weight to the concrete value and add the result
                        # to the bias.
                        # This is an early concretization if you want so.
                        val = get_bounds(v)
                        lb_term, ub_term = in_w.dot_weights(val)
                        lb_bias = lb_bias + lb_term
                        ub_bias = ub_bias + ub_term

                lb_bias = lb_bias + in_bnds.lb_bias
                ub_bias = ub_bias + in_bnds.ub_bias

        in_weights = [pop(v) for v in jaxpr.invars]
        return LiRPABounds.from_weights(
            in_weights, lb_bias, ub_bias, in_domains, full_target_shape
        )


_primitive_backwards_lirpa_rules: dict[Primitive | Marker, BackwardsLiRPARule] = {}


def register_backwards_lirpa_rule(
    primitive: Primitive | Marker, rule: BackwardsLiRPARule
):
    """Registers a new backwards LiRPA rule.

    Args:
        primitive: The primitive (or marker) that is handled by this rule.
        rule: The backwards LiRPA rule for ``primitive``.
            Accepts a pair of weight arrays (lower bound and upper bound, may be ``Zero``)
            for each output of ``primitive``,
            one ``Bounds`` instance or array for each output of ``primitive``,
            one ``Bounds`` instance or array for each input of ``primitive``,
            a backwards lirpa callable for the keyword ``backwards_lirpa``
            (this is for computing bounds on nested jaxprs),
            and all keyword arguments that are required for evaluating ``primitive``.

            It returns a sequence of weights (tuple/list) for the inputs of
            ``primitive`` and two bias arrays, one to add to the linear lower bound
            and one to add to the linear upper bound.

            The ``backwards_lirpa`` callable computes backwards LiRPA bounds given
            a Jaxpr, output weights, and bounds on the arguments of the Jaxpr.
            The return value is as for ``backwards_lirpa``.
    """
    _primitive_backwards_lirpa_rules[primitive] = rule


# ==============================================================================
# Backwards LiRPA Primitive Rules
# ==============================================================================


def nonlinear_backwards_lirpa_rule[P](
    lirpa_params: Callable[
        [P, Sequence[Bounds | Real[Array, "..."]], Bounds | Real[Array, "..."], ...],
        tuple[
            tuple[Real[Array, "..."], Real[Array, "..."]],
            ...,
            tuple[Real[Array, "..."] | Zero, Real[Array, "..."] | Zero],
        ],
    ],
    params: P,
    out_weights: Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
    in_bounds: Sequence[Bounds | jax.Array],
    full_in_shapes: Sequence[tuple[int, ...]],
    full_out_shapes: Sequence[tuple[int, ...]],
    in_batch_axes: Sequence[tuple[int, ...]],
    out_batch_axes: Sequence[tuple[int, ...]],
    batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
    backwards_lirpa: Callable,
    **kwargs,
) -> LiRPABounds[Real[Array, "..."] | Zero]:
    """Backwards LiRPA rule for element-wise non-linear functions.

    This function generalizes the two rules for non-linear functions in Table 6
    of [Xu et al., 2020] to n-ary operations.

    [Xu et al., 2020]: Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang,
        Minlie Huang, Bhavya Kailkhura, Xue Lin, Cho-Jui Hsieh: Automatic Perturbation
        Analysis for Scalable Certified Robustness and Beyond. NeurIPS 2020

    Args:
        lirpa_params: A function that computes the weight and bias parameters of
            the linear relaxation of the non-linear function.
            The weight parameters are called ``\\underline{\\alpha}``,
            ``\\overline{\\alpha}``, ``\\underline{\\beta}``, and ``\\overline{\\beta}``
            in Table 6 (bottom-most cell) in [Xu et al., 2020].
            The bias parameters are ``\\underline{\\gamma}`` and ``\\overline{\\gamma}``.

            The ``lirpa_params`` callable  takes a sequence of ``Bounds`` instances
            (one for each input) and another ``Bounds`` instance for the output,
            as well as all keyword arguments of the underlying non-linear
            function as arguments.
            Individual bounds for the inputs may be replaced by simple ``jax.Arrays``,
            which means that the corresponding input is not bounded but has the given
            concrete value.

            The callable returns the weight parameters as pairs for each input
            (i.e. ``(\\underline{\\alpha}, \\overline{\\alpha})`` are returned as a pair)
            and an additional final pair of bias parameters.
            The bias parameters may also be ``Zero``s, which stands for a scalar zero.
        params: External parameters of this non-linear LiRPA rule.
        out_weights: The output weights to propagate. Needs to have exactly one element.
        in_bounds: ``Bounds`` on each input of the function to bound.
        full_in_shapes: The full input shapes of the function to bound.
        full_out_shapes: The full output shapes of the function to bound.
        in_batch_axes: The batch axes of the inputs.
        out_batch_axes: The batch axes of the outputs.
            Needs to contain a single element.
        batch_axis_mappings: Defines the batch axes of the inputs and outputs and how
            they are mapped to each other.
            Needs to have exactly one element, which is a sequence with the
            same length as ``in_bounds``.
        backwards_lirpa: A callable that computes backwards LiRPA bounds.
        **kwargs: Any keyword arguments of the function to bound.

    Returns:
        The weights and bias arrays from propagating ``out_weights`` backwards.
    """
    out_weights, out_batch_axes, batch_axis_maps = _single_output_asserts(
        out_weights, out_batch_axes, batch_axis_mappings
    )

    # we call the weights alpha and the biases beta here
    *alphas, (beta_lb, beta_ub) = lirpa_params(params, *in_bounds, **kwargs)
    out_w_pos_neg = out_weights.pos_neg_parts()

    def in_ws(alphas, batch_axis_mapping):
        # The out_weights have the shape (*batch_shape, *target_shape, *out_bounds.shape).
        # To multiply the alphas with the output weights, we need to add dimensions
        # for target_shape and move the batch axes to the start.
        num_batch = len(batch_axis_mapping.in_axes)
        num_in = len(out_weights.full_in_shape) - num_batch
        alphas = (
            pull_batch_axes(alpha, (num_batch, 0, num_in), batch_axis_mapping.in_axes)
            for alpha in alphas
        )
        alpha_lb, alpha_ub = (out_weights.expand_input(alpha) for alpha in alphas)
        # The unbatching is necessary since some element-wise operations do broadcasting
        # (for example, the ``lax.max`` in ``jax.nn.relu``).
        out_lb_w_pos, out_ub_w_pos, out_lb_w_neg, out_ub_w_neg = (
            out_weights.unbatch_axes(w, batch_axis_mapping) for w in out_w_pos_neg
        )

        in_lb_w = alpha_lb * out_lb_w_pos + alpha_ub * out_lb_w_neg
        in_ub_w = alpha_ub * out_ub_w_pos + alpha_lb * out_ub_w_neg
        return in_lb_w, in_ub_w

    in_lb_ws, in_ub_ws = zip(
        *(
            in_ws(alph, mapping)
            for alph, mapping in strict_zip(alphas, batch_axis_maps)
        ),
        strict=False,
    )

    dot = out_weights.dot
    out_lb_w_pos, out_ub_w_pos, out_lb_w_neg, out_ub_w_neg = out_w_pos_neg
    in_lb_b = dot(out_lb_w_pos, beta_lb) + dot(out_lb_w_neg, beta_ub)
    in_ub_b = dot(out_ub_w_neg, beta_lb) + dot(out_ub_w_pos, beta_ub)

    # addition of infinite values can cause nan -> replace again by infinite values
    in_lb_b = jnp.nan_to_num(in_lb_b, nan=-jnp.inf)
    in_ub_b = jnp.nan_to_num(in_ub_b, nan=jnp.inf)

    # Some element-wise operations broadcast their arguments
    return out_weights.backwards_step(
        in_lb_ws, in_ub_ws, in_lb_b, in_ub_b, in_bounds, batch_axis_maps
    )


def constant_bounds_backwards_lirpa_rule(
    out_bounds_rule: Callable[[Bounds | jax.Array, ...], Bounds],
    params: None,
    out_weights: Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
    in_bounds: Sequence[Bounds | jax.Array],
    full_in_shapes: Sequence[tuple[int, ...]],
    full_out_shapes: Sequence[tuple[int, ...]],
    in_batch_axes: Sequence[tuple[int, ...]],
    out_batch_axes: Sequence[tuple[int, ...]],
    batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
    backwards_lirpa: Callable,
    **kwargs,
) -> LiRPABounds[Real[Array, "..."] | Zero]:
    """Uses constant output bounds as backwards LiRPA bounds.

    Args:
        out_bounds_rule: Computes output bounds from input bounds.
            This rule produces the constant bounds that are used as LiRPA
            bounds by this lirpa_rule.
        params: External parameters of this LiRPA rule.
            This LiRPA rule does not have external parameters.
        out_weights: The output weights to propagate.
            Needs to have exactly one element.
        in_bounds: ``Bounds`` on each input of ``fun``.
        full_in_shapes: The full input shapes of the function to bound.
        full_out_shapes: The full output shapes of the function to bound.
        in_batch_axes: The batch axes of the inputs.
        out_batch_axes: The batch axes of the output.
            Needs to have exactly one element.
        batch_axis_mappings: Defines the batch axes of the inputs and outputs and how
            they are mapped to each other.
            Needs to have exactly one element, which is a sequence with the
            same length as ``in_bounds``.
        backwards_lirpa: A callable computing backwards LiRPA bounds.
        **kwargs: Any keyword arguments of ``fun``.

    Returns:
        The weights and bias arrays propagating ``out_weights`` backwards in
        backwards LiRPA.
    """
    out_weights, _, _, batch_axis_maps = _single_output_asserts(
        out_weights, full_out_shapes, out_batch_axes, batch_axis_mappings
    )
    out_lb, out_ub = out_bounds_rule(*in_bounds)

    out_lb_w_pos, out_ub_w_pos, out_lb_w_neg, out_ub_w_neg = out_weights.pos_neg_parts()
    dot = out_weights.dot
    in_lb_b = dot(out_lb_w_pos, out_lb) + dot(out_lb_w_neg, out_ub)
    in_ub_b = dot(out_ub_w_pos, out_ub) + dot(out_ub_w_neg, out_lb)

    return out_weights.backwards_step(
        (Zero(),) * len(in_bounds),
        (Zero(),) * len(in_bounds),
        in_lb_b,
        in_ub_b,
        in_bounds,
        batch_axis_maps,
    )


register_backwards_lirpa_rule(
    lax.ge_p,
    partial(
        constant_bounds_backwards_lirpa_rule, partial(ibp_rule_compare_greater, lax.ge)
    ),
)
register_backwards_lirpa_rule(
    lax.gt_p,
    partial(
        constant_bounds_backwards_lirpa_rule, partial(ibp_rule_compare_greater, lax.gt)
    ),
)
register_backwards_lirpa_rule(
    lax.le_p,
    partial(
        constant_bounds_backwards_lirpa_rule, partial(ibp_rule_compare_less, lax.le)
    ),
)
register_backwards_lirpa_rule(
    lax.lt_p,
    partial(
        constant_bounds_backwards_lirpa_rule, partial(ibp_rule_compare_less, lax.lt)
    ),
)


def broadcast_in_dim_backwards_lirpa_rule(
    params: None,
    out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
    in_bounds: Sequence[Bounds | jax.Array],
    full_in_shapes: Sequence[tuple[int, ...]],
    full_out_shapes: Sequence[tuple[int, ...]],
    in_batch_axes: Sequence[tuple[int, ...]],
    out_batch_axes: Sequence[tuple[int, ...]],
    batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
    backwards_lirpa: Callable,
    **kwargs,
) -> LiRPABounds[jax.Array | Zero]:
    """A backwards LiRPA rule for ``lax.broadcast_in_dim``.

    Args:
        params: External parameters of this LiRPA rule.
            This LiRPA rule does not have external parameters.
        out_weights: The output weights to propagate.
            Needs to have exactly one element.
        in_bounds: ``Bounds`` on each input of ``fun``.
            Needs to have exactly one element.
        full_in_shapes: The full input shapes of ``lax.broadcast_in_dim``.
        full_out_shapes: The full output shapes of ``lax.broadcast_in_dim``.
        in_batch_axes: The batch axes of the inputs.
            Needs to have exactly one element.
        out_batch_axes: The batch axes of the output.
            Needs to have exactly one element.
        batch_axis_mappings: Defines the batch axes of the inputs and outputs and how
            they are mapped to each other.
            Needs to have exactly one element, which is a sequence with the
            same length as ``in_bounds``.
        backwards_lirpa: A callable computing backwards LiRPA bounds.
        **kwargs: Any keyword arguments of ``fun``.

    Returns:
        The weights and bias arrays propagating ``out_weights`` backwards in
        backwards LiRPA.
    """
    out_weights, full_out_shape, out_batch_axes, batch_axis_maps = (
        _single_output_asserts(
            out_weights, full_out_shapes, out_batch_axes, batch_axis_mappings
        )
    )
    in_bounds, full_in_shape, _ = _single_input_asserts(
        in_bounds, full_in_shapes, in_batch_axes
    )
    assert len(batch_axis_maps) == 1
    broadcast_axes = kwargs["broadcast_dimensions"]
    batch_axis_map = batch_axis_maps[0]

    in_lb_w, in_ub_w, batch_axis_map = _unbroadcast(
        out_weights, full_in_shape, full_out_shape, batch_axis_map, broadcast_axes
    )
    return out_weights.backwards_step(
        (in_lb_w,), (in_ub_w,), Zero(), Zero(), (in_bounds,), (batch_axis_map,)
    )


def _unbroadcast(
    out_weights: LiRPAWeights[jax.Array | Zero],
    full_in_shape: tuple[int, ...],
    full_out_shape: tuple[int, ...],
    batch_axis_map: BatchAxisMapping,
    broadcast_axes: Sequence[int] | None = None,
) -> tuple[jax.Array, jax.Array, BatchAxisMapping]:
    """Unbatches and sums out broadcasting axes.

    Args:
        broadcast_axes: The ``broadcasting_dimensions`` argument of
            ``jax.lax.broadcast_in_dim``.
            If ``None``, assumes leading additional axes.

    Returns:
        The input lower and upper bound weights and the new batch axis mapping.
    """
    if broadcast_axes is None:
        n_out_total = len(full_out_shape)
        n_added = n_out_total - len(full_in_shape)
        broadcast_axes = tuple(range(n_added, n_out_total))

    # handle broadcasting axes that are batch axes
    ow_unbatched = out_weights.unbatch_weights(batch_axis_map)
    new_batch_axis_map = batch_axis_map.filter_out_axis(batch_axis_map.broadcast_axes)

    # sum out the non-batch broadcasting axes
    added_axes = [i for i in range(len(full_out_shape)) if i not in broadcast_axes]
    size_one_axes = [
        out_i
        for in_i, out_i in enumerate(broadcast_axes)
        if full_in_shape[in_i] != full_out_shape[out_i]
    ]
    reduce_axes = set(added_axes + size_one_axes) - batch_axis_map.out_axes_set
    # Currently, reduce_axes refers to the output.
    # 1. Account for batch axes that are moved to the front in the in_weights shape.
    #    However, batch axes that are broadcast from a size one axis
    #    are also present in the in_weights shape.
    batch_out_plain = batch_axis_map.out_axes_set - set(size_one_axes)
    reduce_axes = [i - sum(ba < i for ba in batch_out_plain) for i in reduce_axes]
    batch_shape, target_shape = ow_unbatched.batch_shape, ow_unbatched.out_shape
    # 2. Shift so that it refers to the in_weights.
    n_prefix = len(batch_shape) + len(target_shape)
    reduce_axes = [n_prefix + i for i in reduce_axes]

    in_shape = non_batch_shape(full_in_shape, new_batch_axis_map.in_axes)
    in_weight_shape = batch_shape + target_shape + in_shape

    def sum_broadcast_axes(w):
        w = jnp.sum(w, reduce_axes)
        return jnp.reshape(w, in_weight_shape)

    in_lb_w, in_ub_w = ow_unbatched
    return sum_broadcast_axes(in_lb_w), sum_broadcast_axes(in_ub_w), new_batch_axis_map


register_backwards_lirpa_rule(
    lax.broadcast_in_dim_p, broadcast_in_dim_backwards_lirpa_rule
)


def add_with_broadcasting_backwards_lirpa_rule(
    alpha: float,
    params: None,
    out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
    in_bounds: Sequence[Bounds | jax.Array],
    full_in_shapes: Sequence[tuple[int, ...]],
    full_out_shapes: Sequence[tuple[int, ...]],
    in_batch_axes: Sequence[tuple[int, ...]],
    out_batch_axes: Sequence[tuple[int, ...]],
    batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
    backwards_lirpa: Callable,
    **kwargs,
) -> LiRPABounds[jax.Array | Zero]:
    """A backwards LiRPA rule for addition and subtraction with broadcasting.

    This backwards LiRPA rule supports affine functions of the form

        f(x, y) = x + alpha * y,

    where ``alpha`` is a fixed float.
    For example, for addition, ``alpha`` is 1.0 and for subtraction, ``alpha``
    is -1.0.

    Args:
        alpha: The multiplicative factor of the second argument.
        params: External parameters of this LiRPA rule.
            This LiRPA rule does not have external parameters.
        out_weights: The output weights to propagate.
            Needs to have exactly one element.
        in_bounds: ``Bounds`` on each input of ``fun``.
            Needs to have exactly two elements.
        full_in_shapes: The full input shapes of ``f``.
        full_out_shapes: The full output shapes of ``f``.
        in_batch_axes: The batch axes of the inputs.
            Needs to have exactly two elements.
        out_batch_axes: The batch axes of the output.
            Needs to have exactly one element.
        batch_axis_mappings: Defines the batch axes of the inputs and outputs and how
            they are mapped to each other.
            Needs to have exactly one element, which is a sequence with the
            same length as ``in_bounds``.
        backwards_lirpa: A callable computing backwards LiRPA bounds.
        **kwargs: Any keyword arguments of ``fun``.

    Returns:
        The weights and bias arrays propagating ``out_weights`` backwards in
        backwards LiRPA.
    """
    out_weights, full_out_shape, out_batch_axes, batch_axis_maps = (
        _single_output_asserts(
            out_weights, full_out_shapes, out_batch_axes, batch_axis_mappings
        )
    )
    assert len(in_bounds) == len(full_in_shapes) == 2
    assert len(in_batch_axes) == len(batch_axis_maps) == 2

    left_ba_map, right_ba_map = batch_axis_maps
    full_left_shape, full_right_shape = full_in_shapes
    left_lb_w, left_ub_w, left_ba_map = _unbroadcast(
        out_weights, full_left_shape, full_out_shape, left_ba_map
    )
    right_lb_w, right_ub_w, right_ba_map = _unbroadcast(
        out_weights, full_right_shape, full_out_shape, right_ba_map
    )

    right_lb_w, right_ub_w = alpha * right_lb_w, alpha * right_ub_w

    return out_weights.backwards_step(
        (left_lb_w, right_lb_w),
        (left_ub_w, right_ub_w),
        Zero(),
        Zero(),
        in_bounds,
        (left_ba_map, right_ba_map),
    )


register_backwards_lirpa_rule(
    lax.add_p, partial(add_with_broadcasting_backwards_lirpa_rule, 1.0)
)
register_backwards_lirpa_rule(
    lax.sub_p, partial(add_with_broadcasting_backwards_lirpa_rule, -1.0)
)


# ------------------------------------------------------------------------------
# Reuse transpose rules.
# ------------------------------------------------------------------------------
# For many linear or bilinear primitives (including identity transformations),
# we can reuse the transpose rule from ``jax.interpreters.ad`` as backwards
# LiRPA rules.
# Steps to reuse a transpose rule:
#  1. Flatten the rear dimensions of the LiRPA out_weight.
#  2. Replace some input bounds with `UndefinedPrimal`.
#  3. Apply a vmapped transpose rule to the partially flattened out_weight.
#  4. Reshape the final dimensions of the obtained in_weight.


def transpose_as_backwards_lirpa_rule_unary(
    primitive: Primitive, keyword_args: tuple[str, ...]
) -> BackwardsLiRPARule:
    """Wraps the transpose rule from ``jax.interpreters.ad`` as a backwards
    LiRPA rule.

    This function can only be used for affine primitives with a single output
    and a single input.
    This function does not support primitives that broadcast.

    Args:
        primitive: The primitive whose transpose rule is to be wrapped as a
            backwards LiRPA rule.
        keyword_args: The keywords of the keyword arguments of ´`primitive``,
            in the order the keyword arguments are defined in the method signature
            of the ``primitive``.

    Returns:
        A backwards LiRPA rule for ``primitive`` that uses the transpose rule
        from ``jax.interpreters.ad`` as a backwards LiRPA rule.
    """
    mapped_args = [True]
    transpose_rule = _vmapped_transpose_rule(primitive, mapped_args, keyword_args)
    return partial(_transpose_rule_as_backwards_lirpa_rule, transpose_rule, mapped_args)


def transpose_as_backwards_lirpa_rule_bilinear(
    primitive, keyword_args: tuple[str, ...]
) -> BackwardsLiRPARule:
    """Use the transpose rule as backwards LiRPA rule for a bilinear function.

    This function can only be used for primitives with two positional arguments and
    a single output.

    Like `transpose_as_backwards_lirpa_rule_unary`, this function does not support
    broadcasting.

    Args:
        primitive: The bilinear primitive whose transpose rule is to be wrapped as a
            backwards LiRPA rule.
        keyword_args: The keywords of the keyword arguments of ``primitive``,
            in the order the keyword arguments are defined in the method signature
            of the ``primitive``.

    Returns:
        A backwards LiRPA rule for ``primitive`` that uses the transpose rule
        from ``jax.interpreters.ad`` as a backwards LiRPA rule.
    """
    # Similar to transpose_as_backwards_lirpa_rule, but we need separate transpose
    # rules depending on which argument has bounds in the linear case.
    lhs_transpose_rule = _vmapped_transpose_rule(primitive, [True, False], keyword_args)
    lhs_rule = partial(
        _transpose_rule_as_backwards_lirpa_rule, lhs_transpose_rule, [True, False]
    )

    rhs_transpose_rule = _vmapped_transpose_rule(primitive, [False, True], keyword_args)
    rhs_rule = partial(
        _transpose_rule_as_backwards_lirpa_rule, rhs_transpose_rule, [False, True]
    )

    def backwards_lirpa_rule(
        params: None,
        out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        **kwargs,
    ) -> LiRPABounds[jax.Array | Zero]:
        assert len(in_bounds) == 2
        lhs, rhs = in_bounds

        if is_bounds(lhs) and is_bounds(rhs):
            raise NotImplementedError("No backwards LiRPA rule for bilinear function.")
        elif is_bounds(lhs):
            return lhs_rule(params, out_weights, in_bounds, **kwargs)
        elif is_bounds(rhs):
            return rhs_rule(params, out_weights, in_bounds, **kwargs)
        assert is_bounds(lhs) or is_bounds(rhs)

    return backwards_lirpa_rule


def transpose_as_backwards_lirpa_rule_binary_left_only(
    primitive, keyword_args: tuple[str, ...], error_message
) -> BackwardsLiRPARule:
    """Use the transpose rule as backwards LiRPA rule for a binary function
    (first argument/left-hand argument only).

    For binary functions that are linear if the second argument is constant,
    for example, division.

    This function can only be used for primitives with two positional arguments and
    a single output.

    Like `transpose_as_backwards_lirpa_rule`, this function supports broadcasting
    along leading axes, but not arbitrary broadcasting.

    Args:
        primitive: The binary primitive whose transpose rule is to be wrapped as a
            backwards LiRPA rule.
        keyword_args: The keywords of the keyword arguments of ´`primitive``,
            in the order the keyword arguments are defined in the method signature
            of the ``primitive``.
        error_message: The error message to use for a ``NotImplementedError`` if
            the second argument is not a constant (has bounds).

    Returns:
        A backwards LiRPA rule for ``primitive`` that uses the transpose rule
        from ``jax.interpreters.ad`` as a backwards LiRPA rule.
    """
    one_side_rule = _vmapped_transpose_rule(primitive, [True, False], keyword_args)
    one_side_rule = partial(
        _transpose_rule_as_backwards_lirpa_rule, one_side_rule, [True, False]
    )

    def backwards_lirpa_rule(
        params: None,
        out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        **kwargs,
    ) -> LiRPABounds[jax.Array | Zero]:
        assert len(in_bounds) == 2
        _, rhs = in_bounds

        if is_bounds(rhs):
            raise NotImplementedError(error_message)
        return one_side_rule(params, out_weights, in_bounds, **kwargs)

    return backwards_lirpa_rule


def _transpose_rule_as_backwards_lirpa_rule(
    transpose_rule: Callable,
    args_propagate: Sequence[bool],
    params: None,
    out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
    in_bounds: Sequence[Bounds | jax.Array],
    full_in_shapes: Sequence[tuple[int, ...]],
    full_out_shapes: Sequence[tuple[int, ...]],
    in_batch_axes: Sequence[tuple[int, ...]],
    out_batch_axes: Sequence[tuple[int, ...]],
    batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
    backwards_lirpa: Callable,
    **kwargs,
) -> LiRPABounds[jax.Array | Zero]:
    """Applies a vmapped transpose rule (``transpose_rule``) as a backwards lirpa rule.

    This function only supports a single bounded argument.
    Exactly one entry of ``args_propagate`` needs to be ``True``.

    Args:
        transpose_rule: The vamapped transpose rule to apply.
        args_propagate: Which arguments of the transpose rule get non-zero input
            weights from propagating backwards.
            These arguments are replaced by ``UndefPrimal``s in the call of
            the transpose rule.
        params: External parameters of this LiRPA rule.
        out_weights: The output weights to propagate.
            Needs to have exactly one element.
        in_bounds: ``Bounds`` on each input of ``fun``.
            Needs to have exactly two elements.
        full_in_shapes: The full input shapes of ``f``.
        full_out_shapes: The full output shapes of ``f``.
        in_batch_axes: The batch axes of the inputs.
            Needs to have exactly two elements.
        out_batch_axes: The batch axes of the output.
            Needs to have exactly one element.
        batch_axis_mappings: Defines the batch axes of the inputs and outputs and how
            they are mapped to each other.
            Needs to have exactly one element, which is a sequence with the
            same length as ``in_bounds``.
        backwards_lirpa: A callable computing backwards LiRPA bounds.
        **kwargs: Any keyword arguments of ``fun``.
    """
    assert sum(args_propagate) == 1
    prop_arg = args_propagate.index(True)

    out_weights, _, _, batch_axis_maps = _single_output_asserts(
        out_weights, full_out_shapes, out_batch_axes, batch_axis_mappings
    )

    # Prepare arguments of transpose_rule
    xs = example_args(in_bounds)
    primals = [
        ad.UndefinedPrimal(x.aval) if do_prop else x
        for x, do_prop in strict_zip(xs, args_propagate)
    ]
    transpose_rule_args = primals + list(kwargs.values())

    # Transform the output weights
    out_weights = out_weights.unbatch_weights(batch_axis_maps[prop_arg])

    def reshape_as_output(w, out_weights):
        return _reshape_as_output(
            w,
            out_weights.batch_shape,
            out_weights.in_shape,
            out_weights.batch_axis_mapping.in_axes,
        )

    def restore_weights_shape(w, in_i, out_shape):
        if is_zero(w):
            return Zero()
        return _restore_weights_shape(w, out_shape, xs[in_i].shape, in_batch_axes[in_i])

    # Apply the transpose rule
    out_ws = [reshape_as_output(w, out_weights) for w in out_weights]
    in_ws = [transpose_rule(w, *transpose_rule_args) for w in out_ws]
    in_ws = [[Zero() if w is None else w for w in ws] for ws in in_ws]
    in_lb_ws, in_ub_ws = (
        [restore_weights_shape(w, i, out_weights.out_shape) for i, w in enumerate(ws)]
        for ws in in_ws
    )

    return out_weights.backwards_step(
        in_lb_ws, in_ub_ws, Zero(), Zero(), in_bounds, batch_axis_maps
    )


def _vmapped_transpose_rule(
    primitive: Primitive,
    args_mapped: Sequence[bool],
    keyword_args: Sequence[str],
) -> Callable:
    """Wrap a transpose rule for use as a backwards LiRPA rule.

    Args:
        primitive: The primitive whose transpose rule to wrap.
        args_mapped: For each positional argument, whether the positional
            argument is mapped over.
        keyword_args: The keywords of the keyword arguments of ´`primitive``,
            in the order the keyword arguments are defined in the method signature
            of the ``primitive``.

    Returns: A vmapped transpose rule accepting weight arrays with the first
        axes flattened.
    """
    base_transpose_rule = ad.primitive_transposes[primitive]
    num_posargs = len(args_mapped)

    def transpose_rule_positional_only(*args):
        # vmap needs all keyword arguments as positional arguments
        # (this is a long-standing issue)
        # => take all positional arguments and turn them into positional
        # and keyword arguments
        out_val, *args = args
        posargs, kwargs = args[:num_posargs], args[num_posargs:]
        kwargs = {key: arg for key, arg in strict_zip(keyword_args, kwargs)}
        transposed = base_transpose_rule(out_val, *posargs, **kwargs)
        return list(transposed)  # transpose rules should always return lists

    # Add a leading extra dimension for the target shape part of the LiRPA out_weight.
    transpose_rule = jax.vmap(
        transpose_rule_positional_only,
        in_axes=[0] + [None] * num_posargs + [None] * len(keyword_args),
        out_axes=[0 if is_mapped else None for is_mapped in args_mapped],
    )
    return transpose_rule


def _register_unary_transpose_rule(primitive, keyword_args=()):
    rule = transpose_as_backwards_lirpa_rule_unary(primitive, keyword_args)
    register_backwards_lirpa_rule(primitive, rule)


_register_unary_transpose_rule(
    lax.convert_element_type_p, ("new_dtype", "weak_type", "sharding")
)
_register_unary_transpose_rule(lax.neg_p)
_register_unary_transpose_rule(lax.reduce_sum_p, ("axes", "out_sharding"))
_register_unary_transpose_rule(
    lax.reduce_window_sum_p,
    (
        "window_dimensions",
        "window_strides",
        "padding",
        "base_dilation",
        "window_dilation",
    ),
)
_register_unary_transpose_rule(lax.reshape_p, ("new_sizes", "dimensions", "sharding"))
_register_unary_transpose_rule(lax.rev_p, ("dimensions",))
_register_unary_transpose_rule(
    lax.slice_p, ("start_indices", "limit_indices", "strides")
)
_register_unary_transpose_rule(lax.squeeze_p, ("dimensions",))
_register_unary_transpose_rule(lax.transpose_p, ("permutation",))

register_backwards_lirpa_rule(
    lax.conv_general_dilated_p,
    transpose_as_backwards_lirpa_rule_bilinear(
        lax.conv_general_dilated_p,
        (
            "window_strides",
            "padding",
            "lhs_dilation",
            "rhs_dilation",
            "dimension_numbers",
            "feature_group_count",
            "batch_group_count",
            "precision",
            "preferred_element_type",
            "out_sharding",
        ),
    ),
)
register_backwards_lirpa_rule(
    lax.dot_general_p,
    transpose_as_backwards_lirpa_rule_bilinear(
        lax.dot_general_p,
        ("dimension_numbers", "precision", "preferred_element_type", "out_sharding"),
    ),
)
register_backwards_lirpa_rule(
    lax.mul_p, transpose_as_backwards_lirpa_rule_bilinear(lax.mul_p, ())
)

register_backwards_lirpa_rule(
    lax.div_p,
    transpose_as_backwards_lirpa_rule_binary_left_only(
        lax.div_p,
        (),
        "No backwards LiRPA rule for division with bounds on the divisor.",
    ),
)


# In principle, padding is an affine transformation and using the transpose
# rule would work fine for both arguments.
# However, it is rare to use a non-constant padding value.
# Therefore, supporting bounds on the padding value would not be worth
# the implementation effort.
register_backwards_lirpa_rule(
    lax.pad_p,
    transpose_as_backwards_lirpa_rule_binary_left_only(
        lax.pad_p,
        (),
        "Bounds on the padding value are currently not supported.",
    ),
)


# TODO: concatenate_p (has arbitrary number of arguments)


def backwards_lirpa_rule_jaxpr(
    params: PyTree[Real[Array, "..."]],
    out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
    in_bounds: Sequence[Bounds | jax.Array],
    jaxpr: ClosedJaxpr,
    full_in_shapes: Sequence[tuple[int, ...]],
    full_out_shapes: Sequence[tuple[int, ...]],
    in_batch_axes: Sequence[tuple[int, ...]],
    out_batch_axes: Sequence[tuple[int, ...]],
    backwards_lirpa: Callable[
        [
            Jaxpr,
            tuple[LiRPAWeights[jax.Array | Zero], ...],
            Bounds | jax.Array,
            ...,
        ],
        tuple[LiRPABounds[jax.Array | Zero], ...],
    ],
) -> tuple[ClosedJaxpr, list[Array], PyTreeDef]:
    """Performs ``backwards_lirpa`` on a Jaxpr, returning a new Jaxpr.

    Args:
        params: External parameters of this LiRPA rule.
            The external parameters need to implement the ``LiRPAParams`` protocol.
            The parameters are passed on to ``backwards_lirpa``.
        out_weights: Output weights to propagate.
            This function does not actually propagate these output weights
            but only uses them to create placeholders.
            This function returns the output weights flattened together
            with their Pytree.
        in_bounds: Input bounds (and concrete values) for backwards LiRPA.
            This function only uses these values to create placeholders.
            Additionally, it returns the input bounds flattened together
            with their Pytree.
        jaxpr: The Jaxpr to produce bounds on.
        full_in_shapes: The shapes of ``jaxpr``'s inputs.
        full_out_shapes: The shapes of ``jaxpr``'s outputs.
        in_batch_axes: The batch axes of the inputs.
        out_batch_axes: The batch axes of the output.
        backwards_lirpa: A callable computing backwards LiRPA bounds given
            a Jaxpr, output weights, and bounds on the arguments of the Jaxpr.
            The return value is as for ``backwards_lirpa``.
            This callable should have a consistent hash
            (use, for example, ``HashablePartial``).

    Returns:
        - A Jaxpr of the backwards LiRPA computation.
        - The flat arguments for the backwards LiRPA jaxpr.
        - The Pytree of the output of performing backwards LiRPA on ``jaxpr``.
    """
    # Need to be tuples for caching of _trace_backwards_lirpa
    in_batch_axes = tuple(in_batch_axes)
    out_batch_axes = tuple(out_batch_axes)
    args_flat, args_tree = jax.tree.flatten((params, out_weights, in_bounds))

    debug_info = jax_api_util.debug_info(
        "backwards_lirpa", backwards_lirpa, args_flat, {}
    )
    bw_lirpa_f = lu.wrap_init(backwards_lirpa, debug_info=debug_info)
    bw_lirpa_f = args_to_kwargs(bw_lirpa_f, (None, "in_batch_axes", "out_batch_axes"))
    bw_lirpa_f = lu.hashable_partial(bw_lirpa_f, jaxpr, in_batch_axes, out_batch_axes)
    bw_lirpa_f, out_tree = _flatten_backwards_lirpa_callable(bw_lirpa_f, args_tree)
    args_avals = tuple(jax.core.get_aval(arg) for arg in args_flat)
    bw_lirpa_jaxpr = _trace_backwards_lirpa(bw_lirpa_f, args_avals)

    return (bw_lirpa_jaxpr, args_flat, out_tree())


@lu.transformation_with_aux
def _flatten_backwards_lirpa_callable(args_tree, *args_flat):
    params, out_weights, in_bounds = jax.tree.unflatten(args_tree, args_flat)
    out = yield (params, out_weights, *in_bounds), {}
    yield jax.tree.flatten(out)


@lu.cache
def _trace_backwards_lirpa(
    backwards_lirpa: lu.WrappedFun,
    args_avals: tuple,
) -> ClosedJaxpr:
    bw_lirpa_jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
        backwards_lirpa, args_avals
    )
    bw_lirpa_jaxpr = ClosedJaxpr(bw_lirpa_jaxpr, consts)
    return bw_lirpa_jaxpr


def _pjit_backwards_lirpa_rule(
    params: PyTree[Real[Array, "..."]],
    out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
    in_bounds: Sequence[Bounds | jax.Array],
    full_in_shapes: Sequence[tuple[int, ...]],
    full_out_shapes: Sequence[tuple[int, ...]],
    in_batch_axes: Sequence[tuple[int, ...]],
    out_batch_axes: Sequence[tuple[int, ...]],
    batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
    backwards_lirpa: Callable[
        [
            Jaxpr,
            tuple[LiRPAWeights[jax.Array | Zero], ...],
            Bounds | jax.Array,
            ...,
        ],
        tuple[LiRPABounds[jax.Array | Zero], ...],
    ],
    jaxpr: ClosedJaxpr,
    in_shardings,
    out_shardings,
    in_layouts,
    out_layouts,
    donated_invars,
    **kwargs,
) -> LiRPABounds[jax.Array | Zero]:
    (bwlirpa_jaxpr, bwlirpa_args_flat, lirpa_bounds_tree) = backwards_lirpa_rule_jaxpr(
        params,
        out_weights,
        in_bounds,
        jaxpr,
        full_in_shapes,
        full_out_shapes,
        in_batch_axes,
        out_batch_axes,
        backwards_lirpa,
    )
    # We have more arguments of different shapes than ``jaxpr``, so we use
    # UNSPECIFIED sharding and None layout for all arguments.
    # Same for the outputs.
    linear_bounds = primitives.jit_p.bind(
        *bwlirpa_args_flat,
        jaxpr=bwlirpa_jaxpr,
        in_shardings=(jax_sharding_impls.UNSPECIFIED,) * len(bwlirpa_args_flat),
        out_shardings=(jax_sharding_impls.UNSPECIFIED,) * lirpa_bounds_tree.num_leaves,
        in_layouts=(None,) * len(bwlirpa_args_flat),
        out_layouts=(None,) * lirpa_bounds_tree.num_leaves,
        donated_invars=(False,) * len(bwlirpa_args_flat),
        **kwargs,
    )
    return jax.tree.unflatten(lirpa_bounds_tree, linear_bounds)


register_backwards_lirpa_rule(primitives.jit_p, _pjit_backwards_lirpa_rule)


# ==============================================================================
# Util
# ==============================================================================


def _single_output_asserts(
    out_weights: Sequence[LiRPAWeights[jax.Array]],
    out_shapes: Sequence[tuple[int, ...]],
    out_batch_axes: Sequence[tuple[int, ...]] | None = None,
    batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]] | None = None,
) -> (
    tuple[
        LiRPAWeights[jax.Array],
        tuple[int, ...],
        tuple[int, ...],
        Sequence[BatchAxisMapping],
    ]
    | tuple[LiRPAWeights[jax.Array], tuple[int, ...], Sequence[BatchAxisMapping]]
    | tuple[LiRPAWeights[jax.Array], tuple[int, ...]]
):
    """Assertions for a single output backwards LiRPA rule.

    Asserts:
     - length of ``out_weights`` is 1
     - length of ``out_shapes`` is 1
     - length of ``out_batch_axes`` is 1
     - length of ``batch_axis_mappings`` is 1
     - ``out_weights[0]`` may not be all zero

    Returns:
        The single output weight, the single output shape,
        the single element of ``out_batch_axes`` (if not ``None``, otherwise omitted),
        and the single element of ``batch_axis_mappings`` (if not ``None``,
        otherwise omitted).
    """
    assert len(out_weights) == 1
    assert not out_weights[0].is_zero_weights
    assert len(out_shapes) == 1

    return_values = (out_weights[0], out_shapes[0])
    if out_batch_axes is not None:
        assert len(out_batch_axes) == 1
        return_values += (out_batch_axes[0],)
    if batch_axis_mappings is not None:
        assert len(batch_axis_mappings) == 1
        return_values += (batch_axis_mappings[0],)
    return return_values


def _single_input_asserts(
    in_bounds: Sequence[Bounds | jax.Array],
    in_shapes: Sequence[tuple[int, ...]],
    in_batch_axes: Sequence[tuple[int, ...]],
) -> tuple[
    Bounds,
    tuple[int, ...],
    tuple[int, ...],
]:
    """Assertions for a single input backwards LiRPA rule.

    Asserts:
     - length of ``in_bounds`` is 1
     - length of ``in_shapes`` is 1
     - length of ``in_batch_axes`` is 1
     - ``in_bounds[0]`` is a Bounds instance.

    Returns:
        The single input bounds instance, the single input shape,
        and the single element of ``in_batch_axes``.
    """
    assert len(in_bounds) == 1
    assert is_bounds(in_bounds[0])
    assert len(in_shapes) == 1
    assert len(in_batch_axes) == 1
    return in_bounds[0], in_shapes[0], in_batch_axes[0]


def _reshape_as_output(
    w: jax.Array,
    batch_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    out_batch_axes: tuple[int, ...],
) -> jax.Array:
    """
    For a ``LiRPAWeight`` that maps from output to target,
    reshapes one element of the ``LiRPAWeight`` ``w`` to have the shape of the
    output, except for a leading axes for flattened the target axes.

    This function assumes ``w`` has the shape
    ``(*batch_shape, *target_shape, *out_shape)`` and
    reshapes it to ``(-1, *full_out_shape)``, where ``full_out_shape``
    consists of ``batch_shape`` and ``out_shape``, combined according to
    ``out_batch_axes``.

    Args:
        w: The weight to reshape.
        batch_shape: The ``batch_shape``.
        out_shape: The ``out_shape``.
        out_batch_axes: The batch axes of the input.
    """
    n_batch, n_out = len(batch_shape), len(out_shape)

    w = jnp.reshape(w, (*batch_shape, -1, *out_shape))
    return incorporate_batch_axes(w, (n_batch, 1, n_out), out_batch_axes)


def _restore_weights_shape(
    w: jax.Array,
    target_shape: tuple[int, ...],
    full_in_shape: tuple[int, ...],
    in_batch_axes: tuple[int, ...],
) -> jax.Array:
    """
    Reshape a weight array that has the full shape of the input with an additional leading
    target axis to have the usual LiRPAWeights shape
    ``(*in_batch_shape, *target_shape, *in_shape)``.

    Concretely, before ``restore_weights_shape``, the weight ``w`` has the shape
    ``(prod(target_shape), *full_in_shape)``, and afterwards it has the shape
    ``(*in_batch_shape, *target_shape, *in_shape)``.
    """
    batch_shape, in_shape = split_shape(full_in_shape, in_batch_axes)
    n_batch, n_in = len(batch_shape), len(in_shape)

    w = pull_batch_axes(w, (n_batch, 1, n_in), in_batch_axes)
    return jnp.reshape(w, (*batch_shape, *target_shape, *in_shape))
