#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import functools
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from math import prod
from typing import Literal, Protocol, TypeGuard

import jax
import jax.core
import jax.numpy as jnp
import jax.tree
import optax
from frozendict import frozendict
from jax import lax
from jax._src.core import JaxprEqn
from jax.extend.core import ClosedJaxpr, Primitive, Var, primitives
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Real

from ...core.batch_axes import BatchAxisMapping
from ...core.markers import Marker, markup_primitive, relu_marker
from ...core.zero import Zero
from ...sets.box import Box
from ...sets.protocols import (
    CanDrawRandomSample,
    HasProjection,
    can_draw_random_sample,
    has_projection,
)
from ...utils.fun_to_jaxpr import bounding_wrapper_of, make_jaxpr_with_bounds
from ...utils.reduce_window import (
    reduce_window_conv_transpose,
    reduce_window_patches,
    select_and_scatter_add2,
)
from ...utils.vardict import VarDict
from ...utils.zip import strict_zip
from ._affinebounds import AffineBounds
from ._bounds import Bounds, all_as_bounds, is_bounds
from ._bwlirpa import (
    COMPUTE_BOUNDS,
    BackwardsLiRPARule,
    _reshape_as_output,
    _restore_weights_shape,
    _single_output_asserts,
    backwards_lirpa,
    fun_to_jaxpr_for_lirpa,
    nonlinear_backwards_lirpa_rule,
    transpose_as_backwards_lirpa_rule_bilinear,
    transpose_as_backwards_lirpa_rule_binary_left_only,
)
from ._ibp import ibp_jaxpr, ibp_rule_reciprocal
from ._lirpabounds import LiRPABounds, LiRPAWeights, incorporate_batch_axes

__all__ = (
    "crown_ibp",
    "alpha_crown",
    "backwards_crown",
    "register_backwards_crown_rule",
    "crown_unary_convex_params",
    "crown_unary_concave_params",
    "crown_unary_s_shaped_params",
    "backwards_crown_relu_rule",
    "backwards_crown_max_rule",
    "backwards_crown_reduce_window_max_rule",
)


# ==============================================================================
# Interface
# ==============================================================================


class CROWNCallable[P](Protocol):
    """Type signature of a parameterized backwards CROWN callable.

    This is the type signature of the functions created by
    ``backwards_crown``.
    Each ``CROWNCallable`` is connected to a ``jaxpr``,
    on which it performs backwards CROWN.
    See ``backwards_crown`` for more details.
    """

    def __call__(
        self,
        params: P,
        *args: Bounds | jax.Array,
    ) -> tuple[AffineBounds[Real[Array, "..."] | Zero], ...]:
        """
        Propagates linear bounds backwards through the ``jaxpr`` encapsulated
        by this ``LiRPACallable`` using the relaxation parameters ``params``
        and the input bounds and concrete arguments provided as ``args``.

        Args:
            ``params``: A mapping from variables in ``jaxpr`` to parameter values
                for the ``\\alpha``-CROWN parameterized linear relaxation.
                For standard, unparameterized CROWN (that this, ``parameters="standard"`` in ``backwards_crown``),
                this argument should be an empty dictionary.
            ``*args``: Constant input bounds and concrete arguments for which to
                perform backwards bound propagation.
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


BACKWARDS_CROWN_PARAM_STRATEGIES = Literal[
    "default", "fixed", "adaptive", "external", "external-full", "external-shared"
]
"""The parameter selection strategies of backwards CROWN rules.

 -  Uses either the ``"fixed"`` or ``"adaptive"`` strategy, depending on the primitive.
    Which option is determined by  the backwards CROWN rules for
    the different primitives.
    This strategy does not use external parameters.
 - The ``"fixed"`` strategy indicates that a rule uses a fixed parameter.
    This strategy does not use external parameters.
 - The ``"adaptive"`` strategy indicates that a rule chooses parameter values
    adaptively, such as the CROWN ReLU rule.
    This strategy does not use external parameters.
 - The ``"external-full"`` strategy indicates that the parameters are determined
    by outside code, that, for example, optimizes this parameter, such
    as in ``\\alpha``-CROWN.
    In contrast to ``"external-shared"``, each entry of an array that is
    computed by an elementwise primitive has its own parameter.
 - The ``"external-shared"`` strategy indicates that the parameters are
    determined by outside code, as for ``"external-full"``, but all values
    in an array share the same parameter value.
    This means, for example, that all ReLU nodes on a ReLU layer share one
    ``\\alpha`` parameter in ``\\alpha``-CROWN.
 - ``"external"`` is an alias of ``"external-full``.
"""


class HasProjectionAndRandomSample(HasProjection, CanDrawRandomSample):
    pass


def is_param_domain(x) -> TypeGuard[HasProjectionAndRandomSample]:
    return has_projection(x) and can_draw_random_sample(x)


PARAM_DOMAIN = PyTree[HasProjectionAndRandomSample]
PARAM_VALUE = PyTree[Float[Array, "..."] | BACKWARDS_CROWN_PARAM_STRATEGIES]


class BackwardsCROWNRule(BackwardsLiRPARule, Protocol):
    """Type signature of a CROWN rule.

    This class adds a method that declares the parameter domains of the rule,
    besides the ``__call__`` method of ``BackwardsLiRPARule``.
    """

    def __call__(
        self,
        params: PyTree[Real[Array, "..."] | BACKWARDS_CROWN_PARAM_STRATEGIES],
        out_weights: Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        *,
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
    ): ...

    def parameter_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        strategies: Sequence[BACKWARDS_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        """Declares the domains of the parameters of this rule.

        This method determines whether there are parameters.
        If there are parameters, it determines the domains of the parameters.
        This also determines the PyTree structure and the shapes of the parameters.

        If a rule requires several parameters for one output, it may use
        a tuple of arrays for the parameters.
        In this case, the domain should be a tuple of array domains.

        This method raises a ``ValueError`` if it receives an
        unsupported parameter strategy.
        All rules needs to support, at-least the ``"default"``,
        ``"external-full"`` and ``"external-shared"`` strategies.
        It may return ``None`` for every strategy if there are no parameters.

        The default implementation returns ``None`` for all outputs.

        Args:
            in_shapes: The shapes of the inputs to the primitive call.
            out_shapes: The shapes of the outputs to the primitive call.
            strategies: The parameter selection strategies for
                each output of the primitive.
                Raise a ``ValueError`` if your method does not support
                one of the given strategies.
            **kwargs: Any keyword arguments of the primitive.

        Raises:
            ValueError: If one or several of the given parameter strategies are
                not supported.

        Returns:
            The domains of the parameters of this rule.
            Returns one parameter domain for each output of the primitive.
            A value of ``None`` indicates that this rule does not require parameters
            for an output.
        """
        ...


# Defined globally for cashing/hashing purposes
def _ibp_bounds(jaxpr, _, *args_flat: Bounds | jax.Array, **__):
    return ibp_jaxpr(jaxpr, intermediate_bounds=True)(*args_flat)


@fun_to_jaxpr_for_lirpa(lirpa_name="CROWN")
def crown(
    fun_or_jaxpr: Callable | ClosedJaxpr,
    compute_bounds: COMPUTE_BOUNDS = "bootstrap",
    in_batch_axes: int | None | tuple[int | None, ...] = ...,
    out_batch_axes: int | None | tuple[int | None, ...] = ...,
) -> Callable[[Bounds | jax.Array, ...], tuple[Bounds | jax.Array, ...]]:
    """Creates a function that performs backwards CROWN on ``fun``.

    By default (``compute_bounds="bootstrap"``), the returned function repeatedly
    applies ``backwards_crown`` to compute bounds on the intermediate values of
    ``fun_or_jaxpr``.
    A final call to ``backwards_crown`` computes bounds on the output of ``fun_or_jaxpr``.
    See ``backwards_crown`` and [Zhang et al., 2018] for more details.

    If ``compute_bounds`` is a callable, the returned function first computes constant bounds
    on the values in ``fun`` using this callable.
    It then applies ``backwards_crown`` to refine these bounds.
    See ``crown_ibp`` for an example.

    [Zhang et al., 2018]: Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh,
        Luca Daniel: Efficient Neural Network Robustness Certification with General
        Activation Functions. NeurIPS 2018: 4944-4953

    Args:
        fun_or_jaxpr: Function or jaxpr to compute bounds on.
            The arguments and the return value of ``fun`` may be any mix of arrays,
            scalars, and standard Python containers (JAX pytrees).
        compute_bounds: How to compute preliminary bounds on the intermediate variables of
            ``jaxpr``.
            If ``compute_bounds`` is ``"bootstrap"``, CROWN is called recursively
            to compute the intermediate bounds.
            Otherwise, `compute_bounds` is a function that takes a Jaxpr and input bounds
            and returns bounds on the intermediate variables of ``jaxpr``.
        in_batch_axes: Batch axes of the inputs.
            See ``backwards_crown`` for more details.
        out_batch_axes: Batch axes of the outputs.
            See ``backwards_crown`` for more details.

    Returns:
        A function ``fun_crown`` that computes output bounds on ``fun`` given
        input bounds on some of the arguments of ``fun``.
        When calling ``fun_crown``, pass ``Bounds`` instances for the arguments with
        input bounds while passing regular ``jax.Arrays`` for the arguments without
        input bounds.
        The output of ``fun_crown`` has the same structure as the output of ``fun``,
        except that all outputs which depends on arguments with input bounds are
        ``Bounds`` instances.

    """
    jaxpr = fun_or_jaxpr  # fun_to_jaxpr_for_lirpa handles functions

    def crown_fun(*args: Bounds | jax.Array):
        bwcrown, _ = backwards_crown(
            jaxpr,
            compute_bounds=compute_bounds,
            in_batch_axes=in_batch_axes,
            out_batch_axes=out_batch_axes,
        )
        return bwcrown({}, *args)

    return crown_fun


@fun_to_jaxpr_for_lirpa(lirpa_name="CROWN-IBP")
def crown_ibp(
    fun_or_jaxpr: Callable | ClosedJaxpr,
    in_batch_axes: int | None | tuple[int | None, ...] = ...,
    out_batch_axes: int | None | tuple[int | None, ...] = ...,
) -> Callable[[Bounds | jax.Array, ...], tuple[Bounds | jax.Array, ...]]:
    """Creates a function that performs backwards CROWN with IBP bounds on ``fun``.

    The returned function first computes constant bounds on the values in ``fun`` using
    ``ibp`` and then applies ``backwards_crown`` to refine these bounds.
    See ``backwards_crown`` and [Zhang et al., 2020] for more details.

    [Zhang et al., 2020]: Huan Zhang, Hongge Chen, Chaowei Xiao, Sven Gowal,
        Robert Stanforth, Bo Li, Duane S. Boning, Cho-Jui Hsieh: Towards Stable and
        Efficient Training of Verifiably Robust Neural Networks. ICLR 2020

    Args:
        fun_or_jaxpr: Function or jaxpr to compute bounds on.
            The arguments and the return value of ``fun`` may be any mix of arrays,
            scalars, and standard Python containers (JAX pytrees).
        in_batch_axes: Batch axes of the inputs.
            See ``backwards_crown`` for more details.
        out_batch_axes: Batch axes of the outputs.
            See ``backwards_crown`` for more details.

    Returns:
        A function ``fun_crown_ibp`` that computes output bounds on ``fun`` given
        input bounds on some of the arguments of ``fun``.
        When calling ``fun_crown_ibp``, pass ``Bounds`` instances for the arguments with
        input bounds while passing regular ``jax.Arrays`` for the arguments without
        input bounds.
        The output of ``fun_crown_ibp`` has the same structure as the output of ``fun``,
        except that all outputs which depends on arguments with input bounds are
        ``Bounds`` instances.

    """
    return crown(
        fun_or_jaxpr,
        compute_bounds=_ibp_bounds,
        in_batch_axes=in_batch_axes,
        out_batch_axes=out_batch_axes,
    )


@fun_to_jaxpr_for_lirpa(lirpa_name="alpha-CROWN")
def alpha_crown(
    target: Callable | ClosedJaxpr,
    optim: optax.GradientTransformation | None = None,
    steps: int = 10,
    default_rng_key: int | PRNGKeyArray = 0,
    compute_bounds: COMPUTE_BOUNDS = _ibp_bounds,
    param_strategies: (
        BACKWARDS_CROWN_PARAM_STRATEGIES | dict[Var, BACKWARDS_CROWN_PARAM_STRATEGIES]
    ) = "external-full",
    in_batch_axes: int | None | tuple[int | None, ...] = ...,
    out_batch_axes: int | None | tuple[int | None, ...] = ...,
) -> Callable[[Bounds | jax.Array, ...], tuple[Bounds | jax.Array, ...]]:
    """Creates a function that performs alpha-CROWN on ``target``.

    The returned function computes optimized CROWN bounds [Xu et al., 2021].
    For this, it first computes fixed bounds using ``compute_bounds`` and then repeatedly
    applies ``backwards_crown``, while optimizing parameters of the ``backwards_crown``
    relaxation using gradient descent.
    See ``backwards_crown`` and [Xu et al., 2021] for more details.

    [Xu et al., 2021]: Kaidi Xu, Huan Zhang, Shiqi Wang, Yihan Wang, Suman Jana,
        Xue Lin, Cho-Jui Hsieh: Fast and Complete: Enabling Complete Neural Network
        Verification with Rapid and Massively Parallel Incomplete Verifiers. ICLR 2021

    Args:
        target: Function or jaxpr to compute bounds on.
            If ``target`` is a function, the arguments and the return value of
            the function may be any mix of arrays,
            scalars, and standard Python containers (JAX pytrees).
        optim: The optimizer to use for optimization.
            Uses ``optax.adam(1e-3)`` by default.
        steps: The number of gradient descent steps (gradient updates) to perform.
        default_rng_key: The default random number generator key to use for sampling
            initial parameters.
            This value can be overriden using the ``rng_key`` argument of the returned
            function.
        compute_bounds: The function to compute bounds on the values in ``fun``.
            Uses ``ibp`` by default.
        param_strategies: The parameter selection strategies each ``backwards_crown`` parameter.
            If ``target`` is a ``Callable``, this must be a single parameter selection
            strategy that is applied for all ``backwards_crown`` parameters.
            If ``target`` is a jaxpr, this can either be a single parameter selection
            strategy for every variable in the jaxpr, or a dictionary
            mapping variables in the jaxpr to a parameter selection strategies.
            By default, uses ``"external-full"`` for all parameters.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
        in_batch_axes: Batch axes of the inputs.
            See ``backwards_crown`` for more details.
        out_batch_axes: Batch axes of the outputs.
            See ``backwards_crown`` for more details.

    Returns:
        A function ``fun_alpha_crown`` that computes output bounds on ``target`` given
        input bounds on some of the arguments of ``target``.
        When calling ``fun_alpha_crown``, pass ``Bounds`` instances for the arguments with
        input bounds while passing regular ``jax.Arrays`` for the arguments without
        input bounds.
        The output of ``fun_alpha_crown`` has the same structure as the output of ``target``,
        except that all outputs which depends on arguments with input bounds are
        ``Bounds`` instances.
        Use the ``rng_key`` argument of ``fun_alpha_crown`` to specify the random number
        generator key to use for sampling initial parameters.
    """
    jaxpr = target  # fun_to_jaxpr_for_lirpa handles functions

    if optim is None:
        optim = optax.adam(1e-3)

    bwcrown, param_domains = backwards_crown(
        jaxpr,
        compute_bounds,
        param_strategies,
        in_batch_axes=in_batch_axes,
        out_batch_axes=out_batch_axes,
    )

    def init_params(key):
        sub_keys = jax.random.split(key, len(param_domains))
        return tuple(
            dom.uniform_sample(key).squeeze(0)
            for dom, key in strict_zip(param_domains, sub_keys)
        )

    bwcrown = jax.jit(bwcrown)

    @jax.jit
    def loss_fn(params, *args):
        lb_params, ub_params = params
        lbs = [out_bounds.lower_bound for out_bounds in bwcrown(lb_params, *args)]
        ubs = [out_bounds.upper_bound for out_bounds in bwcrown(ub_params, *args)]
        # Goal: minimize ub, maximize lb.
        # Since params_lb and params_ub are independent, we can
        # optimise them with one loss.
        return sum(jnp.sum(ub) for ub in ubs) - sum(jnp.sum(lb) for lb in lbs)

    def alpha_crown_fun(
        *args: Bounds | jax.Array, rng_key: int | PRNGKeyArray = default_rng_key
    ):
        if isinstance(rng_key, int):
            rng_key = jax.random.PRNGKey(rng_key)

        lb_key, ub_key = jax.random.split(rng_key)
        lb_params, ub_params = init_params(lb_key), init_params(ub_key)
        params = (lb_params, ub_params)
        optim_state = optim.init(params)

        for _ in range(steps):
            grads = jax.grad(loss_fn)(params, *args)
            updates, optim_state = optim.update(grads, optim_state)
            params = optax.apply_updates(params, updates)

        lbs = bwcrown(lb_params, *args)
        ubs = bwcrown(ub_params, *args)
        # combine separate lower and upper bounds
        return tuple(
            LiRPABounds(
                lb_weights=lb.lb_weights,
                ub_weights=ub.ub_weights,
                lb_bias=lb.lb_bias,
                ub_bias=ub.ub_bias,
                domain=lb.domain,
                full_in_shapes=lb.full_in_shapes,
                full_out_shape=lb.full_out_shape,
                batch_axis_mappings=lb.batch_axis_mappings,
                in_batch_shapes=lb.in_batch_shapes,
                in_shapes=lb.in_shapes,
                out_shape=lb.out_shape,
                out_batch_axes=lb.out_batch_axes,
            )
            for lb, ub in strict_zip(lbs, ubs)
        )

    return alpha_crown_fun


@functools.singledispatch
def backwards_crown(target, *_, **__):
    """Creates a function that performs backwards CROWN on the ``target``.

    If ``target`` is a ``Callable``, this function first converts ``target`` into a
    jaxpr.
    The description below assumes that ``target`` is the jaxpr resulting from the
    conversion.

    The returned function propagates linear bounds backwards through
    the jaxpr ``target`` using ``compute_bounds`` to compute constant bounds on the inputs
    of each equation (these bounds are also called "pre-activation bounds").
    Propagating linear bounds requires overapproximating non-linear components
    of the ``target`` jaxpr using a linear relaxation.

    By default, this function uses the CROWN linear relaxations [Zhang et al., 2018]
    for overapproximating non-linear components (``parameters="standard"``).
    It also supports the ``\\alpha``-CROWN parameterized linear
    relaxations [Xu et al., 2021] (``parameters="external"``).

    Whether to use standard or external parameters can also be specified for
    each variable in the ``target`` jaxpr individually using the ``parameters`` argument.
    Individual primitives can support additional parameter strategies beyond
    ``"default"``, and ``"external"``.

    The returned function takes relaxation parameters and bounds on the input
    variables of ``target`` jaxpr and returns affine bounds for the output variables
    of ``target``.
    See ``CROWNCallable`` for more details.
    The number of relaxation parameters and their domains are determined via the
    second return value of this function.

    [Zhang et al., 2018]: Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh,
        Luca Daniel: Efficient Neural Network Robustness Certification with
        General Activation Functions. NeurIPS 2018: 4944-4953

    [Xu et al., 2021]: Kaidi Xu, Huan Zhang, Shiqi Wang, Yihan Wang, Suman Jana,
        Xue Lin, Cho-Jui Hsieh: Fast and Complete: Enabling Complete Neural Network
        Verification with Rapid and Massively Parallel Incomplete Verifiers. ICLR 2021

    Args:
        target: The callable or jaxpr to propagate bounds through.
        example_args: If ``target`` is a ``Callable``, the ``example_args``
            argument provides example arguments for converting ``target`` into a
            jaxpr.
            If ``target`` is already a jaxpr, this argument must be omitted.
        compute_bounds: A callable that computes preliminary bounds on each
            variable of a jaxpr.
            This callable is also used as a key for caching.
            For this to work reliably, you should define this function
            at the top level so that it has a consistent hash.
        param_strategies: Either a single parameter selection strategy for the
            entire ``target`` or a mapping from variables to parameter
            selection strategies.
            If a mapping is used, variables not contained in the mapping are
            assigned the ``"default"`` parameter selection strategy.

            Use ``"default"`` to use the standard CROWN linear relaxation
            [Zhang et al., 2018] and ``"external"`` to use the
            ``\\alpha``-CROWN parameterized linear relaxation [Xu et al., 2021].
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more parameter selection
            strategies.
        in_batch_axes: An integer, ``None``, or tuple specifying which axes of
            each input of ``target`` is a batch axis.
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
            each output of ``target`` is a batch axis.
            If no ``out_batch_axes`` are specified, the batch axes are inferred using
            ``infer_batch_axes``.
            The format of ``out_batch_axes`` is as for ``in_batch_axes``.

    Returns:
        A function that performs backwards CROWN on ``target`` and
        a tuple of domains for the ``\\alpha``-CROWN parameters.
        When using the ``"default"`` parameter selection strategy,
        the second return value is an empty tuple.
    """
    # this will only be called when target is neither a callable nor a jaxpr
    # (see functoosl.singledispatch)
    raise ValueError(
        f"Can not perform backwards CROWN on {target}. Provide a jaxpr or a callable."
    )


@backwards_crown.register(Callable)
def _backwards_crown_fun(
    target: Callable,
    example_args: tuple[PyTree[Bounds | jax.Array], ...],
    compute_bounds: COMPUTE_BOUNDS,
    param_strategies: (
        BACKWARDS_CROWN_PARAM_STRATEGIES | dict[Var, BACKWARDS_CROWN_PARAM_STRATEGIES]
    ) = "default",
    in_batch_axes: int | None | tuple[int | None, ...] = ...,
    out_batch_axes: int | None | tuple[int | None, ...] = ...,
) -> tuple[
    Callable[
        [tuple[Real[Array, "..."], ...], PyTree[Real[Array, "..."] | Bounds], ...],
        PyTree[AffineBounds[Array]],
    ],
    tuple[PARAM_DOMAIN, ...],
]:
    jaxpr, _, out_tree = make_jaxpr_with_bounds(target, example_args, {})
    crown_fn, param_domains = _backwards_crown_jaxpr(
        jaxpr, compute_bounds, param_strategies, in_batch_axes, out_batch_axes
    )

    # jax_util.wraps docstring argument:
    # {{fun}} placeholder is the function name; {{doc}} is the original docstring.
    docstring = (
        "Computes output lower and upper bounds of {{fun}} using backwards CROWN.\n\n\n"
        "Original documentation of {{fun}}\n"
        "--------------------------------------------------------------------------------\n"
        "\n {{doc}}"
    )

    @bounding_wrapper_of(target, docstring, "backwards CROWN")
    def wrapper_fun(
        params: Sequence[Array], *args: PyTree[Bounds | Array]
    ) -> PyTree[AffineBounds[Array]]:
        args_flat, _ = jax.tree.flatten(args, is_leaf=is_bounds)
        out_bounds_flat = crown_fn(params, *args_flat)
        return jax.tree.unflatten(out_tree, out_bounds_flat)

    return wrapper_fun, param_domains


@backwards_crown.register(ClosedJaxpr)
def _backwards_crown_jaxpr(
    target: ClosedJaxpr,
    compute_bounds: COMPUTE_BOUNDS,
    param_strategies: (
        BACKWARDS_CROWN_PARAM_STRATEGIES | dict[Var, BACKWARDS_CROWN_PARAM_STRATEGIES]
    ) = "default",
    in_batch_axes: int | None | tuple[int | None, ...] = ...,
    out_batch_axes: int | None | tuple[int | None, ...] = ...,
) -> tuple[
    CROWNCallable[tuple[Real[Array, "..."], ...]],
    tuple[PARAM_DOMAIN, ...],
]:
    jaxpr = target

    # these strategies do not have parameters, skip collecting domains
    no_external_params = param_strategies in ("default", "fixed", "adaptive")

    def resolve_alias(strategy):
        match strategy:
            case "external":
                return "external-full"
            case _:
                return strategy

    if isinstance(param_strategies, str):
        param_strategies_ = resolve_alias(param_strategies)
        param_strategies = defaultdict(lambda: param_strategies_)
    else:
        param_strategies = {v: resolve_alias(s) for v, s in param_strategies.items()}
        param_strategies = defaultdict(lambda: "default", param_strategies)

    if no_external_params:
        param_domains = {}
    else:
        param_domains = _collect_parameter_domains(jaxpr, param_strategies)

    def get_params(
        eqn: JaxprEqn, params: PyTree[Real[Array, "..."]]
    ) -> PyTree[Real[Array, "..."]]:
        """
        Assumes that ``params`` is unflattened.
        """
        if eqn.primitive == primitives.jit_p:
            # Pass on all parameters for nested Jaxprs.
            return params

        def get(var: Var):
            if var in param_domains:
                try:
                    # We project the parameters here to guarantee soundness
                    domain = param_domains[var]
                    vals = params[var]
                    vals = jax.tree.map(lambda v, dom: dom.project(v), [vals], [domain])
                    return vals[0]
                except KeyError as ex:
                    raise ValueError(
                        f"No parameter value for variable {var} in {params}."
                    ) from ex
            else:
                return param_strategies[var]

        return [get(var) for var in eqn.outvars]

    bwlirpa = backwards_lirpa(
        jaxpr,
        _primitive_backward_crown_rules,
        compute_bounds,
        get_params,
        in_batch_axes=in_batch_axes,
        out_batch_axes=out_batch_axes,
    )

    flat_domains, params_pytree = jax.tree.flatten(
        param_domains, is_leaf=is_param_domain
    )

    def wrapper_fun(
        params: tuple[Real[Array, "..."], ...], *args: Bounds | Array
    ) -> tuple[AffineBounds[Real[Array, "..."] | Zero], ...]:
        params = jax.tree.unflatten(params_pytree, params)
        return bwlirpa(params, *args)

    wrapper_fun.__doc__ = CROWNCallable.__doc__
    return wrapper_fun, tuple(flat_domains)


def _collect_parameter_domains(
    jaxpr: ClosedJaxpr,
    param_strategies: dict[Var, BACKWARDS_CROWN_PARAM_STRATEGIES],
) -> VarDict[PARAM_DOMAIN]:
    """Collects the parameter domains of the CROWN rules applied for a ``jaxpr``."""
    jaxpr = jaxpr.jaxpr
    domains = {}

    for eqn in jaxpr.eqns:
        prim = markup_primitive(eqn)
        if prim in _primitive_backward_crown_rules:
            rule = _primitive_backward_crown_rules[prim]

            in_shapes = [var.aval.shape for var in eqn.invars]
            out_shapes = [var.aval.shape for var in eqn.outvars]
            strategies = [param_strategies[var] for var in eqn.outvars]
            param_domains = rule.parameter_domains(
                in_shapes, out_shapes, strategies, **eqn.params
            )

            for var, domain in strict_zip(eqn.outvars, param_domains):
                if domain is not None:
                    assert var not in domains
                    domains[var] = domain
        elif prim == primitives.jit_p:
            # Jaxpr variables are globally unique, so we can put all parameters
            # in one domains dictionary.
            domains |= _collect_parameter_domains(eqn.params["jaxpr"], param_strategies)
    return VarDict(domains)


# this dict needs to be hashable for caching in backwards_lirpa
_primitive_backward_crown_rules: frozendict[Primitive | Marker, BackwardsCROWNRule] = (
    frozendict()
)


def register_backwards_crown_rule(
    primitive: Primitive | Marker,
    rule: BackwardsLiRPARule | BackwardsCROWNRule,
):
    """Registers a new backwards CROWN rule.

    Args:
        primitive: The primitive (or marker) that is handled by this rule.
        rule: The CROWN rule for ``primitive``.
    """
    if not hasattr(rule, "parameter_domains"):  # not a BackwardsCROWNRule

        class _BWCROWNRule(BackwardsCROWNRule):
            def __init__(self, lirpa_rule: BackwardsLiRPARule):
                self.lirpa_rule = lirpa_rule

            def __call__(self, *args, **kwargs):
                return self.lirpa_rule(*args, **kwargs)

            def parameter_domains(
                self,
                in_shapes: Sequence[tuple[int, ...]],
                out_shapes: Sequence[tuple[int, ...]],
                strategies: Sequence[BACKWARDS_CROWN_PARAM_STRATEGIES],
                **kwargs,
            ) -> tuple[PARAM_DOMAIN | None, ...]:
                return (None,) * len(out_shapes)

        rule = _BWCROWNRule(rule)

    global _primitive_backward_crown_rules
    _primitive_backward_crown_rules = _primitive_backward_crown_rules | {
        primitive: rule
    }


# ==============================================================================
# CROWN Primitive Rules
# ==============================================================================


class NonlinearBackwardsCROWNRule(BackwardsCROWNRule):
    """A template backwards CROWN rule for element-wise non-linear functions.

    This class only supports a single parameter.
    Based on ``nonlinear_backwards_lirpa_rule``.

    Args:
        compute_params: The function that computes the parameter values of this
            relaxation.
            This function takes the external parameter values or a parameter
            selection strategy, the input bounds, the output bounds, and
            any keyword arguments of the underlying primitive as arguments.
            This function returns a tuple of parameter tuples, which are
            the lower and upper bound parameters for each input of the primitive.
        param_bounds: The bounds on each individual parameter value of this
            relaxation.
    """

    def __init__(
        self,
        compute_params: Callable[
            [
                PARAM_VALUE,
                Sequence[Bounds | jax.Array],
                ...,
            ],
            tuple[
                tuple[jax.Array, jax.Array],
                ...,
                tuple[jax.Array | Zero, jax.Array | Zero],
            ],
        ],
        param_bounds: PARAM_DOMAIN,
    ):
        self.__compute_params = compute_params
        self.__param_domain = param_bounds

    def __call__(
        self,
        params: Sequence[PARAM_VALUE],
        *args,
        **kwargs,
    ) -> tuple[LiRPABounds[jax.Array | Zero], ...] | LiRPABounds[jax.Array | Zero]:
        assert len(params) == 1, "Multiple params for single-output function."
        return nonlinear_backwards_lirpa_rule(
            self.__compute_params, params[0], *args, **kwargs
        )

    def parameter_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        strategies: Sequence[BACKWARDS_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        assert len(strategies) == 1, (
            "Multiple parameter strategies for single-output function."
        )
        strategy = strategies[0]
        in_shape = in_shapes[0]
        match strategy:
            case "default" | "fixed" | "adaptive":
                return (None,)
            case "external" | "external-full":
                full_domain = jax.tree.map(
                    lambda domain: Box(
                        jnp.full(in_shape, domain.lower_bound),
                        jnp.full(in_shape, domain.upper_bound),
                    ),
                    self.__param_domain,
                    is_leaf=is_bounds,
                )
                return (full_domain,)
            case "external-shared":
                return (self.__param_domain,)
            case _:
                raise ValueError(f"Unsupported parameter strategy: {strategy}")


def crown_unary_convex_params(
    fun: Callable[[jax.Array, ...], jax.Array],
    params: jax.Array | BACKWARDS_CROWN_PARAM_STRATEGIES,
    in_bounds: Bounds,
    **kwargs,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Parameters for ``unary_nonlinear_backwards_lirpa_rule`` for unary convex functions.

    Computes the ``alpha_lb``, ``alpha_ub``, ``beta_lb``, ``beta_ub`` parameters for
    a unary convex function.
    For use with ``NonlinearBackwardsCROWNRule``.

    The rule for convex functions is part of the rule for S-shaped
    functions (sigmoid, tanh, arctan), when the input upper bound is
    non-positive (Table 2 and 3 [Zhang et al.,2018]).

    The rule for convex functions requires choosing a point `d` within the input
    bounds that determines the lower bound.
    The lower bound is the tangent at `d` of the function.
    We compute this tangent slope using ``jax.jvp``.

    For choosing ``d``, ``crown_unary_convex_params`` implements two variants:
     - by default and when ``params="fixed"``, ``d`` is selected as the midpoint
       of the input bounds.
     - when external parameters are used, `params` is a value between 0 and 1
       and ``d = in_lb + (in_ub - in_lb) * params``.

    [Zhang et al., 2018]: Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, Luca Daniel:
        Efficient Neural Network Robustness Certification with General Activation Functions. NeurIPS 2018: 4944-4953
        https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html

    Args:
        fun: The unary convex function.
        params: External parameters for ``\\alpha``-CROWN or
            a parameter selection strategy.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
        in_bounds: ``Bounds`` on the input of ``fun``.
        **kwargs: Any keyword arguments of ``fun``.

    Returns:
        The ``alpha_lb``, ``alpha_ub``, ``beta_lb``, and ``beta_ub`` parameters
        for the non-linear backwards CROWN rule.
        See ``NonlinearBackwardsCROWNRule``.
    """
    in_lb, in_ub = in_bounds
    f_in_lb, f_in_ub = fun(in_lb, **kwargs), fun(in_ub, **kwargs)

    alpha_ub = (f_in_ub - f_in_lb) / (in_ub - in_lb)
    beta_ub = f_in_lb - alpha_ub * in_lb
    match params:
        case "default" | "fixed":
            # use the midpoint
            d = (in_ub + in_lb) / 2
        case "adaptive":
            raise ValueError(
                "Adaptive parameters not supported for convex and concave functions."
            )
        case _:
            d = in_lb + (in_ub - in_lb) * params
    f_d, alpha_lb = jax.jvp(partial(fun, **kwargs), (d,), (jnp.ones_like(d),))
    beta_lb = f_d - alpha_lb * d

    return (alpha_lb, alpha_ub), (beta_lb, beta_ub)


def crown_unary_concave_params(
    fun: Callable[[jax.Array, ...], jax.Array],
    params: jax.Array | BACKWARDS_CROWN_PARAM_STRATEGIES,
    in_bounds: Bounds,
    **kwargs,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Parameters for ``unary_nonlinear_backwards_lirpa_rule`` for unary concave functions.

    Computes the ``alpha_lb``, ``alpha_ub``, ``beta_lb``, ``beta_ub`` parameters for
    a unary concave function.
    For use with ``unary_nonlinear_backwards_lirpa_rule``.

    The rule for concave functions is part of the rule for S-shaped
    functions (sigmoid, tanh, arctan), when the input upper bound is
    non-negative (Table 2 and 3 [Zhang et al.,2018]).

    The rule for concave functions requires choosing a point `d` within the input
    bounds that determines the lower bound.
    The upper bound is the tangent at `d` of the function.
    See ``crown_unary_convex_params`` for more details.

    [Zhang et al., 2018]: Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, Luca Daniel:
        Efficient Neural Network Robustness Certification with General Activation Functions. NeurIPS 2018: 4944-4953
        https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html

    Args:
        fun: The unary concave function.
        params: External parameters for ``\alpha``-CROWN or a parameter selection strategy.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
        in_bounds: ``Bounds`` on the input of ``fun``.
        **kwargs: Any keyword arguments of ``fun``.

    Returns:
        The ``alpha_lb``, ``alpha_ub``, ``beta_lb``, and ``beta_ub`` parameters
        for the non-linear backwards CROWN rule.
        See ``NonlinearBackwardsCROWNRule``.
    """

    # if fun is concave, -fun is convex
    def neg_fun(x, **kwargs):
        return -fun(x, **kwargs)

    (alpha_lb, alpha_ub), (beta_lb, beta_ub) = crown_unary_convex_params(
        neg_fun, params, in_bounds, **kwargs
    )
    # Switch lower and upper bounds and negate the parameters
    alpha_lb, beta_lb, alpha_ub, beta_ub = -alpha_ub, -beta_ub, -alpha_lb, -beta_lb
    return (alpha_lb, alpha_ub), (beta_lb, beta_ub)


def crown_unary_s_shaped_params(
    fun: Callable[[jax.Array, ...], jax.Array],
    inflection_point: float,
    params: tuple[Array, Array] | BACKWARDS_CROWN_PARAM_STRATEGIES,
    in_bounds: Bounds,
    **kwargs,
):
    """Parameters for ``NonlinearBackwardsCROWNRule`` for unary S-shaped functions.

    S-shaped functions are functions like sigmoid and tanh, that consist of a convex
    part that is reflected at the inflection point yielding a second, concave part.
    If the function spans both sides of the inflection point, we say that  the function
    is in its mixed phase.

    Computes the ``alpha_lb``, ``alpha_ub``, ``beta_lb``, ``beta_ub`` parameters for
    a unary S-shaped function.
    The rule for S-shaped functions is given in Table 2 and 3 of Zhang et al. [2018].

    This rule requires choosing the slopes of the lower and upper bounds on ``fun``.
    If the function is convex within the input bounds, only the lower bound can be
    parameterized.
    The parameter for the upper bound is ignored in this setting.
    The lower bound slope is chosen as in ``crown_unary_convex_params``.
    Similarly, if the function is concave within the input bounds, only the upper bound
    can be parameterized.
    The upper bound slope is chosen as in ``crown_unary_concave_params``.

    If the function is in its mixed phase, both the lower and upper bounds can be
    parameterized.
    Currently, this function only implements fixed parameters for the mixed phase:
     - The lower bound slope is the tangent slope at the input lower bound.
       Similarly, the upper bound slope is the tangent slope at the input upper bound.
     - TODO: external parameters.

    [Zhang et al., 2018]: Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, Luca Daniel:
        Efficient Neural Network Robustness Certification with General Activation Functions. NeurIPS 2018: 4944-4953
        https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html

    Args:
        fun: The unary S-shaped function.
        inflection_point: The mid-point of the S-shape where ``fun`` changes
            from convex to concave.
            For sigmoid and tanh, the inflection point is zero.
        params: External parameters for ``\alpha``-CROWN or a parameter selection strategy.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
            This value is passed on to ``choose_tangent``.
        in_bounds: ``Bounds`` on the input of ``fun``.
        **kwargs: Any keyword arguments of ``fun``.

    Returns:
        The ``alpha_lb``, ``alpha_ub``, ``beta_lb``, and ``beta_ub`` parameters
        for the non-linear backwards CROWN rule.
        See ``NonlinearBackwardsCROWNRule``.
    """
    in_lb, in_ub = in_bounds
    # Note: Picking the lower/upper bound for the mixed case may be
    # incorrect in some cases, but this is corrected later at (*).
    mixed_phase = (in_lb < inflection_point) & (in_ub > inflection_point)
    match params:
        case "default" | "fixed":
            # Implementation follows the most simple case in
            # https://github.com/Verified-Intelligence/auto_LiRPA/blob/bc476502b64621c23d4fa8e5965a4bead4f80fae/auto_LiRPA/operators/tanh.py
            d_lb = d_ub = (in_ub + in_lb) / 2
            d_lb = jnp.where(mixed_phase, in_lb, d_lb)
            d_ub = jnp.where(mixed_phase, in_ub, d_ub)
        case "adaptive":
            raise ValueError(
                "Adaptive parameters not supported for s-shaped functions."
            )
        case _:
            lb_params, ub_params = params
            d_lb = in_lb + (in_ub - in_lb) * lb_params
            d_ub = in_lb + (in_ub - in_lb) * ub_params
            # TODO: parameterization for mixed phase (issue #41)
            d_lb = jnp.where(mixed_phase, in_lb, d_lb)
            d_ub = jnp.where(mixed_phase, in_ub, d_ub)

    f_d_lb, f_prime_d_lb = jax.jvp(
        partial(fun, **kwargs), (d_lb,), (jnp.ones_like(d_lb),)
    )
    f_d_ub, f_prime_d_ub = jax.jvp(
        partial(fun, **kwargs), (d_ub,), (jnp.ones_like(d_ub),)
    )

    in_lb, in_ub = in_bounds
    f_in_lb, f_prime_in_lb = jax.jvp(
        partial(fun, **kwargs), (in_lb,), (jnp.ones_like(in_lb),)
    )
    f_in_ub, f_prime_in_ub = jax.jvp(
        partial(fun, **kwargs), (in_ub,), (jnp.ones_like(in_ub),)
    )

    convex_alpha_ub = (f_in_ub - f_in_lb) / (in_ub - in_lb)
    convex_beta_ub = f_in_lb - convex_alpha_ub * in_lb
    convex_alpha_lb = f_prime_d_lb
    convex_beta_lb = f_d_lb - convex_alpha_lb * d_lb

    concave_alpha_lb = convex_alpha_ub
    concave_beta_lb = convex_beta_ub
    concave_alpha_ub = f_prime_d_ub
    concave_beta_ub = f_d_ub - concave_alpha_ub * d_ub

    mixed_alpha_ub = f_prime_d_ub
    mixed_beta_ub = f_d_ub - mixed_alpha_ub * d_ub
    mixed_alpha_lb = f_prime_d_lb
    mixed_beta_lb = f_d_lb - mixed_alpha_lb * f_d_lb
    # (*) In some cases, the direct line between fun at lower bound and fun
    # at upper bound is a better upper/lower bound than the chosen tangent
    # line (in some cases, it is also required to use the direct line).
    direct_alpha = convex_alpha_ub
    direct_beta = convex_beta_ub
    mixed_alpha_ub = jnp.where(
        direct_alpha < f_prime_in_ub, direct_alpha, mixed_alpha_ub
    )
    mixed_beta_ub = jnp.where(direct_alpha < f_prime_in_ub, direct_beta, mixed_beta_ub)
    mixed_alpha_lb = jnp.where(
        direct_alpha < f_prime_in_lb, direct_alpha, mixed_alpha_lb
    )
    mixed_beta_lb = jnp.where(direct_alpha < f_prime_in_lb, direct_beta, mixed_beta_lb)

    is_convex = in_ub <= inflection_point
    is_concave = in_lb >= inflection_point

    def select(convex_p, concave_p, mixed_p):
        return jnp.where(is_convex, convex_p, jnp.where(is_concave, concave_p, mixed_p))

    alpha_lb = select(convex_alpha_lb, concave_alpha_lb, mixed_alpha_lb)
    alpha_ub = select(convex_alpha_ub, concave_alpha_ub, mixed_alpha_ub)
    beta_lb = select(convex_beta_lb, concave_beta_lb, mixed_beta_lb)
    beta_ub = select(convex_beta_ub, concave_beta_ub, mixed_beta_ub)

    return (alpha_lb, alpha_ub), (beta_lb, beta_ub)


def crown_relu_params(
    params: Real[Array, "..."] | BACKWARDS_CROWN_PARAM_STRATEGIES,
    in_bounds: Bounds,
    **__,
) -> tuple[
    tuple[Real[Array, "..."], Real[Array, "..."]], tuple[Zero, Real[Array, "..."]]
]:
    """Computes the parameters of the CROWN ReLU relaxation.

    Computes the ``\\underline{\\alpha}``, ``\\overline{\\alpha}``,
    ``\\underline{\\beta}}, and ``\\overline{\\beta}`` parameters of the
    CROWN and ``\\alpha-CROWN`` ReLU rules [Zhang et al., 2018, Xu et al., 2021].

    In case of ``\\alpha``-CROWN, the ``\\underline{\\alpha}`` parameter is
    supplied externally via the ``params`` argument.

    This function is intended to be used with ``NonlinearBackwardsCROWNRule``.

    [Zhang et al., 2018]: Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh,
        Luca Daniel:
        Efficient Neural Network Robustness Certification with General Activation Functions.
        NeurIPS 2018: 4944-4953
        https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html

    [Xu et al., 2021]: Kaidi Xu, Huan Zhang, Shiqi Wang, Yihan Wang, Suman Jana,
        Xue Lin, Cho-Jui Hsieh: Fast and Complete: Enabling Complete Neural Network
        Verification with Rapid and Massively Parallel Incomplete Verifiers. ICLR 2021

    Args:
        params: External parameters for ``\\alpha``-CROWN or a parameter selection strategy.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
        in_bounds: See ``NonlinearBackwardsCROWNRule``.

    Returns:
        The ``alpha_lb``, ``alpha_ub``, ``beta_lb``, and ``beta_ub`` parameters
        for the non-linear backwards CROWN rule.
        See ``NonlinearBackwardsCROWNRule``.
    """
    assert is_bounds(in_bounds)
    in_lb, in_ub = in_bounds

    # weight and bias for undetermined relus (lb < 0 < ub)
    alpha_ub = in_ub / (in_ub - in_lb)
    beta_ub = -in_lb * alpha_ub
    # the slope of the lower bound can be parameterized
    match params:
        case "default" | "adaptive":
            # standard adaptive CROWN strategy
            # (minimizes area between lower bound and ReLU)
            alpha_lb = jnp.where(in_ub >= -in_lb, 1, 0)
        case "fixed":
            # parallel bounds (as in the FastLin relaxation)
            alpha_lb = alpha_ub
        case _:  # external value / alpha-CROWN
            assert isinstance(params, jax.Array)
            # params can be a scalar for the external-shared strategy
            alpha_lb = jnp.broadcast_to(params, in_lb.shape)
    beta_lb = Zero()

    # Account for fixed active/inactive ReLUs
    relu_active = in_lb >= 0
    relu_inactive = in_ub <= 0
    alpha_ub = jnp.where(relu_active, 1, jnp.where(relu_inactive, 0, alpha_ub))
    beta_ub = jnp.where(relu_active | relu_inactive, 0, beta_ub)
    alpha_lb = jnp.where(relu_active, 1, jnp.where(relu_inactive, 0, alpha_lb))

    return (alpha_lb, alpha_ub), (beta_lb, beta_ub)


def crown_max_params(
    params: Real[Array, "..."] | BACKWARDS_CROWN_PARAM_STRATEGIES,
    x_bounds: Bounds | Real[Array, "..."],
    y_bounds: Bounds | Real[Array, "..."],
    **__,
) -> tuple[
    tuple[Real[Array, "..."], Real[Array, "..."]],
    tuple[Real[Array, "..."], Real[Array, "..."]],
    tuple[Zero, Real[Array, "..."]],
]:
    """A generalization of the CROWN rule for ReLU to two arguments based on the
    MILP enconding of max in [Tjeng et al., 2019].

    These parameters reduce to the ReLU bounds if one argument is set to zero.

    [Tjeng et al., 2019]: Vincent Tjeng, Kai Yuanqing Xiao, Russ Tedrake:
        Evaluating Robustness of Neural Networks with Mixed Integer Programming.
        ICLR (Poster) 2019.
        https://openreview.net/forum?id=HyGIdiRqtm

    Args:
        params: External parameters for ``\\alpha``-CROWN or a parameter selection strategy.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
        x_bounds: Constant bounds on the first argument or a concrete value.
        y_bounds: Constant bounds on the second argument or a concrete value.

    Returns:
        The weight and bias parameters (``alpha_lb``, ``alpha_ub``, etc.)
        for the non-linear backwards CROWN rule.
        See ``NonlinearBackwardsCROWNRule``.
    """
    (x_lb, x_ub), (y_lb, y_ub) = all_as_bounds(x_bounds, y_bounds)

    # Terminology:
    # We use alpha for the weight of input x and beta for the weight of input y.
    # We use gamma for the bias.

    # Weight and bias parameters for undetermined maxs (bounds of x and y intersect)
    denominator = x_ub - x_lb + y_ub - y_lb
    alpha_ub = (x_ub - y_lb) / denominator
    beta_ub = (y_ub - x_lb) / denominator
    gamma_ub = (x_ub * y_ub + x_lb * y_lb - x_lb * x_ub - y_lb * y_ub) / denominator
    # The lower bound has the form dx + (1-a)y where a in [0, 1].
    # There are several strategies for selecting a:
    match params:
        case "default" | "adaptive":
            # Adaptive lower bound rule for max (generalization of ReLU rule)
            # The rule follows from computing the integral between max(x,y) and dx + (1-a)y.
            a = jnp.where(x_ub - y_ub + x_lb - y_lb >= 0, 1, 0)
        case "fixed":
            a = 0.5
        case _:  # external value / alpha-CROWN
            assert isinstance(params, jax.Array)
            a = jnp.broadcast_to(params, x_lb.shape)
    alpha_lb = a  # weight for x
    beta_lb = 1 - a  # weight for y
    gamma_lb = Zero()

    # Now account for cases where x_lb >= y_ub or y_lb >= x_ub
    x_dominates = x_lb >= y_ub
    y_dominates = y_lb >= x_ub
    alpha_ub = jnp.where(y_dominates, 0, jnp.where(x_dominates, 1, alpha_ub))
    beta_ub = jnp.where(y_dominates, 1, jnp.where(x_dominates, 0, beta_ub))
    gamma_ub = jnp.where(x_dominates | y_dominates, 0, gamma_ub)
    alpha_lb = jnp.where(y_dominates, 0, jnp.where(x_dominates, 1, alpha_lb))
    beta_lb = jnp.where(y_dominates, 1, jnp.where(x_dominates, 0, beta_lb))

    return (alpha_lb, alpha_ub), (beta_lb, beta_ub), (gamma_lb, gamma_ub)


def crown_min_params(
    params: Real[Array, "..."] | BACKWARDS_CROWN_PARAM_STRATEGIES,
    x_bounds: Bounds | Real[Array, "..."],
    y_bounds: Bounds | Real[Array, "..."],
    **__,
) -> tuple[
    tuple[Real[Array, "..."], Real[Array, "..."]],
    tuple[Real[Array, "..."], Real[Array, "..."]],
    tuple[Real[Array, "..."], Zero],
]:
    """A CROWN rule for the minimum between to two arguments analogously to
    ``crown_max_params``.

    Args:
        params: External parameters for ``\\alpha``-CROWN or a parameter selection strategy.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
        x_bounds: Constant bounds on the first argument or a concrete value.
        y_bounds: Constant bounds on the second argument or a concrete value.

    Returns:
        The ``alpha_lb``, ``alpha_ub``, ``beta_lb``, and ``beta_ub`` parameters
        for the non-linear backwards CROWN rule.
        See ``NonlinearBackwardsCROWNRule``.
    """
    (x_lb, x_ub), (y_lb, y_ub) = all_as_bounds(x_bounds, y_bounds)

    # Terminology:
    # We use alpha for the weight of input x and beta for the weight of input y.
    # We use gamma for the bias.

    # undetermined mins (bounds of x and y intersect)
    denominator = x_lb - x_ub + y_lb - y_ub
    alpha_lb = (x_lb - y_ub) / denominator
    beta_lb = (y_lb - x_ub) / denominator
    gamma_lb = (x_lb * y_lb + x_ub * y_ub - x_lb * x_ub - y_lb * y_ub) / denominator
    # The upper bound has the form ax + (1-a)y where a in [0, 1].
    # There are several strategies for selecting a:
    match params:
        case "default" | "adaptive":
            # Adaptive lower bound rule for min (see max rule)
            a = jnp.where(x_ub - y_ub + x_lb - y_lb >= 0, 0, 1)
        case "fixed":
            a = 0.5
        case _:  # external value / alpha-CROWN
            assert isinstance(params, jax.Array)
            a = jnp.broadcast_to(params, x_lb.shape)
    alpha_ub = a  # weight for x
    beta_ub = 1 - a  # weight for y
    gamma_ub = Zero()

    # Now account for cases where x_lb <= y_ub or y_lb <= x_ub
    x_dominates = x_lb <= y_ub
    y_dominates = y_lb <= x_ub
    alpha_lb = jnp.where(y_dominates, 0, jnp.where(x_dominates, 1, alpha_lb))
    beta_lb = jnp.where(y_dominates, 1, jnp.where(x_dominates, 0, beta_lb))
    gamma_lb = jnp.where(x_dominates | y_dominates, 0, gamma_lb)
    alpha_ub = jnp.where(y_dominates, 0, jnp.where(x_dominates, 1, alpha_ub))
    beta_ub = jnp.where(y_dominates, 1, jnp.where(x_dominates, 0, beta_ub))

    return (alpha_lb, alpha_ub), (beta_lb, beta_ub), (gamma_lb, gamma_ub)


class _BackwardsCROWNReduceWindowMaxRule(BackwardsCROWNRule):
    """A backwards CROWN rule for `lax.reduce_window_max`.

    This rule is a generalization of the CROWN rule for ReLU to arbitrarily many
    arguments based on the MILP enconding of max in [Tjeng et al., 2019].

    This rule uses one scalar parameter per entry of the pooling window.
    For the ``external-shared`` parameter strategy, it shares one set of parameters
    for all entries of the output.
    The shape of the parameter is ``window_shape``.
    For the ``external-full`` parameter strategy, one "window" of parameters is used
    for each entry of the output.
    The shape of the parameter is ``(*output_shape, *window_shape)``.
    This rule can also use fixed parameters or choose the parameters adaptively.

    [Tjeng et al., 2019]: Vincent Tjeng, Kai Yuanqing Xiao, Russ Tedrake:
        Evaluating Robustness of Neural Networks with Mixed Integer Programming.
        ICLR (Poster) 2019.
        https://openreview.net/forum?id=HyGIdiRqtm
    """

    def __call__(
        self,
        params: Sequence[PARAM_VALUE],
        out_weights: Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
        in_bounds: Sequence[Bounds | Real[Array, "..."]],
        *,
        in_batch_axes: Sequence[tuple[int, ...]],
        out_batch_axes: Sequence[tuple[int, ...]],
        batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
        backwards_lirpa: Callable[
            [
                ClosedJaxpr,
                Sequence[LiRPAWeights[Real[Array, "..."] | Zero]],
                Bounds | Real[Array, "..."],
                ...,
            ],
            tuple[LiRPABounds[Real[Array, "..."] | Zero], ...],
        ],
        **kwargs,
    ) -> (
        tuple[LiRPABounds[Real[Array, "..."] | Zero], ...]
        | LiRPABounds[Real[Array, "..."] | Zero]
    ):
        out_weights, out_batch_axes, batch_axis_maps = _single_output_asserts(
            out_weights, out_batch_axes, batch_axis_mappings
        )
        assert len(in_bounds) == 1
        assert is_bounds(in_bounds[0])
        assert len(in_batch_axes) == 1
        in_bounds, in_batch_axes = in_bounds[0], in_batch_axes[0]
        assert len(params) == 1, "Multiple params for single-output function."
        params = params[0]
        kwargs_as_pos = (
            kwargs["window_dimensions"],
            kwargs["window_strides"],
            kwargs["padding"],
            kwargs["base_dilation"],
            kwargs["window_dilation"],
        )

        batch_shape, target_shape, out_shape = out_weights.shape_info
        full_in_shape = in_bounds.lower_bound.shape

        out_weights_reshaped = tuple(
            _reshape_as_output(ow, batch_shape, out_shape, out_batch_axes)
            for ow in out_weights
        )
        in_weights, in_bias = _backwards_crown_reduce_window_max_rule(
            out_weights_reshaped,
            params,
            in_bounds,
            out_batch_axes,
            *kwargs_as_pos,
        )
        in_lb_w, in_ub_w = tuple(
            _restore_weights_shape(w, target_shape, full_in_shape, in_batch_axes)
            for w in in_weights
        )

        def reshape_bias(b):
            # b shape: (-1, out_batch_shape) where -1 is the flattened target shape
            # Goal: full_target_shape
            b = jnp.moveaxis(b, 0, -1).reshape(batch_shape + target_shape)
            n_batch, n_target = len(batch_shape), len(target_shape)
            b = incorporate_batch_axes(b, (n_batch, 0, n_target), out_batch_axes)
            return b

        in_lb_b, in_ub_b = (reshape_bias(b) for b in in_bias)
        return out_weights.backwards_step(
            (in_lb_w,), (in_ub_w,), in_lb_b, in_ub_b, (in_bounds,), batch_axis_maps
        )

    def parameter_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        strategies: Sequence[BACKWARDS_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        assert len(strategies) == 1, (
            "Multiple parameter strategies for single-output function."
        )
        strategy = strategies[0]
        out_shape = out_shapes[0]
        window_dimensions = kwargs["window_dimensions"]

        match strategy:
            case "default" | "adaptive" | "fixed":
                return (None,)
            case "external" | "external-full":
                return (
                    Box(
                        jnp.zeros(out_shape + window_dimensions),
                        jnp.ones(out_shape + window_dimensions),
                    ),
                )
            case "external-shared":
                return (Box(jnp.zeros(window_dimensions), jnp.ones(window_dimensions)),)
            case _:
                raise ValueError(
                    f"Unsupported parameter strategy for reduce_window_max: {strategy}"
                )


@partial(jax.vmap, in_axes=[0] + [None] * 8, out_axes=0)
def _backwards_crown_reduce_window_max_rule(
    out_weight: tuple[Real[Array, "..."], Real[Array, "..."]],
    params: Real[Array, "..."],
    in_bounds: Bounds,
    out_batch_axes: tuple[int, ...],
    window_dimensions: tuple[int, ...],
    window_strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    base_dilation: tuple[int, ...],
    window_dilation: tuple[int, ...],
) -> tuple[
    tuple[Real[Array, "..."], Real[Array, "..."]],
    tuple[Real[Array, "..."], Real[Array, "..."]],
]:
    """Implement the backwards LiRPA unary nonlinear function rule for reduce_window_max.

    Argument shapes (after vmap):
        out_weights: (-1, *full_out_shape)
        in_bounds: full_in_shape
        params: broadcasting-compatible with (*full_out_shape, *window_shape)

    Return value: (in_weights_lb, in_weights_ub), (in_bias_lb, in_bias_ub)
    Output shapes (after vmap):
        in_weights_lb: (-1, *full_in_shape)
        in_weights_ub: (-1, *full_in_shape)
        in_bias_lb: (-1, *out_batch_shape)
        in_bias_ub: (-1, *out_batch_shape)
    where ``out_batch_shape = [full_out_shape[i] for i in out_batch_axes]``.
    """
    # Because of vmap, this function can not have **kwargs (would be vmapped as well)
    kwargs = {
        "window_dimensions": window_dimensions,
        "window_strides": window_strides,
        "padding": padding,
        "base_dilation": base_dilation,
        "window_dilation": window_dilation,
    }

    # We implement the unary nonlinear function rule from Table 6 in Xu et al [2020]
    # for reduce_window_max.
    # Refer to this table for the terminology used below (alpha_ub, usw).
    #
    # [Xu et al., 2020]: Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Kai-Wei Chang,
    #     Minlie Huang, Bhavya Kailkhura, Xue Lin, Cho-Jui Hsieh: Automatic Perturbation
    #     Analysis for Scalable Certified Robustness and Beyond. NeurIPS 2020
    #
    # The lower and upper bound on reduce_window_max are as follows:
    #  - We say one input dominates the others if the input lower bound for that input
    #    is larger than the input upper bound of all other inputs.
    #    If there is a dominant input, `reduce_window_max` is a linear function in the
    #    input domain.
    #    We use the dominant input as lower and upper bound.
    #    If i is the index of the dominant input, alpha_lb_i = alpha_ub_i = 1
    #    and alpha_lb_j = alpha_ub_j = 0 for j != i.
    #  - If there is no dominant input, we use a generalization of the CROWN rule
    #    for ReLU to arbitrarily many arguments based on the MILP enconding of max in
    #    [Tjeng et al., 2019].
    #     * The lower bound is parameterized with alpha_lb_i in [0, 1] and beta_lb = 0.
    #       The alpha_lb parameters are normalized to sum to 1 using softmax.
    #       The parameters can be choosen adaptively to minimize the volume between
    #       the max function and the lower bound.
    #       In this case, the lower bound is the input with the largest ub + lb value.
    #     * The upper bound is the intersection of all upper bounds of the MILP max
    #       encoding of Tjeng et al. [2019].

    # In this function, out_weights has the same shape as the output (due to vmap)
    out_lb_w, out_ub_w = out_weight
    out_lb_w_pos, out_lb_w_neg = jnp.clip(out_lb_w, min=0), jnp.clip(out_lb_w, max=0)
    out_ub_w_pos, out_ub_w_neg = jnp.clip(out_ub_w, min=0), jnp.clip(out_ub_w, max=0)
    in_lb, in_ub = in_bounds

    has_padding = any(lo > 0 or hi > 0 for lo, hi in padding)

    # If there is a dominant input, we can use tight lower and upper bounds.
    _, max_ub, any_dominant = _reduce_window_max_has_dominant_element(
        in_lb, in_ub, **kwargs
    )
    in_lb_b = in_ub_b = Zero()

    # Instead of computing alpha_lb, we directly compute the in_lb_w term of the
    # input weights, since this can be done using the reduce_window_max transpose
    # in most cases, instead of a transposed convolution.
    # The alpha_ub term is later added to the input weights.
    match params:
        case "default" | "adaptive":
            # Choose the entry with the largest mid-point.
            # This minimizes the volume between the max function and the lower bound.
            # If there is a dominant input, this input also has the largest mid value.
            mid = in_ub + in_lb
            in_lb_w = select_and_scatter_add2(mid, out_lb_w_pos, **kwargs)
            in_ub_w = select_and_scatter_add2(mid, out_ub_w_neg, **kwargs)
        case "fixed":
            # This is the default behaviour implemented in auto_LiRPA:
            # Use the input with the largest input lower bound as linear lower bound.
            in_lb_w = select_and_scatter_add2(in_lb, out_lb_w_pos, **kwargs)
            in_ub_w = select_and_scatter_add2(in_lb, out_ub_w_neg, **kwargs)
        case _:  # external value / alpha-CROWN
            alpha_lb = params
            if has_padding:
                # correct for padding: set alpha_lb values for padding entries
                # to -inf (0.0 after softmax)
                is_pad = reduce_window_patches(
                    jnp.zeros_like(in_lb), **kwargs, pad_value=jnp.array(1.0)
                )
                is_pad = jnp.reshape(is_pad, (*is_pad.shape[:-1], *window_dimensions))
                alpha_lb = jnp.broadcast_to(params, is_pad.shape)
                alpha_lb = jnp.where(is_pad, -jnp.inf, alpha_lb)
            window_axes = tuple(range(-len(window_dimensions), 0))
            alpha_lb = jax.nn.softmax(alpha_lb, axis=window_axes)
            # Zero out alpha_lb when we can use a tight lower bound
            alpha_lb = jnp.expand_dims(1 - any_dominant, window_axes) * alpha_lb

            # The select_and_scatter_add2 term is the tight bound that we can use
            # when there is a dominant input.
            # The reduce_window_conv_transpose term is the parameterized lower bound
            # that we use when there is no dominant input.
            # Only one of the terms is non-zero for each output element.
            in_lb_w = select_and_scatter_add2(
                in_lb, any_dominant * out_lb_w_pos, **kwargs
            ) + reduce_window_conv_transpose(in_lb, alpha_lb, out_lb_w_pos, **kwargs)
            in_ub_w = select_and_scatter_add2(
                in_lb, any_dominant * out_ub_w_neg, **kwargs
            ) + reduce_window_conv_transpose(in_lb, alpha_lb, out_ub_w_neg, **kwargs)

    match params:
        case "fixed":
            # For "fixed", we mimic the auto_LiRPA implementation, also for the
            # upper bound.
            # auto_LiRPA uses a constant upper bound (out_ub) unless one input
            # is dominant.

            # alpha_ub is 1 for the dominant element if there is a dominant element
            # and zero otherwise (including all dominated elements).
            in_lb_w = in_lb_w + select_and_scatter_add2(
                in_lb, any_dominant * out_lb_w_neg, **kwargs
            )
            in_ub_w = in_ub_w + select_and_scatter_add2(
                in_lb, any_dominant * out_ub_w_pos, **kwargs
            )

            out_ub = lax.reduce_window_max_p.bind(in_ub, **kwargs)
            beta_ub = (1 - any_dominant) * out_ub
        case _:
            # Step 1: Compute u_i = max_{j!=i} in_ub_j for each entry i in
            # an input window.
            # in_ub_patches shape: (*out_shape, prod(window_))
            pad_val = jnp.min(in_lb) - 1.0  # reduce_window_patches can not handle -inf
            in_ub_patches = reduce_window_patches(in_ub, **kwargs, pad_value=pad_val)
            max_idx = jnp.argmax(in_ub_patches, axis=-1)
            max_idx = (*jnp.indices(max_idx.shape), max_idx)
            max_cancelled = in_ub_patches.at[*max_idx].set(-jnp.inf)
            second_max = jnp.max(max_cancelled, axis=-1)
            u = jnp.broadcast_to(jnp.expand_dims(max_ub, axis=-1), in_ub_patches.shape)
            u = u.at[*max_idx].set(second_max)

            # Step 2: m_i = u_i - in_lb_i
            in_lb_patches = reduce_window_patches(in_lb, **kwargs, pad_value=pad_val)
            if has_padding:
                is_pad = reduce_window_patches(
                    jnp.zeros_like(in_lb), **kwargs, pad_value=jnp.array(1.0)
                )
            else:
                is_pad = jnp.full_like(in_lb_patches, fill_value=0.0)
            m = u - in_lb_patches
            # Step 3: prod(m), prod_{j!=i} m_j, and sum_i prod_{j!=i} m_j
            # Note on padding values: reduce max never selects them, so we want
            # to give them an alpha_ub value of 0.0.
            prod_m = jnp.prod(jnp.where(is_pad, 1.0, m), axis=-1)
            # Compute prod_{j!=i} m_j as prod(m) / m_i
            # => need to handle division by zero later
            prod_others = jnp.expand_dims(prod_m, axis=-1) / m
            prod_others = jnp.where(is_pad, 0.0, prod_others)
            denom = jnp.sum(prod_others, axis=-1)

            alpha_ub = prod_others / jnp.expand_dims(denom, axis=-1)
            beta_ub = (prod(window_dimensions) - 1) * prod_m / denom

            # If there is a dominant input, denom can be zero, so that
            # alpha_ub and beta_ub are NaN.
            # Since we would set them to zero in these cases anyway, we
            # can replace the NaNs with zeros.
            alpha_ub = jnp.nan_to_num(alpha_ub, nan=0.0)
            beta_ub = jnp.nan_to_num(beta_ub, nan=0.0)

            # Zero out alpha_ub and beta_ub if we can use a tight upper bound
            alpha_ub = jnp.where(jnp.expand_dims(any_dominant, -1), 0.0, alpha_ub)
            alpha_ub = jnp.reshape(alpha_ub, alpha_ub.shape[:-1] + window_dimensions)
            beta_ub = jnp.where(any_dominant, 0.0, beta_ub)

            # The select_and_scatter_add2 and reduce_window_conv_transpose terms
            # are never non-zero at the same time for the same entry.
            # See lower bound for more details.
            in_lb_w = (
                in_lb_w
                + select_and_scatter_add2(in_lb, any_dominant * out_lb_w_neg, **kwargs)
                + reduce_window_conv_transpose(in_lb, alpha_ub, out_lb_w_neg, **kwargs)
            )
            in_ub_w = (
                in_ub_w
                + select_and_scatter_add2(in_lb, any_dominant * out_ub_w_pos, **kwargs)
                + reduce_window_conv_transpose(in_lb, alpha_ub, out_ub_w_pos, **kwargs)
            )

    axs = list(range(len(beta_ub.shape)))
    in_lb_b = in_lb_b + jnp.einsum(
        out_lb_w_neg, [..., *axs], beta_ub, axs, out_batch_axes
    )
    in_ub_b = in_ub_b + jnp.einsum(
        out_ub_w_pos, [..., *axs], beta_ub, axs, out_batch_axes
    )

    return (in_lb_w, in_ub_w), (in_lb_b, in_ub_b)


def _reduce_window_max_has_dominant_element(in_lb, in_ub, **reduce_window_kwargs):
    def any_dominant_reducer(x, y):
        lb_x, ub_x, was_dominant_x = x
        lb_y, ub_y, was_dominant_y = y

        x_dominant = lax.ge(lb_x, ub_y)
        y_dominant = lax.ge(lb_y, ub_x)
        dominant = (was_dominant_x & x_dominant) | (was_dominant_y & y_dominant)

        max_lb, max_ub = lax.max(lb_x, lb_y), lax.max(ub_x, ub_y)
        return max_lb, max_ub, dominant

    true = jnp.full(in_lb.shape, fill_value=True)
    max_lb, max_ub, any_dominant = lax.reduce_window(
        (in_lb, in_ub, true),
        (jnp.array(-jnp.inf, dtype=in_lb.dtype),) * 2 + (jnp.array(True),),
        any_dominant_reducer,
        **reduce_window_kwargs,
    )
    return max_lb, max_ub, any_dominant


# Piecewise-Linear Functions
# ------------------------------------------------------------------------------

backwards_crown_relu_rule = NonlinearBackwardsCROWNRule(
    crown_relu_params, Box(jnp.array(0.0), jnp.array(1.0))
)
"""Backwards CROWN rule for ReLU.

Implements the CROWN [Zhang et al., 2018] and ``\\alpha``-CROWN [Xu et al., 2021]
rules for ReLU.

[Zhang et al., 2018]: Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, Luca Daniel:
    Efficient Neural Network Robustness Certification with General Activation Functions.
    NeurIPS 2018: 4944-4953
    https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html

[Xu et al., 2021]: Kaidi Xu, Huan Zhang, Shiqi Wang, Yihan Wang, Suman Jana,
    Xue Lin, Cho-Jui Hsieh: Fast and Complete: Enabling Complete Neural Network
    Verification with Rapid and Massively Parallel Incomplete Verifiers. ICLR 2021
"""


backwards_crown_max_rule = NonlinearBackwardsCROWNRule(
    crown_max_params, Box(jnp.array(0.0), jnp.array(1.0))
)
"""Backwards CROWN rule for max.

Generalises the ReLU rule from [Zhang et al., 2018] based on the MILP enconding
of max from [Tjeng et al., 2019].

[Zhang et al., 2018]: Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, Luca Daniel:
    Efficient Neural Network Robustness Certification with General Activation Functions. NeurIPS 2018: 4944-4953
    https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html

[Tjeng et al., 2019]: Vincent Tjeng, Kai Yuanqing Xiao, Russ Tedrake:
    Evaluating Robustness of Neural Networks with Mixed Integer Programming.
    ICLR (Poster) 2019.
    https://openreview.net/forum?id=HyGIdiRqtm

Args:
    out_weights: The output weights to propagate.
    in_bounds: ``Bounds`` on the input of the ReLU.

Returns:
    The weights and bias arrays propagating ``out_weights`` backwards in
    backwards CROWN.
"""

backwards_crown_min_rule = NonlinearBackwardsCROWNRule(
    crown_min_params, Box(jnp.array(0.0), jnp.array(1.0))
)
"""Backwards CROWN rule for min, analogous to the backwards CROWN rule for max.

Args:
    out_weights: The output weights to propagate.
    in_bounds: ``Bounds`` on the input of the ReLU.

Returns:
    The weights and bias arrays propagating ``out_weights`` backwards in
    backwards CROWN.
"""


register_backwards_crown_rule(relu_marker, backwards_crown_relu_rule)
register_backwards_crown_rule(lax.max_p, backwards_crown_max_rule)
register_backwards_crown_rule(lax.min_p, backwards_crown_min_rule)
backwards_crown_reduce_window_max_rule = _BackwardsCROWNReduceWindowMaxRule()
register_backwards_crown_rule(
    lax.reduce_window_max_p, backwards_crown_reduce_window_max_rule
)


# Convex Functions
# ------------------------------------------------------------------------------


def _register_convex_crown(primitive):
    register_backwards_crown_rule(
        primitive,
        NonlinearBackwardsCROWNRule(
            partial(crown_unary_convex_params, primitive.bind),
            Box(jnp.array(0.0), jnp.array(1.0)),
        ),
    )


_register_convex_crown(lax.abs_p)
_register_convex_crown(lax.cosh_p)
_register_convex_crown(lax.exp_p)
_register_convex_crown(lax.expm1_p)
_register_convex_crown(lax.rsqrt_p)
_register_convex_crown(lax.square_p)


# Concave Functions
# ------------------------------------------------------------------------------


def _register_concave_crown(primitive):
    register_backwards_crown_rule(
        primitive,
        NonlinearBackwardsCROWNRule(
            partial(crown_unary_concave_params, primitive.bind),
            Box(jnp.array(0.0), jnp.array(1.0)),
        ),
    )


_register_concave_crown(lax.acosh_p)
_register_concave_crown(lax.log_p)
_register_concave_crown(lax.log1p_p)
_register_concave_crown(lax.sqrt_p)


# S-Shaped Functions
# ------------------------------------------------------------------------------


def _register_standard_s_shaped_crown(primitive, inflection_point=0):
    register_backwards_crown_rule(
        primitive,
        NonlinearBackwardsCROWNRule(
            partial(crown_unary_s_shaped_params, primitive.bind, inflection_point),
            (
                Box(jnp.array(0.0), jnp.array(1.0)),
                Box(jnp.array(0.0), jnp.array(1.0)),
            ),
        ),
    )


_register_standard_s_shaped_crown(lax.atan_p)
_register_standard_s_shaped_crown(lax.asinh_p)
_register_standard_s_shaped_crown(lax.cbrt_p)  # cubic root
_register_standard_s_shaped_crown(lax.erf_p)
_register_standard_s_shaped_crown(lax.logistic_p)
_register_standard_s_shaped_crown(lax.tanh_p)


# Integer Power
# ------------------------------------------------------------------------------


def crown_integer_pow_params(
    params: Real[Array, "..."] | BACKWARDS_CROWN_PARAM_STRATEGIES,
    in_bounds: Bounds[Real[Array, "..."]] | Real[Array, "..."],
    *,
    y: int,
    **__,
) -> tuple[
    tuple[Real[Array, "..."], Real[Array, "..."]],
    tuple[Zero, Real[Array, "..."]],
]:
    """A backwards CROWN rule for the integer power primitive.

    Handles:
      - y = -1 (reciprocal): convex for x > 0, concave for x < 0.
      - y > 0, odd: convex on [0, ∞), concave on (-∞, 0].
      - y > 0, even: convex everywhere.
    Todo:
       - y < -1: convex for x > 0, concave for x < 0 (not yet implemented).

    Returns:
        The weight and bias parameters (``alpha_lb``, ``alpha_ub``, etc.)
        for the non-linear backwards CROWN rule.
        See ``NonlinearBackwardsCROWNRule``.
    """
    in_lb, in_ub = all_as_bounds(in_bounds)[0]

    if y == 0:
        alpha = jnp.zeros_like(in_lb)
        beta = jnp.ones_like(in_lb)
        return (alpha, alpha), (beta, beta)

    if y == -1:  # Reciprocal case
        convex_params = crown_unary_convex_params(
            partial(lax.integer_pow, y=y), params, in_bounds
        )
        concave_params = crown_unary_concave_params(
            partial(lax.integer_pow, y=y), params, in_bounds
        )
        mixed_alpha = jnp.zeros_like(in_lb)
        mixed_beta_lb = jnp.full_like(in_lb, -jnp.inf)
        mixed_beta_ub = jnp.full_like(in_lb, jnp.inf)
        mixed_params = ((mixed_alpha, mixed_alpha), (mixed_beta_lb, mixed_beta_ub))

        def select(convex, concave, mixed):
            return jnp.where(in_lb >= 0, convex, jnp.where(in_ub <= 0, concave, mixed))

        return jax.tree.map(select, convex_params, concave_params, mixed_params)
    elif y > 0 and y % 2 == 0:  #  Even powers
        return crown_unary_convex_params(
            lambda x: lax.integer_pow(x, y=y), params, (in_lb, in_ub)
        )
    elif y > 0 and y % 2 == 1:  #  Odd powers
        raise NotImplementedError(
            f"Backwards CROWN for integer_pow with y={y} not yet implemented."
        )
    elif y < -1:  # Negative exponents
        raise NotImplementedError(
            f"Backwards CROWN for integer_pow with y={y} not yet implemented."
        )
    else:
        raise AssertionError()


register_backwards_crown_rule(
    lax.integer_pow_p,
    NonlinearBackwardsCROWNRule(
        crown_integer_pow_params, Box(jnp.array(0.0), jnp.array(1.0))
    ),
)

# Element-wise Multiplication
# ------------------------------------------------------------------------------


def bilinear_mul_params(
    params: tuple[Real[Array, "..."], Real[Array, "..."]]
    | BACKWARDS_CROWN_PARAM_STRATEGIES,
    x_bounds: Bounds,
    y_bounds: Bounds,
    **__,
) -> tuple[
    tuple[Real[Array, "..."], Real[Array, "..."]],
    tuple[Real[Array, "..."], Real[Array, "..."]],
    tuple[Zero, Real[Array, "..."]],
]:
    """Backwards rule for CROWN for element-wise multiplication of two bounded values.

    This class implements the adaptive multiplications bounds from [Huang et al., 2025].
    It supplements the transpose rule for multiplication that is applied if
    only one of the two arguments of multiplication is bounded.

    It adds an adaptive parameter rule not present in [Huang et al., 2025].

    [Huang et al., 2025] Pei Huang, Dennis Wei, Omri Isac, Haoze Wu, Min Wu,
        Clark Barrett: Parameterized Abstract Interpretation for Transformer
        Verification
        https://www-cs.stanford.edu/~minwu/assets/pdf/AAAI2026Parameterized.pdf

    Args:
        params: External parameters for ``\\alpha``-CROWN or a parameter selection strategy.
            See ``BACKWARDS_CROWN_PARAM_STRATEGIES`` for more details.
        x_bounds: Constant bounds on the first argument or a concrete value.
        y_bounds: Constant bounds on the second argument or a concrete value.

    Returns:
        The weight and bias parameters (``alpha_lb``, ``alpha_ub``, etc.)
        for the non-linear backwards CROWN rule.
        See ``NonlinearBackwardsCROWNRule``.
    """
    (x_lb, x_ub), (y_lb, y_ub) = all_as_bounds(x_bounds, y_bounds)

    # Terminology:
    # We use alpha for the weight of input x and beta for the weight of input y.
    # We use gamma for the bias.

    mid_x, ran_x = (x_ub + x_lb) / 2, (x_ub - x_lb) / 2
    mid_y, ran_y = (y_ub + y_lb) / 2, (y_ub - y_lb) / 2

    match params:
        case "default" | "adaptive":
            # TODO: these adaptive should be validated again
            a_lb = jnp.where(
                x_ub * y_ub / 2 - x_ub * y_lb - x_lb * y_ub + 3 * x_lb * y_lb / 2 >= 0,
                -1.0,
                1.0,
            )
            a_ub = jnp.where(x_lb * y_ub - x_ub * y_lb >= 0, -1.0, 1.0)
        case "fixed":
            a_lb = a_ub = 0.0
        case _:  # external value / alpha-CROWN
            a_lb, a_ub = params
            a_lb = jnp.broadcast_to(a_lb, x_lb.shape)
            a_ub = jnp.broadcast_to(a_ub, x_lb.shape)

    # Equation (20a) in [Huang et al., 2025]
    alpha_lb = mid_y + a_lb * ran_y
    beta_lb = mid_x + a_lb * ran_x
    gamma_lb = -mid_x * mid_y - ran_x * ran_y - a_lb * (mid_x * ran_y + mid_y * ran_x)

    # Equation (20b) in [Huang et al., 2025]
    alpha_ub = mid_y - a_ub * ran_y
    beta_ub = mid_x + a_ub * ran_x
    gamma_ub = -mid_x * mid_y + ran_x * ran_y + a_ub * (mid_x * ran_y - mid_y * ran_x)

    return (alpha_lb, alpha_ub), (beta_lb, beta_ub), (gamma_lb, gamma_ub)


class MulBackwardsCROWNRule(BackwardsCROWNRule):
    """A backwards CROWN rule for element-wise multiplication.

    This rule uses `transpose_as_backwards_lirpa_rule_bilinear``
    if one of the arguments is not bounded and uses ``nonlinear_backwards_lirpa_rule``
    for the bilinear case.
    """

    def __init__(self):
        self.__tranpose_rule = transpose_as_backwards_lirpa_rule_bilinear(lax.mul_p, ())
        self.__bilinear_params = bilinear_mul_params
        self.__param_domain = (Box(jnp.array(-1.0), jnp.array(1.0)),) * 2

    def __call__(
        self,
        params: Sequence[PARAM_VALUE],
        out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        **kwargs,
    ) -> tuple[LiRPABounds[jax.Array | Zero], ...] | LiRPABounds[jax.Array | Zero]:
        assert len(params) == 1, "Multiple params for single-output function."
        assert len(in_bounds) == 2, "Multiplication takes two arguments."
        x, y = in_bounds
        if is_bounds(x) and is_bounds(y):
            return nonlinear_backwards_lirpa_rule(
                self.__bilinear_params, params[0], out_weights, in_bounds, **kwargs
            )
        else:
            return self.__tranpose_rule(None, out_weights, in_bounds, **kwargs)

    def parameter_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        strategies: Sequence[BACKWARDS_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        assert len(strategies) == 1, (
            "Multiple parameter strategies for single-output function."
        )
        strategy = strategies[0]
        in_shape = in_shapes[0]
        match strategy:
            case "default" | "fixed" | "adaptive":
                return (None,)
            case "external" | "external-full":
                full_domain = jax.tree.map(
                    lambda domain: Box(
                        jnp.full(in_shape, domain.lower_bound),
                        jnp.full(in_shape, domain.upper_bound),
                    ),
                    self.__param_domain,
                    is_leaf=is_bounds,
                )
                return (full_domain,)
            case "external-shared":
                return (self.__param_domain,)
            case _:
                raise ValueError(f"Unsupported parameter strategy: {strategy}")


register_backwards_crown_rule(lax.mul_p, MulBackwardsCROWNRule())


# Division
# ------------------------------------------------------------------------------
# division is equivalent to multiplication by the reciprocal


class DivBackwardsCROWNRule(BackwardsCROWNRule):
    """A backwards CROWN rule for element-wise division.

    This rule uses `transpose_as_backwards_lirpa_rule_binary_left_only``
    if only the dividend is bounded.
    Otherwise, it combines the CRONW rules for element-wise multiplication
    and the reciprocal (integer powers), since ``x/y = x * y^{-1}``.
    """

    def __init__(self):
        self.__tranpose_rule = transpose_as_backwards_lirpa_rule_binary_left_only(
            lax.div_p,
            (),
            "This should not happen. Please file a bug report in formalax.",
        )
        self.__mul_rule = _primitive_backward_crown_rules[lax.mul_p]
        self.__int_pow_rule = _primitive_backward_crown_rules[lax.integer_pow_p]

    def __call__(
        self,
        params: Sequence[PARAM_VALUE],
        out_weights: Sequence[LiRPAWeights[jax.Array | Zero]],
        in_bounds: Sequence[Bounds | jax.Array],
        batch_axis_mappings: Sequence[Sequence[BatchAxisMapping]],
        **kwargs,
    ) -> tuple[LiRPABounds[jax.Array | Zero], ...] | LiRPABounds[jax.Array | Zero]:
        assert len(params) == 1, "Multiple params for single-output function."
        assert len(in_bounds) == 2, "Division takes two arguments."
        params = params[0]
        x, y = in_bounds

        if not is_bounds(y):
            return self.__tranpose_rule(
                None,
                out_weights,
                in_bounds,
                batch_axis_mappings=batch_axis_mappings,
                **kwargs,
            )

        if not isinstance(params, tuple):
            mul_params = reciprocal_params = (params,)
        else:
            mul_params, reciprocal_params = params

        reciprocal_bounds = ibp_rule_reciprocal(y)
        mul_lirpa_bounds = self.__mul_rule(
            mul_params,
            out_weights,
            (x, reciprocal_bounds),
            batch_axis_mappings=batch_axis_mappings,
            **kwargs,
        )
        x_lirpa_weights, y_inv_lirpa_weights = mul_lirpa_bounds.weights_iter()
        y_lirpa_bounds = self.__int_pow_rule(
            reciprocal_params,
            (y_inv_lirpa_weights,),
            (y,),
            y=-1,  # exponent
            batch_axis_mappings=((batch_axis_mappings[0][1],),),
            **kwargs,
        )

        x_w_lb, x_w_ub = x_lirpa_weights
        ((y_w_lb, y_w_ub),) = y_lirpa_bounds.weights_iter()
        in_lb_b = y_lirpa_bounds.lb_bias + mul_lirpa_bounds.lb_bias
        in_ub_b = y_lirpa_bounds.ub_bias + mul_lirpa_bounds.ub_bias
        # handle the unbounded case: x_w_lb and x_w_ub get nan if
        # reciprocal_bounds has a inf/-inf bound.
        x_w_lb = jnp.nan_to_num(x_w_lb, nan=0.0)
        x_w_ub = jnp.nan_to_num(x_w_ub, nan=0.0)
        return out_weights[0].backwards_step(
            (x_w_lb, y_w_lb),
            (x_w_ub, y_w_ub),
            in_lb_b,
            in_ub_b,
            in_bounds,
            batch_axis_mappings[0],
        )

    def parameter_domains(
        self,
        in_shapes: Sequence[tuple[int, ...]],
        out_shapes: Sequence[tuple[int, ...]],
        strategies: Sequence[BACKWARDS_CROWN_PARAM_STRATEGIES],
        **kwargs,
    ) -> tuple[PARAM_DOMAIN | None, ...]:
        mul_domains = self.__mul_rule.parameter_domains(
            in_shapes, out_shapes, strategies
        )
        reciprocal_domains = self.__int_pow_rule.parameter_domains(
            in_shapes, out_shapes, strategies
        )
        return ((mul_domains, reciprocal_domains),)


register_backwards_crown_rule(lax.div_p, DivBackwardsCROWNRule())
