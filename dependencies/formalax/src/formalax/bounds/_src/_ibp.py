#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import typing
from functools import partial
from typing import Callable, Concatenate, Protocol, Sequence

import jax
import jax._src.ad_util as jax_ad_util
import jax._src.util as jax_util
import jax.core
import jax.extend.linear_util as lu
import jax.numpy as jnp
import jax.tree
from frozendict import frozendict
from jax import lax
from jax.extend.core import ClosedJaxpr, Primitive, Var, primitives
from jaxtyping import Array, PyTree, PyTreeDef, Real

from ...core.markers import markup_primitive_call, relu_marker
from ...sets import Singleton
from ...sets.box import Box
from ...utils import future_store
from ...utils import jaxpr as jaxpr_utils
from ...utils.caching import HashablePartial
from ...utils.name_stack import transform_name_stack_ctx
from ...utils.pytree import flatten_fun
from ...utils.wraps import wraps
from ...utils.zip import strict_zip
from ._bounds import (
    Bounds,
    all_as_bounds,
    duplicate_for_bounds,
    is_bounds,
)

__all__ = (
    "ibp",
    "ibp_jaxpr",
    "register_ibp_rule",
    "ibp_rule_linear",
    "ibp_rule_bilinear",
    "ibp_rule_mul",
    "ibp_rule_compare_greater",
    "ibp_rule_compare_less",
    "ibp_rule_compare_eq",
    "ibp_rule_monotonic_non_increasing",
    "ibp_rule_monotonic_non_decreasing",
    "ibp_rule_strongly_concave",
    "ibp_rule_strongly_convex",
    "ibp_rule_reciprocal",
    "ibp_rule_jaxpr",
)

# ==============================================================================
# Interface
# ==============================================================================


class IBPRule(Protocol):
    """Type signature of an IBP rule.

    IBP rules are registered for primitives or ``formalax.Marker`` instances.
    They receive the bounded arguments and parameters of a primitive call
    and return the output bounds.

    IBP rules can be marked as recursive by setting ``rule.recursive = True``
    on the rule instance.
    In this case, the rule additionally receives an IBP callable as an additional
    keyword argument ``ibp``.
    """

    def __call__(
        self,
        *args: Bounds[jax.Array] | jax.Array,
        **kwargs,
    ) -> Bounds[jax.Array] | jax.Array: ...


def ibp(fun: Callable, ibp_rules: dict[Primitive, IBPRule] | None = None) -> Callable:
    """Create a function that performs Interval Bound Propagation (IBP) on ``fun``.

    The created function takes lower and upper bounds on the input and computes
    lower and upper bounds on the output of ``fun``.

    Args:
        fun: Function to compute bounds on.
            The arguments and the return value of ``fun`` may be any mix of arrays,
            scalars, and standard Python containers (JAX pytrees).
        ibp_rules: A dictionary mapping JAX primitives to IBP rules.
            Uses the default IBP rules if not provided.

    Returns:
        A function ``fun_ibp`` that computes output bounds on ``fun`` given
        input bounds on some of the arguments of ``fun``.
        When calling ``fun_ibp``, pass ``Bounds`` instances for the arguments with
        input bounds while passing regular ``jax.Arrays`` for the arguments without
        input bounds.
        The output of ``fun_ibp`` has the same structure as the output of ``fun``,
        except that all outputs which depends on arguments with input bounds are
        ``Bounds`` instances.

    Examples:
        >>> lb_in, ub_in = 0.0, 1.0
        >>> lb_out, ub_out = ibp(jax.numpy.exp)(Box(lb_in, ub_in))
        >>> print(lb_out)
        1.0
        >>> print(ub_out)
        2.7182817
    """
    if ibp_rules is None:
        ibp_rules = primitive_ibp_rules

    ibp_f_docstring = (
        "Computes output lower and upper bounds of {fun} using "
        "Interval Bound Propagation (IBP).\n"
    )
    if fun.__doc__:
        ibp_f_docstring += "\n\nOriginal documentation of {fun}:\n\n"
        ibp_f_docstring += fun.__doc__

    @wraps(fun, doc=ibp_f_docstring)
    def ibp_f(*args, **kwargs) -> PyTree:
        args_flat, in_tree = jax.tree.flatten(args, is_leaf=is_bounds)
        flat_fun, out_tree = flatten_fun(fun, in_tree, is_bounds)
        out_flat = _ibp(flat_fun, ibp_rules, transform_stack=True)(*args_flat, **kwargs)
        return jax.tree.unflatten(out_tree(), out_flat)

    return ibp_f


def ibp_jaxpr(
    jaxpr: ClosedJaxpr,
    intermediate_bounds: bool = False,
    ibp_rules: dict[Primitive, IBPRule] | None = None,
) -> Callable[
    ...,
    dict[Var, Box[Real[Array, "..."]] | tuple[Box[Real[Array, "..."]], ...]],
]:
    """Create a function that performs Interval Bound Propagation (IBP) on ``jaxpr``.

    In difference to ``ibp``, this function can also return the intermediate bounds
    for all variables in the ``jaxpr``.

    Args:
        jaxpr: A Jaxpr representing the computation to compute bounds on.
        intermediate_bounds: If ``True``, the returned function returns a dictionary
            mapping the variables in ``jaxpr`` to their (intermediate) IBP bounds.
            If ``False``, the returned function returns a dictionary containing only
            the IBP bounds for the output variables.
        ibp_rules: A dictionary mapping JAX primitives to IBP rules.
            Uses the default IBP rules if not provided.
    Returns:
        A function ``fun_ibp`` that computes bounds on the evaluation
        of ``jaxpr``.
        When calling ``fun_ibp``, pass ``Bounds`` instances for the arguments with
        input bounds while passing regular ``jax.Arrays`` for the arguments without
        input bounds.

        The output of ``fun_ibp`` is a dictionary mapping variables in ``jaxpr``
        to bounds computed using IBP.
        If ``intermediate_bounds`` is ``True``, this dictionary contains bounds
        for all variables in ``jaxpr``.
        If ``intermediate_bounds`` is ``False``, the dictionary only contains bounds
        for the output variables.

        In both cases, the dictionary may map some variables to ``jax.Arrays`` instead
        of ``Bounds`` instances.
        This is the case if the value of a variable is independent of the input variables
        which have ``Bounds`` as values.
    """
    if ibp_rules is None:
        ibp_rules = primitive_ibp_rules

    def ibp_f(
        *args: Bounds | Real[Array, "..."],
    ) -> dict[Var, Bounds | Real[Array, "..."]]:
        f = jaxpr_utils.jaxpr_as_fun(jaxpr, intermediate_values=intermediate_bounds)
        flat_fun, out_tree = flatten_fun(f, None, is_bounds)
        out_flat = _ibp(flat_fun, ibp_rules, transform_stack=True)(*args)
        return jax.tree.unflatten(out_tree(), out_flat)

    ibp_f.__doc__ = (
        f"Computes lower and upper bounds on the variables in the following Jaxpr "
        f"using Interval Bound Propagation (IBP):\n\n {jaxpr}\n\n"
        f"Args:\n"
        f"    args: The ``Bounds`` or concrete values (``jax.Array``) for each "
        f"input variable of the Jaxpr.\n\n"
        f"Returns:\n"
        f"    A mapping from variables in the Jaxpr to ``Bounds`` or concrete values."
    )
    return ibp_f


# ==============================================================================
# Implementation
# ==============================================================================


def _ibp(
    fun: Callable, ibp_rules: dict[Primitive, IBPRule], transform_stack: bool = False
) -> Callable:
    """
    Creates trace and tracers for IBP.

    Args:
        fun: The function to perform IBP on.
        args: Arguments to ``fun``.
        kwargs: Keyword arguments to ``fun``.

    Returns:
        A function that performs IBP on ``fun``.
    """
    ibp_rules = frozendict(ibp_rules)

    with transform_name_stack_ctx("ibp", transform_stack):

        def ibp_trace_f(
            *args: Bounds | Real[Array, "..."], **kwargs
        ) -> tuple[Bounds | Real[Array, "..."], ...]:
            with jax.core.take_current_trace() as parent_trace:
                trace = IBPTrace(parent_trace, ibp_rules)
                in_tracers = [
                    IBPTracer(trace, *arg.concrete) if is_bounds(arg) else arg
                    for arg in args
                ]
                with jax.core.set_current_trace(trace):
                    outs = fun(*in_tracers, **kwargs)
                return tuple(map(trace.unwrap, outs))

        return ibp_trace_f


class IBPTrace(jax.core.Trace["IBPTracer"]):
    """
    A trace sending pairs of lower and upper bounds (represented by `IBPTracers`)
    through a function to perform Interval Bound Propagation (IBP).
    """

    def __init__(
        self, parent_trace: jax.core.Trace, ibp_rules: frozendict[Primitive, IBPRule]
    ):
        super().__init__()
        self.parent_trace = parent_trace
        self.ibp_rules = ibp_rules
        self.ibp_callable = HashablePartial(_ibp, ibp_rules=ibp_rules)

    def unwrap(self, val) -> Bounds | jax.Array:
        if isinstance(val, IBPTracer):
            if val.is_point:
                return val.lower_bound
            else:
                return val.bounds
        else:
            return val

    def process_primitive(self, primitive, tracers: Sequence["IBPTracer"], params):
        args = [self.unwrap(t) for t in tracers]

        if not any(is_bounds(x) for x in args):
            return primitive.bind_with_trace(self.parent_trace, tracers, params)

        prim = markup_primitive_call(primitive, *args, **params)
        ibp_rule = self.ibp_rules.get(prim)
        if not ibp_rule:
            raise NotImplementedError(f"No IBP rule implemented for {primitive}.")

        with jax.core.set_current_trace(self.parent_trace):
            outs = self._call_rule(ibp_rule, *args, **params)

        if primitive.multiple_results:
            # Some outputs may not be Bounds if they do not depend on the
            # bounded inputs
            return [
                IBPTracer(self, *arg.concrete) if is_bounds(arg) else arg
                for arg in outs
            ]
        else:
            return IBPTracer(self, *outs.concrete) if is_bounds(outs) else outs

    def _call_rule(self, ibp_rule: IBPRule, *args, **params):
        if hasattr(ibp_rule, "recursive") and ibp_rule.recursive:
            return ibp_rule(*args, **params, ibp=self.ibp_callable)
        else:
            return ibp_rule(*args, **params)

    def process_custom_jvp_call(
        self,
        primitive: Primitive,
        fun: Callable,
        jvp: Callable,
        tracers: Sequence["IBPTracer"],
        *,
        symbolic_zeros,
    ):
        args = [self.unwrap(t) for t in tracers]
        if not any(is_bounds(x) for x in args):
            return primitive.bind_with_trace(
                self.parent_trace,
                (fun, jvp, tracers),
                {"symbolic_zeros": symbolic_zeros},
            )

        prim = markup_primitive_call(
            primitive, fun, jvp, tracers, symbolic_zeros=symbolic_zeros
        )
        if ibp_rule := self.ibp_rules.get(prim):
            out_vals = self._call_rule(
                ibp_rule,
                primitive,
                self.parent_trace,
                fun,
                jvp,
                *args,
                symbolic_zeros=symbolic_zeros,
            )
        else:
            # Try to apply IBP directly to the function and its JVP.
            fun_args, fun_in_tree = jax.tree.flatten(args)  # also flatten Bounds
            _, jvp_in_tree = jax.tree.flatten(args + args)

            fun_is_lu = isinstance(fun, lu.WrappedFun)
            jvp_is_lu = isinstance(jvp, lu.WrappedFun)

            if fun_is_lu:
                fun = fun.call_wrapped
            if jvp_is_lu:
                jvp = jvp.call_wrapped

            fun = _ibp(fun, ibp_rules=self.ibp_rules)
            jvp = _ibp(jvp, ibp_rules=self.ibp_rules)
            fun, fun_out_tree = flatten_fun(fun, fun_in_tree)
            jvp, jvp_out_tree = flatten_fun(jvp, jvp_in_tree)

            if fun_is_lu:
                fun = lu.wrap_init(fun)
            if jvp_is_lu:
                jvp = lu.wrap_init(jvp)

            out_vals = primitive.bind_with_trace(
                self.parent_trace,
                (fun, jvp, *fun_args),
                {"symbolic_zeros": symbolic_zeros},
            )

            was_jvp, out_tree = future_store.which_store(jvp_out_tree, fun_out_tree)
            if was_jvp:
                out_vals = jax.tree.unflatten(out_tree, list(out_vals) + list(out_vals))
                out_vals = out_vals[: len(out_vals) // 2]
            else:
                out_vals = jax.tree.unflatten(out_tree, out_vals)
        return [
            IBPTracer(self, *arg.concrete) if is_bounds(arg) else arg
            for arg in out_vals
        ]


class IBPTracer(jax.core.Tracer):
    """
    A tracer that is sent through functions to perform Interval Bound Propagation (IBP).

    It stores a (constant) lower bound and a (constant) upper bound that together
    represent an interval.
    """

    __slots__ = (
        "lower_bound",
        "upper_bound",
        "is_point",
    )

    def __init__(
        self,
        trace: IBPTrace,
        lower_bound,
        upper_bound,
        is_point: bool = False,
    ):
        """
        Create a new `IBPTracer`.

        Args:
            trace: The `IBPTrace` for this tracer.
            lower_bound: A (constant) lower bound.
            upper_bound: A (constant) upper bound.
            is_point: Whether the lower bound is equal to the upper bound so that the
                interval `[lower_bound, upper_bound]` represents a point.
        """
        super().__init__(trace)
        assert (
            not hasattr(lower_bound, "shape") or lower_bound.shape == upper_bound.shape
        )
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_point = is_point

    @property
    def aval(self):
        return jax.core.get_aval(self.lower_bound)

    def full_lower(self):
        if self.is_point:
            return jax.core.full_lower(self.lower_bound)
        else:
            return self

    @property
    def bounds(self) -> Bounds:
        return Box(self.lower_bound, self.upper_bound)


primitive_ibp_rules: dict[Primitive, IBPRule] = {}


def register_ibp_rule(primitive: Primitive, rule: IBPRule, recursive: bool = False):
    """Registers a new IBP rule.

    Args:
        primitive: The primitive that is handled by this rule.
        rule: The IBP rule for computing upper and lower bounds on the output given
            upper and lower bounds on the input.
            The ``rule`` accepts the same arguments as the primitive.
            Each positional arguments is either a ``Bounds`` instance,
            or a single concrete value (an array).
            The ``rule`` returns a ``Bounds`` instance.
        recursive: Whether the rule is recursive.
            Sets ``rule.recursive = True`` if ``recursive`` is ``True``.
            See ``IBPRule`` for more details.
    """
    if recursive:
        rule.recursive = True
    primitive_ibp_rules[primitive] = rule


# ==============================================================================
# Primitive Rules
# ==============================================================================


def ibp_rule_linear[T: jax.Array, **P](
    linear_fun: Callable[Concatenate[T, T, P], T],
    lhs: T | Bounds[T],
    rhs: T | Bounds[T],
    **kwargs: P.kwargs,
) -> Bounds[T]:
    """Performs Interval Bound Propagation (IBP) for linear functions.

    Implements the midpoint rule for linear transformations from [Gowal et al., 2018]:
    Assuming the ``rhs`` has input bounds (``lb_rhs`` and ``ub_rhs``)
    and the ``lhs`` is a concrete value:

        mid_rhs = (lb_rhs + ub_rhs) / 2
        range_rhs = (ub_rhs - lb_rhs) / 2
        mid_out = linear_fun(lhs, mid_rhs)
        range_out = linear_fun(abs(lhs), range_rhs)
        lb_out = mid_out - range_out
        ub_out = mid_out + range_out

    This function raises an error if both the ``lhs`` and the ``rhs``
    having input bounds.
    Have a look at ``ibp_rule_bilinear`` if you want to support this case.

    [Gowal et al., 2018]: Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth,
        Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy A. Mann,
        Pushmeet Kohli: On the Effectiveness of Interval Bound Propagation for
        Training Verifiably Robust Models. CoRR abs/1810.12715 (2018)

    Args:
        linear_fun: The linear function (matrix multiplication, convolution).
            Generally, this function does not have a *bias* term.
        lhs: The left hand-side argument of ``linear_fun``. Either a concrete
            value or a pair of input bounds.
        rhs: The right hand side argument of ``linear_fun``. Either a concrete
            value or a pair of input bounds.
        **kwargs: Further keyword arguments that are passed on to ``linear_fun``.

    Returns:
        Output bounds for ``linear_fun`` under the input bounds on ``lhs``
        and ``rhs``.
    """

    def raise_error(*args, **kwargs) -> typing.Never:
        raise NotImplementedError(
            f"IBP for {linear_fun} with bounds for both arguments "
            f"is not implemented yet."
        )

    return ibp_rule_bilinear(linear_fun, raise_error, lhs, rhs, **kwargs)


def ibp_rule_bilinear[T: jax.Array, **P](
    bilinear_fun: Callable[Concatenate[T, T, P], T],
    bilinear_ibp_rule: Callable[Concatenate[Bounds[T], Bounds[T], P], Bounds[T]],
    lhs: T | Bounds[T],
    rhs: T | Bounds[T],
    **kwargs: P.kwargs,
) -> Bounds[T]:
    """Performs Interval Bound Propagation (IBP) for bilinear functions.

    A bilinear function is a function with two arguments that is linear
    in each argument if the value of the other arguments is fixed.
    An example is ``f(x, y) = x * y``.

    This function implements the midpoint rule (see ``ibp_rule_linear``) if either
    ``lhs`` or ``rhs`` are single values (not ``Bounds``), in which case ``bilinear_fun``
    is linear.
    If both ``lhs`` and ``rhs`` are ``Bounds``, it delegates to
    ``bilinear_ibp_rule``.

    Args:
        bilinear_fun: The linear function (matrix multiplication, convolution).
            Generally, this function does not have a *bias* term.
        bilinear_ibp_rule: A function computing bounds (IBP rule) on
            ``bilinear_fun`` when both ``lhs`` and ``rhs`` are ``Bounds``,
            so that ``bilinear_fun`` is actually bilinear and not just
            linear.
        lhs: The left hand-side argument of ``linear_fun``. Either a concrete
            value or a pair of input bounds.
        rhs: The right hand side argument of ``linear_fun``. Either a concrete
            value or a pair of input bounds.
        kwargs: Further keyword arguments that are passed on to ``linear_fun``.

    Returns:
        Output bounds for ``bilinear_fun`` under the input bounds on ``lhs``
        and ``rhs``.
    """
    if is_bounds(lhs) and is_bounds(rhs):
        return bilinear_ibp_rule(lhs, rhs, **kwargs)
    elif is_bounds(lhs):
        return _ibp_rule_linear_rhs_bounded(
            lambda x, y, **kwargs: bilinear_fun(y, x, **kwargs), rhs, lhs, **kwargs
        )
    elif is_bounds(rhs):
        return _ibp_rule_linear_rhs_bounded(bilinear_fun, lhs, rhs, **kwargs)
    else:
        val = bilinear_fun(lhs, rhs, **kwargs)
        return Singleton(val)


def _ibp_rule_linear_rhs_bounded[T: jax.Array, **P](
    linear_fun: Callable[Concatenate[T, T, P], T],
    lhs: T,
    rhs: Bounds[T],
    **kwargs: P.kwargs,
) -> Bounds[T]:
    lb_in, ub_in = rhs.concrete
    mid_in = (ub_in + lb_in) / 2.0
    range_in = (ub_in - lb_in) / 2.0
    mid_out = linear_fun(lhs, mid_in, **kwargs)
    range_out = linear_fun(jnp.abs(lhs), range_in, **kwargs)
    return Box(mid_out - range_out, mid_out + range_out)


def ibp_rule_mul[T: jax.Array](x: T | Bounds[T], y: T | Bounds[T]) -> Bounds[T]:
    """Performs Interval Bound Propagation (IBP) for element-wise multiplication.

    Element-wise multiplication ``f(x, y) = x * y`` (``lax.mul``) is a bilinear function.
    This function uses the midpoint rule (see ``ibp_rule_linear``) if either
    ``x`` or ``y`` are single values (not ``Bounds``), in which case
    ``mul`` is linear.

    If both ``x`` and ``y`` are bounds, this function computes the lower and
    upper bounds as

        a1 = lb_x * lb_y
        a2 = lb_x * ub_y
        a3 = ub_x * lb_y
        a4 = ub_x * ub_y
        lb_out = min(a1, a2, a3, a4)
        ub_out = max(a1, a2, a3, a4)

    Args:
        x: The left hand-side argument of ``mul``. Either a concrete<
            value or a pair of input bounds.
        y: The right hand side argument of ``mul``. Either a concrete
            value or a pair of input bounds.

    Returns:
        Output bounds for ``mul`` under the input bounds on ``x`` and ``y``.
    """

    def mul_bilinear(x, y):
        x_lb, x_ub = x.concrete
        y_lb, y_ub = y.concrete

        lblb = lax.mul(x_lb, y_lb)
        lbub = lax.mul(x_lb, y_ub)
        ublb = lax.mul(x_ub, y_lb)
        ubub = lax.mul(x_ub, y_ub)

        out_lb = jnp.minimum(jnp.minimum(lblb, lbub), jnp.minimum(ublb, ubub))
        out_ub = jnp.maximum(jnp.maximum(lblb, lbub), jnp.maximum(ublb, ubub))
        return Box(out_lb, out_ub)

    return ibp_rule_bilinear(lax.mul, mul_bilinear, x, y)


def ibp_rule_monotonic_non_decreasing[T: jax.Array](
    fun: Callable[..., T], *args: T | Bounds[T], **kwargs
) -> Bounds[T] | tuple[Bounds[T], ...]:
    """Performs Interval Bound Propagation (IBP) for monotonic non-decreasing functions.

    If ``lb_1, ..., lb_n`` are the lower bounds of the arguments ``*args``
    and ``ub_1, ..., ub_n`` are the upper bounds of the arguments,
    this function computes the output bounds ``lb_out`` and ``ub_out`` as

        lb_out = fun(lb_1, ..., lb_n)
        ub_out = fun(ub_1, ..., ub_n)

    Args:
        fun: The monotonic non-decreasing function (sum, maximum) to bound.
        *args: Lower and upper bounds or concrete values for each argument
            of ``fun``.
        kwargs: Further keyword arguments that are passed on to ``fun``.

    Returns:
        Output bounds for ``fun`` under the input bounds on the arguments.
    """
    in_bounds = all_as_bounds(*args)
    in_lbs, in_ubs = strict_zip(*[b.concrete for b in in_bounds])
    out_lbs = fun(*in_lbs, **kwargs)
    out_ubs = fun(*in_ubs, **kwargs)
    if isinstance(out_lbs, tuple | list):
        return tuple(Box(lb, ub) for lb, ub in strict_zip(out_lbs, out_ubs))
    else:
        return Box(out_lbs, out_ubs)


def ibp_rule_monotonic_non_increasing[T: jax.Array](
    fun: Callable[..., T], *args: T | Bounds[T], **kwargs
) -> Bounds[T]:
    """Performs Interval Bound Propagation (IBP) for monotonic non-increasing functions.

    If ``lb_1, ..., lb_n`` are the lower bounds of the arguments ``*args``
    and ``ub_1, ..., ub_n`` are the upper bounds of the arguments,
    this function computes the output bounds ``lb_out`` and ``ub_out`` as

        lb_out = fun(ub_1, ..., ub_n)
        ub_out = fun(lb_1, ..., lb_n)

    Args:
        fun: The monotonic non-decreasing function (negation, reciprocal) to bound.
        *args: Lower and upper bounds or concrete values for each argument
            of ``fun``.
        kwargs: Further keyword arguments that are passed on to ``fun``.

    Returns:
        Output bounds for ``fun`` under the input bounds on the arguments.
    """
    in_bounds = all_as_bounds(*args)
    in_lbs, in_ubs = strict_zip(*[b.concrete for b in in_bounds])
    out_lb = fun(*in_ubs, **kwargs)
    out_ub = fun(*in_lbs, **kwargs)
    return Box(out_lb, out_ub)


def ibp_rule_strongly_convex[T: jax.Array](
    fun: Callable[..., T], minimum: tuple[float, float], x: T | Bounds[T], **kwargs
) -> Bounds[T]:
    """Performs Interval Bound Propagation (IBP) for strongly convex unary functions.

    If ``lb`` and ``ub`` are the bounds on the argument and ``x_{min}`` is the
    unique minimizer of ``fun``, this function computes the output bounds ``lb_out``
    and ``ub_out`` as

        lb_out = fun(x_{min}) if lb < x_{min} < ub else min(fun(lb), fun(ub))
        ub_out = max(fun(lb), fun(ub))

    Args:
        fun: The strongly convex unary function (abs, square) to bound.
        minimum: The point where ``fun`` attains its minimum as
            ``(minimizer, minimum)``, where the minimizer is the function argument
            and the minimum is the function value at the minimizer.
        x: Lower and upper bounds or concrete values for the argument of ``fun``.
        kwargs: Further keyword arguments that are passed on to ``fun``.

    Returns:
        Output bounds for ``fun`` under the input bounds on the arguments.
    """
    x_lb, x_ub = all_as_bounds(x)[0].concrete
    f_lb, f_ub = fun(x_lb, **kwargs), fun(x_ub, **kwargs)
    x_min, f_min = minimum

    out_lb = jnp.where((x_lb < x_min) & (x_min < x_ub), f_min, jnp.minimum(f_lb, f_ub))
    out_ub = jnp.maximum(f_lb, f_ub)
    return Box(out_lb, out_ub)


def ibp_rule_strongly_concave[T: jax.Array](
    fun: Callable[..., T], maximum: tuple[float, float], x: T | Bounds[T], **kwargs
) -> Bounds[T]:
    """Performs Interval Bound Propagation (IBP) for strongly concave unary functions.

    If ``lb`` and ``ub`` are the bounds on the argument and ``x_{max}`` is the
    unique maximizer of ``fun``, this function computes the output bounds ``lb_out``
    and ``ub_out`` as

        lb_out = min(fun(lb), fun(ub))
        ub_out = fun(x_{max}) if lb < x_{max} < ub else max(fun(lb), fun(ub))

    Args:
        fun: The strongly concave unary function (abs, square) to bound.
        maximum: The point where ``fun`` attains its maximum as
            ``(maximizer, maximum)``, where the maximizer is the function argument
            and the maximum is the function value at the maximizer.
        x: Lower and upper bounds or concrete values for the argument of ``fun``.
        kwargs: Further keyword arguments that are passed on to ``fun``.

    Returns:
        Output bounds for ``fun`` under the input bounds on the arguments.
    """
    x_lb, x_ub = all_as_bounds(x)[0].concrete
    f_lb, f_ub = fun(x_lb, **kwargs), fun(x_ub, **kwargs)
    x_max, f_max = maximum

    out_lb = jnp.minimum(f_lb, f_ub)
    out_ub = jnp.where((x_lb < x_max) & (x_max < x_ub), f_max, jnp.maximum(f_lb, f_ub))
    return Box(out_lb, out_ub)


def ibp_rule_reciprocal[T: jax.Array](x: T | Bounds[T]) -> Bounds[T]:
    x_lb, x_ub = all_as_bounds(x)[0].concrete
    out_lb = jnp.where((x_lb < 0) & (x_ub >= 0), -jnp.inf, 1 / x_ub)
    out_ub = jnp.where((x_lb <= 0) & (x_ub > 0), jnp.inf, 1 / x_lb)
    return Box(out_lb, out_ub)


def ibp_rule_jaxpr(
    jaxpr: ClosedJaxpr,
    *args: Bounds | Real[Array, "..."],
    ibp: Callable,
) -> tuple[ClosedJaxpr, tuple[Real[Array, "..."], ...], PyTree]:
    """Performs Interval Bound Propagation (IBP) on a Jaxpr, returning a new Jaxpr.

    Args:
        jaxpr: The Jaxpr to produce bounds on.
        args: Bounded and concrete arguments for the Jaxpr.
        ibp: An IBP callable with the same signature as ``ibp``.
            Make your rule recursive by setting ``rule.recursive = True``
            to obtain the right IBP callable.

    Returns:
        - A Jaxpr of the IBP computation.
        - The flattened arguments.
        - The aval Pytree of the output of the IBP computation.
    """
    args_flat, in_tree = jax.tree.flatten(args)
    avals_in = duplicate_for_bounds(jaxpr.in_avals, args)
    ibp_jaxpr, out_shapes = _ibp_jaxpr(jaxpr, avals_in, in_tree, ibp)
    return ibp_jaxpr, args_flat, out_shapes


@jax_util.weakref_lru_cache
def _ibp_jaxpr(
    jaxpr: ClosedJaxpr,
    avals_in: tuple[jax.Array, ...],
    in_tree: PyTreeDef,
    ibp: Callable,
) -> tuple[ClosedJaxpr, PyTree]:
    f = jax.extend.core.jaxpr_as_fun(jaxpr)
    f_ibp = ibp(f)

    avals_in = jax.tree.unflatten(in_tree, avals_in)
    jaxpr_out, out_shapes = jax.make_jaxpr(f_ibp, return_shape=True)(*avals_in)
    return jaxpr_out, out_shapes


# Broadcasting, reshaping, padding, etc. are linear transformations with
# non-negative weights (all one or zero).
# ------------------------------------------------------------------------------
register_ibp_rule(
    lax.broadcast_in_dim_p,
    partial(ibp_rule_monotonic_non_decreasing, lax.broadcast_in_dim_p.bind),
)
register_ibp_rule(
    lax.concatenate_p,
    partial(ibp_rule_monotonic_non_decreasing, lax.concatenate),
)
register_ibp_rule(
    lax.convert_element_type_p,
    partial(ibp_rule_monotonic_non_decreasing, lax.convert_element_type_p.bind),
)
register_ibp_rule(
    lax.copy_p, partial(ibp_rule_monotonic_non_decreasing, lax.copy_p.bind)
)
register_ibp_rule(lax.pad_p, partial(ibp_rule_monotonic_non_decreasing, lax.pad))
register_ibp_rule(
    lax.reshape_p, partial(ibp_rule_monotonic_non_decreasing, lax.reshape_p.bind)
)
register_ibp_rule(lax.rev_p, partial(ibp_rule_monotonic_non_decreasing, lax.rev))
register_ibp_rule(lax.slice_p, partial(ibp_rule_monotonic_non_decreasing, lax.slice))
register_ibp_rule(
    lax.split_p, partial(ibp_rule_monotonic_non_decreasing, lax.split_p.bind)
)
register_ibp_rule(
    lax.squeeze_p, partial(ibp_rule_monotonic_non_decreasing, lax.squeeze)
)
register_ibp_rule(
    lax.transpose_p, partial(ibp_rule_monotonic_non_decreasing, lax.transpose)
)

# Element-Wise Non-Decreasing Functions
# ------------------------------------------------------------------------------
register_ibp_rule(lax.acosh_p, partial(ibp_rule_monotonic_non_decreasing, lax.acosh))
register_ibp_rule(lax.add_p, partial(ibp_rule_monotonic_non_decreasing, lax.add))
register_ibp_rule(lax.atan_p, partial(ibp_rule_monotonic_non_decreasing, lax.atan))
register_ibp_rule(lax.clamp_p, partial(ibp_rule_monotonic_non_decreasing, lax.clamp))
register_ibp_rule(lax.exp_p, partial(ibp_rule_monotonic_non_decreasing, lax.exp))
register_ibp_rule(lax.expm1_p, partial(ibp_rule_monotonic_non_decreasing, lax.expm1))
register_ibp_rule(lax.exp2_p, partial(ibp_rule_monotonic_non_decreasing, lax.exp2))
register_ibp_rule(lax.log_p, partial(ibp_rule_monotonic_non_decreasing, lax.log))
register_ibp_rule(lax.log1p_p, partial(ibp_rule_monotonic_non_decreasing, lax.log1p))
register_ibp_rule(
    lax.logistic_p, partial(ibp_rule_monotonic_non_decreasing, lax.logistic)
)
register_ibp_rule(lax.max_p, partial(ibp_rule_monotonic_non_decreasing, lax.max))
register_ibp_rule(lax.min_p, partial(ibp_rule_monotonic_non_decreasing, lax.min))
register_ibp_rule(
    lax.reduce_sum_p, partial(ibp_rule_monotonic_non_decreasing, lax.reduce_sum)
)
register_ibp_rule(
    lax.reduce_max_p, partial(ibp_rule_monotonic_non_decreasing, lax.reduce_max)
)
register_ibp_rule(
    lax.reduce_min_p, partial(ibp_rule_monotonic_non_decreasing, lax.reduce_min)
)
register_ibp_rule(
    lax.reduce_window_sum_p,
    partial(ibp_rule_monotonic_non_decreasing, lax.reduce_window_sum_p.bind),
)
register_ibp_rule(
    lax.reduce_window_max_p,
    partial(ibp_rule_monotonic_non_decreasing, lax.reduce_window_max_p.bind),
)
register_ibp_rule(
    lax.reduce_window_min_p,
    partial(ibp_rule_monotonic_non_decreasing, lax.reduce_window_min_p.bind),
)
register_ibp_rule(
    lax.select_and_scatter_add_p,
    partial(ibp_rule_monotonic_non_decreasing, lax.select_and_scatter_add_p.bind),
)
register_ibp_rule(lax.sign_p, partial(ibp_rule_monotonic_non_decreasing, lax.sign))
register_ibp_rule(lax.sqrt_p, partial(ibp_rule_monotonic_non_decreasing, lax.sqrt))
register_ibp_rule(lax.tanh_p, partial(ibp_rule_monotonic_non_decreasing, lax.tanh))

register_ibp_rule(
    jax_ad_util.add_jaxvals_p,
    partial(ibp_rule_monotonic_non_decreasing, jax_ad_util.add_jaxvals),
)

# Element-Wise Non-Increasing Functions
# ------------------------------------------------------------------------------
register_ibp_rule(lax.neg_p, partial(ibp_rule_monotonic_non_increasing, lax.neg))
register_ibp_rule(lax.rsqrt_p, partial(ibp_rule_monotonic_non_increasing, lax.rsqrt))


# Convex Functions
# ------------------------------------------------------------------------------
register_ibp_rule(lax.abs_p, partial(ibp_rule_strongly_convex, lax.abs, (0, 0)))
register_ibp_rule(lax.cosh_p, partial(ibp_rule_strongly_convex, lax.cosh, (0, 1)))
register_ibp_rule(lax.square_p, partial(ibp_rule_strongly_convex, lax.square, (0, 0)))


def _ibp_integer_pow_rule[T: jax.Array](x: Bounds[T] | T, y: int):
    if y == -1:
        return ibp_rule_reciprocal(x)
    elif y < 0:
        raise NotImplementedError(
            f"IBP for integer_pow with negative exponent ({y=}) is not implemented yet."
        )
    elif y % 2 == 1:
        return ibp_rule_monotonic_non_decreasing(lax.integer_pow, x, y=y)
    else:
        return ibp_rule_strongly_convex(lax.integer_pow, (0, 0), x, y=y)


register_ibp_rule(lax.integer_pow_p, _ibp_integer_pow_rule)


# General Linear Maps
# ------------------------------------------------------------------------------
register_ibp_rule(
    lax.conv_general_dilated_p,
    partial(ibp_rule_linear, lax.conv_general_dilated),
)
register_ibp_rule(lax.dot_general_p, partial(ibp_rule_linear, lax.dot_general))
register_ibp_rule(lax.mul_p, ibp_rule_mul)


def _sub_ibp_rule[T: jax.Array](x: T | Bounds[T], y: T | Bounds[T]) -> Bounds[T]:
    (x_lb, x_ub), (y_lb, y_ub) = (b.concrete for b in all_as_bounds(x, y))
    return Box(lax.sub(x_lb, y_ub), lax.sub(x_ub, y_lb))


register_ibp_rule(lax.sub_p, _sub_ibp_rule)


# Non-Linear Functions
# ------------------------------------------------------------------------------


def _div_ibp_rule[T: jax.Array](x: T | Bounds[T], y: T | Bounds[T]) -> Bounds[T]:
    if is_bounds(y):
        y_reciprocal = ibp_rule_reciprocal(y)
    else:
        y_reciprocal = jnp.reciprocal(y)
    return ibp_rule_mul(x, y_reciprocal)


register_ibp_rule(lax.div_p, _div_ibp_rule)


# Comparison Operators (less than, greater than, ...)
# ------------------------------------------------------------------------------
# Assume False < True (which is true in Python).
# The rules for >= are (similar for >):
# [lb1, ub1] >= [lb2, ub2] = [True, True] if lb1 >= ub2
# [lb1, ub1] >= [lb2, ub2] = [False, False] if ub1 < lb2
# [lb1, ub1] >= [lb2, ub2] = [False, True] otherwise
#
# The rules for <= are:
# [lb1, ub1] <= [lb2, ub2] = [True, True] if ub1 <= lb2
# [lb1, ub1] <= [lb2, ub2] = [False, False] if lb1 > ub2
# [lb1, ub1] <= [lb2, ub2] = [False, True] otherwise


def ibp_rule_compare_greater[T: jax.Array](
    greater: Callable, x: T | Bounds[T], y: T | Bounds[T]
) -> Bounds[T]:
    """IBP rule for greater than and greater equal.

    Args:
        greater: The comparison function.
        x: bounds on the first argument.
        y: bounds on the second argument.

    Returns:
        bounds on the comparison of ``x`` and ``y``.
    """
    (x_lb, x_ub), (y_lb, y_ub) = all_as_bounds(x, y)
    return Box(greater(x_lb, y_ub), greater(x_ub, y_lb))


def ibp_rule_compare_less[T: jax.Array](
    less: Callable, x: T | Bounds[T], y: T | Bounds[T]
) -> Bounds[T]:
    """IBP rule for less than and less equal.

    Args:
        less: The comparison function.
        x: bounds on the first argument.
        y: bounds on the second argument.

    Returns:
        bounds on the comparison of ``x`` and ``y``.
    """
    (x_lb, x_ub), (y_lb, y_ub) = (b.concrete for b in all_as_bounds(x, y))
    return Box(less(x_ub, y_lb), less(x_lb, y_ub))


register_ibp_rule(lax.ge_p, partial(ibp_rule_compare_greater, lax.ge))
register_ibp_rule(lax.gt_p, partial(ibp_rule_compare_greater, lax.gt))
register_ibp_rule(lax.le_p, partial(ibp_rule_compare_less, lax.le))
register_ibp_rule(lax.lt_p, partial(ibp_rule_compare_less, lax.lt))


def ibp_rule_compare_eq[T: jax.Array](x: T | Bounds[T], y: T | Bounds[T]) -> Bounds[T]:
    """IBP rule for equals."""
    (x_lb, x_ub), (y_lb, y_ub) = (b.concrete for b in all_as_bounds(x, y))
    intersect = (x_lb <= y_ub) & (y_lb <= y_ub)
    equal_points = (x_lb == x_ub) & (x_ub == y_lb) & (y_lb == y_ub)
    return Box(equal_points, intersect)


register_ibp_rule(lax.eq_p, ibp_rule_compare_eq)


# Conditionals
# ------------------------------------------------------------------------------


def _select_n_ibp_rule[T: jax.Array](
    which: T | Bounds[T], *cases: T | Bounds[T]
) -> Bounds[T]:
    assert len(cases) > 0
    cases = all_as_bounds(*cases)

    if not is_bounds(which):
        in_lbs, in_ubs = strict_zip(*[b.concrete for b in cases])
        out_lbs = lax.select_n(which, *in_lbs)
        out_ubs = lax.select_n(which, *in_ubs)
        return Box(out_lbs, out_ubs)

    # Take the minimum lower bound and maximum upper bound from all cases
    # having an index that lies in `which`.
    # Note: this rule does not allow computing gradients because of
    #       maximum and minimum, which have no transpose rules
    which_lb, which_ub = which
    inf = jnp.full_like(cases[0].lower_bound, fill_value=jnp.inf)
    ninf = jnp.full_like(cases[0].lower_bound, fill_value=-jnp.inf)
    out_lb = inf
    out_ub = ninf
    for i, case in enumerate(cases):
        i_in_which = (which_lb <= i) & (i <= which_ub)
        ith_lb = lax.select(i_in_which, case.lower_bound, inf)
        ith_ub = lax.select(i_in_which, case.upper_bound, ninf)
        out_lb = jnp.minimum(out_lb, ith_lb)
        out_ub = jnp.maximum(out_ub, ith_ub)
    return Box(out_lb, out_ub)


register_ibp_rule(lax.select_n_p, _select_n_ibp_rule)


# JIT'ed Functions
# ------------------------------------------------------------------------------


def _jit_ibp_rule[T: jax.Array](
    *args: T | Bounds[T],
    jaxpr: ClosedJaxpr,
    in_shardings,
    out_shardings,
    in_layouts,
    out_layouts,
    donated_invars,
    ibp: Callable,
    **kwargs,
):
    jaxpr_ibp, flat_args, shapes_out = ibp_rule_jaxpr(jaxpr, *args, ibp=ibp)

    duplicate_in = partial(duplicate_for_bounds, with_bounds=args)
    duplicate_out = partial(duplicate_for_bounds, with_bounds=shapes_out)
    out_flat = primitives.jit_p.bind(
        *flat_args,
        jaxpr=jaxpr_ibp,
        in_shardings=duplicate_in(in_shardings),
        out_shardings=duplicate_out(out_shardings),
        in_layouts=duplicate_in(in_layouts),
        out_layouts=duplicate_out(out_layouts),
        donated_invars=duplicate_in(donated_invars),  # FIXME: all False?
        **kwargs,
    )
    return jax.tree.unflatten(jax.tree.structure(shapes_out), out_flat)


register_ibp_rule(primitives.jit_p, _jit_ibp_rule, recursive=True)


# Special Functions
# ------------------------------------------------------------------------------


def _jax_nn_relu_ibp_rule(
    primitive: jax.extend.core.Primitive,
    parent_trace: jax.core.Trace,
    fun: Callable,
    jvp: Callable,
    *args: Bounds | jax.Array,
    symbolic_zeros,
):
    assert len(args) == 1, "ReLU got multiple arguments."
    # the case that the argument is an array is filtered out by IBPTrace.
    assert is_bounds(args[0]), "IBP rule invoked on non-bounds argument."

    # We could use the monotonous non-decreasing rule here, but
    # jax.nn.relu being custom_jvp_call_p complicates things, since
    # we need to call fun/jvp exactly once.
    # For this reason, we compute the output lower bound by binding to
    # custom_jvp_call_p and compute the output upper bound using a separate
    # jax.nn.relu call.

    fun = _apply_monotonous_non_decreasing_rule(fun.call_wrapped)
    jvp = _apply_monotonous_non_decreasing_rule(jvp.call_wrapped)

    fun_args, fun_in_tree = jax.tree.flatten(args)
    _, jvp_in_tree = jax.tree.flatten(args + args)

    fun, fun_out_tree = flatten_fun(fun, fun_in_tree)
    jvp, jvp_out_tree = flatten_fun(jvp, jvp_in_tree)

    fun = lu.wrap_init(fun)
    jvp = lu.wrap_init(jvp)
    out_vals = primitive.bind_with_trace(
        parent_trace, (fun, jvp, *fun_args), {"symbolic_zeros": symbolic_zeros}
    )

    was_jvp, out_tree = future_store.which_store(jvp_out_tree, fun_out_tree)
    if was_jvp:
        out_vals = jax.tree.unflatten(out_tree, list(out_vals) + list(out_vals))
        return out_vals[: len(out_vals) // 2]
    else:
        return jax.tree.unflatten(out_tree, out_vals)


def _apply_monotonous_non_decreasing_rule(f: Callable) -> Callable:
    def f_wrapped(*args):
        with transform_name_stack_ctx("jax_nn_relu_ibp", True):
            with jax.core.take_current_trace() as parent_trace:
                trace = _ReLUIBPTrace(parent_trace)
                in_tracers = [_ReLUIBPTracer(trace, *arg) for arg in args]
                with jax.core.set_current_trace(trace):
                    out_tracers = f(*in_tracers)
                return tuple(map(trace.unwrap, out_tracers))

    return f_wrapped


class _ReLUIBPTrace(jax.core.Trace["_ReLUIBPTracer"]):
    """Applies the monotonous non-decreasing rule.

    Intended for use with jax.nn.relu only.
    """

    def __init__(self, parent_trace: jax.core.Trace):
        super().__init__()
        self.parent_trace = parent_trace

    def unwrap(self, val) -> tuple[jax.Array, jax.Array]:
        return val.bounds if isinstance(val, _ReLUIBPTracer) else val

    def process_primitive(self, primitive, tracers: Sequence["_ReLUIBPTracer"], params):
        args = [self.unwrap(t) for t in tracers]
        if not any(is_bounds(a) for a in args):
            return primitive.bind_with_trace(self.parent_trace, args, params)

        in_lbs = [a.lower_bound if is_bounds(a) else a for a in args]
        in_ubs = [a.upper_bound if is_bounds(a) else a for a in args]
        out_lbs = primitive.bind_with_trace(self.parent_trace, in_lbs, params)
        out_ubs = primitive.bind_with_trace(self.parent_trace, in_ubs, params)
        if primitive.multiple_results:
            return [
                _ReLUIBPTracer(self, lb, ub) for lb, ub in strict_zip(out_lbs, out_ubs)
            ]
        else:
            return _ReLUIBPTracer(self, out_lbs, out_ubs)

    def process_custom_jvp_call(
        self,
        primitive: Primitive,
        fun: Callable,
        jvp: Callable,
        tracers: Sequence["_ReLUIBPTracer"],
        *,
        symbolic_zeros,
    ):
        args = [self.unwrap(t) for t in tracers]
        out_bounds = _jax_nn_relu_ibp_rule(
            primitive, self.parent_trace, fun, jvp, *args, symbolic_zeros=symbolic_zeros
        )
        return [_ReLUIBPTracer(self, *b.concrete) for b in out_bounds]


class _ReLUIBPTracer(jax.core.Tracer):
    __slots__ = ("lower_bound", "upper_bound")

    def __init__(
        self,
        trace: _ReLUIBPTrace,
        lower_bound,
        upper_bound,
    ):
        super().__init__(trace)
        if hasattr(lower_bound, "shape"):
            assert lower_bound.shape == upper_bound.shape
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def aval(self):
        return jax.core.get_aval(self.lower_bound)

    def full_lower(self):
        return self

    @property
    def bounds(self) -> Bounds:
        return Box(self.lower_bound, self.upper_bound)


register_ibp_rule(relu_marker, _jax_nn_relu_ibp_rule)
