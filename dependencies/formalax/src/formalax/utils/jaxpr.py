#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from functools import partial
from typing import Callable

from jax.core import Atom
from jax.extend.core import ClosedJaxpr, Jaxpr, Literal, Var
from jaxtyping import Array, Shaped

from .name_stack import eqn_name_stack_ctx
from .vardict import VarDict
from .zip import strict_zip


def jaxpr_as_fun(
    closed_jaxpr: ClosedJaxpr, *, intermediate_values: bool = True
) -> Callable[..., VarDict[Shaped[Array, "..."]]]:
    """Returns a function that evaluates ``jaxpr``.

    Args:
        closed_jaxpr: The Jaxpr (``jax.core.ClosedJaxpr``) to evaluate.
        intermediate_values: Whether the returned function returns only the output
            values (``intermediate_values=False``) or also the values of the
            intermediate variables (``intermediate_values=True``).
            See ``eval_jaxpr`` for more details.

    Returns:
        A function that evaluates ``jaxpr``.
    """
    return partial(
        eval_jaxpr,
        closed_jaxpr.jaxpr,
        closed_jaxpr.consts,
        intermediate_values=intermediate_values,
        propagate_source_info=True,
    )


def eval_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    intermediate_values: bool = False,
    propagate_source_info: bool = True,
) -> tuple[Shaped[Array, "..."], ...] | VarDict[Shaped[Array, "..."]]:
    """Evaluates a Jaxpr. Can return only the outputs or all intermediate values.

    Args:
        jaxpr: The Jaxpr to evaluate.
        consts: The values of the constants in ``jaxpr``.
        *args: The values for the input variables in ``jaxpr``.
        intermediate_values: Whether to return only the values of the
            output variables (``intermediate_values=False``), or return
            the values of all variables appearing in ``jaxpr``.
        propagate_source_info: See ``jax.core.eval_jaxpr``.

    Returns:
        If ``intermeditate_values`` is ``False``, returns the tuple of output values.
        If ``intermeditate_values`` is ``True``, returns a mapping that contains
        the values of all variables appearing in ``jaxpr``.
    """
    # based on
    # https://github.com/google/jax/blob/5e418f5ab2692d4791816e85ed82eb0834a579cb/jax/_src/core.py#L480
    env: dict[Var, Shaped[Array, "..."]] = {}

    def read(v: Atom) -> Shaped[Array, "..."]:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Shaped[Array, "..."]):
        env[v] = val

    for constvar, const in strict_zip(jaxpr.constvars, consts):
        write(constvar, const)

    for invar, arg in strict_zip(jaxpr.invars, args):
        write(invar, arg)

    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)

        with eqn_name_stack_ctx(eqn, propagate_source_info), eqn.ctx.manager:
            ans = eqn.primitive.bind(
                *subfuns, *[read(v) for v in eqn.invars], **bind_params
            )

        if eqn.primitive.multiple_results:
            for outvar, val in strict_zip(eqn.outvars, ans):
                write(outvar, val)
        else:
            write(eqn.outvars[0], ans)

    if intermediate_values:
        return VarDict(env)
    else:
        return tuple(read(v) for v in jaxpr.outvars)


def all_vars(jaxpr: Jaxpr | ClosedJaxpr) -> tuple[Var, ...]:
    """Returns all variables in a Jaxpr."""
    if isinstance(jaxpr, ClosedJaxpr):
        jaxpr = jaxpr.jaxpr

    all_vars = jaxpr.invars + jaxpr.outvars
    all_vars += sum((eqn.invars for eqn in jaxpr.eqns), [])
    all_vars += sum((eqn.outvars for eqn in jaxpr.eqns), [])
    all_vars = {var for var in all_vars if not isinstance(var, Literal)}
    return tuple(all_vars)


class HashableLiteral:
    """Adds identity hashing to a ``jax.core.Literal``."""

    __slots__ = ("literal",)

    def __init__(self, literal: Literal):
        self.literal = literal

    @property
    def val(self):
        return self.literal.val

    @property
    def aval(self):
        return self.literal.aval

    def hash(self):
        return self.literal.hash

    def __eq__(self, other):
        if isinstance(other, HashableLiteral):
            other = other.literal
        if not isinstance(other, Literal):
            return False
        if hasattr(self.literal, "hash") and hasattr(other, "hash"):
            return self.literal.val == other.val
        else:
            return self.literal is other

    def __hash__(self):
        if hasattr(self.literal, "hash"):
            return self.literal.hash
        else:
            return hash(id(self.literal))

    def __repr__(self):
        if hasattr(self.literal, "hash"):
            return repr(self.literal)
        else:
            return f"HashableLiteral(val={self.literal.val}, id={id(self.literal)})"
