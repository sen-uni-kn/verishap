import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import jax
from jax import numpy as jnp
from jaxtyping import Array, PyTree, Real

from formalax.core.zero import Zero, apply_non_zero, zero

from ...sets.box import Box
from ...sets.protocols import MinMaxSplittable
from ._bounds import Bounds, is_bounds


class AffineBounds[T: Real[Array, "..."] | Zero](Bounds[T], Protocol):
    """An affine lower and upper bound on a set of ``jax.Arrays``.

    This protocol defines the interface for pairs of affine lower
    and upper bounds on a set of ``Arrays``.
    Implementing classes may use different affine transformations, for example,
    simple matrix multiplication, or batched tensor products.

    ``AffineBounds`` instances accept pytrees as arguments for functions like
    ``lower_bound_at`` or ``concretize``.
    These pytree need to have the same number of leaves as the ``AffineBounds``
    instance has weights.
    Each leaf of the input pytree is matched to a weight based on the order of the
    flattened pytree.

    Subclasses of ``AffineBounds`` may differ by the format in which
    the weights and biases are stored.
    """

    @property
    def domain(self) -> PyTree[Bounds[T], "W"]:
        """The input domain of these affine bounds.

        This is the domain for which the affine bounds were computed and
        are valid.
        """
        ...

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def lb_weights(self) -> PyTree[T, "W"]:
        """Returns the weights of the affine lower bound."""
        ...

    @property
    def lb_bias(self) -> T:
        """Returns the bias of the affine lower bound."""
        ...

    @property
    def ub_weights(self) -> PyTree[T, "W"]:
        """Returns the weights of the affine upper bound."""
        ...

    @property
    def ub_bias(self) -> T:
        """Returns the bias of the affine upper bound."""
        ...

    # --------------------------------------------------------------------------
    # Affine Bounds
    # --------------------------------------------------------------------------

    def lower_bound_at(self, xs: PyTree[T, "X"]) -> T:
        """Evaluates the affine lower bound for the inputs ``xs``."""
        ...

    def upper_bound_at(self, xs: PyTree[T, "X"]) -> T:
        """Evaluates the affine upper bound for the inputs ``xs``."""
        ...

    def concretize(self, *xs_bounds: PyTree[T, "X"]) -> Box[T]:
        """Compute a constant lower and upper bound from the affine bounds.

        Args:
            xs_bounds: The bounds on the inputs.
                This needs to be a pytree of ``Bounds`` instances
                or ``jax.Arrays`` with the same pytree structure as the weights
                of the affine bounds.
                For example, ``self.weights_lb``.

        Returns:
            Constant bounds ``Box(lb, ub)`` such that
            ``lb <= self.lower_bound(xs) <= self.upper_bound(xs) <= ub`` for
            all ``xs`` satisfying ``xs_bounds``.
        """
        ...

    # --------------------------------------------------------------------------
    # Bounds Interface
    # --------------------------------------------------------------------------

    @property
    def lower_bound(self) -> T:
        """A concrete lower bound on this affine lower bound with in ``self.domain``."""
        ...

    @property
    def upper_bound(self) -> T:
        """A concrete lower bound on this affine lower bound with in ``self.domain``."""
        ...

    @property
    def concrete(self) -> MinMaxSplittable[T]:
        """A concrete lower and upper bound on this affine lower bound with in ``self.domain``."""
        ...


_UNASSIGNED = object()  # sentinel for _LazyAttr


class _LazyAttr[T]:
    __slots__ = ("__val",)

    def __init__(self):
        self.__val = _UNASSIGNED

    def get(self, compute: Callable[[], T]) -> T:
        if self.__val is _UNASSIGNED:
            self.__val = compute()
        return self.__val


@dataclass(eq=False, frozen=True, slots=True, kw_only=True)
class AffineBoundsABC[T](AffineBounds[T], ABC):
    """Abstract base class for ``AffineBounds`` implementations.

    This class implements ``concretize``, ``lower_bound_at``, ``upper_bound_at``,
    ``lower_bound``, ``upper_bound``, and ``concrete`` using the ``affine`` method.
    The values of ``lower_bounds``, ``upper_bounds``, and ``concrete`` are cached
    to provide an efficient ``Bounds`` interface.

    Stores weights and the domain as a tuple of arrays.
    """

    lb_weights: tuple[T, ...]
    ub_weights: tuple[T, ...]
    lb_bias: T
    ub_bias: T
    domain: tuple[Bounds[T], ...]
    __concrete_bounds: _LazyAttr[T] = dataclasses.field(
        default_factory=_LazyAttr, init=False, repr=False
    )

    @abstractmethod
    def affine(
        self, xs: tuple[T, ...], weights: tuple[T, ...], bias: T | Zero = zero
    ) -> T:
        """Computes the affine transformation of ``xs`` with ``weights`` and ``bias``.

        The arguments ``xs`` and ``weights`` need to be pytrees of arrays
        with the same structure as ``self.lb_weights`` and ``self.ub_weights``.

        Args:
            xs: The inputs to the affine transformation.
            weights: The weights of the affine transformation.
            bias: The bias of the affine transformation.
                Can be ``Zero``, which stands for a scalar zero.

        Returns:
            An affine transformation like ``\\sum_i xs_i @ weights_i + bias``.
        """
        raise NotImplementedError()

    def lower_bound_at(self, xs: PyTree[T, "X"]) -> T:
        xs, _ = tuple(jax.tree.flatten(xs))
        return self.affine(xs, self.lb_weights, self.lb_bias)

    def upper_bound_at(self, xs: PyTree[T, "X"]) -> T:
        xs, _ = tuple(jax.tree.flatten(xs))
        return self.affine(xs, self.ub_weights, self.ub_bias)

    def concretize(self, *xs_bounds: PyTree[T, "X"]) -> Box[T]:
        xs, _ = jax.tree.flatten(
            xs_bounds,
            is_leaf=is_bounds,
        )
        in_lbs = tuple(x.lower_bound if is_bounds(x) else x for x in xs)
        in_ubs = tuple(x.upper_bound if is_bounds(x) else x for x in xs)

        def weights_pos_neg(weights):
            weights_pos = jax.tree.map(apply_non_zero(jnp.clip, min=0), weights)
            weights_neg = jax.tree.map(apply_non_zero(jnp.clip, max=0), weights)
            return weights_pos, weights_neg

        weights_pos, weights_neg = weights_pos_neg(self.lb_weights)
        out_lb = self.affine(in_lbs, weights_pos, self.lb_bias) + self.affine(
            in_ubs, weights_neg
        )

        weights_pos, weights_neg = weights_pos_neg(self.ub_weights)
        out_ub = self.affine(in_ubs, weights_pos, self.ub_bias) + self.affine(
            in_lbs, weights_neg
        )

        return Box(out_lb, out_ub)

    # --------------------------------------------------------------------------
    # Bounds interface with auto-concretization and caching
    # --------------------------------------------------------------------------

    @property
    def concrete(self) -> Box[T]:
        return self.__concrete_bounds.get(lambda: self.concretize(self.domain))

    @property
    def lower_bound(self) -> T:
        return self.concrete.lower_bound

    @property
    def upper_bound(self) -> T:
        return self.concrete.upper_bound
