from dataclasses import dataclass
from typing import Callable, Protocol, Tuple

import jax.numpy as jnp


class SatisfactionFunction(Protocol):
    """
    A protocol for satisfaction functions used in falsification attacks
    """

    def __call__(
        self, inputs: jnp.ndarray, network: Callable[[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, bool]:
        """
        Computes the satisfaction measure and boolean satisfaction state.

        Returns:
            A tuple of (satisfaction_value, is_satisfied)
            - satisfaction_value: scalar or vector of satisfaction scores
            - is_satisfied: bool or array of bools indicating satisfaction
        """
        ...


@dataclass
class Bounds:
    lower_bound: jnp.ndarray
    upper_bound: jnp.ndarray
