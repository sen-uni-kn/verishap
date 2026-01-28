#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from abc import ABC, abstractmethod
from typing import Callable

import chex
import jax
import jax.numpy as jnp

from ..module_cases import ModuleTestCase, request_case_fixture


class BoundsTest(ABC):
    """Base class for test classes that test bound computation methods."""

    @abstractmethod
    def compute_bounds(self, module_case: ModuleTestCase, **kwargs) -> Callable:
        raise NotImplementedError()

    def get_bounds(self, module_case: ModuleTestCase, **kwargs):
        out_concrete = module_case.module(*module_case.concrete_args)
        out_lb, out_ub = self.compute_bounds(module_case, **kwargs)(
            *module_case.bounded_args
        ).concrete
        return out_concrete, out_lb, out_ub

    def get_bounds_case_fixture(
        self, case_fixture, request, rng_seed=None, batch_size=None, **kwargs
    ):
        module_case = request_case_fixture(case_fixture, request, rng_seed, batch_size)
        return self.get_bounds(module_case, **kwargs)

    def assert_grad_bounds_margin(self, module_case: ModuleTestCase):
        jax.config.update("jax_enable_x64", True)
        flat_args, args_tree = jax.tree.flatten(module_case.bounded_args)

        def bounds_margin(*flat_args):
            compute_bounds_args = jax.tree.unflatten(args_tree, flat_args)
            out_lb, out_ub = self.compute_bounds(module_case)(*compute_bounds_args)
            return jnp.mean(out_lb - out_ub)

        grad_fn = jax.grad(bounds_margin, list(range(len(flat_args))))
        grad = grad_fn(*flat_args)
        print(grad)
        chex.assert_numerical_grads(bounds_margin, flat_args, order=1)

    def assert_grad_bounds_margin_case_fixture(
        self, case_fixture, request, rng_seed=None, batch_size=None
    ):
        module_case = request_case_fixture(case_fixture, request, rng_seed, batch_size)
        self.assert_grad_bounds_margin(module_case)
