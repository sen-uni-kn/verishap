#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from functools import partial

import jax.numpy as jnp
import pytest
from frozendict import frozendict
from jax import lax

from formalax import Box
from formalax.bounds import backwards_lirpa
from formalax.bounds._src._bwcrown import _ibp_bounds

# * import necessary to import various test fixtures
from ..module_cases import *  # noqa: F403, F401
from ..module_cases import (
    ModuleTestCase,
    add_case,
    add_constant_left_case,
    add_constant_right_case,
    affine_layer_case1,
    affine_layer_case2,
    broadcast_add_case,
    broadcast_add_case2,
    conv_layer_case,
    div_by_constant_case,
    ge_case,
    gt_case,
    le_case,
    linear_nn_case,
    linear_nn_jit_case,
    lt_case,
    mul_constant_left_case,
    mul_constant_right_case,
    neg_case,
    reshape_case,
    sub_case,
    sub_constant_case,
    sub_from_constant_case,
    transpose_case,
)
from .base_class import BoundsTest

test_lirpa_once_cases = (
    neg_case.__name__,
    add_case.__name__,
    broadcast_add_case.__name__,
    broadcast_add_case2.__name__,
    add_constant_right_case.__name__,
    add_constant_left_case.__name__,
    sub_case.__name__,
    sub_constant_case.__name__,
    sub_from_constant_case.__name__,
    mul_constant_right_case.__name__,
    mul_constant_left_case.__name__,
    div_by_constant_case.__name__,
    reshape_case.__name__,
    transpose_case.__name__,
    ge_case.__name__,
    gt_case.__name__,
    le_case.__name__,
    lt_case.__name__,
)

test_lirpa_multiple_cases = (
    affine_layer_case1.__name__,
    affine_layer_case2.__name__,
    conv_layer_case.__name__,
    linear_nn_case.__name__,
    linear_nn_jit_case.__name__,
)


def get_compute_bounds(fun, compute_intermediate):
    bwlirpa = backwards_lirpa(
        fun, nonlinear_rules=frozendict({}), compute_bounds=compute_intermediate
    )
    return partial(bwlirpa, {})


@pytest.mark.parametrize("intermediate_bounds", ["bootstrap", _ibp_bounds])
class TestLirpa(BoundsTest):
    def compute_bounds(self, module_case: ModuleTestCase, **kwargs):
        return get_compute_bounds(module_case.module, kwargs["intermediate_bounds"])

    @pytest.mark.parametrize("case", test_lirpa_once_cases)
    def test_bounds_once(self, case, intermediate_bounds, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, intermediate_bounds=intermediate_bounds
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))

    @pytest.mark.parametrize("case", test_lirpa_multiple_cases)
    def test_bounds_repeat5(self, case, intermediate_bounds, argument_seeds5, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, argument_seeds5, intermediate_bounds=intermediate_bounds
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))

    @pytest.mark.parametrize("broadcast_dims", [(1, 2), (0, 1), (0, 2)])
    def test_broadcast(self, broadcast_dims, intermediate_bounds):
        in_shape = (2, 1)
        out_shape = (2, 2, 3)
        in_lb, in_ub = jnp.zeros(in_shape), jnp.ones(in_shape)

        out_lb, out_ub = get_compute_bounds(
            lambda x: lax.broadcast_in_dim(x, out_shape, broadcast_dims),
            intermediate_bounds,
        )(Box(in_lb, in_ub)).concrete

        assert out_lb.shape == out_shape
        assert out_ub.shape == out_shape
        assert (out_lb == jnp.zeros(out_shape)).all()
        assert (out_ub == jnp.ones(out_shape)).all()
