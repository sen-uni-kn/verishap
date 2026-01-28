#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax.numpy as jnp
import pytest
from jax import lax

from formalax import Box, crown_ibp

# * import necessary to import various test fixtures
from ..module_cases import *  # noqa: F403, F401
from ..module_cases import (
    ModuleTestCase,
    abs_case,
    acasxu_case,
    atan_case,
    div_by_constant_case,
    div_by_neg_case,
    div_by_pos_case,
    div_case,
    emnist_flax_conv_case,
    flax_max_pool_case1,
    flax_max_pool_case2,
    flax_max_pool_case3,
    int_pow_even_case,
    jax_relu_case,
    max_case,
    mnist_equinox_conv_batchnorm_case,
    mnist_flax_mlp_case,
    mnist_ibp_training_flax_conv_case,
    mnist_onnx_conv_case,
    mnist_onnx_fully_connected_case,
    mul_case,
    reciprocal_case,
    reciprocal_negative_case,
    reciprocal_positive_case,
    scalar_out_vmapped_avg_of_vmapped_case,
    shallow_nn_case,
    shallow_nn_jit_case,
    shallow_vmapped_case,
    shallow_vmapped_jit_case,
    sigmoid_case,
    simple_relu_case,
    square_case,
    tanh_case,
    tanh_two_layer_nn_case,
    vmapped_avg_of_vmapped_case,
)
from .base_class import BoundsTest

test_crown_ibp_once_cases = (
    simple_relu_case.__name__,
    max_case.__name__,
    abs_case.__name__,
    atan_case.__name__,
    sigmoid_case.__name__,
    tanh_case.__name__,
    square_case.__name__,
    int_pow_even_case.__name__,
    reciprocal_case.__name__,
    reciprocal_positive_case.__name__,
    reciprocal_negative_case.__name__,
    div_case.__name__,
    div_by_pos_case.__name__,
    div_by_neg_case.__name__,
    div_by_constant_case.__name__,
    mul_case.__name__,
    jax_relu_case.__name__,
    flax_max_pool_case1.__name__,
    flax_max_pool_case2.__name__,
    flax_max_pool_case3.__name__,
    acasxu_case.__name__,
    mnist_onnx_fully_connected_case.__name__,
    mnist_onnx_conv_case.__name__,
    mnist_ibp_training_flax_conv_case.__name__,
    emnist_flax_conv_case.__name__,
    mnist_flax_mlp_case.__name__,
    mnist_equinox_conv_batchnorm_case.__name__,
)

test_crown_ibp_multiple_cases = (
    shallow_nn_case.__name__,
    shallow_nn_jit_case.__name__,
    shallow_vmapped_case.__name__,
    shallow_vmapped_jit_case.__name__,
    tanh_two_layer_nn_case.__name__,
    vmapped_avg_of_vmapped_case.__name__,
    scalar_out_vmapped_avg_of_vmapped_case.__name__,
)


class TestCrownIBP(BoundsTest):
    def compute_bounds(self, module_case: ModuleTestCase, **kwargs):
        return crown_ibp(module_case.module)

    @pytest.mark.parametrize("case", test_crown_ibp_once_cases)
    def test_bounds_once(self, case, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(case, request)

        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))

    @pytest.mark.parametrize("case", test_crown_ibp_multiple_cases)
    def test_bounds_repeat5(self, case, argument_seeds5, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, argument_seeds5
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))

    @pytest.mark.parametrize("broadcast_dims", [(1, 2), (0, 1), (0, 2)])
    def test_broadcast(self, broadcast_dims):
        in_shape = (2, 1)
        out_shape = (2, 2, 3)
        in_lb, in_ub = jnp.zeros(in_shape), jnp.ones(in_shape)

        out_lb, out_ub = crown_ibp(
            lambda x: lax.broadcast_in_dim(x, out_shape, broadcast_dims)
        )(Box(in_lb, in_ub)).concrete

        assert out_lb.shape == out_shape
        assert out_ub.shape == out_shape
        assert (out_lb == jnp.zeros(out_shape)).all()
        assert (out_ub == jnp.ones(out_shape)).all()
