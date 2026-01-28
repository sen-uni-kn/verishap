#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax.numpy as jnp
import pytest

from formalax import crown

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
    flax_max_pool_case1,
    flax_max_pool_case2,
    flax_max_pool_case3,
    int_pow_even_case,
    jax_relu_case,
    max_case,
    mnist_onnx_fully_connected_case,
    mul_case,
    reciprocal_case,
    reciprocal_negative_case,
    reciprocal_positive_case,
    shallow_nn_case,
    shallow_nn_jit_case,
    shallow_vmapped_case,
    shallow_vmapped_jit_case,
    sigmoid_case,
    simple_relu_case,
    square_case,
    tanh_case,
    tanh_two_layer_nn_case,
)
from .base_class import BoundsTest

test_crown_once_cases = (
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
    # These have too high dimensional hidden dimensions to run CROWN
    # mnist_onnx_conv_case.__name__,
    # mnist_ibp_training_flax_conv_case.__name__,
    # emnist_flax_conv_case.__name__,
    # mnist_flax_mlp_case.__name__,
    # mnist_equinox_conv_batchnorm_case.__name__,
)

test_crown_multiple_cases = (
    shallow_nn_case.__name__,
    shallow_nn_jit_case.__name__,
    shallow_vmapped_case.__name__,
    shallow_vmapped_jit_case.__name__,
    tanh_two_layer_nn_case.__name__,
)


class TestCrown(BoundsTest):
    def compute_bounds(self, module_case: ModuleTestCase, **kwargs):
        return crown(module_case.module)

    @pytest.mark.parametrize("case", test_crown_once_cases)
    def test_bounds_once(self, case, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(case, request)
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))

    @pytest.mark.parametrize("case", test_crown_multiple_cases)
    def test_bounds_repeat5(self, case, argument_seeds5, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, argument_seeds5
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))
