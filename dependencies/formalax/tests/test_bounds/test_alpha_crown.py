#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from functools import partial

import jax
import jax.numpy as jnp
import pytest

from formalax.bounds import alpha_crown, backwards_crown
from formalax.bounds._src._bwcrown import _ibp_bounds
from formalax.utils.zip import strict_zip

from ..module_cases import (
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
from ..random_utils import get_seed
from .base_class import BoundsTest

alpha_crown_test_once_cases = [
    simple_relu_case.__name__,
    max_case.__name__,
    abs_case.__name__,
    tanh_case.__name__,
    sigmoid_case.__name__,
    atan_case.__name__,
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
]

alpha_crown_test_multiple_cases = [
    shallow_nn_case.__name__,
    shallow_nn_jit_case.__name__,
    shallow_vmapped_case.__name__,
    shallow_vmapped_jit_case.__name__,
    tanh_two_layer_nn_case.__name__,
]


@pytest.mark.parametrize("strategy", ["external-full", "external-shared", "fixed"])
class TestAlphaCROWNRandomParams(BoundsTest):
    def compute_bounds(self, module_case, **kwargs):
        bwcrown, param_domains = backwards_crown(
            module_case.module,
            module_case.bounded_args,
            _ibp_bounds,
            param_strategies=kwargs["strategy"],
        )

        seed = get_seed("alpha_crown")
        rng_keys = jax.random.split(jax.random.PRNGKey(seed), len(param_domains))
        params = [
            jax.random.uniform(key, shape=domain.lower_bound.shape)
            for domain, key in strict_zip(param_domains, rng_keys)
        ]
        return partial(bwcrown, params)

    @pytest.mark.parametrize("case", alpha_crown_test_once_cases)
    def test_bounds_once(self, case, strategy, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, strategy=strategy
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))

    @pytest.mark.parametrize("case", alpha_crown_test_multiple_cases)
    def test_bounds_repeat5(self, case, argument_seeds5, strategy, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, argument_seeds5, strategy=strategy
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))


@pytest.mark.parametrize("strategy", ["external-full", "external-shared"])
class TestAlphaCROWNOptimizeParams(BoundsTest):
    def compute_bounds(self, module_case, **kwargs):
        return alpha_crown(module_case.module, steps=1)

    @pytest.mark.parametrize("case", alpha_crown_test_once_cases)
    def test_bounds_once(self, case, strategy, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, strategy=strategy
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))

    @pytest.mark.parametrize("case", alpha_crown_test_multiple_cases)
    def test_bounds_repeat5(self, case, argument_seeds5, strategy, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, argument_seeds5, strategy=strategy
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | jnp.isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | jnp.isclose(out_ub, out_concrete))
