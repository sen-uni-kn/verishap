#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import jax.numpy as jnp
import pytest

from formalax.verify import input_splitting_bab

from ..module_cases import (
    reduce_max_case1,
    reduce_min_case1,
    reduce_sum_case1,
    request_case_fixture,
    scalar_add_case,
    scalar_add_constant_left_case,
    scalar_add_constant_right_case,
    scalar_broadcast_add_case,
    scalar_jax_relu_case,
    scalar_max_case,
    scalar_mul_case,
    scalar_mul_constant_left_case,
    scalar_mul_constant_right_case,
    scalar_neg_case,
    scalar_out_acasxu_case,
    scalar_out_affine_layer_case,
    scalar_out_conv_layer_case,
    scalar_out_flax_avg_pool_case,
    scalar_out_flax_max_pool_case,
    scalar_out_flax_min_pool_case,
    scalar_out_linear_nn_case,
    scalar_out_linear_nn_jit_case,
    scalar_out_reshape_case,
    scalar_out_shallow_nn_case,
    scalar_out_shallow_nn_jit_case,
    scalar_out_shallow_vmapped_case,
    scalar_out_shallow_vmapped_jit_case,
    scalar_out_tanh_two_layer_nn_case,
    scalar_out_transpose_case,
    scalar_sigmoid_case,
    scalar_simple_relu_case,
)

bab_input_splitting_test_once_cases = (
    reduce_sum_case1.__name__,
    reduce_max_case1.__name__,
    reduce_min_case1.__name__,
    scalar_out_reshape_case.__name__,
    scalar_out_transpose_case.__name__,
    scalar_out_flax_max_pool_case.__name__,
    scalar_out_flax_min_pool_case.__name__,
    scalar_out_flax_avg_pool_case.__name__,
    scalar_out_acasxu_case.__name__,
)

bab_input_splitting_test_multiple_cases = (
    scalar_neg_case.__name__,
    scalar_add_case.__name__,
    scalar_broadcast_add_case.__name__,
    scalar_add_constant_right_case.__name__,
    scalar_add_constant_left_case.__name__,
    scalar_mul_case.__name__,
    scalar_mul_constant_right_case.__name__,
    scalar_mul_constant_left_case.__name__,
    scalar_simple_relu_case.__name__,
    scalar_max_case.__name__,
    scalar_jax_relu_case.__name__,
    scalar_sigmoid_case.__name__,
    scalar_out_affine_layer_case.__name__,
    scalar_out_conv_layer_case.__name__,
    scalar_out_shallow_nn_case.__name__,
    scalar_out_shallow_nn_jit_case.__name__,
    scalar_out_shallow_vmapped_case.__name__,
    scalar_out_shallow_vmapped_jit_case.__name__,
    scalar_out_linear_nn_case.__name__,
    scalar_out_linear_nn_jit_case.__name__,
    scalar_out_tanh_two_layer_nn_case.__name__,  # has a fixed batch size
)


class TestBaBInputSplitting:
    iterations = 5

    @pytest.mark.parametrize("target", ["min", "max", "img"])
    @pytest.mark.parametrize("case", bab_input_splitting_test_once_cases)
    def test_verify_once(self, target, case, request):
        module_case = request_case_fixture(case, request)
        out_concrete = module_case.module(*module_case.concrete_args)

        bounds = input_splitting_bab(module_case.module, target=target)(
            *module_case.bounded_args
        )
        for _ in range(self.iterations):
            lb, ub = next(bounds)
            print(lb, ub)
            assert jnp.all(lb <= ub)
            if target in ("min", "img"):
                assert jnp.all((lb <= out_concrete) | jnp.isclose(lb, out_concrete))
            if target in ("max", "img"):
                assert jnp.all((ub >= out_concrete) | jnp.isclose(ub, out_concrete))

    @pytest.mark.parametrize("target", ["min", "max", "img"])
    @pytest.mark.parametrize("case", bab_input_splitting_test_multiple_cases)
    def test_verify_repeat5(self, target, case, argument_seeds5, request):
        module_case = request_case_fixture(case, request, argument_seeds5)
        out_concrete = module_case.module(*module_case.concrete_args)

        bounds = input_splitting_bab(module_case.module, target=target)(
            *module_case.bounded_args
        )
        for _ in range(self.iterations):
            lb, ub = next(bounds)
            assert jnp.all(lb <= ub)
            if target in ("min", "img"):
                assert jnp.all((lb <= out_concrete) | jnp.isclose(lb, out_concrete))
            if target in ("max", "img"):
                assert jnp.all((ub >= out_concrete) | jnp.isclose(ub, out_concrete))
