#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax
import jax.numpy as jnp
import pytest
from jax import nn

from formalax import Box, ibp, ibp_jaxpr
from formalax.bounds.utils import is_bounds

from ..module_cases import (
    ModuleTestCase,
    abs_case,
    acasxu_case,
    add_case,
    add_constant_left_case,
    add_constant_right_case,
    affine_layer_case1,
    affine_layer_case2,
    atan_case,
    batch_bounds,
    broadcast_add_case,
    broadcast_add_case2,
    conv_layer_case,
    div_by_constant_case,
    div_by_neg_case,
    div_by_pos_case,
    div_case,
    emnist_flax_conv_case,
    flax_avg_pool_case1,
    flax_avg_pool_case2,
    flax_avg_pool_case3,
    flax_max_pool_case1,
    flax_max_pool_case2,
    flax_max_pool_case3,
    flax_min_pool_case1,
    flax_min_pool_case2,
    flax_min_pool_case3,
    ge_case,
    gt_case,
    jax_relu_case,
    le_case,
    linear_nn_case,
    linear_nn_jit_case,
    lt_case,
    max_case,
    mnist_equinox_conv_batchnorm_case,
    mnist_flax_mlp_case,
    mnist_ibp_training_flax_conv_case,
    mnist_onnx_conv_case,
    mnist_onnx_fully_connected_case,
    mul_case,
    mul_constant_left_case,
    mul_constant_right_case,
    neg_case,
    reciprocal_case,
    reciprocal_negative_case,
    reciprocal_positive_case,
    reduce_max_case1,
    reduce_max_case2,
    reduce_min_case1,
    reduce_min_case2,
    reduce_sum_case1,
    reduce_sum_case2,
    request_case_fixture,
    reshape_case,
    scalar_out_vmapped_avg_of_vmapped_case,
    shallow_nn_case,
    shallow_nn_jit_case,
    shallow_vmapped_case,
    shallow_vmapped_jit_case,
    sigmoid_case,
    simple_relu_case,
    square_case,
    sub_case,
    sub_constant_case,
    sub_from_constant_case,
    tanh_case,
    tanh_two_layer_nn_case,
    transpose_case,
    vmapped_avg_of_vmapped_case,
)
from ..nets.simple import shallow_nn
from .base_class import BoundsTest

ibp_test_once_cases = (
    neg_case.__name__,
    add_case.__name__,
    broadcast_add_case.__name__,
    broadcast_add_case2.__name__,
    add_constant_right_case.__name__,
    add_constant_left_case.__name__,
    sub_case.__name__,
    sub_constant_case.__name__,
    sub_from_constant_case.__name__,
    mul_case.__name__,
    mul_constant_right_case.__name__,
    mul_constant_left_case.__name__,
    div_case.__name__,
    div_by_pos_case.__name__,
    div_by_neg_case.__name__,
    div_by_constant_case.__name__,
    reshape_case.__name__,
    transpose_case.__name__,
    simple_relu_case.__name__,
    max_case.__name__,
    abs_case.__name__,
    atan_case.__name__,
    sigmoid_case.__name__,
    tanh_case.__name__,
    square_case.__name__,
    reciprocal_case.__name__,
    reciprocal_positive_case.__name__,
    reciprocal_negative_case.__name__,
    ge_case.__name__,
    gt_case.__name__,
    le_case.__name__,
    lt_case.__name__,
    jax_relu_case.__name__,  # jit'ed
    reduce_sum_case1.__name__,
    reduce_sum_case2.__name__,
    reduce_max_case1.__name__,
    reduce_max_case2.__name__,
    reduce_min_case1.__name__,
    reduce_min_case2.__name__,
    flax_max_pool_case1.__name__,
    flax_max_pool_case2.__name__,
    flax_max_pool_case3.__name__,
    flax_min_pool_case1.__name__,
    flax_min_pool_case2.__name__,
    flax_min_pool_case3.__name__,
    flax_avg_pool_case1.__name__,
    flax_avg_pool_case2.__name__,
    flax_avg_pool_case3.__name__,
    acasxu_case.__name__,
    mnist_onnx_fully_connected_case.__name__,
    mnist_onnx_conv_case.__name__,
    mnist_ibp_training_flax_conv_case.__name__,
    emnist_flax_conv_case.__name__,
    mnist_flax_mlp_case.__name__,
    mnist_equinox_conv_batchnorm_case.__name__,
)

ibp_test_multiple_cases = (
    affine_layer_case1.__name__,
    affine_layer_case2.__name__,
    conv_layer_case.__name__,
    shallow_nn_case.__name__,
    shallow_nn_jit_case.__name__,
    shallow_vmapped_case.__name__,
    shallow_vmapped_jit_case.__name__,
    tanh_two_layer_nn_case.__name__,
    linear_nn_case.__name__,
    linear_nn_jit_case.__name__,
    vmapped_avg_of_vmapped_case.__name__,
    scalar_out_vmapped_avg_of_vmapped_case.__name__,
)


def isclose(a, b):
    return jnp.isclose(a, b, atol=1e-4)


class TestIBP(BoundsTest):
    def compute_bounds(self, module_case: ModuleTestCase, **kwargs):
        return ibp(module_case.module)

    @pytest.mark.parametrize("case", ibp_test_once_cases)
    def test_bounds_once(self, case, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(case, request)
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | isclose(out_ub, out_concrete))

    @pytest.mark.parametrize("case", ibp_test_multiple_cases)
    def test_bounds_repeat5(self, case, argument_seeds5, request):
        out_concrete, out_lb, out_ub = self.get_bounds_case_fixture(
            case, request, argument_seeds5
        )
        assert jnp.all(out_lb <= out_ub)
        assert jnp.all((out_lb <= out_concrete) | isclose(out_lb, out_concrete))
        assert jnp.all((out_ub >= out_concrete) | isclose(out_ub, out_concrete))

    def test_intermediate_bounds(self):
        # shallow NN: 2 -> 3 -> 1
        w1 = jnp.array([[0.5, 0.5], [0.2, -0.3], [-0.1, -0.8]])
        b1 = jnp.array([[0.1, 0.0, -0.4]])

        w2 = jnp.array([[0.2, -0.1, 0.6]])
        b2 = jnp.array([[1.3]])

        example_input = jnp.zeros((1, 2))
        jaxpr = jax.make_jaxpr(shallow_nn)(example_input, w1, b1, w2, b2)

        # make sure the Jaxpr has the expected structure
        assert jaxpr.eqns[0].primitive == jax.lax.transpose_p
        assert jaxpr.eqns[1].primitive == jax.lax.dot_general_p
        assert jaxpr.eqns[2].primitive == jax.lax.add_p
        # ReLU:
        assert jaxpr.eqns[3].primitive == jax.custom_derivatives.custom_jvp_call_p
        assert jaxpr.eqns[4].primitive == jax.lax.transpose_p
        assert jaxpr.eqns[5].primitive == jax.lax.dot_general_p
        assert jaxpr.eqns[6].primitive == jax.lax.add_p

        in_bounds = Box(
            lower_bound=jnp.array([[-1.0, -1.0]]), upper_bound=jnp.array([[1.0, 1.0]])
        )
        var_bounds = ibp_jaxpr(jaxpr, intermediate_bounds=True)(
            in_bounds, w1, b1, w2, b2
        )

        x_var, w1_var, b1_var, w2_var, b2_var = jaxpr.jaxpr.invars
        assert is_bounds(var_bounds[x_var])
        assert jnp.allclose(var_bounds[x_var].lower_bound, in_bounds.lower_bound)
        assert jnp.allclose(var_bounds[x_var].upper_bound, in_bounds.upper_bound)
        assert var_bounds[w1_var] is w1
        assert var_bounds[b1_var] is b1
        assert var_bounds[w2_var] is w2
        assert var_bounds[b2_var] is b2

        w1_t_var = jaxpr.eqns[0].outvars[0]
        assert not is_bounds(var_bounds[w1_t_var])

        def check_var_bounds(var, true_lb, true_ub):
            assert is_bounds(var_bounds[var])
            actual_lb, actual_ub = var_bounds[var]
            assert jnp.allclose(actual_lb, true_lb)
            assert jnp.allclose(actual_ub, true_ub)

        first_linear_var = jaxpr.eqns[1].outvars[0]
        first_linear_true_lb = jnp.array([[-1.0, -0.5, -0.9]])
        first_linear_true_ub = jnp.array([[1.0, 0.5, 0.9]])
        check_var_bounds(first_linear_var, first_linear_true_lb, first_linear_true_ub)

        first_add_var = jaxpr.eqns[2].outvars[0]
        first_add_true_lb = jnp.array([[-0.9, -0.5, -1.3]])
        first_add_true_ub = jnp.array([[1.1, 0.5, 0.5]])
        check_var_bounds(first_add_var, first_add_true_lb, first_add_true_ub)

        relu_var = jaxpr.eqns[3].outvars[0]
        relu_true_lb = jnp.array([[0.0, 0.0, 0.0]])
        relu_true_ub = jnp.array([[1.1, 0.5, 0.5]])
        check_var_bounds(relu_var, relu_true_lb, relu_true_ub)

        w2_t_var = jaxpr.eqns[4].outvars[0]
        assert not is_bounds(var_bounds[w2_t_var])

        second_linear_var = jaxpr.eqns[5].outvars[0]
        second_linear_true_lb = jnp.array([[-0.05]])
        second_linear_true_ub = jnp.array([[0.2 * 1.1 + 0.6 * 0.5]])
        check_var_bounds(
            second_linear_var, second_linear_true_lb, second_linear_true_ub
        )

        second_add_var = jaxpr.eqns[6].outvars[0]
        second_add_true_lb = second_linear_true_lb + 1.3
        second_add_true_ub = second_linear_true_ub + 1.3
        check_var_bounds(second_add_var, second_add_true_lb, second_add_true_ub)

        out_var = jaxpr.jaxpr.outvars[0]
        check_var_bounds(out_var, second_add_true_lb, second_add_true_ub)

    @pytest.mark.parametrize(
        "case",
        [
            "simple_relu_case",
            "jax_relu_case",
        ],
    )
    def test_grad_once(self, case, request):
        self.assert_grad_bounds_margin_case_fixture(case, request)

    @pytest.mark.parametrize(
        "case",
        [
            "simple_relu_case",
            "jax_relu_case",
            "affine_layer_case2",
            "shallow_nn_case",
            "shallow_nn_jit_case",
            "shallow_vmapped_case",
            "shallow_vmapped_jit_case",
        ],
    )
    def test_grad_repeat5(self, case, argument_seeds5, request):
        self.assert_grad_bounds_margin_case_fixture(case, request, argument_seeds5)

    @pytest.mark.parametrize(
        "rng_seed", [pytest.param(f"smears-{i}", id=f"seed{i}") for i in range(5)]
    )
    @pytest.mark.parametrize(
        "case",
        ["shallow_nn_case"],
    )
    def test_smears(self, case, rng_seed, request):
        """
        Smears use IBP bounds on the Jacobian to estimate the impact of different
        input dimensions on a function.
        Their use is in selecting input dimensions to split in branch and bound.
        """
        module_case = request_case_fixture(case, request, rng_seed)

        bounded_args = batch_bounds(module_case.bounded_args)
        vmap_axes = [0 if is_bounds(arg) else None for arg in bounded_args]
        jac = jax.vmap(jax.jacrev(module_case.module), vmap_axes)
        jac_concrete = jac(*module_case.concrete_args)
        jac_lb, jac_ub = ibp(jac)(*bounded_args)
        assert jnp.all(jac_lb <= jac_ub)
        assert jnp.all((jac_lb <= jac_concrete) | isclose(jac_lb, jac_concrete))
        assert jnp.all((jac_ub >= jac_concrete) | isclose(jac_ub, jac_concrete))

    def test_relu_smears(self):
        # simple_relu yields trivial bounds on the Jacobian
        smears_fun = ibp(jax.jacrev(nn.relu))

        smears = smears_fun(Box(jnp.array(-1.0), jnp.array(1.0)))
        assert smears == Box(jnp.array(0.0), jnp.array(1.0))

        smears = smears_fun(Box(jnp.array(1.0), jnp.array(2.0)))
        assert smears == Box(jnp.array(1.0), jnp.array(1.0))

        smears = smears_fun(Box(jnp.array(-2.0), jnp.array(-1.0)))
        assert smears == Box(jnp.array(0.0), jnp.array(0.0))

    @pytest.mark.parametrize(
        "rng_seed",
        [pytest.param(f"smears-affine-{i}", id=f"seed{i}") for i in range(5)],
    )
    @pytest.mark.parametrize(
        "case",
        ["affine_layer_case2"],
    )
    def test_affine_smears(self, case, rng_seed, request):
        module_case = request_case_fixture(case, request, rng_seed)
        affine_layer = module_case.module
        _, weight, bias = module_case.bounded_args
        in_bounds, _, _ = module_case.bounded_args

        smears_fun = ibp(jax.jacrev(affine_layer))

        smears = smears_fun(in_bounds, weight, bias)
        # the Jacobian is independent of the input, so it should not be
        # a Bounds object.
        assert not is_bounds(smears)
        assert jnp.allclose(smears, jnp.transpose(weight))
