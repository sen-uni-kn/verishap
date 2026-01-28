#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax
import jax.numpy as jnp
import pytest
from jax import lax

from formalax.core.batch_axes import infer_batch_axes

# * import necessary to import various test fixtures
from ..module_cases import *  # noqa: F403, F401
from ..module_cases import request_case_fixture


class TestBatchAxes:
    """Tests ``formalax.core.batch_axes``."""

    @pytest.mark.parametrize(
        "case",
        [
            "neg_case",
            "add_case",
            "sub_case",
            "mul_case",
            "simple_relu_case",
            "max_case",
            "jax_relu_case",
            "abs_case",
            "atan_case",
            "sigmoid_case",
            "tanh_case",
            "square_case",
        ],
    )
    def test_all_axes_batch_axes(self, case, request):
        module_case = request_case_fixture(case, request)
        jaxpr = jax.make_jaxpr(module_case.module)(*module_case.concrete_args)
        batch_axes = infer_batch_axes(jaxpr)

        for var in jaxpr.jaxpr.outvars:
            for i in range(len(var.aval.shape)):
                assert i in batch_axes[var]

    @pytest.mark.parametrize(
        "case",
        [
            "add_constant_right_case",
            "add_constant_left_case",
            "sub_constant_case",
            "sub_from_constant_case",
            "mul_constant_right_case",
            "mul_constant_left_case",
            "div_by_constant_case",
        ],
    )
    def test_no_batch_axes(self, case, request):
        module_case = request_case_fixture(case, request)
        jaxpr = jax.make_jaxpr(module_case.module)(*module_case.concrete_args)
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=((),) * len(jaxpr.in_avals))

        print(jaxpr)
        for var in jaxpr.jaxpr.outvars:
            print(var, batch_axes[var])
            assert len(batch_axes[var]) == 0

    @pytest.mark.parametrize(
        ("in_shape", "in_batch_axes", "out_shape", "true_out_batch_axes"),
        [
            ((4, 8, 16), {0, 1, 2}, (4, 8, 16), {0, 1, 2}),
            ((4, 8, 16), {0, 1, 2}, (512,), set()),
            ((4, 8, 16), {0, 1, 2}, (16, 4, 8), set()),
            ((4, 8, 16), {0, 1, 2}, (4, 128), {0}),
            ((4, 8, 16), {0, 1}, (4, 128), {0}),
            ((4, 8, 16), {0, 2}, (4, 128), {0}),
            ((4, 8, 16), {1, 2}, (4, 128), set()),
            ((4, 8, 16), {1}, (4, 128), set()),
            ((4, 8, 16), {2}, (32, 16), {1}),
            ((4, 8, 16), {0, 2}, (32, 16), {1}),
            ((4, 8, 16), {0, 1}, (32, 16), set()),
            ((4, 8, 16), {0}, (32, 16), set()),
            ((4, 8, 16), set(), (1, 512, 1), {0, 2}),
            ((4, 8, 16), {1}, (8, 1, 64), {1}),
            ((4, 8, 16), {1, 2}, (16, 4, 8), set()),
        ],
    )
    def test_reshape(self, in_shape, out_shape, in_batch_axes, true_out_batch_axes):
        jaxpr = jax.make_jaxpr(lambda x: jnp.reshape(x, out_shape))(jnp.zeros(in_shape))
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes,))
        out_batch_axes = batch_axes[jaxpr.jaxpr.outvars[0]]
        assert len(set(out_batch_axes)) == len(out_batch_axes)
        assert set(out_batch_axes) == true_out_batch_axes

    @pytest.mark.parametrize("in_batch_axes", [{0, 1}, {0}, {1}])
    def test_broadcast_add(self, in_batch_axes):
        x1 = jnp.zeros((8, 16))
        x2 = jnp.zeros(())
        jaxpr = jax.make_jaxpr(lambda x, y: x + y)(x1, x2)
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes, ...))

        x2_var, out_var = jaxpr.jaxpr.invars[1], jaxpr.jaxpr.outvars[0]
        x2_out_mapping = batch_axes.mapping(x2_var, out_var)
        assert len(x2_out_mapping.in_axes_set) == 0
        assert len(x2_out_mapping.broadcast_axes) == len(in_batch_axes)
        assert x2_out_mapping.out_axes_set == in_batch_axes

    def test_broadcast_in_dim(self):
        x = jnp.zeros((4, 1, 2, 1))
        shape = (4, 3, 3, 2, 3)
        dims = (0, 1, 3, 4)
        jaxpr = jax.make_jaxpr(lambda x: lax.broadcast_in_dim(x, shape, dims))(x)
        batch_axes = infer_batch_axes(jaxpr)

        x_var, out_var = jaxpr.jaxpr.invars[0], jaxpr.jaxpr.outvars[0]
        mapping = batch_axes.mapping(x_var, out_var)
        assert mapping.broadcast_axes == {1, 2, 4}
        assert mapping.source(0) == 0
        assert mapping.source(1) == 1
        assert mapping.source(2) is None
        assert mapping.source(3) == 2
        assert mapping.source(4) == 3

    @pytest.mark.parametrize(
        "case",
        [
            "flax_max_pool_case1",
            "flax_max_pool_case2",
            "flax_max_pool_case3",
            "mnist_onnx_fully_connected_case",
            "mnist_onnx_conv_case",
            "mnist_ibp_training_flax_conv_case",
            "emnist_flax_conv_case",
            "mnist_flax_mlp_case",
            "affine_layer_case1",
            "affine_layer_case2",
            "conv_layer_case",
            "shallow_nn_case",
            "shallow_nn_jit_case",
            "shallow_vmapped_case",
            "shallow_vmapped_jit_case",
            "tanh_two_layer_nn_case",
        ],
    )
    def test_leading_batch_axis(self, case, request):
        module_case = request_case_fixture(case, request)
        jaxpr = jax.make_jaxpr(module_case.module)(*module_case.concrete_args)
        batch_axes = infer_batch_axes(jaxpr)

        assert 0 in batch_axes[jaxpr.jaxpr.invars[0]]
        assert 0 in batch_axes[jaxpr.jaxpr.outvars[0]]

    @pytest.mark.parametrize(
        "case",
        [
            "affine_layer_case1",
            "affine_layer_case2",
            "conv_layer_case",
            "shallow_nn_case",
            "shallow_nn_jit_case",
            "shallow_vmapped_case",
            "shallow_vmapped_jit_case",
            "mnist_onnx_conv_case",
            "mnist_ibp_training_flax_conv_case",
            "emnist_flax_conv_case",
        ],
    )
    def test_only_leading_batch_axis(self, case, request):
        module_case = request_case_fixture(case, request)
        jaxpr = jax.make_jaxpr(module_case.module)(*module_case.concrete_args)
        batch_axes = infer_batch_axes(jaxpr)

        input_var = jaxpr.jaxpr.invars[0]
        print(jaxpr)
        assert (0,) == batch_axes[input_var]

    @pytest.mark.parametrize(
        ("in_shape", "in_batch_axes", "out_shape", "dimensions", "true_out_batch_axes"),
        [
            ((4, 8, 16), {0, 1, 2}, (16, 4, 8), (0, 1, 2), set()),
            ((4, 8, 16), {0, 1, 2}, (16, 4, 8), (2, 0, 1), {0, 1, 2}),
        ],
    )
    def test_reshape_with_dimensions(
        self, in_shape, in_batch_axes, out_shape, dimensions, true_out_batch_axes
    ):
        jaxpr = jax.make_jaxpr(
            lambda x: lax.reshape(x, out_shape, dimensions=dimensions)
        )(jnp.zeros(in_shape))
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes,))
        out_batch_axes = batch_axes[jaxpr.jaxpr.outvars[0]]
        assert len(set(out_batch_axes)) == len(out_batch_axes)
        assert set(out_batch_axes) == true_out_batch_axes

    @pytest.mark.parametrize(
        ("in_shape", "in_batch_axes", "pad_config", "true_out_batch_axes"),
        [
            ((4, 8), {0, 1}, ((1, 1, 0), (1, 1, 0)), {0, 1}),
            ((4, 8), {0}, ((1, 1, 0), (1, 1, 0)), {0}),
            ((4, 8), set(), ((1, 1, 0), (1, 1, 0)), set()),
        ],
    )
    def test_pad(self, in_shape, in_batch_axes, pad_config, true_out_batch_axes):
        jaxpr = jax.make_jaxpr(lambda x: lax.pad(x, 0.0, pad_config))(
            jnp.zeros(in_shape)
        )
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes,))
        out_batch_axes = batch_axes[jaxpr.jaxpr.outvars[0]]
        assert set(out_batch_axes) == true_out_batch_axes

    @pytest.mark.parametrize(
        ("in_shape", "in_batch_axes", "dimensions", "true_out_batch_axes"),
        [
            ((4, 8, 16), {0, 1, 2}, (1,), {0, 1}),
            ((4, 8, 16), {0, 1}, (1,), {0}),
        ],
    )
    def test_reduce(self, in_shape, in_batch_axes, dimensions, true_out_batch_axes):
        jaxpr = jax.make_jaxpr(
            lambda x: lax.reduce(x, jnp.zeros(()), lax.add, dimensions=dimensions)
        )(jnp.zeros(in_shape))
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes, set()))
        out_batch_axes = batch_axes[jaxpr.jaxpr.outvars[0]]
        assert set(out_batch_axes) == true_out_batch_axes

    @pytest.mark.parametrize(
        ("op", "dtype"),
        [
            (lax.reduce_sum, jnp.float32),
            (lax.reduce_prod, jnp.float32),
            (lax.reduce_min, jnp.int32),
            (lax.reduce_max, jnp.int32),
            (lax.reduce_or, jnp.int32),
            (lax.reduce_and, jnp.int32),
            (lax.reduce_xor, jnp.int32),
        ],
    )
    @pytest.mark.parametrize(
        ("in_shape", "in_batch_axes", "axes", "true_out_batch_axes"),
        [
            ((4, 8, 16), {0, 1, 2}, (1,), {0, 1}),
            ((4, 8, 16), {0, 1}, (1,), {0}),
        ],
    )
    def test_reduce_op(
        self, op, dtype, in_shape, in_batch_axes, axes, true_out_batch_axes
    ):
        jaxpr = jax.make_jaxpr(lambda x: op(x, axes=axes))(
            jnp.zeros(in_shape, dtype=dtype)
        )
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes, set()))
        out_batch_axes = batch_axes[jaxpr.jaxpr.outvars[0]]
        assert set(out_batch_axes) == true_out_batch_axes

    @pytest.mark.parametrize(
        ("in_shape", "in_batch_axes", "dimensions", "true_out_batch_axes"),
        [
            ((4, 1, 16), {0, 1, 2}, (1,), {0, 1}),
            ((4, 1, 16), {0, 1}, (1,), {0}),
        ],
    )
    def test_squeeze(self, in_shape, in_batch_axes, dimensions, true_out_batch_axes):
        jaxpr = jax.make_jaxpr(lambda x: lax.squeeze(x, dimensions=dimensions))(
            jnp.zeros(in_shape)
        )
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes, set()))
        out_batch_axes = batch_axes[jaxpr.jaxpr.outvars[0]]
        assert set(out_batch_axes) == true_out_batch_axes

    @pytest.mark.parametrize(
        ("in_shape", "in_batch_axes", "axis", "sizes", "true_out_batch_axes"),
        [
            ((4, 8, 16), {0, 1}, 1, (2, 6), {0}),
            ((4, 8, 16), {0, 1}, 2, (2, 14), {0, 1}),
            ((4, 8, 16), {0, 2}, 1, (1,) * 8, {0, 2}),
        ],
    )
    def test_split(self, in_shape, in_batch_axes, axis, sizes, true_out_batch_axes):
        jaxpr = jax.make_jaxpr(lambda x: lax.split(x, sizes, axis=axis))(
            jnp.zeros(in_shape)
        )
        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes,))
        out_batch_axes = batch_axes[jaxpr.jaxpr.outvars[0]]

        assert set(out_batch_axes) == true_out_batch_axes

    @pytest.mark.parametrize(
        (
            "in_shape",
            "in_batch_axes",
            "start_indices",
            "limit_indices",
            "true_out_batch_axes",
        ),
        [
            ((4, 8, 16), {0, 1}, (0, 0, 0), (4, 8, 16), {0, 1}),
            ((4, 8, 16), {0, 1, 2}, (0, 2, 0), (4, 6, 16), {0, 2}),
            ((4, 8, 16), {1}, (0, 0, 0), (4, 8, 16), {1}),
            ((4, 8, 16), {2}, (0, 0, 4), (4, 8, 12), set()),
            ((4, 8, 16), {0, 1}, (0, 0, 0), (4, 8, 8), {0, 1}),
        ],
    )
    def test_slice(
        self, in_shape, in_batch_axes, start_indices, limit_indices, true_out_batch_axes
    ):
        jaxpr = jax.make_jaxpr(
            lambda x: lax.slice(
                x, start_indices=start_indices, limit_indices=limit_indices
            )
        )(jnp.zeros(in_shape))

        batch_axes = infer_batch_axes(jaxpr, in_batch_axes=(in_batch_axes,))
        out_batch_axes = batch_axes[jaxpr.jaxpr.outvars[0]]

        assert len(set(out_batch_axes)) == len(out_batch_axes)
        assert set(out_batch_axes) == true_out_batch_axes
