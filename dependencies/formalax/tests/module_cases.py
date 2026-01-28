#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from dataclasses import dataclass
from functools import partial
from typing import Callable, Generator, Literal

import jax
import jax.numpy as jnp
import pytest
from flax import linen
from jax import lax, nn

from formalax import Bounds, Box
from formalax.utils.zip import strict_zip

from .nets import (
    affine_layer,
    conv_layer,
    linear_nn,
    linear_nn_jit,
    shallow_nn,
    shallow_nn_jit,
    shallow_vmapped,
    shallow_vmapped_jit,
    simple_relu,
    tanh_two_layer_nn,
    vmapped_avg_of_vmapped,
)
from .random_utils import rand_interval, rand_uniform, rng_key_iter
from .utils import batch_bounds

# ==============================================================================
# Module Case Fixtures
# ==============================================================================
#
# Module case fixtures bundle a callable module, such as a neural network, with
# arguments for calling the module and bounds for testing bounds computations.
# Each module case fixture is a factory fixture that yields a function that
# returns a module and arguments for calling the module.
# The module and its arguments are bundled as a ModuleTestCase.
# Using factory fixtures allows for parameterizing the fixture, for example,
# with different random seeds or different batch sizes.


@dataclass
class ModuleTestCase:
    """A module case for a bounds computation method."""

    module: Callable
    concrete_args: list[jax.Array]
    bounded_args: list[jax.Array | Bounds]
    in_batch_axes: int | None | tuple[int | None, ...] = ...
    out_batch_axes: int | None = ...

    @property
    def batch_axes(self) -> dict[str, int | None]:
        val = {}
        if self.in_batch_axes is not ...:
            val["in_batch_axes"] = self.in_batch_axes
        if self.out_batch_axes is not ...:
            val["out_batch_axes"] = self.out_batch_axes
        return val


class ModuleTestCaseFactory:
    """A factory for creating ModuleTestCases."""

    def __init__(self):
        self.module = None
        self.module_head = None
        self.args_generator = None
        self.in_batch_axes = ...
        self.out_batch_axes = ...
        self.default_rng_seed = "Aug9"
        self.default_batch_size = 100

    def with_module(self, module: Callable):
        self.module = module
        return self

    def with_module_head(self, module_head: Callable):
        """A "head" is a layer or module applied to the output of the main module."""
        self.module_head = module_head
        return self

    def with_batch_axes(self, in_axes=..., out_axes=...):
        if in_axes is not ...:
            self.in_batch_axes = in_axes
        if out_axes is not ...:
            self.out_batch_axes = out_axes
        return self

    def with_default_rng_seed(self, default_rng_seed):
        self.default_rng_seed = default_rng_seed
        return self

    def with_default_batch_size(self, default_batch_size):
        self.default_batch_size = default_batch_size
        return self

    def with_random_arguments(
        self,
        argument_shapes,
        bounded_arguments=None,
        batch_bounds=False,
        signs: Literal["pos", "neg", "any"] = "any",
    ):
        self.args_generator = (
            "random_args",
            argument_shapes,
            bounded_arguments,
            batch_bounds,
            signs,
        )
        return self

    def with_randomly_perturbed_images(
        self,
        image_shape: tuple[int, int, int],
    ):
        self.args_generator = ("randomly_perturbed_images", image_shape)
        return self

    def __call__(self, request, rng_seed=None, batch_size=None):
        assert self.module is not None
        assert self.args_generator is not None

        if rng_seed is None:
            rng_seed = self.default_rng_seed

        if batch_size is None:
            batch_size = self.default_batch_size

        module = self.module
        if isinstance(module, str):
            module = request.getfixturevalue(module)

        head = self.module_head
        if isinstance(head, str):
            head = request.getfixturevalue(head)

        if head is not None:
            main_module = module

            def eval_module_and_head(*args, **kwargs):
                return head(main_module(*args, **kwargs))

            module = eval_module_and_head

        match self.args_generator:
            case (
                "random_args",
                argument_shapes,
                bounded_arguments,
                batch_bounds,
                signs,
            ):
                concrete_args, bounded_args = self.random_arguments(
                    argument_shapes,
                    bounded_arguments,
                    batch_bounds,
                    signs,
                    rng_seed,
                )
            case ("randomly_perturbed_images", image_shape):
                concrete_args, bounded_args = self.randomly_perturbed_images(
                    batch_size, image_shape, rng_seed
                )
                concrete_args = [concrete_args]
                bounded_args = [bounded_args]
            case _:
                raise ValueError(f"Unknown args generator: {self._args_generator}")

        batch_axes_args = {}
        if self.in_batch_axes is not ...:
            batch_axes_args["in_batch_axes"] = self.in_batch_axes
        if self.out_batch_axes is not ...:
            batch_axes_args["out_batch_axes"] = self.out_batch_axes

        return ModuleTestCase(
            module,
            concrete_args,
            bounded_args,
            **batch_axes_args,
        )

    @staticmethod
    def random_arguments(
        argument_shapes,
        bounded_arguments=None,
        batch_bounds_=False,
        signs: Literal["pos", "neg", "any"] = "any",
        rng_seed="Aug9",
    ) -> tuple[list[jax.Array], list[jax.Array | Bounds]]:
        """Randomly generate bounds for parameters and parameter values."""
        if bounded_arguments is None:
            bounded_arguments = (True,) * len(argument_shapes)

        match signs:
            case "pos":
                lower = 1e-4
                upper = 10
            case "neg":
                lower = -10
                upper = -1e-4
            case "any":
                lower = -10
                upper = 10

        rng_keys = rng_key_iter(rng_seed)
        concrete_args = []
        bounded_args = []
        for has_bounds, shape in strict_zip(bounded_arguments, argument_shapes):
            if not has_bounds:
                val = rand_uniform(shape, next(rng_keys), lower, upper)
                concrete_args.append(val)
                bounded_args.append(val)
            else:
                arg_lb, arg_ub = rand_interval(
                    shape, next(rng_keys), lower, upper, sign=signs
                )
                vals = rand_uniform(
                    (100,) + shape, next(rng_keys), lower=arg_lb, upper=arg_ub
                )
                vals = jnp.vstack(
                    (vals, jnp.expand_dims(arg_lb, 0), jnp.expand_dims(arg_ub, 0))
                )
                concrete_args.append(vals)
                bounded_args.append(Box(arg_lb, arg_ub))
        if batch_bounds_:
            bounded_args = batch_bounds(bounded_args)
        return concrete_args, bounded_args

    @staticmethod
    def randomly_perturbed_images(
        batch_size: int,
        image_shape: tuple[int, int, int],
        rng_seed: str = "Jan1",
    ) -> tuple[jax.Array, Bounds[jax.Array]]:
        rng_keys = rng_key_iter(rng_seed)
        images = rand_uniform(
            (batch_size, *image_shape), next(rng_keys), lower=0, upper=1
        )
        eps = rand_uniform((batch_size, 1, 1, 1), next(rng_keys), lower=1e-8, upper=0.2)

        in_lb = jnp.clip(images - eps, 0.0, 1.0)
        in_ub = jnp.clip(images + eps, 0.0, 1.0)
        return images, Box(in_lb, in_ub)


def request_case_fixture(case_fixture, request, rng_seed=None, batch_size=None):
    fixture_factory = request.getfixturevalue(case_fixture)
    return fixture_factory(request, rng_seed, batch_size)


@pytest.fixture(
    scope="module",
    params=[pytest.param(f"bounds{i}", id=f"seed{i}") for i in range(3)],
)
def argument_seeds3(request):
    """A fixture leading to a test being repeated 3 times with different random seeds."""
    return request.param


@pytest.fixture(
    scope="module",
    params=[pytest.param(f"bounds{i}", id=f"seed{i}") for i in range(5)],
)
def argument_seeds5(request):
    """A fixture leading to a test being repeated 5 times with different random seeds."""
    return request.param


@pytest.fixture(
    scope="module",
    params=[pytest.param(f"bounds{i}", id=f"seed{i}") for i in range(10)],
)
def argument_seeds10(request):
    """A fixture leading to a test being repeated 10 times with different random seeds."""
    return request.param


# ==============================================================================
# Text Fixture Creating Utils
# ==============================================================================


def make_fixture(name, fun, *args, test_order="advanced", **kwargs):
    """Creates a new fixture calling fun with args and kwargs.

    The fixture is registered to the current module.
    If name is ``bar`` and the current module is ``foo``,
    the fixture can be accessed as ``foo.bar`` and imported
    using ``from foo import bar``.

    Args:
        name: Name of the fixture
        fun: Function to call when fixture is used
        *args: Arguments to pass to fun
        test_order: Test execution order priority. One of:
            - "basic": Run first (simple operations like neg, add, sub)
            - "advanced": Run second (complex operations like relu, maxpool)
            - "integration": Run last (complete networks)
        **kwargs: Keyword arguments to pass to fun
    """

    def fixture():
        return fun(*args, **kwargs)

    fixture.__name__ = name

    # Store test order information as an attribute
    fixture._test_order = test_order

    pytest_fixture = pytest.fixture(fixture)
    # Also store the test order on the pytest fixture
    pytest_fixture._test_order = test_order

    return pytest_fixture


def make_image_model_fixture(
    net_fixture_, image_shape
) -> Callable[[], ModuleTestCaseFactory]:
    def image_model_fixture() -> ModuleTestCaseFactory:
        yield (
            ModuleTestCaseFactory()
            .with_module(net_fixture_)
            .with_randomly_perturbed_images(image_shape)
            .with_default_batch_size(100)
            .with_default_rng_seed("mnist_onnx")
        )

    image_model_fixture.__name__ = net_fixture_ + "_case"
    return image_model_fixture


def sum_outputs(
    get_case_factory: Callable[[], ModuleTestCaseFactory], axis: int = -1
) -> ModuleTestCaseFactory:
    """Sums out the outputs of a module to create a scalar output."""

    def case_fn():
        return get_case_factory().with_module_head(partial(jnp.sum, axis=axis))

    return case_fn


def make_image_robustness_fixture(
    net_fixture_, image_shape, true_class=0
) -> Callable[[], ModuleTestCaseFactory]:
    def robustness_sat_fn(scores: jax.Array) -> jax.Array:
        return scores[:, true_class] - jnp.max(scores, axis=-1)

    def robustness_fixture() -> ModuleTestCaseFactory:
        case_factory = next(make_image_model_fixture(net_fixture_, image_shape)())
        return case_factory.with_module_head(robustness_sat_fn)

    robustness_fixture.__name__ = net_fixture_ + "_robustness_case"
    return robustness_fixture


# ==============================================================================
# Module Cases: Elementwise Linear
# ==============================================================================


def make_neg_case(in_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory().with_module(lax.neg).with_random_arguments([in_shape])
    )


neg_case = make_fixture("neg_case", make_neg_case, test_order="basic")
scalar_neg_case = make_fixture(
    "scalar_neg_case", make_neg_case, in_shape=(1,), test_order="basic"
)


def make_add_case(a_shape=(8, 8), b_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a + b)
        .with_random_arguments([a_shape, b_shape])
    )


add_case = make_fixture("add_case", make_add_case, test_order="basic")
scalar_add_case = make_fixture(
    "scalar_add_case", make_add_case, a_shape=(1,), b_shape=(1,), test_order="basic"
)


def make_broadcast_add_case(a_shape=(8, 8), b_shape=()) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a + b)
        .with_random_arguments(
            [a_shape, b_shape],
            bounded_arguments=[True, False],
        )
    )


broadcast_add_case = make_fixture(
    "broadcast_add_case", make_broadcast_add_case, test_order="basic"
)
broadcast_add_case2 = make_fixture(
    "broadcast_add_case2",
    make_broadcast_add_case,
    a_shape=(8, 8),
    b_shape=(1,),
    test_order="basic",
)
scalar_broadcast_add_case = make_fixture(
    "scalar_broadcast_add_case",
    make_broadcast_add_case,
    a_shape=(1,),
    b_shape=(),
    test_order="basic",
)


def make_add_constant_right_case(a_shape=(64,), b_shape=(64,)) -> ModuleTestCaseFactory:
    # since the second argument is fixed, the size 64 axis can not be a batch axis
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a + b)
        .with_random_arguments([a_shape, b_shape], [True, False])
        .with_batch_axes(in_axes=None)
    )


add_constant_right_case = make_fixture(
    "add_constant_right_case", make_add_constant_right_case, test_order="basic"
)
scalar_add_constant_right_case = make_fixture(
    "scalar_add_constant_right_case",
    make_add_constant_right_case,
    a_shape=(1,),
    b_shape=(1,),
    test_order="basic",
)


def make_add_constant_left_case(a_shape=(64,), b_shape=(64,)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a + b)
        .with_random_arguments([a_shape, b_shape], [False, True])
        .with_batch_axes(in_axes=None)
    )


add_constant_left_case = make_fixture(
    "add_constant_left_case", make_add_constant_left_case, test_order="basic"
)
scalar_add_constant_left_case = make_fixture(
    "scalar_add_constant_left_case",
    make_add_constant_left_case,
    a_shape=(1,),
    b_shape=(1,),
    test_order="basic",
)


def make_sub_case(a_shape=(8, 8), b_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a - b)
        .with_random_arguments([a_shape, b_shape])
    )


sub_case = make_fixture("sub_case", make_sub_case, test_order="basic")
scalar_sub_case = make_fixture(
    "scalar_sub_case", make_sub_case, a_shape=(1,), b_shape=(1,), test_order="basic"
)


def make_sub_constant_case(a_shape=(64,), b_shape=(64,)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a - b)
        .with_random_arguments([a_shape, b_shape], [True, False])
        .with_batch_axes(in_axes=None)
    )


sub_constant_case = make_fixture(
    "sub_constant_case", make_sub_constant_case, test_order="basic"
)
scalar_sub_constant_case = make_fixture(
    "scalar_sub_constant_case",
    make_sub_constant_case,
    a_shape=(1,),
    b_shape=(1,),
    test_order="basic",
)


def make_sub_from_constant_case(a_shape=(64,), b_shape=(64,)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a - b)
        .with_random_arguments([a_shape, b_shape], [False, True])
        .with_batch_axes(in_axes=None)
    )


sub_from_constant_case = make_fixture(
    "sub_from_constant_case", make_sub_from_constant_case, test_order="basic"
)
scalar_sub_from_constant_case = make_fixture(
    "scalar_sub_from_constant_case",
    make_sub_from_constant_case,
    a_shape=(1,),
    b_shape=(1,),
    test_order="basic",
)


# ==============================================================================
# Module Cases: Elementwise Mul/Div
# ==============================================================================


def make_mul_case(a_shape=(8, 8), b_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a * b)
        .with_random_arguments(
            [a_shape, b_shape],
        )
    )


mul_case = make_fixture("mul_case", make_mul_case, test_order="basic")
scalar_mul_case = make_fixture(
    "scalar_mul_case", make_mul_case, a_shape=(1,), b_shape=(1,), test_order="basic"
)


def make_mul_constant_right_case(a_shape=(64,), b_shape=(64,)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a * b)
        .with_random_arguments([a_shape, b_shape], [True, False])
        .with_batch_axes(in_axes=None)
    )


mul_constant_right_case = make_fixture(
    "mul_constant_right_case", make_mul_constant_right_case, test_order="basic"
)
scalar_mul_constant_right_case = make_fixture(
    "scalar_mul_constant_right_case",
    make_mul_constant_right_case,
    a_shape=(1,),
    b_shape=(1,),
    test_order="basic",
)


def make_mul_constant_left_case(a_shape=(64,), b_shape=(64,)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a * b)
        .with_random_arguments([a_shape, b_shape], [False, True])
        .with_batch_axes(in_axes=None)
    )


mul_constant_left_case = make_fixture(
    "mul_constant_left_case", make_mul_constant_left_case, test_order="basic"
)
scalar_mul_constant_left_case = make_fixture(
    "scalar_mul_constant_left_case",
    make_mul_constant_left_case,
    a_shape=(1,),
    b_shape=(1,),
    test_order="basic",
)

def make_div_case(a_shape=(8, 8), b_shape=(8, 8), signs="any") -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a / b)
        .with_random_arguments(
            [a_shape, b_shape], signs=signs
        )
    )


div_case = make_fixture(
    "div_case", make_div_case, test_order="basic"
)
scalar_div_case = make_fixture(
    "scalar_div_case",
    make_div_case,
    a_shape=(1,),
    b_shape=(1,),
    test_order="basic",
)
div_by_pos_case = make_fixture(
    "div_by_pos_case",
    make_div_case,
    signs="pos",
    test_order="basic",
)
div_by_neg_case = make_fixture(
    "div_by_neg_case",
    make_div_case,
    signs="neg",
    test_order="basic",
)


def make_div_by_constant_case(a_shape=(64,), b_shape=(64,)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a, b: a / b)
        .with_random_arguments([a_shape, b_shape], [True, False])
        .with_batch_axes(in_axes=None)
    )


div_by_constant_case = make_fixture(
    "div_by_constant_case", make_div_by_constant_case, test_order="basic"
)
scalar_div_by_constant_case = make_fixture(
    "scalar_div_by_constant_case",
    make_div_by_constant_case,
    a_shape=(1,),
    b_shape=(1,),
    test_order="basic",
)


# ==============================================================================
# Module Cases: Reshape/Transpose
# ==============================================================================


def make_reshape_case(in_shape=(16,)) -> ModuleTestCaseFactory:
    reshape = partial(jnp.reshape, shape=(-1, 4, 4))
    return (
        ModuleTestCaseFactory()
        .with_module(reshape)
        .with_random_arguments([in_shape])
        .with_batch_axes(in_axes=None)
    )


reshape_case = make_fixture("reshape_case", make_reshape_case, test_order="basic")
scalar_out_reshape_case = make_fixture(
    "scalar_out_reshape_case",
    sum_outputs(make_reshape_case, axis=(-2, -1)),
    test_order="basic",
)


def make_transpose_case(in_shape=(16, 8, 4)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(jnp.transpose)
        .with_random_arguments([in_shape], batch_bounds=True)
    )


transpose_case = make_fixture("transpose_case", make_transpose_case, test_order="basic")
scalar_out_transpose_case = make_fixture(
    "scalar_out_transpose_case",
    sum_outputs(make_transpose_case, axis=(0, 1, 2)),
    test_order="basic",
)


# ==============================================================================
# Module Cases: Elementwise Non-Linear
# ==============================================================================


def make_simple_relu_case(in_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(simple_relu)
        .with_random_arguments([in_shape])
    )


simple_relu_case = make_fixture(
    "simple_relu_case", make_simple_relu_case, test_order="basic"
)
scalar_simple_relu_case = make_fixture(
    "scalar_simple_relu_case",
    make_simple_relu_case,
    in_shape=(1,),
    test_order="basic",
)


def make_jax_relu_case(in_shape=(8, 8)) -> ModuleTestCaseFactory:
    """Jit'ed ReLU from JAX."""
    return (
        ModuleTestCaseFactory().with_module(nn.relu).with_random_arguments([in_shape])
    )


jax_relu_case = make_fixture("jax_relu_case", make_jax_relu_case, test_order="advanced")
scalar_jax_relu_case = make_fixture(
    "scalar_jax_relu_case", make_jax_relu_case, in_shape=(1,), test_order="advanced"
)


def make_abs_case(in_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory().with_module(jnp.abs).with_random_arguments([in_shape])
    )


abs_case = make_fixture("abs_case", make_abs_case, test_order="basic")
scalar_abs_case = make_fixture(
    "scalar_abs_case", make_abs_case, in_shape=(1,), test_order="basic"
)


def make_atan_case(in_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory().with_module(jnp.atan).with_random_arguments([in_shape])
    )


atan_case = make_fixture("atan_case", make_atan_case, test_order="basic")
scalar_atan_case = make_fixture(
    "scalar_atan_case", make_atan_case, in_shape=(1,), test_order="basic"
)


def make_sigmoid_case(in_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(nn.sigmoid)
        .with_random_arguments([in_shape])
    )


sigmoid_case = make_fixture("sigmoid_case", make_sigmoid_case, test_order="basic")
scalar_sigmoid_case = make_fixture(
    "scalar_sigmoid_case", make_sigmoid_case, in_shape=(1,), test_order="basic"
)


def make_tanh_case(in_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory().with_module(nn.tanh).with_random_arguments([in_shape])
    )


tanh_case = make_fixture("tanh_case", make_tanh_case, test_order="basic")
scalar_tanh_case = make_fixture(
    "scalar_tanh_case", make_tanh_case, in_shape=(1,), test_order="basic"
)


def make_square_case(in_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(jnp.square)
        .with_random_arguments([in_shape])
    )


square_case = make_fixture("square_case", make_square_case, test_order="basic")
scalar_square_case = make_fixture(
    "scalar_square_case", make_square_case, in_shape=(1,), test_order="basic"
)


# ==============================================================================
# Module Cases: Elementwise Reductions
# ==============================================================================


def make_max_case(a_shape=(8, 8), b_shape=(8, 8)) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(jnp.maximum)
        .with_random_arguments([a_shape, b_shape])
    )


max_case = make_fixture("max_case", make_max_case, test_order="advanced")
scalar_max_case = make_fixture(
    "scalar_max_case", make_max_case, a_shape=(1,), b_shape=(1,), test_order="advanced"
)


# ==============================================================================
# Module Cases: Axis Reductions
# ==============================================================================


def make_reduce_sum_case(in_shape=(64,), axis=-1) -> ModuleTestCaseFactory:
    reduce_sum = partial(jnp.sum, axis=axis)
    return (
        ModuleTestCaseFactory()
        .with_module(reduce_sum)
        .with_random_arguments([in_shape])
    )


reduce_sum_case1 = make_fixture(
    "reduce_sum_case1", make_reduce_sum_case, test_order="advanced"
)
reduce_sum_case2 = make_fixture(
    "reduce_sum_case2",
    make_reduce_sum_case,
    in_shape=(8, 8),
    axis=-2,
    test_order="advanced",
)


def make_reduce_max_case(in_shape=(64,), axis=-1) -> ModuleTestCaseFactory:
    reduce_max = partial(jnp.max, axis=axis)
    return (
        ModuleTestCaseFactory()
        .with_module(reduce_max)
        .with_random_arguments([in_shape])
    )


reduce_max_case1 = make_fixture(
    "reduce_max_case1", make_reduce_max_case, test_order="advanced"
)
reduce_max_case2 = make_fixture(
    "reduce_max_case2",
    make_reduce_max_case,
    in_shape=(8, 8),
    axis=-2,
    test_order="advanced",
)


def make_reduce_min_case(in_shape=(64,), axis=-1) -> ModuleTestCaseFactory:
    reduce_min = partial(jnp.min, axis=axis)
    return (
        ModuleTestCaseFactory()
        .with_module(reduce_min)
        .with_random_arguments([in_shape])
    )


reduce_min_case1 = make_fixture(
    "reduce_min_case1", make_reduce_min_case, test_order="advanced"
)
reduce_min_case2 = make_fixture(
    "reduce_min_case2",
    make_reduce_min_case,
    in_shape=(8, 8),
    axis=-2,
    test_order="advanced",
)


# ==============================================================================
# Module Cases: Pooling
# ==============================================================================


def make_pooling_case(
    op: Callable, image_shape, window_shape, strides, padding, name: str
) -> Callable[[], ModuleTestCaseFactory]:
    pool = partial(op, window_shape=window_shape, strides=strides, padding=padding)

    def pool_fixture() -> Generator[ModuleTestCaseFactory, None, None]:
        yield (
            ModuleTestCaseFactory()
            .with_module(pool)
            .with_random_arguments([image_shape])
        )

    pool_fixture.__name__ = name
    return pool_fixture


flax_max_pool_case1 = pytest.fixture(
    make_pooling_case(
        linen.max_pool,
        (4, 10, 10, 2),
        (2, 2),
        (1, 1),
        "SAME",
        name="flax_max_pool_case1",
    )
)
flax_max_pool_case2 = pytest.fixture(
    make_pooling_case(
        linen.max_pool,
        (4, 28, 28, 1),
        (3, 3),
        (3, 3),
        "VALID",
        name="flax_max_pool_case2",
    )
)
flax_max_pool_case3 = pytest.fixture(
    make_pooling_case(
        linen.max_pool,
        (4, 32, 32, 3),
        (5, 5),
        (2, 4),
        "VALID",
        name="flax_max_pool_case3",
    )
)
scalar_out_flax_max_pool_case = pytest.fixture(
    make_pooling_case(
        linen.max_pool,
        (1, 2, 2, 1),
        (2, 2),
        (2, 2),
        "VALID",
        name="scalar_out_flax_max_pool_case",
    )
)
flax_min_pool_case1 = pytest.fixture(
    make_pooling_case(
        linen.pooling.min_pool,
        (4, 10, 10, 2),
        (2, 2),
        (1, 1),
        "SAME",
        name="flax_min_pool_case1",
    )
)
flax_min_pool_case2 = pytest.fixture(
    make_pooling_case(
        linen.pooling.min_pool,
        (4, 28, 28, 1),
        (3, 3),
        (3, 3),
        "VALID",
        name="flax_min_pool_case2",
    )
)
flax_min_pool_case3 = pytest.fixture(
    make_pooling_case(
        linen.pooling.min_pool,
        (4, 32, 32, 3),
        (5, 5),
        (2, 4),
        "VALID",
        name="flax_min_pool_case3",
    )
)
scalar_out_flax_min_pool_case = pytest.fixture(
    make_pooling_case(
        linen.pooling.min_pool,
        (1, 2, 2, 1),
        (2, 2),
        (2, 2),
        "VALID",
        name="scalar_out_flax_min_pool_case",
    )
)
flax_avg_pool_case1 = pytest.fixture(
    make_pooling_case(
        linen.pooling.avg_pool,
        (4, 10, 10, 2),
        (2, 2),
        (1, 1),
        "SAME",
        name="flax_avg_pool_case1",
    )
)
flax_avg_pool_case2 = pytest.fixture(
    make_pooling_case(
        linen.pooling.avg_pool,
        (4, 28, 28, 1),
        (3, 3),
        (3, 3),
        "VALID",
        name="flax_avg_pool_case2",
    )
)
flax_avg_pool_case3 = pytest.fixture(
    make_pooling_case(
        linen.pooling.avg_pool,
        (4, 32, 32, 3),
        (5, 5),
        (2, 4),
        "VALID",
        name="flax_avg_pool_case3",
    )
)
scalar_out_flax_avg_pool_case = pytest.fixture(
    make_pooling_case(
        linen.pooling.avg_pool,
        (1, 2, 2, 1),
        (2, 2),
        (2, 2),
        "VALID",
        name="scalar_out_flax_avg_pool_case",
    )
)


# ==============================================================================
# Module Cases: Affine Layers
# ==============================================================================


def make_affine_layer_case(
    arg_shapes=((3,), (3, 2), (2,)), has_bounds=(True, False, True)
) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(affine_layer)
        .with_random_arguments(arg_shapes, has_bounds)
        .with_batch_axes(in_axes=None)
    )


affine_layer_case1 = make_fixture("affine_layer_case1", make_affine_layer_case)
affine_layer_case2 = make_fixture(
    "affine_layer_case2",
    make_affine_layer_case,
    arg_shapes=[(10,), (10, 32), (32,)],
    has_bounds=[True, False, True],
)
scalar_out_affine_layer_case = make_fixture(
    "scalar_out_affine_layer_case",
    sum_outputs(make_affine_layer_case, axis=-1),
)


def make_conv_layer_case(
    arg_shapes=((2, 8, 8), (5, 2, 3, 3), (5, 6, 6)), has_bounds=(True, False, False)
) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(conv_layer)
        .with_random_arguments(arg_shapes, has_bounds, batch_bounds=True)
    )


conv_layer_case = make_fixture("conv_layer_case", make_conv_layer_case)
scalar_out_conv_layer_case = make_fixture(
    "scalar_out_conv_layer_case", sum_outputs(make_conv_layer_case, axis=(-3, -2, -1))
)


# ==============================================================================
# Module Cases: Random Parameter Neural Networks
# ==============================================================================


def make_shallow_nn_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(shallow_nn)
        .with_random_arguments(
            [
                (8,),  # + batch input
                (20, 8),
                (20,),
                (5, 20),
                (5,),
            ],
            [True, False, False, False, False],
        )
    )


shallow_nn_case = make_fixture(
    "shallow_nn_case", make_shallow_nn_case, test_order="integration"
)
scalar_out_shallow_nn_case = make_fixture(
    "scalar_out_shallow_nn_case",
    sum_outputs(make_shallow_nn_case, axis=-1),
    test_order="integration",
)


def make_shallow_nn_jit_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(shallow_nn_jit)
        .with_random_arguments(
            [
                (8,),
                (20, 8),
                (20,),
                (5, 20),
                (5,),
            ],
            [True, False, False, False, False],
        )
    )


shallow_nn_jit_case = make_fixture(
    "shallow_nn_jit_case", make_shallow_nn_jit_case, test_order="integration"
)
scalar_out_shallow_nn_jit_case = make_fixture(
    "scalar_out_shallow_nn_jit_case",
    sum_outputs(make_shallow_nn_jit_case, axis=-1),
    test_order="integration",
)


def make_shallow_vmapped_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(shallow_vmapped)
        .with_random_arguments(
            [(20,), (128, 20), (128,), (10, 128), (10,)],
            [True, False, False, False, False],
            batch_bounds=True,
        )
    )


shallow_vmapped_case = make_fixture(
    "shallow_vmapped_case", make_shallow_vmapped_case, test_order="integration"
)
scalar_out_shallow_vmapped_case = make_fixture(
    "scalar_out_shallow_vmapped_case",
    sum_outputs(make_shallow_vmapped_case, axis=-1),
    test_order="integration",
)


def make_shallow_vmapped_jit_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(shallow_vmapped_jit)
        .with_random_arguments(
            [(20,), (128, 20), (128,), (10, 128), (10,)],
            [True, False, False, False, False],
            batch_bounds=True,
        )
    )


shallow_vmapped_jit_case = make_fixture(
    "shallow_vmapped_jit_case", make_shallow_vmapped_jit_case, test_order="integration"
)
scalar_out_shallow_vmapped_jit_case = make_fixture(
    "scalar_out_shallow_vmapped_jit_case",
    sum_outputs(make_shallow_vmapped_jit_case, axis=-1),
    test_order="integration",
)


def make_linear_nn_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(linear_nn)
        .with_random_arguments(
            [
                (8,),  # + batch input
                (20, 8),
                (20,),
                (5, 20),
                (5,),
            ],
            [True, False, False, False, False],
        )
    )


linear_nn_case = make_fixture(
    "linear_nn_case", make_linear_nn_case, test_order="integration"
)
scalar_out_linear_nn_case = make_fixture(
    "scalar_out_linear_nn_case",
    sum_outputs(make_linear_nn_case, axis=-1),
    test_order="integration",
)


def make_linear_nn_jit_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(linear_nn_jit)
        .with_random_arguments(
            [
                (8,),
                (20, 8),
                (20,),
                (5, 20),
                (5,),
            ],
            [True, False, False, False, False],
        )
    )


linear_nn_jit_case = make_fixture(
    "linear_nn_jit_case", make_linear_nn_jit_case, test_order="integration"
)
scalar_out_linear_nn_jit_case = make_fixture(
    "scalar_out_linear_nn_jit_case",
    sum_outputs(make_linear_nn_jit_case, axis=-1),
    test_order="integration",
)


def make_tanh_two_layer_nn_case(batch_size: int = 8) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(tanh_two_layer_nn)
        .with_random_arguments(
            [
                (batch_size, 32),
                (32, 256),
                (256,),
                (256, 128),
                (128,),
                (128, 10),
                (10,),
            ],
            [True, False, False, False, False, False, False],
        )
    )


tanh_two_layer_nn_case = make_fixture(
    "tanh_two_layer_nn_case", make_tanh_two_layer_nn_case, test_order="integration"
)
scalar_out_tanh_two_layer_nn_case = make_fixture(
    "scalar_out_tanh_two_layer_nn_case",
    sum_outputs(partial(make_tanh_two_layer_nn_case, batch_size=1), axis=-1),
    test_order="integration",
)


def make_vmapped_avg_of_vmapped_case(batch_size: int = 10) -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(vmapped_avg_of_vmapped)
        .with_random_arguments(
            [
                (
                    batch_size,
                    20,
                ),
                (128, 20),
                (128,),
                (1, 128),
                (1,),
            ],
            [True, False, False, False, False],
            batch_bounds=True,
        )
    )


vmapped_avg_of_vmapped_case = make_fixture(
    "vmapped_avg_of_vmapped_case",
    make_vmapped_avg_of_vmapped_case,
    test_order="integration",
)
scalar_out_vmapped_avg_of_vmapped_case = make_fixture(
    "scalar_out_vmapped_avg_of_vmapped_case",
    partial(make_vmapped_avg_of_vmapped_case, batch_size=1),
    test_order="integration",
)

# ==============================================================================
# Module Cases: Trained Neural Networks
# ==============================================================================


def make_acasxu_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module("acasxu_network")
        .with_random_arguments([(5,)])
    )


acasxu_case = make_fixture("acasxu_case", make_acasxu_case, test_order="integration")
scalar_out_acasxu_case = make_fixture(
    "scalar_out_acasxu_case",
    sum_outputs(make_acasxu_case, axis=-1),
    test_order="integration",
)


# the mnist_ networks are fixtures defined in tests/nets/
mnist_onnx_fully_connected_case = pytest.fixture(
    make_image_model_fixture("mnist_onnx_fully_connected", (1, 28, 28))
)
mnist_onnx_conv_case = pytest.fixture(
    make_image_model_fixture("mnist_onnx_conv", (1, 28, 28))
)
mnist_ibp_training_flax_conv_case = pytest.fixture(
    make_image_model_fixture("mnist_ibp_training_flax_conv", (28, 28, 1))
)
emnist_flax_conv_case = pytest.fixture(
    make_image_model_fixture("emnist_flax_conv", (28, 28, 1))
)
mnist_flax_mlp_case = pytest.fixture(
    make_image_model_fixture("mnist_flax_mlp", (28, 28, 1))
)
mnist_equinox_conv_batchnorm_case = pytest.fixture(
    make_image_model_fixture("mnist_equinox_conv_batchnorm", (1, 28, 28))
)

mnist_onnx_fully_connected_robustness_case = pytest.fixture(
    make_image_robustness_fixture("mnist_onnx_fully_connected", (1, 28, 28))
)
mnist_onnx_conv_robustness_case = pytest.fixture(
    make_image_robustness_fixture("mnist_onnx_conv", (1, 28, 28))
)
mnist_ibp_training_flax_conv_robustness_case = pytest.fixture(
    make_image_robustness_fixture("mnist_ibp_training_flax_conv", (28, 28, 1))
)
emnist_flax_conv_robustness_case = pytest.fixture(
    make_image_robustness_fixture("emnist_flax_conv", (28, 28, 1))
)
mnist_flax_mlp_robustness_case = pytest.fixture(
    make_image_robustness_fixture("mnist_flax_mlp", (28, 28, 1))
)
mnist_equinox_conv_batchnorm_robustness_case = pytest.fixture(
    make_image_robustness_fixture("mnist_equinox_conv_batchnorm", (1, 28, 28))
)


# ==============================================================================
# Module Cases: Elementwise Comparisons
# ==============================================================================


@pytest.fixture
def ge_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lax.ge)
        .with_random_arguments([(8, 8), (8, 8)])
    )


@pytest.fixture
def gt_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lax.gt)
        .with_random_arguments([(8, 8), (8, 8)])
    )


@pytest.fixture
def le_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lax.le)
        .with_random_arguments([(8, 8), (8, 8)])
    )


@pytest.fixture
def lt_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lax.lt)
        .with_random_arguments([(8, 8), (8, 8)])
    )


# ==============================================================================
# Module Cases: Integer Power Fixtures
# ==============================================================================


@pytest.fixture
def reciprocal_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda x: lax.integer_pow(x, -1))
        .with_random_arguments([(8, 8)])
    )


@pytest.fixture
def reciprocal_positive_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda x: lax.integer_pow(x, -1))
        .with_random_arguments([(8, 8)], signs="pos")
    )


@pytest.fixture
def reciprocal_negative_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda x: lax.integer_pow(x, -1))
        .with_random_arguments([(8, 8)], signs="neg")
    )


@pytest.fixture
def int_pow_even_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda x: lax.integer_pow(x, 4))
        .with_random_arguments([(8, 8)])
    )


@pytest.fixture
def int_pow_odd_positive_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda x: lax.integer_pow(x, 3))
        .with_random_arguments([(8, 8)], signs="pos")
    )


@pytest.fixture
def int_pow_odd_negative_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda x: lax.integer_pow(x, 7))
        .with_random_arguments([(8, 8)], signs="pos")
    )


# ==============================================================================
# Module Cases: Special Primitives Fixtures
# ==============================================================================


@pytest.fixture
def squeeze_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda x: jnp.squeeze(x))
        .with_random_arguments([(8, 1, 8)])
    )


@pytest.fixture
def convert_element_type_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda x: lax.convert_element_type(x, jnp.int32))
        .with_random_arguments([(8, 8)])
    )


@pytest.fixture
def min_array_scalar_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda a: jnp.minimum(a, 0.5))
        .with_random_arguments([(8, 8)])
    )


@pytest.fixture
def select_n_case() -> ModuleTestCaseFactory:
    return (
        ModuleTestCaseFactory()
        .with_module(lambda i, a, b, c: lax.select_n(i, [a, b, c]))
        .with_random_arguments([(1,), (8, 8), (8, 8), (8, 8)])
    )
