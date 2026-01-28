#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
import jax.numpy as jnp
import pytest
from jax import lax

from formalax.bounds._src._lirpabounds import LiRPAWeights

from ..random_utils import rand_uniform, rng_key_iter


class TestIdentityWeights:
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.int32, jnp.uint8])
    @pytest.mark.parametrize("shape", [(10,), (2, 2), (5, 7, 3), (3, 2, 4, 1, 2)])
    def test_identity_weights_for(self, shape, dtype):
        rng_keys = rng_key_iter("identity_as")
        array = rand_uniform(shape, next(rng_keys))
        array = jnp.asarray(array, dtype=dtype)

        lb_weight, ub_weight = LiRPAWeights.identity_for(array, batch_axes=())
        assert jnp.allclose(lb_weight, ub_weight)
        identity = lb_weight

        assert identity.shape == (*shape, *shape)
        assert identity.dtype == array.dtype

        times_identity = jnp.tensordot(identity, array, axes=len(shape))
        assert jnp.allclose(times_identity, array)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.int32, jnp.uint8])
    @pytest.mark.parametrize("shape", [(10,), (2, 2), (3, 2, 4, 1, 2)])
    @pytest.mark.parametrize("batch_axis", [0, -1])
    def test_batched_identity_weights_for(self, shape, batch_axis, dtype):
        rng_keys = rng_key_iter("identity_as")
        array = rand_uniform(shape, next(rng_keys))
        array = jnp.asarray(array, dtype=dtype)

        lb_weight, ub_weight = LiRPAWeights.identity_for(
            array, batch_axes=(batch_axis,)
        )
        assert jnp.allclose(lb_weight, ub_weight)
        identity = lb_weight

        batch_axis = len(shape) + batch_axis if batch_axis < 0 else batch_axis
        batch_size = array.shape[batch_axis]
        data_shape = array.shape[:batch_axis] + array.shape[batch_axis + 1 :]

        assert identity.shape == (batch_size,) + data_shape + data_shape
        assert identity.dtype == array.dtype

        times_identity = lax.dot_general(
            identity,
            array,
            dimension_numbers=(
                (
                    tuple(len(shape) + i for i in range(len(shape) - 1)),
                    tuple(i for i in range(len(array.shape)) if i != batch_axis),
                ),
                ((0,), (batch_axis,)),
            ),
        )
        # dot general always places the batch axis up front
        times_identity = jnp.moveaxis(times_identity, 0, batch_axis)
        assert jnp.allclose(times_identity, array)
