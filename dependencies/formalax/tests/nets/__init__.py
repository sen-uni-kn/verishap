#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from .acasxu import acasxu_network
from .equinox import mnist_equinox_conv_batchnorm
from .flax import (
    emnist_flax_conv,
    emnist_flax_mlp,
    mnist_flax_mlp,
    mnist_ibp_training_flax_conv,
)
from .onnx import mnist_onnx_conv, mnist_onnx_fully_connected
from .simple import (
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

__all__ = [
    acasxu_network,
    emnist_flax_mlp,
    emnist_flax_conv,
    mnist_flax_mlp,
    mnist_ibp_training_flax_conv,
    mnist_onnx_conv,
    mnist_onnx_fully_connected,
    mnist_equinox_conv_batchnorm,
    affine_layer,
    conv_layer,
    shallow_nn,
    shallow_nn_jit,
    shallow_vmapped,
    shallow_vmapped_jit,
    simple_relu,
    tanh_two_layer_nn,
    linear_nn,
    linear_nn_jit,
    vmapped_avg_of_vmapped,
]
