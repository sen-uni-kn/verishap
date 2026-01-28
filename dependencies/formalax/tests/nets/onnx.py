#  Copyright (c) 2024. The Formalax Authors.
#  Licensed under the MIT license.
from typing import Callable

import onnx
import pytest
from jaxonnxruntime import backend as jax_onnx_backend


def load_onnx_model(model_path) -> Callable:
    onnx_model = onnx.load(model_path)
    model_in_jax = jax_onnx_backend.BackendRep(onnx_model)

    return lambda *args: model_in_jax.run(args)[0]


@pytest.fixture
def mnist_onnx_fully_connected(resource_dir) -> Callable:
    return load_onnx_model(resource_dir / "mnist_fully_connected.onnx")


@pytest.fixture
def mnist_onnx_conv(resource_dir) -> Callable:
    return load_onnx_model(resource_dir / "mnist_conv.onnx")
