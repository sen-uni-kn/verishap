#  Copyright (c) 2025. The Formalax Authors.
#  Licensed under the MIT license.
import os
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import requests


class ACASXuModel(eqx.Module):
    in_means: jnp.ndarray
    in_ranges: jnp.ndarray
    in_min: jnp.ndarray
    in_max: jnp.ndarray
    out_means: jnp.ndarray
    out_ranges: jnp.ndarray
    weights: list
    biases: list

    def __init__(
        self,
        in_means,
        in_ranges,
        in_min,
        in_max,
        out_means,
        out_ranges,
        weights,
        biases,
    ):
        self.in_means = in_means
        self.in_ranges = in_ranges
        self.in_min = in_min
        self.in_max = in_max
        self.out_means = out_means
        self.out_ranges = out_ranges
        self.weights = weights
        self.biases = biases

    def __call__(self, x):
        # clip and normalize the input
        x = jnp.clip(x, self.in_min, self.in_max)
        x = (x - self.in_means) / self.in_ranges

        for w, b in zip(self.weights[:-1], self.biases[:-1], strict=True):
            x = x @ w.T + b
            x = jax.nn.relu(x)
        y = x @ self.weights[-1].T + self.biases[-1]

        # de-normalize the output
        y = y * self.out_ranges + self.out_means
        return y


def load_nnet(
    path: os.PathLike,
) -> Callable:
    """
    Loads a ReLU activated neural network model from a `.nnet` file.

    :param path: The path to the `.nnet` file
    :return: The neural network and the input space bounds from the `.nnet` file
    """
    with open(path) as f:
        line = f.readline()
        count = 1
        while line[0:2] == "//":
            line = f.readline()
            count += 1
        # num_layers doesn't include the inputs module!
        num_layers, input_size, output_size, _ = [
            int(x) for x in line.strip().split(",")[:-1]
        ]
        line = f.readline()

        # inputs module size, layer1size, layer2size...
        layer_sizes = [int(x) for x in line.strip().split(",")[:-1]]

        # the next line contains a flag that is not used; ignore
        f.readline()
        # symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        input_minimums = [float(x) for x in line.strip().split(",") if x != ""]
        while len(input_minimums) < input_size:
            input_minimums.append(min(input_minimums))
        input_minimums = jnp.array(input_minimums)

        line = f.readline()
        input_maximums = [float(x) for x in line.strip().split(",") if x != ""]
        while len(input_maximums) < input_size:
            input_maximums.append(max(input_maximums))
        input_maximums = jnp.array(input_maximums)

        line = f.readline()
        means = [float(x) for x in line.strip().split(",")[:-1]]
        # if there are too little means given (we also need one for the output)
        # fill up with 0, which will cause no modifications in the data
        if len(means) < input_size + 1:
            means.append(0.0)
        means = jnp.array(means)

        line = f.readline()
        ranges = [float(x) for x in line.strip().split(",")[:-1]]
        # same as with means
        if len(ranges) < input_size + 1:
            ranges.append(1.0)
        ranges = jnp.array(ranges)

        weights = []
        biases = []

        for layer_idx in range(num_layers):
            current_layer_size = layer_sizes[layer_idx + 1]

            weight_matrix = []
            for _ in range(current_layer_size):
                line = f.readline()
                weight_matrix.append([float(x) for x in line.strip().split(",")[:-1]])
            weights.append(jnp.array(weight_matrix))

            bias_vector = []
            for _ in range(current_layer_size):
                line = f.readline()
                bias_vector.append(float(line.strip().split(",")[0]))
            biases.append(jnp.array(bias_vector))

    network = ACASXuModel(
        means[:-1],
        ranges[:-1],
        input_minimums,
        input_maximums,
        means[-1],
        ranges[-1],
        weights,
        biases,
    )
    return network


def acasxu_network(
    i1: int,
    i2: int,
    root=".models/acasxu",
    base_url="https://raw.githubusercontent.com/guykatzz/ReluplexCav2017/60b482eec832c891cb59c0966c9821e40051c082/nnet/",
):
    """
    Load an ACAS Xu network from the `root` directory.
    Download the network from the Reluplex GitHub repository if necessary.
    """
    i1, i2 = int(i1), int(i2)
    if i1 < 1 or i1 > 5:
        raise ValueError("The first network index must be between 1 and 5")
    if i2 < 1 or i2 > 9:
        raise ValueError("The second network index must be between 1 and 9")

    net_file = f"ACASXU_run2a_{i1}_{i2}_batch_2000.nnet"
    root_dir = Path(root)
    if not root_dir.exists():
        root_dir.mkdir(parents=True)
    target_file = root_dir / net_file
    if not target_file.exists():
        url = base_url + net_file
        print(f"Downloading ACAS Xu network {i1}, {i2} from {url}.")
        result = requests.get(url)
        if not result.ok:
            raise ValueError(
                f"Failed to download ACAS Xu network {i1}, {i2} from {url}."
            )

        root_dir.mkdir(exist_ok=True)
        target_file.touch()
        with open(target_file, "wb") as file:
            file.write(result.content)
    return load_nnet(target_file)
