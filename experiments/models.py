# Copyright 2025 David Boetius
import io
from collections.abc import Sequence
from functools import partial
from typing import Literal, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
from equinox import nn
from jaxtyping import Array, Float, PyTree


class ZScoreNorm(eqx.Module):
    mean: Float[Array, " n"]
    std: Float[Array, " n"]

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        return (x - self.mean) / self.std


class ZScoreUnnorm(eqx.Module):
    mean: Float[Array, " n"]
    std: Float[Array, " n"]

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        return x * self.std + self.mean


class MLP(eqx.Module):
    """A Multi-Layer Perceptron model for classifying tabular data."""

    input_norm: ZScoreNorm
    output_norm: ZScoreUnnorm
    layers: list
    input_dim: int
    output_dim: int
    hidden_dim: int
    hidden_layers: int

    def __init__(
        self,
        input_dim,
        output_dim,
        key,
        hidden_dim=32,
        hidden_layers=2,
        input_norm_stats: tuple[Float[Array, " n"], Float[Array, " n"]] | None = None,
        output_norm_stats: tuple[Float[Array, " m"], Float[Array, " m"]] | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        if input_norm_stats is None:
            input_mean, input_std = jnp.zeros(input_dim), jnp.ones(input_dim)
        else:
            input_mean, input_std = input_norm_stats
        self.input_norm = ZScoreNorm(input_mean, input_std)

        if output_norm_stats is None:
            if output_dim == 1:
                output_mean, output_std = jnp.zeros(()), jnp.ones(())
            else:
                output_mean, output_std = jnp.zeros(output_dim), jnp.ones(output_dim)
        else:
            output_mean, output_std = output_norm_stats
        self.output_norm = ZScoreUnnorm(output_mean, output_std)

        keys = jax.random.split(key, hidden_layers + 1)
        layer_sizes = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(
                eqx.nn.Linear(layer_sizes[i - 1], layer_sizes[i], key=keys[i])
            )
            layers.append(jax.nn.relu)
        self.layers = layers[:-1]  # remove training ReLU layer

    def __call__(self, x: Float[Array, " n"]) -> Float[Array, " m"]:
        x = self.input_norm(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_norm(x)

    def save(self, file: str, extra_info: dict | None = None):
        if extra_info is None:
            extra_info = {}
        # Input and output norm are saved as arrays by equinox.
        info_dict = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
        } | extra_info
        with open(file, "wb") as f:
            yaml_str = io.StringIO()
            yaml.YAML().dump(info_dict, yaml_str)
            f.write(yaml_str.getvalue().encode("utf-8"))
            f.write(b"\n---\n")
            eqx.tree_serialise_leaves(f, self)

    @classmethod
    def load(cls, file: str):
        with open(file, "rb") as f:
            yaml_lines = []
            while True:
                line = f.readline().decode("utf-8")
                if line.startswith("---"):
                    break
                yaml_lines.append(line)
            info_dict = yaml.YAML().load(io.StringIO("\n".join(yaml_lines)))
            model = cls(
                info_dict["input_dim"],
                info_dict["output_dim"],
                jax.random.PRNGKey(0),
                info_dict["hidden_dim"],
                info_dict["hidden_layers"],
                input_norm_stats=None,
                output_norm_stats=None,
            )
            model = eqx.tree_deserialise_leaves(f, model)
            return model


# ==============================================================================
# Vision Models
# ==============================================================================


CONV_LAYER_DEFINITION = dict[
    Literal[
        "channels",
        "conv_kernel_size",
        "conv_stride",
        "conv_padding",
        "pool_kernel_size",
        "pool_stride",
        "pool_padding",
    ],
    int,
]
CONV_LAYER_DEFAULTS = {
    "channels": 4,
    "conv_kernel_size": 5,
    "conv_stride": 1,
    "conv_padding": 2,
    "pool_kernel_size": 2,
    "pool_stride": 2,
    "pool_padding": 0,
}


class CNN(eqx.Module):
    """A simple CNN model for classification.

    Adapted from https://docs.kidger.site/equinox/examples/mnist/
    """

    layers: list
    input_shape: tuple[int, ...]
    output_dim: int

    def __init__(
        self,
        input_shape,
        output_dim,
        key,
        conv_layers: Sequence[CONV_LAYER_DEFINITION] = (
            {"channels": 4},
            {"channels": 8},
        ),
        fc_in_sizes: Sequence[int] = (512, 64),
    ):
        self.input_shape = tuple(input_shape)
        self.output_dim = output_dim

        conv_keys, fc_keys = jax.random.split(key, 2)
        conv_keys = jax.random.split(conv_keys, len(conv_layers))
        layers = [partial(jnp.reshape, shape=input_shape)]
        in_channels = input_shape[-3]
        for defn, key in zip(conv_layers, conv_keys, strict=True):
            out_channels = defn.get("channels", CONV_LAYER_DEFAULTS["channels"])
            conv = eqx.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=defn.get(
                    "conv_kernel_size", CONV_LAYER_DEFAULTS["conv_kernel_size"]
                ),
                stride=defn.get("conv_stride", CONV_LAYER_DEFAULTS["conv_stride"]),
                padding=defn.get("conv_padding", CONV_LAYER_DEFAULTS["conv_padding"]),
                use_bias=False,
                key=key,
            )
            pool = eqx.nn.AvgPool2d(
                kernel_size=defn.get(
                    "pool_kernel_size", CONV_LAYER_DEFAULTS["pool_kernel_size"]
                ),
                stride=defn.get("pool_stride", CONV_LAYER_DEFAULTS["pool_stride"]),
                padding=defn.get("pool_padding", CONV_LAYER_DEFAULTS["pool_padding"]),
            )
            norm = eqx.nn.BatchNorm(out_channels, axis_name="batch", mode="batch")
            layers += [conv, pool, norm, jax.nn.relu]
            in_channels = out_channels

        fc_keys = jax.random.split(fc_keys, len(fc_in_sizes))
        layers.append(jnp.ravel)
        fc_out_sizes = tuple(fc_in_sizes[1:]) + (output_dim,)
        for i in range(len(fc_in_sizes)):
            layers += [
                eqx.nn.Linear(fc_in_sizes[i], fc_out_sizes[i], key=fc_keys[i]),
                jax.nn.relu,
            ]
        layers = layers[:-1]  # remove last ReLU layer
        self.layers = tuple(layers)

    def __call__(
        self, x: Float[Array, " c h w"] | Float[Array, " h w"], state: PyTree
    ) -> tuple[Float[Array, ""], PyTree]:
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)
        if self.output_dim == 1:
            x = x.squeeze(axis=-1)
        return x, state

    @classmethod
    def save(
        cls, model: "CNN", state: PyTree, file: str, extra_info: dict | None = None
    ):
        conv_layers = []
        fc_in_sizes = []
        prev_conv = {}
        for layer in model.layers:
            if isinstance(layer, eqx.nn.Conv2d):
                prev_conv = {
                    "channels": layer.out_channels,
                    "conv_kernel_size": layer.kernel_size,
                    "conv_stride": layer.stride,
                    "conv_padding": layer.padding,
                }
            elif isinstance(layer, eqx.nn.AvgPool2d):
                assert len(prev_conv) > 0
                prev_conv["pool_kernel_size"] = layer.kernel_size
                prev_conv["pool_stride"] = layer.stride
                prev_conv["pool_padding"] = layer.padding
                conv_layers.append(prev_conv)
                prev_conv = {}
            elif isinstance(layer, eqx.nn.Linear):
                fc_in_sizes.append(layer.in_features)

        if extra_info is None:
            extra_info = {}
        info_dict = {
            "input_shape": model.input_shape,
            "output_dim": model.output_dim,
            "conv_layers": conv_layers,
            "fc_in_sizes": fc_in_sizes,
        } | extra_info
        with open(file, "wb") as f:
            yaml_str = io.StringIO()
            yaml.YAML().dump(info_dict, yaml_str)
            f.write(yaml_str.getvalue().encode("utf-8"))
            f.write(b"\n---\n")
            eqx.tree_serialise_leaves(f, (model, state))

    @classmethod
    def load(cls, file: str) -> tuple["CNN", PyTree]:
        """Loads a CNN model and its state from a file."""
        with open(file, "rb") as f:
            yaml_lines = []
            while True:
                line = f.readline().decode("utf-8")
                if line.startswith("---"):
                    break
                yaml_lines.append(line)
            info_dict = yaml.YAML().load(io.StringIO("\n".join(yaml_lines)))
            model, state = eqx.nn.make_with_state(cls)(
                info_dict["input_shape"],
                info_dict["output_dim"],
                jax.random.PRNGKey(0),
                info_dict["conv_layers"],
                info_dict["fc_in_sizes"],
            )
            model, state = eqx.tree_deserialise_leaves(f, (model, state))
            return model, state


# ------------------------------------------------------------------------------
# ResNet
# ------------------------------------------------------------------------------
# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py



class BatchNorm(nn.StatefulLayer):
    bn: nn.BatchNorm

    def __init__(self, planes: int):
        self.bn = nn.BatchNorm(planes, axis_name="batch", mode="batch")

    def __call__(self, x: Array, state: PyTree, key = None) -> tuple[Array, PyTree]:
        return self.bn(x, state)


def conv3x3(
    in_planes: int,
    out_planes: int,
    key: jax.random.PRNGKey,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        use_bias=False,
        dilation=dilation,
        key=key,
    )


def conv1x1(
    in_planes: int, out_planes: int, key: jax.random.PRNGKey, stride: int = 1
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        use_bias=False,
        key=key,
    )


class BasicBlock(nn.StatefulLayer):
    expansion: int = 1
    conv1: nn.Conv2d
    bn1: BatchNorm
    conv2: nn.Conv2d
    bn2: BatchNorm
    downsample: eqx.Module | None
    stride: int

    def __init__(
        self,
        inplanes: int,
        planes: int,
        key: jax.random.PRNGKey,
        stride: int = 1,
        downsample: eqx.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., eqx.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        keys = jax.random.split(key, 2)
        self.conv1 = conv3x3(inplanes, planes, keys[0], stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, keys[1])
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: Array, state: PyTree, key = None) -> tuple[Array, PyTree]:
        identity = x

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)

        if self.downsample is not None:
            identity, state = self.downsample(x, state)

        out += identity
        out = jax.nn.relu(out)

        return out, state


class Bottleneck(nn.StatefulLayer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4
    conv1: nn.Conv2d
    bn1: BatchNorm
    conv2: nn.Conv2d
    bn2: BatchNorm
    conv3: nn.Conv2d
    bn3: BatchNorm
    downsample: eqx.Module | None
    stride: int

    def __init__(
        self,
        inplanes: int,
        planes: int,
        key: jax.random.PRNGKey,
        stride: int = 1,
        downsample: eqx.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., eqx.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm
        keys = jax.random.split(key, 3)
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, keys[0])
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, keys[1], stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, keys[2])
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x: Array, state: PyTree, key = None) -> tuple[Array, PyTree]:
        identity = x

        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)
        out = jax.nn.relu(out)

        out = self.conv3(out)
        out, state = self.bn3(out, state)

        if self.downsample is not None:
            identity, state = self.downsample(x, state)

        out += identity
        out = jax.nn.relu(out)

        return out, state


class ResNet(nn.StatefulLayer):
    _norm_layer: eqx.Module
    inplanes: int
    dilation: int
    groups: int
    base_width: int
    conv1: nn.Conv2d
    bn1: BatchNorm
    maxpool: nn.MaxPool2d
    layer1: nn.Sequential
    layer2: nn.Sequential
    layer3: nn.Sequential
    layer4: nn.Sequential
    avgpool: nn.AdaptiveAvgPool2d
    fc: nn.Linear

    def __init__(
        self,
        key: jax.random.PRNGKey,
        block: type[BasicBlock | Bottleneck] = BasicBlock,
        layers: tuple[int, ...] = (2, 2, 2, 2),
        groups: int = 1,
        width_per_group: int = 64,
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        norm_layer = BatchNorm
        self._norm_layer = norm_layer

        keys = jax.random.split(key, 6)

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=keys[0],
        )
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], key=keys[1])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, key=keys[2])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, key=keys[3])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, key=keys[4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes, key=keys[5])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        key: jax.random.PRNGKey,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        keys = jax.random.split(key, blocks + 1)

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                [
                    conv1x1(self.inplanes, planes * block.expansion, keys[0], stride),
                    norm_layer(planes * block.expansion),
                ]
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                keys[1],
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    keys[i + 1],
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(layers)

    def __call__(
        self, x: Float[Array, " c h w"], state: PyTree, key = None
    ) -> tuple[Float[Array, ""], PyTree]:
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = jax.nn.relu(x)
        x = self.maxpool(x)

        x, state = self.layer1(x, state)
        x, state = self.layer2(x, state)
        x, state = self.layer3(x, state)
        x, state = self.layer4(x, state)

        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.fc(x)

        return x, state

def resnet18(key: jax.random.PRNGKey, in_channels: int = 3, num_classes: int = 1) -> ResNet:
    return ResNet(key, block=BasicBlock, layers=(2, 2, 2, 2), in_channels=in_channels, num_classes=num_classes)

