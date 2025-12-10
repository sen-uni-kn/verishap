# Copyright 2025 David Boetius
"""Resnet in Jax.

Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import Array, Float, PyTree


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
            3,
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
        self.fc = nn.Linear(512 * block.expansion, 1, key=keys[5])

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
        self, x: Float[Array, " 3 32 32"], state: PyTree, key = None
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
        x = x.squeeze(axis=-1)

        return x, state

def resnet18(key: jax.random.PRNGKey) -> ResNet:
    return ResNet(key, block=BasicBlock, layers=(2, 2, 2, 2))
