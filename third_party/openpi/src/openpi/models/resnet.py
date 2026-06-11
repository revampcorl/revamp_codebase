# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small ResNet stem used by the vendored OpenPI ViT module.

The TurnOnSinkFaucet pi0 config in this release uses the SigLIP/ViT path with
``resnet=None``. OpenPI's ``vit.py`` still imports this module at import time,
so the release keeps compatible layers here instead of depending on an
unvendored upstream file.
"""

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class StdConv(nn.Module):
    """Convolution layer with weight-standardized kernels."""

    features: int
    kernel_size: tuple[int, int]
    strides: tuple[int, int] = (1, 1)
    padding: str | tuple[tuple[int, int], tuple[int, int]] = "SAME"
    use_bias: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            self.kernel_size + (x.shape[-1], self.features),
            self.param_dtype,
        )
        mean = jnp.mean(kernel, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(kernel, axis=(0, 1, 2), keepdims=True)
        kernel = (kernel - mean) / jnp.sqrt(var + 1e-5)
        y = jax.lax.conv_general_dilated(
            x,
            kernel.astype(self.dtype),
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, (self.features,), self.param_dtype)
            y = y + bias.astype(self.dtype)
        return y


class ResidualUnit(nn.Module):
    """Bottleneck residual block matching the interface expected by ViT."""

    nout: int
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        width = self.nout // 4

        y = StdConv(width, (1, 1), name="conv1")(x)
        y = nn.GroupNorm(name="gn1")(y)
        y = nn.relu(y)
        y = StdConv(width, (3, 3), strides=self.strides, name="conv2")(y)
        y = nn.GroupNorm(name="gn2")(y)
        y = nn.relu(y)
        y = StdConv(self.nout, (1, 1), name="conv3")(y)
        y = nn.GroupNorm(name="gn3")(y)

        if residual.shape != y.shape:
            residual = StdConv(self.nout, (1, 1), strides=self.strides, name="conv_proj")(residual)
            residual = nn.GroupNorm(name="gn_proj")(residual)

        return nn.relu(residual + y)


class ResNetStage(nn.Module):
    """Sequence of bottleneck residual blocks."""

    block_size: int
    nout: int
    first_stride: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        for index in range(self.block_size):
            stride = self.first_stride if index == 0 else (1, 1)
            x = ResidualUnit(self.nout, stride, name=f"unit{index + 1}")(x)
        return x
