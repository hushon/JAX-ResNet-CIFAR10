# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Resnet."""

import types
from typing import Mapping, Optional, Sequence, Union, Any

from haiku._src import basic
from haiku._src import batch_norm
from haiku._src import conv
from haiku._src import module
from haiku._src import pool
import jax
import jax.numpy as jnp

# If forking replace this block with `import haiku as hk`.
# hk = types.ModuleType("haiku")
import haiku as hk
hk.Module = module.Module
hk.BatchNorm = batch_norm.BatchNorm
hk.Conv2D = conv.Conv2D
hk.Linear = basic.Linear
hk.max_pool = pool.max_pool
del basic, batch_norm, conv, module, pool


class BlockV1(hk.Module):
    """ResNet V1 block with optional bottleneck."""

    def __init__(
        self,
        channels: int,
        stride: Union[int, Sequence[int]],
        bn_config: Mapping[str, float],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.999)

        conv_0 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=3,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="conv_0")
        bn_0 = hk.BatchNorm(name="batchnorm_0", **bn_config)

        conv_1 = hk.Conv2D(
            output_channels=channels,
            kernel_shape=3,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="conv_1")

        bn_1 = hk.BatchNorm(name="batchnorm_1", **bn_config)
        self.layers = ((conv_0, bn_0), (conv_1, bn_1))
        if stride != 1:
            self.shortcut = lambda x: jnp.pad(x[:, ::2, ::2, :], ((
                0, 0), (0, 0), (0, 0), (channels//4, channels//4)), "constant")
        else:
            self.shortcut = lambda x: x

    def __call__(self, inputs, is_training, test_local_stats):
        out = shortcut = inputs

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out)
            out = bn_i(out, is_training, test_local_stats)
            if i < len(self.layers) - 1:  # Don't apply relu on last layer
                out = jax.nn.relu(out)

        shortcut = self.shortcut(shortcut)
        return jax.nn.relu(out + shortcut)


class BlockGroup(hk.Module):
    """Higher level block for ResNet implementation."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        stride: Union[int, Sequence[int]],
        bn_config: Mapping[str, float],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                BlockV1(channels=channels,
                          stride=(stride if i == 0 else 1),
                          bn_config=bn_config,
                          name="block_%d" % (i)))

    def __call__(self, inputs, is_training, test_local_stats):
        out = inputs
        for block in self.blocks:
            out = block(out, is_training, test_local_stats)
        return out


def check_length(length, value, name):
    if len(value) != length:
        raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.Module):
    """ResNet model."""

    CONFIGS = {
        20: {
            "blocks_per_group": (3, 3, 3),
            "channels_per_group": (16, 32, 64),
        },
        32: {
            "blocks_per_group": (5, 5, 5),
            "channels_per_group": (16, 32, 64),
        },
        44: {
            "blocks_per_group": (7, 7, 7),
            "channels_per_group": (16, 32, 64),
        },
        56: {
            "blocks_per_group": (9, 9, 9),
            "channels_per_group": (16, 32, 64),
        },
        110: {
            "blocks_per_group": (18, 18, 18),
            "channels_per_group": (16, 32, 64),
        },
    }

    BlockGroup = BlockGroup  # pylint: disable=invalid-name
    BlockV1 = BlockV1  # pylint: disable=invalid-name

    def __init__(
        self,
        blocks_per_group: Sequence[int],
        num_classes: int,
        bn_config: Optional[Mapping[str, float]] = None,
        channels_per_group: Sequence[int] = (16, 32, 64),
        logits_config: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """Constructs a ResNet model.

        Args:
          blocks_per_group: A sequence of length 4 that indicates the number of
            blocks created in each group.
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers. By default the
            ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
            ``False``.
          bottleneck: Whether the block should bottleneck or not. Defaults to
            ``True``.
          channels_per_group: A sequence of length 4 that indicates the number
            of channels used for each block in each group.
          use_projection: A sequence of length 4 that indicates whether each
            residual block should use projection.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(name=name)

        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        logits_config = dict(logits_config or {})
        logits_config.setdefault("w_init", jnp.zeros)
        logits_config.setdefault("name", "logits")

        # Number of blocks in each group for ResNet.
        # check_length(4, blocks_per_group, "blocks_per_group")
        # check_length(4, channels_per_group, "channels_per_group")

        self.initial_conv = hk.Conv2D(
            output_channels=16,
            kernel_shape=3,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="initial_conv")
        self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm",
                                                **bn_config)

        self.block_groups = []
        strides = (1, 2, 2)
        for i in range(3):
            self.block_groups.append(
                BlockGroup(channels=channels_per_group[i],
                           num_blocks=blocks_per_group[i],
                           stride=strides[i],
                           bn_config=bn_config,
                           name="block_group_%d" % (i)))

        self.logits = hk.Linear(num_classes, **logits_config)

    def __call__(self, inputs, is_training, test_local_stats=False):
        out = inputs
        out = self.initial_conv(out)
        out = self.initial_batchnorm(out, is_training, test_local_stats)
        out = jax.nn.relu(out)

        for block_group in self.block_groups:
            out = block_group(out, is_training, test_local_stats)

        out = jnp.mean(out, axis=[1, 2])
        return self.logits(out)


class ResNet20(ResNet):
    """ResNet20."""

    def __init__(self,
                 num_classes: int,
                 bn_config: Optional[Mapping[str, float]] = None,
                 logits_config: Optional[Mapping[str, Any]] = None,
                 name: Optional[str] = None):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(num_classes=num_classes,
                         bn_config=bn_config,
                         logits_config=logits_config,
                         name=name,
                         **ResNet.CONFIGS[20])


class ResNet32(ResNet):
    """ResNet32."""

    def __init__(self,
                 num_classes: int,
                 bn_config: Optional[Mapping[str, float]] = None,
                 logits_config: Optional[Mapping[str, Any]] = None,
                 name: Optional[str] = None):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(num_classes=num_classes,
                         bn_config=bn_config,
                         logits_config=logits_config,
                         name=name,
                         **ResNet.CONFIGS[32])


class ResNet44(ResNet):
    """ResNet44."""

    def __init__(self,
                 num_classes: int,
                 bn_config: Optional[Mapping[str, float]] = None,
                 logits_config: Optional[Mapping[str, Any]] = None,
                 name: Optional[str] = None):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(num_classes=num_classes,
                         bn_config=bn_config,
                         logits_config=logits_config,
                         name=name,
                         **ResNet.CONFIGS[44])


class ResNet56(ResNet):
    """ResNet56."""

    def __init__(self,
                 num_classes: int,
                 bn_config: Optional[Mapping[str, float]] = None,
                 logits_config: Optional[Mapping[str, Any]] = None,
                 name: Optional[str] = None):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(num_classes=num_classes,
                         bn_config=bn_config,
                         logits_config=logits_config,
                         name=name,
                         **ResNet.CONFIGS[56])


class ResNet110(ResNet):
    """ResNet110."""

    def __init__(self,
                 num_classes: int,
                 bn_config: Optional[Mapping[str, float]] = None,
                 logits_config: Optional[Mapping[str, Any]] = None,
                 name: Optional[str] = None):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(num_classes=num_classes,
                         bn_config=bn_config,
                         logits_config=logits_config,
                         name=name,
                         **ResNet.CONFIGS[110])
