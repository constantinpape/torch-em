# This file implements 3d resnets, based on the implementations from torchvision:
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

# from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
# from torchvision.utils import _log_api_usage_once


__all__ = [
    "ResNet3d",
    "resnet3d_18",
    "resnet3d_34",
    "resnet3d_50",
    "resnet3d_101",
    "resnet3d_152",
    "resnext3d_50_32x4d",
    "resnext3d_101_32x8d",
    "resnext3d_101_64x4d",
    "wide_resnet3d_50_2",
    "wide_resnet3d_101_2",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
    """@private
    """
    # 3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """@private
    """
    # 1x1 convolution
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """@private
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """@private
    """
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):
    """@private
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int,
        out_channels: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        stride_conv1: bool = True,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(
            in_channels, self.inplanes, kernel_size=7, stride=2 if stride_conv1 else 1, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Any,
    progress: bool,
    **kwargs: Any,
) -> ResNet3d:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet3d(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def resnet3d_18(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a residual network for 3d data with 18 layers.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The 3D ResNet.
    """
    return _resnet(
        BasicBlock, [2, 2, 2, 2], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def resnet3d_34(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a residual network for 3d data with 34 layers.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The 3D ResNet.
    """
    return _resnet(
        BasicBlock, [3, 4, 6, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def resnet3d_50(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a residual network for 3d data with 50 layers.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The 3D ResNet.
    """
    return _resnet(
        Bottleneck, [3, 4, 6, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def resnet3d_101(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a residual network for 3d data with 101 layers.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The 3D ResNet.
    """
    return _resnet(
        Bottleneck, [3, 4, 23, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def resnet3d_152(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a residual network for 3d data with 152 layers.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The 3D ResNet.
    """
    return _resnet(
        Bottleneck, [3, 8, 36, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def resnext3d_50_32x4d(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a residual network for 3d data with 50 layers and ResNext layer design.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The 3D ResNext.
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(
        Bottleneck, [3, 4, 6, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def resnext3d_101_32x8d(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a residual network for 3d data with 101 layers and ResNext layer design.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The 3D ResNext.
    """
    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 8)
    return _resnet(
        Bottleneck, [3, 4, 23, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def resnext3d_101_64x4d(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a residual network for 3d data with 101 layers and ResNext layer design.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The 3D ResNext.
    """
    _ovewrite_named_param(kwargs, "groups", 64)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(
        Bottleneck, [3, 4, 23, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def wide_resnet3d_50_2(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a wide residual network for 3d data with 50 layers.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The wide 3D ResNet.
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(
        Bottleneck, [3, 4, 6, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )


def wide_resnet3d_101_2(in_channels: int, out_channels: int, **kwargs: Any) -> ResNet3d:
    """Get a wide residual network for 3d data with 101 layers.

    The implementation of this network is based on torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        kwargs: Additional keyword arguments for the ResNet.

    Returns:
        The wide 3D ResNet.
    """
    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(
        Bottleneck, [3, 4, 23, 3], weights=None, progress=False,
        in_channels=in_channels, out_channels=out_channels, **kwargs
    )
