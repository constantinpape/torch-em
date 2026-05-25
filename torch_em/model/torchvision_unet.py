"""UNet variants with pretrained torchvision backbones as encoders.

Supported 2D backbones (pretrained on ImageNet):
    ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
    ResNeXt: resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
    Wide ResNet: wide_resnet50_2, wide_resnet101_2
    VGG: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
    DenseNet: densenet121, densenet161, densenet169, densenet201
    MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
    EfficientNet: efficientnet_b0..b7, efficientnet_v2_s/m/l
    ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
    RegNet: regnet_x_400mf/800mf/1_6gf/3_2gf/8gf/16gf/32gf, regnet_y_400mf/800mf/1_6gf/3_2gf/8gf/16gf/32gf
    MnasNet: mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
    GoogLeNet: googlenet
    ShuffleNet: shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
    Swin Transformer: swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b

Supported 3D backbones (pretrained on Kinetics-400):
    r3d_18, r2plus1d_18, mc3_18
"""

import importlib
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.feature_extraction import create_feature_extractor

from .unet import UNetBase, Decoder, ConvBlock2d, ConvBlock3d, Upsampler2d, Upsampler3d


# Registry entry: (module_name, fn_name, node_names, channels, scale_factors, pre_skip_factor)
#
# node_names: all feature nodes [skip0, ..., skip_{N-1}, bottleneck], shallowest first
# channels: channel count at each node (same order as node_names)
# scale_factors: spatial downsampling between consecutive nodes (len = len(nodes) - 1)
# pre_skip_factor: spatial downsampling from the raw input to skip0 (int for 2D, (D,H,W) tuple for 3D)
#
# Backbones with pre_skip_factor=2 start at H/2; those with pre_skip_factor=4 start at H/4.
# The final_upsample in TorchvisionUNet2d restores output to input resolution.

TV = "torchvision.models"

# Node name lists shared across families
N_RESNET = ["relu", "layer1", "layer2", "layer3", "layer4"]
N_VGG11 = ["features.4", "features.9", "features.14", "features.19", "features.20"]
N_VGG11_BN = ["features.6", "features.13", "features.20", "features.27", "features.28"]
N_VGG13 = ["features.8", "features.13", "features.18", "features.23", "features.24"]
N_VGG13_BN = ["features.12", "features.19", "features.26", "features.33", "features.34"]
N_VGG16 = ["features.8", "features.15", "features.22", "features.29", "features.30"]
N_VGG16_BN = ["features.12", "features.22", "features.32", "features.42", "features.43"]
N_VGG19 = ["features.8", "features.17", "features.26", "features.35", "features.36"]
N_VGG19_BN = ["features.12", "features.25", "features.38", "features.51", "features.52"]
N_DENSENET = [
    "features.relu0", "features.transition1.conv",
    "features.transition2.conv", "features.transition3.conv", "features.norm5",
]
N_MOBILENET_V2 = ["features.1", "features.3", "features.6", "features.13", "features.18"]
N_MOBILENET_V3_S = ["features.1", "features.2", "features.4", "features.9"]
N_MOBILENET_V3_L = ["features.2", "features.4", "features.7", "features.16"]
N_EFF_B = ["features.1", "features.2", "features.3", "features.5", "features.7"]
N_EFF_V2 = ["features.1", "features.2", "features.3", "features.5", "features.6"]
N_CONVNEXT = ["features.1", "features.3", "features.5", "features.7"]
N_REGNET = ["stem", "trunk_output.block1", "trunk_output.block2", "trunk_output.block3", "trunk_output.block4"]
N_MNASNET = ["layers.8", "layers.9", "layers.10", "layers.12"]
N_GOOGLENET = ["conv3.relu", "inception3b.cat", "inception4e.cat", "inception5b.cat"]
N_SHUFFLENET = ["maxpool", "stage2.3.view_1", "stage3.7.view_1", "conv5.2"]
# Swin stage3 has 6 blocks for tiny and 18 for small/base, so two node lists are needed.
N_SWIN_T = [
    "features.1.1.add_1", "features.3.1.add_1", "features.5.5.add_1", "features.7.1.add_1",
]
N_SWIN_SB = [
    "features.1.1.add_1", "features.3.1.add_1", "features.5.17.add_1", "features.7.1.add_1",
]

# Common channel and scale-factor lists
C_VGG = [128, 256, 512, 512, 512]
SF4 = [2, 2, 2, 2]  # depth=4 (pre_skip=2), scale factor 2 at each of 4 inter-node steps
SF3 = [2, 2, 2]  # depth=3 (pre_skip=2 or 4)

# 3D backbone helpers
NODES_3D = ["layer1", "layer2", "layer3", "layer4"]
SF_3D_ISO = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]
SF_3D_MC3 = [(1, 2, 2), (1, 2, 2), (1, 2, 2)]

BACKBONE_REGISTRY_2D = {
    # ResNet (depth=4, pre_skip=2)
    "resnet18": (TV, "resnet18", N_RESNET, [64, 64, 128, 256, 512], SF4, 2),
    "resnet34": (TV, "resnet34", N_RESNET, [64, 64, 128, 256, 512], SF4, 2),
    "resnet50": (TV, "resnet50", N_RESNET, [64, 256, 512, 1024, 2048], SF4, 2),
    "resnet101": (TV, "resnet101", N_RESNET, [64, 256, 512, 1024, 2048], SF4, 2),
    "resnet152": (TV, "resnet152", N_RESNET, [64, 256, 512, 1024, 2048], SF4, 2),
    # ResNeXt (depth=4, pre_skip=2; same node structure as ResNet)
    "resnext50_32x4d": (TV, "resnext50_32x4d", N_RESNET, [64, 256, 512, 1024, 2048], SF4, 2),
    "resnext101_32x8d": (TV, "resnext101_32x8d", N_RESNET, [64, 256, 512, 1024, 2048], SF4, 2),
    "resnext101_64x4d": (TV, "resnext101_64x4d", N_RESNET, [64, 256, 512, 1024, 2048], SF4, 2),
    # Wide ResNet (depth=4, pre_skip=2; same node structure as ResNet)
    "wide_resnet50_2": (TV, "wide_resnet50_2", N_RESNET, [64, 256, 512, 1024, 2048], SF4, 2),
    "wide_resnet101_2": (TV, "wide_resnet101_2", N_RESNET, [64, 256, 512, 1024, 2048], SF4, 2),
    # VGG (depth=4, pre_skip=2; nodes are last-ReLU before each pool, then the final MaxPool)
    "vgg11": (TV, "vgg11", N_VGG11, C_VGG, SF4, 2),
    "vgg11_bn": (TV, "vgg11_bn", N_VGG11_BN, C_VGG, SF4, 2),
    "vgg13": (TV, "vgg13", N_VGG13, C_VGG, SF4, 2),
    "vgg13_bn": (TV, "vgg13_bn", N_VGG13_BN, C_VGG, SF4, 2),
    "vgg16": (TV, "vgg16", N_VGG16, C_VGG, SF4, 2),
    "vgg16_bn": (TV, "vgg16_bn", N_VGG16_BN, C_VGG, SF4, 2),
    "vgg19": (TV, "vgg19", N_VGG19, C_VGG, SF4, 2),
    "vgg19_bn": (TV, "vgg19_bn", N_VGG19_BN, C_VGG, SF4, 2),
    # DenseNet (depth=4, pre_skip=2; nodes are relu0, transition.conv layers, then norm5)
    "densenet121": (TV, "densenet121", N_DENSENET, [64, 128, 256, 512, 1024], SF4, 2),
    "densenet161": (TV, "densenet161", N_DENSENET, [96, 192, 384, 1056, 2208], SF4, 2),
    "densenet169": (TV, "densenet169", N_DENSENET, [64, 128, 256, 640, 1664], SF4, 2),
    "densenet201": (TV, "densenet201", N_DENSENET, [64, 128, 256, 896, 1920], SF4, 2),
    # MobileNet V2 (depth=4, pre_skip=2)
    "mobilenet_v2": (TV, "mobilenet_v2", N_MOBILENET_V2, [16, 24, 32, 96, 1280], SF4, 2),
    # MobileNet V3 (depth=3, pre_skip=4; patchify-style stem goes H->H/4)
    "mobilenet_v3_small": (TV, "mobilenet_v3_small", N_MOBILENET_V3_S, [16, 24, 40, 96], SF3, 4),
    "mobilenet_v3_large": (TV, "mobilenet_v3_large", N_MOBILENET_V3_L, [24, 40, 80, 960], SF3, 4),
    # EfficientNet B (depth=4, pre_skip=2)
    "efficientnet_b0": (TV, "efficientnet_b0", N_EFF_B, [16, 24, 40, 112, 320], SF4, 2),
    "efficientnet_b1": (TV, "efficientnet_b1", N_EFF_B, [16, 24, 40, 112, 320], SF4, 2),
    "efficientnet_b2": (TV, "efficientnet_b2", N_EFF_B, [16, 24, 48, 120, 352], SF4, 2),
    "efficientnet_b3": (TV, "efficientnet_b3", N_EFF_B, [24, 32, 48, 136, 384], SF4, 2),
    "efficientnet_b4": (TV, "efficientnet_b4", N_EFF_B, [24, 32, 56, 160, 448], SF4, 2),
    "efficientnet_b5": (TV, "efficientnet_b5", N_EFF_B, [24, 40, 64, 176, 512], SF4, 2),
    "efficientnet_b6": (TV, "efficientnet_b6", N_EFF_B, [32, 40, 72, 200, 576], SF4, 2),
    "efficientnet_b7": (TV, "efficientnet_b7", N_EFF_B, [32, 48, 80, 224, 640], SF4, 2),
    # EfficientNet V2 (depth=4, pre_skip=2)
    "efficientnet_v2_s": (TV, "efficientnet_v2_s", N_EFF_V2, [24, 48, 64, 160, 256], SF4, 2),
    "efficientnet_v2_m": (TV, "efficientnet_v2_m", N_EFF_V2, [24, 48, 80, 176, 304], SF4, 2),
    "efficientnet_v2_l": (TV, "efficientnet_v2_l", N_EFF_V2, [32, 64, 96, 224, 384], SF4, 2),
    # ConvNeXt (depth=3, pre_skip=4; patchify stem goes H->H/4; stages are features.1/3/5/7)
    "convnext_tiny": (TV, "convnext_tiny", N_CONVNEXT, [96, 192, 384, 768], SF3, 4),
    "convnext_small": (TV, "convnext_small", N_CONVNEXT, [96, 192, 384, 768], SF3, 4),
    "convnext_base": (TV, "convnext_base", N_CONVNEXT, [128, 256, 512, 1024], SF3, 4),
    "convnext_large": (TV, "convnext_large", N_CONVNEXT, [192, 384, 768, 1536], SF3, 4),
    # RegNet X (depth=4, pre_skip=2)
    "regnet_x_400mf": (TV, "regnet_x_400mf", N_REGNET, [32, 32, 64, 160, 400], SF4, 2),
    "regnet_x_800mf": (TV, "regnet_x_800mf", N_REGNET, [32, 64, 128, 288, 672], SF4, 2),
    "regnet_x_1_6gf": (TV, "regnet_x_1_6gf", N_REGNET, [32, 72, 168, 408, 912], SF4, 2),
    "regnet_x_3_2gf": (TV, "regnet_x_3_2gf", N_REGNET, [32, 96, 192, 432, 1008], SF4, 2),
    "regnet_x_8gf": (TV, "regnet_x_8gf", N_REGNET, [32, 80, 240, 720, 1920], SF4, 2),
    "regnet_x_16gf": (TV, "regnet_x_16gf", N_REGNET, [32, 256, 512, 896, 2048], SF4, 2),
    "regnet_x_32gf": (TV, "regnet_x_32gf", N_REGNET, [32, 336, 672, 1344, 2520], SF4, 2),
    # RegNet Y (depth=4, pre_skip=2)
    "regnet_y_400mf": (TV, "regnet_y_400mf", N_REGNET, [32, 48, 104, 208, 440], SF4, 2),
    "regnet_y_800mf": (TV, "regnet_y_800mf", N_REGNET, [32, 64, 144, 320, 784], SF4, 2),
    "regnet_y_1_6gf": (TV, "regnet_y_1_6gf", N_REGNET, [32, 48, 120, 336, 888], SF4, 2),
    "regnet_y_3_2gf": (TV, "regnet_y_3_2gf", N_REGNET, [32, 72, 216, 576, 1512], SF4, 2),
    "regnet_y_8gf": (TV, "regnet_y_8gf", N_REGNET, [32, 224, 448, 896, 2016], SF4, 2),
    "regnet_y_16gf": (TV, "regnet_y_16gf", N_REGNET, [32, 224, 448, 1232, 3024], SF4, 2),
    "regnet_y_32gf": (TV, "regnet_y_32gf", N_REGNET, [32, 232, 696, 1392, 3712], SF4, 2),
    # MnasNet (depth=3, pre_skip=4)
    "mnasnet0_5": (TV, "mnasnet0_5", N_MNASNET, [16, 24, 40, 96], SF3, 4),
    "mnasnet0_75": (TV, "mnasnet0_75", N_MNASNET, [24, 32, 64, 144], SF3, 4),
    "mnasnet1_0": (TV, "mnasnet1_0", N_MNASNET, [24, 40, 80, 192], SF3, 4),
    "mnasnet1_3": (TV, "mnasnet1_3", N_MNASNET, [32, 56, 104, 248], SF3, 4),
    # GoogLeNet (depth=3, pre_skip=4; skips at last feature before each maxpool, bottleneck at inception5b)
    "googlenet": (TV, "googlenet", N_GOOGLENET, [192, 480, 832, 1024], SF3, 4),
    # ShuffleNet V2 (depth=3, pre_skip=4; skip0=maxpool, skips at end of stage2/3, bottleneck=conv5)
    "shufflenet_v2_x0_5": (TV, "shufflenet_v2_x0_5", N_SHUFFLENET, [24, 48, 96, 1024], SF3, 4),
    "shufflenet_v2_x1_0": (TV, "shufflenet_v2_x1_0", N_SHUFFLENET, [24, 116, 232, 1024], SF3, 4),
    "shufflenet_v2_x1_5": (TV, "shufflenet_v2_x1_5", N_SHUFFLENET, [24, 176, 352, 1024], SF3, 4),
    "shufflenet_v2_x2_0": (TV, "shufflenet_v2_x2_0", N_SHUFFLENET, [24, 244, 488, 2048], SF3, 4),
    # Swin Transformer (depth=3, pre_skip=4; NHWC outputs require permute; t/s share channels, b is wider)
    "swin_t": (TV, "swin_t", N_SWIN_T, [96, 192, 384, 768], SF3, 4, True),
    "swin_s": (TV, "swin_s", N_SWIN_SB, [96, 192, 384, 768], SF3, 4, True),
    "swin_b": (TV, "swin_b", N_SWIN_SB, [128, 256, 512, 1024], SF3, 4, True),
    "swin_v2_t": (TV, "swin_v2_t", N_SWIN_T, [96, 192, 384, 768], SF3, 4, True),
    "swin_v2_s": (TV, "swin_v2_s", N_SWIN_SB, [96, 192, 384, 768], SF3, 4, True),
    "swin_v2_b": (TV, "swin_v2_b", N_SWIN_SB, [128, 256, 512, 1024], SF3, 4, True),
}

BACKBONE_REGISTRY_3D = {
    "r3d_18": ("torchvision.models.video", "r3d_18", NODES_3D, [64, 128, 256, 512], SF_3D_ISO, (1, 2, 2)),
    "r2plus1d_18": ("torchvision.models.video", "r2plus1d_18", NODES_3D, [64, 128, 256, 512], SF_3D_ISO, (1, 2, 2)),
    "mc3_18": ("torchvision.models.video", "mc3_18", NODES_3D, [64, 128, 256, 512], SF_3D_MC3, (1, 2, 2)),
}


def _load_backbone(module_name, fn_name, pretrained):
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    return fn(weights="DEFAULT" if pretrained else None)


class TorchvisionEncoder(nn.Module):
    """@private"""

    def __init__(
        self,
        backbone: nn.Module,
        node_names: List[str],
        skip_channels: List[int],
        bottleneck_channels: int,
        skip_targets: List[int],
        bottleneck_target: int,
        in_channels: int,
        conv_cls,
        nhwc: bool = False,
    ):
        super().__init__()

        depth = len(skip_channels)
        assert len(node_names) == depth + 1

        return_nodes = {name: f"skip{i}" for i, name in enumerate(node_names[:-1])}
        return_nodes[node_names[-1]] = "bottleneck"

        self.input_proj = conv_cls(in_channels, 3, kernel_size=1) if in_channels != 3 else None
        self.extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        self._depth = depth

        self.skip_projs = nn.ModuleList([
            conv_cls(inc, outc, kernel_size=1) if inc != outc else nn.Identity()
            for inc, outc in zip(skip_channels, skip_targets)
        ])
        self.bottleneck_proj = (
            conv_cls(bottleneck_channels, bottleneck_target, kernel_size=1)
            if bottleneck_channels != bottleneck_target else nn.Identity()
        )

        self.return_outputs = True
        self._in_channels = in_channels
        self._out_channels = bottleneck_target
        self._nhwc = nhwc

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    def __len__(self):
        return self._depth

    def forward(self, x):
        if self.input_proj is not None:
            x = self.input_proj(x)

        features = self.extractor(x)
        if self._nhwc:
            features = {k: v.permute(0, 3, 1, 2).contiguous() for k, v in features.items()}
        skips = [proj(features[f"skip{i}"]) for i, proj in enumerate(self.skip_projs)]
        bottleneck = self.bottleneck_proj(features["bottleneck"])

        if self.return_outputs:
            return bottleneck, skips
        return bottleneck


def _build_encoder_and_decoder(
    backbone_name,
    registry,
    depth,
    initial_features,
    gain,
    in_channels,
    pretrained,
    conv_block_impl,
    sampler_impl,
    conv_cls,
    **conv_block_kwargs,
):
    entry = registry[backbone_name]
    module_name, fn_name, all_nodes, all_channels, all_scale_factors, pre_skip_factor = entry[:6]
    nhwc = entry[6] if len(entry) > 6 else False

    max_depth = len(all_nodes) - 1
    if depth > max_depth:
        raise ValueError(f"Backbone '{backbone_name}' supports at most depth={max_depth}, got {depth}.")

    used_nodes = all_nodes[:depth + 1]
    used_channels = all_channels[:depth + 1]
    skip_channels = used_channels[:depth]
    bottleneck_channels = used_channels[depth]
    scale_factors = all_scale_factors[:depth]

    features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]

    # Required projected skip channels (shallow to deep):
    # decoder level i (deep to shallow) needs features_decoder[i] - features_decoder[i+1] channels
    # from encoder_inputs[i] = skip at depth-1-i (after reversal in UNetBase)
    skip_targets_deep_first = [features_decoder[i] - features_decoder[i + 1] for i in range(depth)]
    skip_targets = list(reversed(skip_targets_deep_first))  # shallow to deep

    # Base: initial_features * gain^(depth-1) -> initial_features * gain^depth
    bottleneck_target = features_decoder[0] // gain

    backbone = _load_backbone(module_name, fn_name, pretrained)

    encoder = TorchvisionEncoder(
        backbone=backbone,
        node_names=used_nodes,
        skip_channels=skip_channels,
        bottleneck_channels=bottleneck_channels,
        skip_targets=skip_targets,
        bottleneck_target=bottleneck_target,
        in_channels=in_channels,
        conv_cls=conv_cls,
        nhwc=nhwc,
    )

    base = conv_block_impl(bottleneck_target, features_decoder[0], **conv_block_kwargs)

    decoder = Decoder(
        features=features_decoder,
        scale_factors=scale_factors[::-1],
        conv_block_impl=conv_block_impl,
        sampler_impl=sampler_impl,
        **conv_block_kwargs,
    )

    return encoder, base, decoder, scale_factors, pre_skip_factor, features_decoder


class TorchvisionUNetBase(UNetBase):
    """@private"""

    def __init__(
        self,
        encoder,
        base,
        decoder,
        out_conv,
        scale_factors,
        pre_skip_factor,
        final_activation,
        postprocessing,
        check_shape,
        perform_range_checks,
        norm_mean,
        norm_std,
    ):
        super().__init__(
            encoder=encoder,
            base=base,
            decoder=decoder,
            out_conv=out_conv,
            final_activation=final_activation,
            postprocessing=postprocessing,
            check_shape=check_shape,
        )
        self._scale_factors = scale_factors
        self._pre_skip_factor = pre_skip_factor
        self.perform_range_checks = perform_range_checks
        self.register_buffer("norm_mean", norm_mean)
        self.register_buffer("norm_std", norm_std)

    def _check_shape(self, x):
        spatial = x.shape[2:]
        pre = self._pre_skip_factor
        if not isinstance(pre, (list, tuple)):
            pre = (pre,) * len(spatial)
        for dim, (sh, pf) in enumerate(zip(spatial, pre)):
            total = pf
            for sf in self._scale_factors:
                total = total * (sf[dim] if isinstance(sf, (list, tuple)) else sf)
            if sh % total != 0:
                raise ValueError(
                    f"Input spatial shape {tuple(spatial)} is not compatible with this backbone and depth "
                    f"(dim {dim} must be divisible by {total})."
                )

    def _apply_upsample(self, decoded: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mean is not None and self.perform_range_checks:
            actual_min, actual_max = torch.aminmax(x.detach())
            if actual_min < 0.0 or actual_max > 1.0:
                raise ValueError(
                    f"Input is outside the expected [0, 1] range for pretrained normalization: "
                    f"got [{actual_min.item():.4f}, {actual_max.item():.4f}]."
                )

        if self.norm_mean is not None:
            x = (x - self.norm_mean) / self.norm_std

        out = super().forward(x)
        return [self._apply_upsample(t) for t in out] if isinstance(out, list) else self._apply_upsample(out)


class TorchvisionUNet2d(TorchvisionUNetBase):
    """A 2D U-Net that uses a pretrained torchvision backbone as the encoder.

    Skip connections from the backbone are projected to match the standard
    feature progression (initial_features * gain ** level). The decoder is
    identical to the one used by UNet2d. A final bilinear upsample restores
    the output to the original input resolution.

    When pretrained=True and in_channels=3, inputs are automatically normalized
    with ImageNet statistics (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    and are expected in [0, 1].

    Supported backbones (pretrained on ImageNet):
        ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
        ResNeXt: resnext50_32x4d, resnext101_32x8d, resnext101_64x4d
        Wide ResNet: wide_resnet50_2, wide_resnet101_2
        VGG: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
        DenseNet: densenet121, densenet161, densenet169, densenet201
        MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
        EfficientNet: efficientnet_b0..b7, efficientnet_v2_s/m/l
        ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
        RegNet: regnet_x_400mf/800mf/1_6gf/3_2gf/8gf/16gf/32gf, regnet_y_400mf/800mf/1_6gf/3_2gf/8gf/16gf/32gf
        MnasNet: mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
        GoogLeNet: googlenet
        ShuffleNet: shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
        Swin Transformer: swin_t, swin_s, swin_b, swin_v2_t, swin_v2_s, swin_v2_b

    Args:
        backbone: Name of the torchvision backbone to use.
        out_channels: Number of output channels.
        in_channels: Number of input channels. Must be 3 when pretrained=True.
            If != 3 and pretrained=False, a learned 1x1 projection maps the input
            to 3 channels before the backbone.
        depth: Number of encoder/decoder levels. Most backbones support depth=4;
            ConvNeXt, MobileNetV3, and MnasNet support depth=3.
        initial_features: Controls decoder channel widths: level i has
            initial_features * gain ** i channels.
        gain: Multiplier for decoder features per level.
        pretrained: Whether to load ImageNet-pretrained backbone weights.
        perform_range_checks: Whether to validate that inputs are in [0, 1] before
            normalization. Disable to avoid GPU sync overhead during training.
        final_activation: Activation applied after the output convolution.
        postprocessing: Optional postprocessing module or name.
        check_shape: Whether to validate the input shape.
        conv_block_kwargs: Additional kwargs forwarded to ConvBlock2d (e.g. norm).
    """

    def __init__(
        self,
        backbone: str,
        out_channels: int,
        in_channels: int = 3,
        depth: int = 4,
        initial_features: int = 32,
        gain: int = 2,
        pretrained: bool = True,
        perform_range_checks: bool = True,
        final_activation: Optional[Union[str, nn.Module]] = None,
        postprocessing: Optional[Union[str, nn.Module]] = None,
        check_shape: bool = True,
        **conv_block_kwargs,
    ):
        if backbone not in BACKBONE_REGISTRY_2D:
            raise ValueError(f"Unknown 2D backbone '{backbone}'. Choose from: {list(BACKBONE_REGISTRY_2D)}")
        if pretrained and in_channels != 3:
            raise ValueError(
                "pretrained=True requires in_channels=3. The backbone was pretrained on 3-channel inputs "
                "and cannot be meaningfully initialized from pretrained weights with a different channel count."
            )

        encoder, base, decoder, scale_factors, pre_skip_factor, features_decoder = _build_encoder_and_decoder(
            backbone_name=backbone, registry=BACKBONE_REGISTRY_2D, depth=depth,
            initial_features=initial_features, gain=gain, in_channels=in_channels,
            pretrained=pretrained, conv_block_impl=ConvBlock2d, sampler_impl=Upsampler2d,
            conv_cls=nn.Conv2d, **conv_block_kwargs,
        )
        out_conv = None if out_channels is None else nn.Conv2d(features_decoder[-1], out_channels, kernel_size=1)

        if pretrained:
            norm_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            norm_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        else:
            norm_mean = norm_std = None

        super().__init__(
            encoder=encoder, base=base, decoder=decoder, out_conv=out_conv,
            scale_factors=scale_factors, pre_skip_factor=pre_skip_factor,
            final_activation=final_activation, postprocessing=postprocessing,
            check_shape=check_shape, perform_range_checks=perform_range_checks,
            norm_mean=norm_mean, norm_std=norm_std,
        )

    def _apply_upsample(self, decoded: torch.Tensor) -> torch.Tensor:
        return F.interpolate(decoded, scale_factor=float(self._pre_skip_factor), mode="bilinear", align_corners=False)


class TorchvisionUNet3d(TorchvisionUNetBase):
    """A 3D U-Net that uses a pretrained torchvision video backbone as the encoder.

    The video backbone (trained on Kinetics-400) uses true 3D convolutions and can
    process volumetric inputs (B, C, D, H, W) directly. Skip connections are
    projected to match the standard feature progression. A final trilinear upsample
    restores the output to the input resolution (the backbone stem downsamples H and W
    by 2 before the first skip level).

    When pretrained=True and in_channels=3, inputs are automatically normalized with
    Kinetics-400 statistics (mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.21699))
    and are expected in [0, 1].

    Supported backbones (pretrained on Kinetics-400):
        r3d_18, r2plus1d_18, mc3_18

    Args:
        backbone: Name of the torchvision video backbone to use.
        out_channels: Number of output channels.
        in_channels: Number of input channels. Must be 3 when pretrained=True.
            If != 3 and pretrained=False, a learned 1x1 projection maps the input
            to 3 channels before the backbone.
        depth: Number of encoder/decoder levels (max 3 for supported video backbones).
        initial_features: Controls decoder channel widths: level i has
            initial_features * gain ** i channels.
        gain: Multiplier for decoder features per level.
        pretrained: Whether to load Kinetics-400-pretrained backbone weights.
        perform_range_checks: Whether to validate that inputs are in [0, 1] before
            normalization. Disable to avoid GPU sync overhead during training.
        final_activation: Activation applied after the output convolution.
        postprocessing: Optional postprocessing module or name.
        check_shape: Whether to validate the input shape.
        conv_block_kwargs: Additional kwargs forwarded to ConvBlock3d (e.g. norm).
    """

    def __init__(
        self,
        backbone: str,
        out_channels: int,
        in_channels: int = 3,
        depth: int = 3,
        initial_features: int = 32,
        gain: int = 2,
        pretrained: bool = True,
        perform_range_checks: bool = True,
        final_activation: Optional[Union[str, nn.Module]] = None,
        postprocessing: Optional[Union[str, nn.Module]] = None,
        check_shape: bool = True,
        **conv_block_kwargs,
    ):
        if backbone not in BACKBONE_REGISTRY_3D:
            raise ValueError(f"Unknown 3D backbone '{backbone}'. Choose from: {list(BACKBONE_REGISTRY_3D)}")
        if pretrained and in_channels != 3:
            raise ValueError(
                "pretrained=True requires in_channels=3. The backbone was pretrained on 3-channel inputs "
                "and cannot be meaningfully initialized from pretrained weights with a different channel count."
            )

        encoder, base, decoder, scale_factors, pre_skip_factor, features_decoder = _build_encoder_and_decoder(
            backbone_name=backbone, registry=BACKBONE_REGISTRY_3D, depth=depth,
            initial_features=initial_features, gain=gain, in_channels=in_channels,
            pretrained=pretrained, conv_block_impl=ConvBlock3d, sampler_impl=Upsampler3d,
            conv_cls=nn.Conv3d, **conv_block_kwargs,
        )
        out_conv = None if out_channels is None else nn.Conv3d(features_decoder[-1], out_channels, kernel_size=1)

        if pretrained:
            norm_mean = torch.tensor((0.43216, 0.394666, 0.37645)).view(1, 3, 1, 1, 1)
            norm_std = torch.tensor((0.22803, 0.22145, 0.216989)).view(1, 3, 1, 1, 1)
        else:
            norm_mean = norm_std = None

        super().__init__(
            encoder=encoder, base=base, decoder=decoder, out_conv=out_conv,
            scale_factors=scale_factors, pre_skip_factor=pre_skip_factor,
            final_activation=final_activation, postprocessing=postprocessing,
            check_shape=check_shape, perform_range_checks=perform_range_checks,
            norm_mean=norm_mean, norm_std=norm_std,
        )

    def _apply_upsample(self, decoded: torch.Tensor) -> torch.Tensor:
        scale = [float(f) for f in self._pre_skip_factor]
        return F.interpolate(decoded, scale_factor=scale, mode="trilinear", align_corners=False)
