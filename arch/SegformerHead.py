"""SegFormer-style multi-level fusion using only PyTorch."""

import torch
from torch import nn
import torch.nn.functional as F


class _ConvProjection(nn.Module):
    """Match ConvModule with norm_cfg=None, ReLU, and its default initialization."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activate = nn.ReLU(inplace=True)
        # MMCV ConvModule calls mmengine.kaiming_init at construction time.
        # nn.Conv2d's own default initialization has a different distribution.
        nn.init.kaiming_normal_(self.conv.weight, a=0, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.activate(self.conv(x))



class SegformerHeadv2(nn.Module):
    """Project each level, concatenate, fuse, and predict segmentation logits.

    in_channels always describes the original patch embedding widths.
    With use_cls_token=True, inputs are (feature_map, cls_token) pairs and
    projection input channels are doubled internally. Otherwise inputs are
    plain feature maps. No MMSegmentation, MMCV, or MMEngine is needed.
    """

    def __init__(self, interpolate_mode="bilinear", num_classes=3,
                 in_channels=(768, 768, 768, 768),
                 channels=384, image_shape=(512, 512), dropout_ratio=0.1,
                 align_corners=False, upsample_rate=4, use_cls_token=True):
        super().__init__()
        if not in_channels or min(in_channels) <= 0 or channels <= 0:
            raise ValueError("Feature and decoder channels must be positive")
        if num_classes <= 0 or upsample_rate < 1:
            raise ValueError("Invalid classes or upsample rate")
        if interpolate_mode not in ("bilinear", "bicubic", "nearest"):
            raise ValueError(f"Unsupported interpolation: {interpolate_mode}")
        self.in_channels = tuple(in_channels)
        self.use_cls_token = use_cls_token
        # Match LinearHead: callers pass original widths, not CLS-concatenated widths.
        self.projection_channels = tuple(c * (2 if use_cls_token else 1)
                                         for c in self.in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.img_resolution = image_shape
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners
        self.upsample_rate = upsample_rate
        self.convs = nn.ModuleList([
            _ConvProjection(c, channels) for c in self.projection_channels
        ])
        self.fusion_conv = _ConvProjection(
            channels * len(self.in_channels), channels
        )
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def _resize(self, x, size):
        kwargs = {} if self.interpolate_mode == "nearest" else {
            "align_corners": self.align_corners
        }
        return F.interpolate(x, size=size, mode=self.interpolate_mode, **kwargs)

    def forward(self, inputs, output_size=None):
        if len(inputs) != len(self.convs):
            raise ValueError(f"Expected {len(self.convs)} feature levels, got {len(inputs)}")
        features = []
        for index, item in enumerate(inputs):
            if self.use_cls_token:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError("use_cls_token=True requires (feature_map, cls_token) pairs")
                feature, cls_token = item
                if feature.ndim != 4 or cls_token.shape != feature.shape[:2]:
                    raise ValueError("Expected feature [B,C,H,W] and CLS [B,C]")
            else:
                if isinstance(item, (tuple, list)):
                    raise ValueError("use_cls_token=False requires plain feature maps")
                feature = item
            if feature.ndim != 4 or feature.shape[1] != self.in_channels[index]:
                raise ValueError(f"Feature {index} must have {self.in_channels[index]} channels")
            if self.use_cls_token:
                cls_map = cls_token[:, :, None, None].expand_as(feature)
                feature = torch.cat((feature, cls_map), dim=1)
            features.append(feature)

        fusion_size = tuple(s * self.upsample_rate for s in features[0].shape[-2:])
        projected = [self._resize(conv(x), fusion_size)
                     for conv, x in zip(self.convs, features)]
        fused = torch.cat(projected, dim=1)
        logits = self.conv_seg(self.dropout(self.fusion_conv(fused)))
        size = self.img_resolution if output_size is None else output_size
        return self._resize(logits, size)
