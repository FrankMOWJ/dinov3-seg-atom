"""DINOv3 with Q/V LoRA and a segmentation decoder."""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SegformerHead import SegformerHeadv2
from .linear_head import LinearHead
from .backbone import load_dinov3

CKPT_FOLDER = '/home/cpsc4880_wm455/project_cpsc4880/cpsc4880_wm455/dinov3_linear/ckpt'

@dataclass
class SegmentationOutput:
    loss: torch.Tensor
    logits: torch.Tensor


class CrossEntropyLoss(nn.Module):
    def forward(self, pred, target, ignore_index=255):
        if target.ndim == 4:
            if target.shape[1] != 1:
                raise ValueError("Segmentation targets must have one channel")
            target = target[:, 0]
        if ignore_index is None:
            return F.cross_entropy(pred, target)
        return F.cross_entropy(pred, target, ignore_index=ignore_index)


class _LoRA_qkv(nn.Module):
    """Keep DINOv3's original masked-bias QKV layer, adding Q/V adapters."""

    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        # Official DINOv3 attention reads these attributes from self.qkv.
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q + self.linear_b_q(self.linear_a_q(x))
        v = v + self.linear_b_v(self.linear_a_v(x))
        return torch.cat((q, k, v), dim=-1)


class PDZSeg(nn.Module):
    """Frozen official DINOv3 + trainable Q/V LoRA + segmentation head.

    backbone_weights accepts an official checkpoint path or URL.
    backbone_repo may point to a cloned DINOv3 repository for offline use.
    pretrained=False is for architecture tests or training from scratch.
    Both the SegFormer and official LinearHead paths use only PyTorch.
    use_cls_token defaults to False for LinearHead and True for SegFormer.
    """

    BACKBONES = {
        "small": ("dinov3_vits16", 384, (2, 5, 8, 11)),
        "base": ("dinov3_vitb16", 768, (2, 5, 8, 11)),
        "large": ("dinov3_vitl16", 1024, (4, 11, 17, 23)),
    }

    def __init__(self, backbone_size="base", r=4, image_shape=(512, 512),
                 num_classes=4, decode_type="linear4", decoder_type="segformer",
                 lora_layer=None, max_epoch=100, backbone_weights=None,
                 backbone_repo="facebookresearch/dinov3", pretrained=True,
                 decoder_channels=None, use_cls_token=None):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        if backbone_size not in self.BACKBONES:
            raise ValueError(f"DINOv3 backbone_size must be one of {tuple(self.BACKBONES)}")
        # "linear"：只提取 DINOv3 最后一层的特征，送入解码器。 "linear4"：提取 四个不同深度层的特征，融合后进行分割。
        if decode_type not in ("linear", "linear4"):
            raise ValueError("decode_type must be 'linear' or 'linear4'")
        if decoder_type != "linear" and not decoder_type.startswith("segformer"):
            raise ValueError(f"Unknown segmentation decoder: {decoder_type}")
        if decoder_type == "linear" and decoder_channels is not None:
            raise ValueError("decoder_channels applies only to the SegFormer decoder")
        if num_classes < 2:
            raise ValueError("Cross-entropy segmentation requires num_classes >= 2")
        if isinstance(image_shape, int):
            image_shape = (image_shape, image_shape)
        if len(image_shape) != 2 or min(image_shape) <= 0:
            raise ValueError("image_shape must be a positive (height, width)")
        self.backbone_size = backbone_size
        self.backbone_name, self.embedding_dim, layers = self.BACKBONES[backbone_size]
        self.n = layers if decode_type == "linear4" else 1
        self.image_shape = tuple(image_shape)
        self.num_classes = num_classes
        self.rank = r
        self.decode_type = decode_type
        self.decoder_type = decoder_type
        self.use_cls_token = decoder_type != "linear" if use_cls_token is None else use_cls_token
        self.decoder_channels = decoder_channels
        if decoder_type != "linear" and decoder_channels is None:
            self.decoder_channels = self.embedding_dim // 2
        self.epoch = 0
        self.max_epoch = max_epoch

        load_kwargs = {"pretrained": pretrained}
        if backbone_weights is not None:
            load_kwargs["weights"] = str(backbone_weights)
        self.dinov3 = load_dinov3(backbone_repo, self.backbone_name, **load_kwargs)
        self.patch_size = self.dinov3.patch_size
        self.lora_layer = (list(range(len(self.dinov3.blocks)))
                           if lora_layer is None else list(lora_layer))
        if len(set(self.lora_layer)) != len(self.lora_layer) or any(
                i < 0 or i >= len(self.dinov3.blocks) for i in self.lora_layer):
            raise ValueError("lora_layer must contain unique, valid block indices")

        for param in self.dinov3.parameters():
            param.requires_grad = False

        # Adapters are registered through dinov3.blocks; these lists are for I/O.
        self.w_As = []
        self.w_Bs = []
        for i, block in enumerate(self.dinov3.blocks):
            if i not in self.lora_layer:
                continue
            qkv = block.attn.qkv
            dim = qkv.in_features
            adapters = [nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False),
                        nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)]
            for adapter in adapters:
                adapter.to(device=qkv.weight.device, dtype=qkv.weight.dtype)
            a_q, b_q, a_v, b_v = adapters
            self.w_As.extend((a_q, a_v))
            self.w_Bs.extend((b_q, b_v))
            block.attn.qkv = _LoRA_qkv(qkv, a_q, b_q, a_v, b_v)
        self.reset_parameters()

        levels = 4 if decode_type == "linear4" else 1
        # Both heads double input channels internally when CLS is enabled.
        self.inchannels = [self.embedding_dim] * levels
        if decoder_type == "linear":
            self.decode_head = LinearHead(
                in_channels=self.inchannels, n_output_channels=num_classes,
                use_cls_token=self.use_cls_token,
            )
        else:
            self.decode_head = SegformerHeadv2(
                num_classes=num_classes,
                in_channels=self.inchannels, channels=self.decoder_channels,
                image_shape=self.image_shape, use_cls_token=self.use_cls_token,
            )
        self.cross_entropy_loss = CrossEntropyLoss()

    def reset_parameters(self):
        for adapter in self.w_As:
            nn.init.kaiming_uniform_(adapter.weight, a=math.sqrt(5))
        for adapter in self.w_Bs:
            nn.init.zeros_(adapter.weight)

    def _checkpoint_metadata(self):
        return {"format_version": 2, "backbone_name": self.backbone_name,
                "rank": self.rank, "lora_layers": self.lora_layer,
                "num_classes": self.num_classes, "decode_type": self.decode_type,
                "decoder_type": self.decoder_type, "decoder_channels": self.decoder_channels,
                "use_cls_token": self.use_cls_token}

    def save_parameters(self, filename):
        """Save adapters, decoder, and metadata, omitting frozen weights."""
        if Path(filename).suffix not in (".pt", ".pth"):
            raise ValueError("Checkpoint filename must end in .pt or .pth")
        state = {f"w_a_{i:03d}": layer.weight.detach().cpu()
                 for i, layer in enumerate(self.w_As)}
        state.update({f"w_b_{i:03d}": layer.weight.detach().cpu()
                      for i, layer in enumerate(self.w_Bs)})
        state.update({k: v.detach().cpu() for k, v in self.decode_head.state_dict().items()})
        state["_metadata"] = self._checkpoint_metadata()
        torch.save(state, filename)

    def load_parameters(self, filename, device="cpu"):
        """Restore in place, preserving existing optimizer references."""
        state = torch.load(filename, map_location=device, weights_only=True)
        metadata = state.get("_metadata")
        if isinstance(metadata, dict):
            metadata = dict(metadata)
            # Earlier DINOv3 SegFormer checkpoints always concatenated CLS.
            if str(metadata.get("decoder_type", "")).startswith("segformer"):
                metadata.setdefault("use_cls_token", True)
        if metadata != self._checkpoint_metadata():
            raise ValueError("Checkpoint architecture does not match this DINOv3 model. "
                             "Legacy DINOv2 adapter checkpoints require retraining.")
        copies = [(f"w_a_{i:03d}", layer.weight) for i, layer in enumerate(self.w_As)]
        copies += [(f"w_b_{i:03d}", layer.weight) for i, layer in enumerate(self.w_Bs)]
        copies += list(self.decode_head.state_dict().items())
        for key, target in copies:
            if key not in state or state[key].shape != target.shape:
                raise ValueError(f"Missing or incompatible checkpoint tensor: {key}")
        with torch.no_grad():
            for key, target in copies:
                target.copy_(state[key])

    def _predict_logits(self, pixel_values):
        if pixel_values.ndim != 4 or pixel_values.shape[1] not in (1, 3):
            raise ValueError("Expected images [B,1,H,W] or [B,3,H,W]")
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        height, width = pixel_values.shape[-2:]
        # Preserve every pixel for sizes such as the old 532px input.
        pad_h, pad_w = (-height) % self.patch_size, (-width) % self.patch_size
        if pad_h or pad_w:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="replicate")
        features = self.dinov3.get_intermediate_layers(
            pixel_values, n=self.n, reshape=True,
            return_class_token=self.use_cls_token, norm=False
        )
        if self.decoder_type.startswith("segformer"):
            logits = self.decode_head(features, output_size=pixel_values.shape[-2:])
        else:
            # Use forward() for training so the official head's dropout is kept.
            logits = self.decode_head(features)
            logits = F.interpolate(logits, size=pixel_values.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits[:, :, :height, :width]

    def forward(self, pixel_values, seg_gt):
        if seg_gt is None:
            raise ValueError("Segmentation ground truth is required; use interface() for inference")
        logits = self._predict_logits(pixel_values)
        return SegmentationOutput(loss=self.cross_entropy_loss(logits, seg_gt), logits=logits)

    def interface(self, pixel_values):
        return self._predict_logits(pixel_values)

if __name__ == '__main__':
    model = PDZSeg(
        backbone_size="base",
        decoder_type="segformer",
        use_cls_token=True,
        pretrained=True,
    ).eval()
    images = torch.randn(2, 3, 512, 512)
    print("input:", images.shape)

    with torch.no_grad():
        features = model.dinov3.get_intermediate_layers(
            images, n=model.n, reshape=True,
            return_class_token=model.use_cls_token, norm=False,
        )
        for i, feature in enumerate(features):
            if model.use_cls_token:
                spatial, cls_token = feature
                print(f"feature[{i}]:", spatial.shape, "CLS:", cls_token.shape)
            else:
                print(f"feature[{i}]:", feature.shape)

        output = model.interface(images)
        print("output:", output.shape)
