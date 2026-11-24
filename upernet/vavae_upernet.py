"""面向 VA-VAE 骨干网络的精简版 UPerNet 语义分割头。

模块命名与 HuggingFace UperNetForSemanticSegmentation 保持一致，
便于直接加载 `openmmlab/upernet-*` 预训练分割头权重。
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vavae_backbone import VAVAEBackbone


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


class PyramidPoolingBlock(nn.Module):
    def __init__(self, scale, in_channels, channels):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.AdaptiveAvgPool2d(scale), ConvModule(in_channels, channels, kernel_size=1)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, pool_scales, in_channels, channels):
        super().__init__()
        self.align_corners = False
        # 按 "0"、"1"、... 注册子模块，与 HuggingFace checkpoint 的键名对齐。
        self._blocks = []
        for i, scale in enumerate(pool_scales):
            block = PyramidPoolingBlock(scale, in_channels, channels)
            self.add_module(str(i), block)
            self._blocks.append(block)

    def forward(self, x):
        outs = [x]
        for block in self._blocks:
            pooled = block(x)
            outs.append(
                F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners)
            )
        return torch.cat(outs, dim=1)


class UperNetHead(nn.Module):
    def __init__(self, in_channels, channels=512, num_classes=150, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.channels = channels
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

        self.psp_modules = PyramidPoolingModule(pool_scales, in_channels[-1], channels)
        self.bottleneck = ConvModule(
            in_channels[-1] + len(pool_scales) * channels, channels, kernel_size=3, padding=1
        )

        self.lateral_convs = nn.ModuleList(
            [ConvModule(c, channels, kernel_size=1) for c in in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvModule(channels, channels, kernel_size=3, padding=1) for _ in in_channels[:-1]]
        )
        self.fpn_bottleneck = ConvModule(
            len(in_channels) * channels, channels, kernel_size=3, padding=1
        )

    def forward(self, features):
        if len(features) != len(self.lateral_convs) + 1:
            raise ValueError(f"Expected {len(self.lateral_convs) + 1} feature maps, got {len(features)}")

        laterals = [conv(features[i]) for i, conv in enumerate(self.lateral_convs)]
        psp = self.bottleneck(self.psp_modules(features[-1]))
        laterals.append(psp)

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False
            )

        fpn_outs = [conv(laterals[i]) for i, conv in enumerate(self.fpn_convs)]
        fpn_outs.append(laterals[-1])
        target_size = fpn_outs[0].shape[2:]
        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size, mode="bilinear", align_corners=False)

        return self.classifier(self.fpn_bottleneck(torch.cat(fpn_outs, dim=1)))


class VAVAEUPerNet(nn.Module):
    def __init__(self, ckpt_path, num_classes=150, embed_dim=32, ch_mult=(1, 1, 2, 2, 4)):
        super().__init__()
        self.backbone = VAVAEBackbone(ckpt_path, embed_dim=embed_dim, ch_mult=ch_mult)
        self.decode_head = UperNetHead(self.backbone.output_channels, num_classes=num_classes)
        self.backbone_frozen = False

    def freeze_backbone(self):
        """冻结 VA-VAE encoder 并在无梯度上下文中执行前向传播。

        骨干网络不包含随机初始化层，冻结后即严格只训练 UPerNet head；
        同时跳过 encoder 的反向图构建，可节省训练时间与激活显存。
        """
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.eval()
        self.backbone_frozen = True

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            # 保持 encoder 的 BatchNorm 统计量与 Dropout 处于评估模式。
            self.backbone.eval()
        return self

    def forward(self, x):
        if self.backbone_frozen:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

    def load_hf_head(self, hf_ckpt_path):
        """从 openmmlab/upernet-* 权重中加载形状匹配的 UPerNet head 参数。

        Swin-Tiny 的特征通道为 [96, 192, 384, 768]，而 VA-VAE 输出为
        [256, 256, 512, 512]，因此依赖输入通道维度的张量（lateral_convs、
        psp_modules、bottleneck）会自动跳过并保持随机初始化。
        """
        ckpt = torch.load(hf_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        own_state = self.decode_head.state_dict()
        head_state, skipped = {}, []
        for k, v in state_dict.items():
            if not k.startswith("decode_head."):
                continue
            name = k[len("decode_head."):]
            # 旧版 transformers 权重的 PPM 键名为 "psp_modules.N.1.*"，
            # 新版为 "psp_modules.N.layers.1.*"，此处做兼容映射。
            parts = name.split(".")
            if len(parts) >= 4 and parts[0] == "psp_modules" and parts[1].isdigit() and parts[2].isdigit():
                name = f"psp_modules.{parts[1]}.layers.{'.'.join(parts[2:])}"
            if name not in own_state:
                skipped.append((name, "not in model"))
            elif own_state[name].shape != v.shape:
                skipped.append((name, f"{tuple(v.shape)} != {tuple(own_state[name].shape)}"))
            else:
                head_state[name] = v
        result = self.decode_head.load_state_dict(head_state, strict=False)
        print(f"loaded {len(head_state)} head tensors; size-mismatch/skipped={len(skipped)}; missing={len(result.missing_keys)}")
        for name, why in skipped[:8]:
            print(f"  skipped {name}: {why}")
        return result


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = os.path.join(root, "vavae-imagenet256-f16d32-dinov2.pt")
    model = VAVAEUPerNet(ckpt, num_classes=150).eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(x)
    print("input:", tuple(x.shape))
    print("logits:", tuple(logits.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
