"""从 VA-VAE encoder 提取多尺度特征，供语义分割头使用。"""

import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "LightningDiT"))

# autoencoder.py 顶层导入了 tqdm 但实际未使用，在精简环境下用占位模块兜底。
try:
    import tqdm  # noqa: F401
except ImportError:
    stub = types.ModuleType("tqdm")
    stub.__spec__ = __import__("importlib.machinery", fromlist=["ModuleSpec"]).ModuleSpec("tqdm", None)
    stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = stub

from tokenizer.autoencoder import AutoencoderKL  # noqa: E402


class VAVAEBackbone(nn.Module):
    """封装 VA-VAE encoder，输出 1/4、1/8、1/16 与 1/32 四级特征图。

    在 f16d32 配置下输出通道数为 [256, 256, 512, 512]，输入图像需归一化至 [-1, 1]。
    """

    def __init__(self, ckpt_path, embed_dim=32, ch_mult=(1, 1, 2, 2, 4)):
        super().__init__()
        self.embed_dim = embed_dim
        self.ch_mult = tuple(ch_mult)

        # 不在此处直接传 ckpt_path：新版 PyTorch 默认 weights_only=True，
        # 而 Lightning 格式的 checkpoint 需要使用 weights_only=False 手动加载。
        self.vae = AutoencoderKL(
            embed_dim=embed_dim, ch_mult=self.ch_mult, ckpt_path=None
        )
        self._load_encoder_weights(ckpt_path)

        self.output_channels = [
            128 * self.ch_mult[2],
            128 * self.ch_mult[3],
            128 * self.ch_mult[4],
            128 * self.ch_mult[4],
        ]
        # 1/32 特征直接对 1/16 特征做平均池化得到，不引入新增可训练层，
        # 从而保证冻结 backbone 时严格仅训练分割头。

    def _load_encoder_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
            if not any(bad in k for bad in ("foundation_model", "loss.", "optimizer", "scheduler"))
        }
        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
        encoder_missing = [k for k in missing if k.startswith("encoder.")]
        if encoder_missing:
            raise RuntimeError("Missing VA-VAE encoder weights: " + ", ".join(encoder_missing[:10]))

    def forward(self, x):
        encoder = self.vae.encoder
        h = encoder.conv_in(x)

        features = []
        for i_level in range(encoder.num_resolutions):
            blocks = encoder.down[i_level].block
            atten = encoder.down[i_level].attn
            for i_block in range(len(blocks)):
                h = blocks[i_block](h, None)
                if i_block < len(atten):
                    h = atten[i_block](h)
            if i_level == 2:
                features.append(h)  # 1/4，256 通道
            if i_level == 3:
                features.append(h)  # 1/8，256 通道
            if i_level != encoder.num_resolutions - 1:
                h = encoder.down[i_level].downsample(h)

        h = encoder.mid.block_1(h, None)
        h = encoder.mid.attn_1(h)
        h = encoder.mid.block_2(h, None)
        features.append(h)  # 1/16，512 通道
        features.append(F.avg_pool2d(h, kernel_size=2, stride=2))  # 1/32，512 通道
        return features


if __name__ == "__main__":
    ckpt = os.path.join(ROOT, "vavae-imagenet256-f16d32-dinov2.pt")
    model = VAVAEBackbone(ckpt).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feats = model(x)
    print("input:", tuple(x.shape))
    for i, feat in enumerate(feats, start=1):
        print(f"feature_map_{i}: {tuple(feat.shape)}  stride={x.shape[-1] // feat.shape[-1]}")
