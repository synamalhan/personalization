"""PyTorch backbones wrapped so the Core ML model takes raw RGB pixels (0-255) and does the
ImageNet mean/std normalization itself. Core ML's ImageType has one scalar `scale`, which cannot
express a per-channel std, so normalization lives inside the model instead.
The caller must still resize to 256 (short side) and center-crop to 224 (see research/config.py).
"""
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
import config  # noqa: E402


class Normalized(nn.Module):
    def __init__(self, embedder: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.register_buffer("mean", torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pixels):  # pixels: (N, 3, 224, 224), RGB, values 0..255
        return self.embedder((pixels * config.PIXEL_SCALE - self.mean) / self.std)


class _DinoCLS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # DINOv2 resizes its position embeddings with bicubic interpolation, which coremltools
        # cannot convert. For a fixed 224x224 input the result is constant, so compute it once
        # and return it as-is (identical values).
        with torch.no_grad():
            pos = model.interpolate_pos_encoding(torch.zeros(1, 1, model.embed_dim),
                                                 config.CROP_SIZE, config.CROP_SIZE)
        model.interpolate_pos_encoding = lambda x, w, h: pos

    def forward(self, x):
        return self.model.forward_features(x)["x_norm_clstoken"]


class _DinoV3CLS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).pooler_output  # layer-normed CLS token


def build_embedder(name: str) -> nn.Module:
    """The normalized-input embedder in eval mode (fp32)."""
    if name == "mobilenetv3":
        weights = getattr(models.MobileNet_V3_Large_Weights, config.MOBILENET_WEIGHTS)
        net = models.mobilenet_v3_large(weights=weights)
        net.classifier = nn.Identity()  # pooled 960-d features
        return Normalized(net).eval()
    if name == "dinov2":
        net = torch.hub.load(config.DINOV2_HUB_REPO, config.DINOV2_HUB_MODEL)
        return Normalized(_DinoCLS(net)).eval()
    if name == "dinov3":
        from transformers import AutoModel
        return Normalized(_DinoV3CLS(AutoModel.from_pretrained(config.DINOV3_HF_MODEL))).eval()
    raise ValueError(name)
