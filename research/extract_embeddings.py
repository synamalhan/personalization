"""Extract raw (un-normalized) embeddings for Flowers102 and Oxford-IIIT Pets.

Writes results/emb_<dataset>_<backbone>.npz with arrays X (float32, N x D) and y (int64, N).
With --limit N the file is named emb_<dataset>_<backbone>_limit<N>.npz so a quick test
can never be mistaken for (or overwrite) a full extraction.

Usage:
    python research/extract_embeddings.py --limit 64            # quick test, all combos
    python research/extract_embeddings.py --datasets pets --backbones dinov2 --limit 64
    python research/extract_embeddings.py                       # full run
"""
import argparse
import os
import time
from pathlib import Path

# Must be set before torch is imported: lets ops that MPS lacks fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, models, transforms

import config

REPO_ROOT = Path(__file__).resolve().parent.parent


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(config.RESIZE_SHORT_SIDE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.CROP_SIZE),
        transforms.ToTensor(),  # PIL RGB uint8 -> float [0, 1], shape C x H x W
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def load_dataset(name: str, transform) -> ConcatDataset:
    """All labeled splits combined."""
    root = REPO_ROOT / config.DATA_DIR
    if name == "flowers102":
        # train + val + test are all labeled (8189 images, 102 classes).
        parts = [datasets.Flowers102(root, split=s, transform=transform, download=True)
                 for s in ("train", "val", "test")]
    elif name == "pets":
        # trainval + test are all labeled (7349 images, 37 classes; target = breed index).
        parts = [datasets.OxfordIIITPet(root, split=s, target_types="category",
                                        transform=transform, download=True)
                 for s in ("trainval", "test")]
    else:
        raise ValueError(name)
    return ConcatDataset(parts)


class MobileNetV3Embedder(nn.Module):
    """MobileNetV3-Large with the classifier removed: pooled 960-d features."""

    def __init__(self):
        super().__init__()
        weights = getattr(models.MobileNet_V3_Large_Weights, config.MOBILENET_WEIGHTS)
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier = nn.Identity()  # forward() now returns flatten(avgpool(features))
        self.model = model

    def forward(self, x):
        return self.model(x)


class DinoV2Embedder(nn.Module):
    """DINOv2 ViT-S/14: the final-layer-normed CLS token (384-d)."""

    def __init__(self):
        super().__init__()
        # NOTE: torch.hub downloads the repo's Python code from GitHub and runs it.
        self.model = torch.hub.load(config.DINOV2_HUB_REPO, config.DINOV2_HUB_MODEL)

    def forward(self, x):
        return self.model.forward_features(x)["x_norm_clstoken"]


class DinoV3Embedder(nn.Module):
    """DINOv3 ViT-S/16: the layer-normed CLS token (384-d), via Hugging Face transformers."""

    def __init__(self):
        super().__init__()
        from transformers import AutoModel  # imported here so the other backbones don't need transformers
        self.model = AutoModel.from_pretrained(config.DINOV3_HF_MODEL)

    def forward(self, x):
        return self.model(pixel_values=x).pooler_output


def build_backbone(name: str, device: torch.device) -> nn.Module:
    model = {"mobilenetv3": MobileNetV3Embedder, "dinov2": DinoV2Embedder, "dinov3": DinoV3Embedder}[name]()
    return model.eval().to(device)


@torch.inference_mode()
def embed(model: nn.Module, dataset, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    feats, labels = [], []
    for i, (images, targets) in enumerate(loader):
        feats.append(model(images.to(device)).float().cpu().numpy())
        labels.append(targets.numpy())
        print(f"\r  batch {i + 1}/{len(loader)}", end="", flush=True)
    print()
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels).astype(np.int64)


def maybe_subset(dataset, limit: int | None):
    """Seeded random subset (not the first N, which would cover only a few classes)."""
    if limit is None or limit >= len(dataset):
        return dataset
    rng = np.random.default_rng(config.SEED)
    indices = np.sort(rng.choice(len(dataset), size=limit, replace=False))
    return Subset(dataset, indices.tolist())


def report(tag: str, X: np.ndarray, y: np.ndarray) -> None:
    norms = np.linalg.norm(X, axis=1)
    print(f"  {tag}: X {X.shape} {X.dtype}, y {y.shape} {y.dtype}, classes={len(np.unique(y))}")
    print(f"    NaN={int(np.isnan(X).sum())} inf={int(np.isinf(X).sum())} "
          f"norm min/mean/max={norms.min():.3f}/{norms.mean():.3f}/{norms.max():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="use only N random images per dataset")
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS), choices=config.DATASETS)
    parser.add_argument("--backbones", nargs="+", default=list(config.BACKBONES), choices=config.BACKBONES)
    args = parser.parse_args()

    device = pick_device()
    print(f"device: {device}  torch {torch.__version__}")
    out_dir = REPO_ROOT / config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = build_transform()
    suffix = f"_limit{args.limit}" if args.limit else ""

    for ds_name in args.datasets:
        dataset = maybe_subset(load_dataset(ds_name, transform), args.limit)
        print(f"{ds_name}: {len(dataset)} images")
        for bb_name in args.backbones:
            print(f" backbone {bb_name}")
            model = build_backbone(bb_name, device)
            start = time.perf_counter()
            X, y = embed(model, dataset, device)
            print(f"  took {time.perf_counter() - start:.1f}s")
            report(f"{ds_name}/{bb_name}", X, y)
            path = out_dir / f"emb_{ds_name}_{bb_name}{suffix}.npz"
            np.savez(path, X=X, y=y)
            print(f"  saved {path.relative_to(REPO_ROOT)}")
            del model


if __name__ == "__main__":
    main()
