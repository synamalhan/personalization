"""Create the 20 fixed gate-check images and their Python embeddings.

Picks GATE_N_CLASSES x GATE_PER_CLASS Flowers102 test images with a fixed seed, saves them as JPEG
files into the app's Resources (git-ignored: dataset license not verified), then embeds the SAVED
JPEG files with the Python pipeline, so the phone and the laptop start from identical bytes.

Writes results/gate_python_<backbone>.npz (X, y) and <GATE_DIR>/manifest.json.
"""
import json

import numpy as np
from PIL import Image
from torchvision import datasets

import config
from common import REPO_ROOT
from extract_embeddings import build_backbone, build_transform, pick_device
import torch


def main():
    ds = datasets.Flowers102(REPO_ROOT / config.DATA_DIR, split="test", download=False)
    labels = np.array(ds._labels)
    rng = np.random.default_rng(config.SEED)
    classes = np.sort(rng.choice(np.unique(labels), size=config.GATE_N_CLASSES, replace=False))
    out_dir = REPO_ROOT / config.GATE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("gate_*.jpg"):
        old.unlink()

    manifest = []
    for c in classes:
        for i in rng.choice(np.flatnonzero(labels == c), size=config.GATE_PER_CLASS, replace=False):
            name = f"gate_{len(manifest):02d}_c{int(c):03d}.jpg"
            ds[int(i)][0].convert("RGB").save(out_dir / name, quality=95)
            manifest.append(dict(file=name, label=int(c)))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    transform, device = build_transform(), pick_device()
    for backbone in config.BACKBONES:
        model = build_backbone(backbone, device)
        with torch.inference_mode():
            X = np.concatenate([model(transform(Image.open(out_dir / m["file"]).convert("RGB"))[None].to(device))
                                .float().cpu().numpy() for m in manifest])
        y = np.array([m["label"] for m in manifest])
        np.savez(REPO_ROOT / config.RESULTS_DIR / f"gate_python_{backbone}.npz", X=X, y=y)
        print(f"{backbone}: X {X.shape}, NaN {int(np.isnan(X).sum())}")
    print(f"{len(manifest)} images in {config.GATE_DIR}")


if __name__ == "__main__":
    main()
