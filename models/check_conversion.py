"""Compare Core ML embeddings with the original PyTorch model on a few hundred images, and
record what input the converted model expects.

    python models/check_conversion.py [--backbone mobilenetv3] [--n 300]

Images: seeded random sample of Flowers102 (test split) from research/data, preprocessed
exactly as research/config.py says (resize 256, center crop 224). The PyTorch path normalizes in
torch (ToTensor + Normalize); the Core ML path gets the cropped PIL image and normalizes inside the
model. Writes models/conversion_<backbone>.json.
"""
import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torchvision import datasets, transforms

from backbones import build_embedder, config
from convert import NAMES, OUT_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent


def cosine(a, b):
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="mobilenetv3", choices=list(NAMES))
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()

    crop = transforms.Compose([
        transforms.Resize(config.RESIZE_SHORT_SIDE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.CROP_SIZE),
    ])
    ds = datasets.Flowers102(REPO_ROOT / config.DATA_DIR, split="test", download=False)
    idx = np.random.default_rng(config.SEED).choice(len(ds), size=args.n, replace=False)
    images = [crop(ds[int(i)][0].convert("RGB")) for i in idx]

    # Reference: PyTorch fp32, normalization in torch exactly as in extract_embeddings.py.
    to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)])
    net = build_embedder(args.backbone)
    with torch.inference_mode():
        ref = np.concatenate([net.embedder(torch.stack([to_tensor(im) for im in images[i:i + 50]])).numpy()
                              for i in range(0, len(images), 50)])
    print(f"reference embeddings {ref.shape}")

    result = dict(backbone=args.backbone, coremltools=ct.__version__, torch=torch.__version__,
                  n_images=args.n, models={})
    for precision in ("fp16", "int8"):
        path = OUT_DIR / f"{NAMES[args.backbone]}_{precision}.mlpackage"
        model = ct.models.MLModel(str(path))
        out = np.stack([model.predict({"image": im})["embedding"].reshape(-1) for im in images])
        cos = cosine(ref, out)
        result["models"][precision] = dict(
            mean_cosine=float(cos.mean()), min_cosine=float(cos.min()),
            max_abs_diff=float(np.abs(ref - out).max()), nan=int(np.isnan(out).sum()))
        print(f"{precision}: mean cosine {cos.mean():.6f}  min {cos.min():.6f}  "
              f"max|diff| {np.abs(ref - out).max():.4f}  NaN {int(np.isnan(out).sum())}")

    spec = model.get_spec()
    inp = spec.description.input[0]
    result["expected_input"] = dict(
        name=inp.name, width=inp.type.imageType.width, height=inp.type.imageType.height,
        color_space="RGB", pixel_range="0-255 (no scaling by the caller)",
        normalization="inside the model: (pixel/255 - mean) / std with ImageNet mean/std",
        caller_must="resize shorter side to 256 (bilinear), center crop 224x224, RGB order",
        output=spec.description.output[0].name)
    (OUT_DIR / f"conversion_{args.backbone}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["expected_input"], indent=2))


if __name__ == "__main__":
    main()
