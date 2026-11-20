"""Convert a backbone to Core ML .mlpackage files at fp16 and int8.

    python models/convert.py [--backbone mobilenetv3|dinov2]

Outputs (git-ignored; weights may carry license limits):
    models/<Name>_fp16.mlpackage   weights and compute in float16
    models/<Name>_int8.mlpackage   weights quantized to int8 (linear, symmetric, per-channel),
                                   compute still float16 (weight-only quantization)

Model input: "image", 224x224 RGB. Model output: "embedding".
Verified against coremltools 9.0 docs/signatures; see check_conversion.py for the measured results.
"""
import argparse
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import torch

from backbones import build_embedder

OUT_DIR = Path(__file__).resolve().parent
NAMES = {"mobilenetv3": "MobileNetV3Large", "dinov2": "DINOv2Small"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="mobilenetv3", choices=list(NAMES))
    args = p.parse_args()
    print(f"coremltools {ct.__version__}, torch {torch.__version__}")

    model = build_embedder(args.backbone)
    example = torch.rand(1, 3, 224, 224) * 255
    traced = torch.jit.trace(model, example)

    fp16 = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1.0,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    fp16_path = OUT_DIR / f"{NAMES[args.backbone]}_fp16.mlpackage"
    fp16.save(str(fp16_path))
    print("saved", fp16_path.name)

    config = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    int8 = cto.linear_quantize_weights(fp16, config=config)
    int8_path = OUT_DIR / f"{NAMES[args.backbone]}_int8.mlpackage"
    int8.save(str(int8_path))
    print("saved", int8_path.name)


if __name__ == "__main__":
    main()
