"""Gate check: compare on-device embeddings with the Python embeddings of the same 20 images.

    python research/compare_device.py path/to/gate_device_mobilenetv3.json [more json files]

Reports cosine similarity per image (target: ~1.0 for fp16), then fits both heads on 2 images per
class from the PYTHON embeddings and compares their predictions for all 20 images using Python vs
device embeddings. Python embeddings: results/gate_python_<backbone>.npz (prepare_gate_images.py).
"""
import argparse
import json

import numpy as np

import config
from common import REPO_ROOT
from heads import HEADS

# Likely causes if the numbers do not match, most likely first.
CAUSES = ["resize method (bilinear/antialiasing differs between Pillow and Core Graphics)",
          "color order or color space (RGB vs BGR; Display P3 vs sRGB decoding)",
          "normalization constants (mean/std, or the 0-255 vs 0-1 scale)",
          "crop (center-crop offset or rounding; wrong orientation of the photo)"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("device_json", nargs="+")
    args = p.parse_args()
    manifest = json.loads((REPO_ROOT / config.GATE_DIR / "manifest.json").read_text())
    files = [m["file"] for m in manifest]

    for path in args.device_json:
        dev = json.load(open(path))
        backbone = dev["backbone"]
        with np.load(REPO_ROOT / config.RESULTS_DIR / f"gate_python_{backbone}.npz") as f:
            Xp, y = f["X"], f["y"]
        by_file = {it["file"]: np.array(it["embedding"], dtype=np.float32) for it in dev["items"]}
        missing = [f for f in files if f not in by_file]
        if missing:
            print(f"{backbone}: device file is missing {len(missing)} images, e.g. {missing[:3]}")
            continue
        Xd = np.stack([by_file[f] for f in files])
        cos = (Xp * Xd).sum(1) / (np.linalg.norm(Xp, axis=1) * np.linalg.norm(Xd, axis=1))
        print(f"\n== {backbone} ({dev['device']}, {dev['build']} build, "
              f"{dev['seconds_per_image'] * 1000:.0f} ms/image) ==")
        for f, c in zip(files, cos):
            print(f"  {f}  cosine {c:.6f}")
        print(f"  mean {cos.mean():.6f}  min {cos.min():.6f}  max {cos.max():.6f}   "
              f"norm ratio device/python {np.mean(np.linalg.norm(Xd, axis=1) / np.linalg.norm(Xp, axis=1)):.4f}")

        # Heads: fit on 2 images per class (Python embeddings), predict all 20 both ways.
        support_idx = np.concatenate([np.flatnonzero(y == c)[:2] for c in np.unique(y)])
        support = {c: Xp[[i for i in support_idx if y[i] == c]] for c in np.unique(y)}
        for name, head_cls in HEADS.items():
            head = head_cls().fit(support)
            pp, pd = head.predict_proba(Xp), head.predict_proba(Xd)
            same = (pp.argmax(1) == pd.argmax(1)).mean()
            acc_p = (head.labels[pp.argmax(1)] == y).mean()
            acc_d = (head.labels[pd.argmax(1)] == y).mean()
            print(f"  {name}: same predicted class {same:.0%}, accuracy python {acc_p:.0%} device {acc_d:.0%}, "
                  f"max |probability diff| {np.abs(pp - pd).max():.4f}")
        summary = dict(backbone=backbone, device=dev["device"], build=dev["build"], n_images=len(files),
                       mean_cosine=float(cos.mean()), min_cosine=float(cos.min()),
                       heads={name: None for name in HEADS})
        for name, head_cls in HEADS.items():
            head = head_cls().fit(support)
            pp, pd = head.predict_proba(Xp), head.predict_proba(Xd)
            summary["heads"][name] = dict(same_class=float((pp.argmax(1) == pd.argmax(1)).mean()),
                                          max_prob_diff=float(np.abs(pp - pd).max()))
        (REPO_ROOT / config.RESULTS_DIR / f"gate_comparison_{backbone}.json").write_text(json.dumps(summary, indent=2))
        if cos.mean() < 0.999:
            print("  Below target. Likely causes, in order: " + "; ".join(f"{i + 1}) {c}" for i, c in enumerate(CAUSES)))


if __name__ == "__main__":
    main()
