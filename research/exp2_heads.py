"""Experiment 2: which head is better?

Same episodes for both heads, so we report the PAIRED difference (logistic - nearest class
mean) per episode with a 95% CI. A CI that excludes 0 means the heads really differ.
Raw per-episode rows: results/exp2_heads.csv (one row per episode with both heads' scores).
"""
import argparse

import config
from common import mean_ci
from exp1_shots import add_args, collect, write_csv


def main():
    p = argparse.ArgumentParser()
    add_args(p)
    p.add_argument("--out", default="exp2_heads.csv")
    args = p.parse_args()
    rows = collect(args)
    if not rows:
        raise SystemExit("no data")

    by_episode = {}
    for r in rows:
        key = (r["dataset"], r["backbone"], r["k_shot"], r["seed"])
        by_episode.setdefault(key, {})[r["head"]] = r
    paired = []
    for (ds, bb, k, seed), h in sorted(by_episode.items()):
        a, b = h["nearest_class_mean"], h["logistic"]
        paired.append(dict(dataset=ds, backbone=bb, k_shot=k, seed=seed,
                           ncm_accuracy=a["accuracy"], logistic_accuracy=b["accuracy"],
                           ncm_macro_f1=a["macro_f1"], logistic_macro_f1=b["macro_f1"],
                           diff_accuracy=b["accuracy"] - a["accuracy"],
                           diff_macro_f1=b["macro_f1"] - a["macro_f1"]))
    write_csv(paired, args.out)

    print(f"Logistic minus nearest class mean, paired over {args.episodes} episodes (mean [95% CI])")
    print(f"{'dataset':<11}{'backbone':<13}{'k':<4}{'d accuracy':<26}{'d macro-F1':<26}")
    for ds, bb, k in sorted({(r["dataset"], r["backbone"], r["k_shot"]) for r in paired}):
        sel = [r for r in paired if (r["dataset"], r["backbone"], r["k_shot"]) == (ds, bb, k)]
        cells = []
        for col in ("diff_accuracy", "diff_macro_f1"):
            m, lo, hi = mean_ci([r[col] for r in sel])
            cells.append(f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]")
        print(f"{ds:<11}{bb:<13}{k:<4}{cells[0]:<26}{cells[1]:<26}")


if __name__ == "__main__":
    main()
