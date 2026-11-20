"""Make the figures and results/summary.md from the raw CSVs in results/.

    python research/make_figures.py [--suffix _limit2000]

Reads exp1_shots, exp2_heads, exp5_forgetting, exp6_unknown (+ optional benchmark_release.csv from the
phone and models/conversion_*.json). Figures go to results/figures/. Anything whose input is missing is
skipped AND listed in summary.md as "not available", so gaps stay visible. Worst cases are kept in.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import REPO_ROOT, mean_ci

RESULTS = REPO_ROOT / config.RESULTS_DIR
FIGS = RESULTS / "figures"
# Validated categorical slots (blue, orange): color = backbone everywhere; line style / hatch = head.
BACKBONE_COLOR = {"mobilenetv3": "#2a78d6", "dinov2": "#eb6834", "dinov3": "#1baf7a"}
BACKBONE_NAME = {"mobilenetv3": "MobileNetV3", "dinov2": "DINOv2-small", "dinov3": "DINOv3-small"}
HEAD_STYLE = {"nearest_class_mean": "-", "logistic": "--"}
HEAD_NAME = {"nearest_class_mean": "nearest class mean", "logistic": "logistic"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True, "font.size": 9,
                     "savefig.dpi": 160, "savefig.bbox": "tight"})


def read(name, suffix):
    path = RESULTS / f"{name}{suffix}.csv"
    return pd.read_csv(path) if path.exists() else None


def ci_table(df, keys, col):
    rows = []
    for k, g in df.groupby(keys):
        m, lo, hi = mean_ci(g[col].values)
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), mean=m, lo=lo, hi=hi, n=len(g)))
    return pd.DataFrame(rows)


def fig_shots(exp1, suffix):
    t = ci_table(exp1, ["dataset", "backbone", "head", "k_shot"], "accuracy")
    datasets = sorted(t.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for (bb, head), g in t[t.dataset == ds].groupby(["backbone", "head"]):
            g = g.sort_values("k_shot")
            ax.plot(g.k_shot, g["mean"], HEAD_STYLE[head], color=BACKBONE_COLOR[bb], lw=2,
                    marker="o", ms=4, label=f"{BACKBONE_NAME[bb]}, {HEAD_NAME[head]}")
            ax.fill_between(g.k_shot, g.lo, g.hi, color=BACKBONE_COLOR[bb], alpha=0.15, lw=0)
        ax.set_xscale("log"); ax.set_xticks(sorted(t.k_shot.unique())); ax.set_xticklabels(sorted(t.k_shot.unique()))
        ax.set_title(f"{ds}: top-1 accuracy vs photos per class", loc="left"); ax.set_xlabel("photos per class (k)")
        ax.set_ylabel("top-1 accuracy (mean, 95% CI)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIGS / "fig1_accuracy_vs_shots.png"); plt.close(fig)


def fig_forgetting(exp5, suffix):
    t = ci_table(exp5, ["dataset", "backbone", "head", "mode", "n_classes"], "acc_old")
    panels = sorted(set(zip(t.dataset, t.backbone)))
    mode_color = {"replay": "#2a78d6", "sequential": "#eb6834", "newest_only": "#1baf7a"}
    mode_name = {"replay": "replay (refit on all)", "sequential": "sequential, no replay",
                 "newest_only": "fit on newest class only"}
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8), squeeze=False, sharey=True)
    for ax, (ds, bb) in zip(axes[0], panels):
        for (head, mode), g in t[(t.dataset == ds) & (t.backbone == bb)].groupby(["head", "mode"]):
            g = g.sort_values("n_classes")
            ax.plot(g.n_classes, g["mean"], HEAD_STYLE[head], color=mode_color[mode], lw=2,
                    label=f"{mode_name[mode]}, {HEAD_NAME[head]}")
            ax.fill_between(g.n_classes, g.lo, g.hi, color=mode_color[mode], alpha=0.15, lw=0)
        ax.set_title(f"{ds} / {BACKBONE_NAME[bb]}", loc="left"); ax.set_xlabel("classes taught")
    axes[0][0].set_ylabel("accuracy on earlier classes")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    fig.suptitle("Forgetting. Nearest class mean: sequential equals replay (its means never change). "
                 "Logistic: sequential forgets.", x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.savefig(FIGS / "fig2_forgetting.png"); plt.close(fig)


def fig_unknown(exp6):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, col, title in zip(axes, ["auroc", "ece"], ["AUROC: unseen vs taught (higher is better)", "ECE, 15 bins (lower is better)"]):
        t = ci_table(exp6[exp6[col].notna()], ["dataset", "backbone", "head"], col)
        heads = ["nearest_class_mean", "logistic"] + (["cosine_rule"] if col == "auroc" else [])
        hatch = {"nearest_class_mean": "", "logistic": "////", "cosine_rule": "...."}
        groups = sorted(set(zip(t.dataset, t.backbone)))
        w = 0.8 / len(heads)
        for gi, (ds, bb) in enumerate(groups):
            for hi_, head in enumerate(heads):
                r = t[(t.dataset == ds) & (t.backbone == bb) & (t["head"] == head)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = gi + (hi_ - (len(heads) - 1) / 2) * w
                ax.bar(x, r["mean"], w * 0.9, color=BACKBONE_COLOR[bb], hatch=hatch[head], edgecolor=SURFACE, lw=0,
                       label=f"{HEAD_NAME.get(head, 'cosine threshold rule')}" if gi == 0 else None)
                ax.errorbar(x, r["mean"], yerr=[[r["mean"] - r.lo], [r.hi - r["mean"]]], color=INK, lw=1, capsize=2)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"{d}\n{BACKBONE_NAME[b]}" for d, b in groups])
        ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
    # Legend entries for the hatch encoding in neutral ink.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#9a9992", hatch=h, edgecolor=SURFACE)
               for h in ["", "////", "...."]]
    axes[0].legend(handles, ["nearest class mean", "logistic", "cosine threshold rule"], frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3)
    fig.savefig(FIGS / "fig3_unknown_auroc_ece.png"); plt.close(fig)


def fig_latency(exp1, suffix):
    path = RESULTS / "benchmark_release.csv"
    if not path.exists():
        return False
    bench = pd.read_csv(path)
    bench = bench[bench.metric.str.startswith("embed_")]
    med = bench.groupby("metric").seconds.median() * 1000
    acc = exp1[(exp1.k_shot == 5)].groupby("backbone").accuracy.mean()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for bb in acc.index:
        if f"embed_{bb}" in med:
            ax.scatter(med[f"embed_{bb}"], acc[bb], s=60, color=BACKBONE_COLOR[bb], label=BACKBONE_NAME[bb])
    ax.set_xlabel("median embedding latency on phone (ms)"); ax.set_ylabel("top-1 accuracy at k=5")
    ax.legend(frameon=False); ax.set_title("Accuracy vs latency", loc="left")
    fig.savefig(FIGS / "fig4_accuracy_vs_latency.png"); plt.close(fig)
    return True


def load_conversion():
    out = {}
    for bb in config.BACKBONES:
        p = REPO_ROOT / "models" / f"conversion_{bb}.json"
        if p.exists():
            out[bb] = json.loads(p.read_text())
    return out


def fig_precision(conv):
    if not conv:
        return False
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ylabels, y = [], 0
    for bb, d in conv.items():
        for prec in ("fp16", "int8"):
            m = d["models"][prec]
            ax.plot([m["min_cosine"], m["mean_cosine"]], [y, y], color=BACKBONE_COLOR[bb], lw=2)
            ax.scatter([m["mean_cosine"]], [y], color=BACKBONE_COLOR[bb], s=36, zorder=3)
            ax.scatter([m["min_cosine"]], [y], color=BACKBONE_COLOR[bb], s=36, marker="x", zorder=3)
            ylabels.append(f"{BACKBONE_NAME[bb]} {prec}"); y += 1
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels); ax.invert_yaxis()
    ax.set_xlabel("cosine similarity to PyTorch embedding (dot = mean, x = worst image)")
    ax.set_title(f"Core ML fp16 vs int8 ({next(iter(conv.values()))['n_images']} images)", loc="left")
    fig.savefig(FIGS / "fig5_fp16_vs_int8.png"); plt.close(fig)
    return True


def fmt(m, lo, hi):
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def write_summary(exp1, exp2, exp5, exp6, conv, suffix, have_latency):
    L = ["# Results summary", "",
         f"Source CSVs: `*{suffix}.csv` in `results/`. " + (
             "**These were run on a 2000-image random subset of Flowers102 only (about 20 images per class), "
             "not the full datasets, and Pets was not run.** Treat the numbers as a preview." if suffix else ""),
         "All intervals are 95% CIs (t-interval across independent episodes or seeds).", ""]
    ds_list = sorted(exp1.dataset.unique())
    L += ["## Exp 1: top-1 accuracy vs photos per class", f"Episodes per cell: {exp1.groupby(['dataset','backbone','head','k_shot']).size().iloc[0]}; "
          f"{config.EXP1_N_WAY}-way.", "", "| dataset | backbone | head | k | accuracy | macro-F1 |", "| --- | --- | --- | --- | --- | --- |"]
    a, f = ci_table(exp1, ["dataset", "backbone", "head", "k_shot"], "accuracy"), ci_table(exp1, ["dataset", "backbone", "head", "k_shot"], "macro_f1")
    for (_, ra), (_, rf) in zip(a.iterrows(), f.iterrows()):
        L.append(f"| {ra.dataset} | {BACKBONE_NAME[ra.backbone]} | {HEAD_NAME[ra['head']]} | {ra.k_shot} | "
                 f"{fmt(ra['mean'], ra.lo, ra.hi)} | {fmt(rf['mean'], rf.lo, rf.hi)} |")
    worst = a.sort_values("mean").iloc[0]
    L += ["", f"**Worst case kept in:** {BACKBONE_NAME[worst.backbone]} / {HEAD_NAME[worst['head']]} at k={worst.k_shot}: "
          f"accuracy {fmt(worst['mean'], worst.lo, worst.hi)}. At k=20 with `--n-query 5` only a few classes are eligible, "
          "so episodes reuse the same classes and the intervals are optimistic.", ""]

    L += ["## Exp 2: logistic minus nearest class mean (paired by episode)", "",
          "| dataset | backbone | k | d accuracy | d macro-F1 |", "| --- | --- | --- | --- | --- |"]
    d1, d2 = ci_table(exp2, ["dataset", "backbone", "k_shot"], "diff_accuracy"), ci_table(exp2, ["dataset", "backbone", "k_shot"], "diff_macro_f1")
    for (_, r1), (_, r2) in zip(d1.iterrows(), d2.iterrows()):
        L.append(f"| {r1.dataset} | {BACKBONE_NAME[r1.backbone]} | {r1.k_shot} | {r1['mean']:+.3f} [{r1.lo:+.3f}, {r1.hi:+.3f}] | "
                 f"{r2['mean']:+.3f} [{r2.lo:+.3f}, {r2.hi:+.3f}] |")
    L.append("")

    last = exp5.n_classes.max()
    L += [f"## Exp 5: accuracy on earlier classes after adding class {last}", "",
          "| dataset | backbone | head | replay | sequential, no replay | newest class only |", "| --- | --- | --- | --- | --- | --- |"]
    t5 = ci_table(exp5[exp5.n_classes == last], ["dataset", "backbone", "head", "mode"], "acc_old")
    for (ds, bb, head), g in t5.groupby(["dataset", "backbone", "head"]):
        cells = [fmt(*g[g["mode"] == m].iloc[0][["mean", "lo", "hi"]]) for m in ("replay", "sequential", "newest_only")]
        L.append(f"| {ds} | {BACKBONE_NAME[bb]} | {HEAD_NAME[head]} | " + " | ".join(cells) + " |")
    L += ["", "Sequential = keep the old head and add the new class from its own examples only. Nearest class mean cannot forget "
          "(its class means do not change), so it equals replay. Logistic forgets: training on the new class alone pushes old logits down. "
          "'Newest class only' is 0 by construction (a head fit on one class can only predict that class); it is the Learner's `replay = false` setting.", ""]

    L += ["## Exp 6: unknown detection (10 taught / 10 unseen classes)", "",
          "| dataset | backbone | head | AUROC | ECE (15 bins) | top-1 on taught |", "| --- | --- | --- | --- | --- | --- |"]
    keys = ["dataset", "backbone", "head"]
    au, ec, ac = (ci_table(exp6[exp6[c].notna()], keys, c) for c in ("auroc", "ece", "acc_known"))
    for _, r in au.iterrows():
        e = ec[(ec.dataset == r.dataset) & (ec.backbone == r.backbone) & (ec["head"] == r["head"])]
        c = ac[(ac.dataset == r.dataset) & (ac.backbone == r.backbone) & (ac["head"] == r["head"])]
        L.append(f"| {r.dataset} | {BACKBONE_NAME[r.backbone]} | {HEAD_NAME.get(r['head'], 'cosine threshold rule')} | {fmt(r['mean'], r.lo, r.hi)} | "
                 f"{fmt(*e.iloc[0][['mean', 'lo', 'hi']]) if len(e) else 'n/a'} | {fmt(*c.iloc[0][['mean', 'lo', 'hi']]) if len(c) else 'n/a'} |")
    rule = exp6[exp6["head"] == "cosine_rule"]
    L += ["", "Operating point of the PersonalizeCore threshold rule (max cosine to class means; threshold = mean - 2 std of leave-one-out similarities):", "",
          "| dataset | backbone | threshold | taught accepted | unseen rejected |", "| --- | --- | --- | --- | --- |"]
    for (ds, bb), g in rule.groupby(["dataset", "backbone"]):
        L.append(f"| {ds} | {BACKBONE_NAME[bb]} | {g.threshold.mean():.3f} | {fmt(*mean_ci(g.known_accepted))} | {fmt(*mean_ci(g.unknown_rejected))} |")
    worst6 = au.sort_values("mean").iloc[0]
    L += ["", f"**Worst case kept in:** lowest AUROC is {BACKBONE_NAME[worst6.backbone]} / {HEAD_NAME.get(worst6['head'], 'cosine threshold rule')}: "
          f"{fmt(worst6['mean'], worst6.lo, worst6.hi)}. MobileNetV3 nearest-class-mean ECE is large because its cosines are compressed and the "
          "softmax temperature (0.1) was not tuned.", ""]

    L += ["## Exp 4: Core ML fp16 vs int8 (embedding cosine vs PyTorch)", ""]
    if conv:
        L += ["| backbone | precision | mean cosine | worst image | max abs diff |", "| --- | --- | --- | --- | --- |"]
        for bb, d in conv.items():
            for prec in ("fp16", "int8"):
                m = d["models"][prec]
                L.append(f"| {BACKBONE_NAME[bb]} | {prec} | {m['mean_cosine']:.4f} | {m['min_cosine']:.4f} | {m['max_abs_diff']:.3f} |")
        L += ["", f"{conv[next(iter(conv))]['n_images']} Flowers102 test images, coremltools {conv[next(iter(conv))]['coremltools']}. "
              "This measures embedding similarity only; few-shot accuracy with int8 embeddings and latency were **not** measured.", ""]
    else:
        L += ["not available", ""]

    L += ["## Gate check: phone vs laptop embeddings", ""]
    gates = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("gate_comparison_*.json"))]
    if gates:
        L += ["| backbone | device / build | images | mean cosine | worst image | same predicted class (nearest mean / logistic) | max probability difference |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for g in gates:
            h = g["heads"]
            real = g["device"].startswith("iPhone")
            L.append(f"| {BACKBONE_NAME[g['backbone']]} | {g['device']} / {g['build']} ({'real iPhone' if real else 'simulator'}) | {g['n_images']} | {g['mean_cosine']:.4f} | "
                     f"{g['min_cosine']:.4f} | {h['nearest_class_mean']['same_class']:.0%} / {h['logistic']['same_class']:.0%} | "
                     f"{max(h['nearest_class_mean']['max_prob_diff'], h['logistic']['max_prob_diff']):.3f} |")
        if any(g["device"].startswith("iPhone") for g in gates):
            L += ["", "Run on a real iPhone (Debug build, default Core ML compute units). The gap to the 0.999 target is most likely the "
                  "resize method (Core Graphics vs Pillow); the simulator, run earlier with CPU-only Core ML, gave nearly the same number "
                  "(0.9947), which points at preprocessing rather than the compute unit. Not yet run for the DINO models.", ""]
        else:
            L += ["", "Run in the iOS Simulator with CPU-only Core ML (the simulator returned all zeros with the default compute units), "
                  "so this is **not** a phone result. Not yet run on a real iPhone.", ""]
    else:
        L += ["not available", ""]

    L += ["## Not available (not run yet)", "",
          "- Exp 3 (accuracy vs latency) and Exp 7 (on-device cost): need the Release benchmark CSV from a real iPhone" + (" (present)" if have_latency else " (missing)") + ".",
          "- Exp 8 (real photos): needs `exp8_real.py` run on the app's exported vectors; none exist.",
          "- Gate check for the DINOv2 and DINOv3 models on a real iPhone (only MobileNetV3 was run).",
          "- Pets dataset and the full Flowers102 run for every experiment above.", ""]
    (RESULTS / "summary.md").write_text("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default=None)
    args = p.parse_args()
    suffix = args.suffix if args.suffix is not None else ("" if (RESULTS / "exp1_shots.csv").exists() else "_limit2000")
    FIGS.mkdir(parents=True, exist_ok=True)
    exp1, exp2, exp5, exp6 = (read(n, suffix) for n in ("exp1_shots", "exp2_heads", "exp5_forgetting", "exp6_unknown"))
    missing = [n for n, d in zip(("exp1", "exp2", "exp5", "exp6"), (exp1, exp2, exp5, exp6)) if d is None]
    if missing:
        raise SystemExit(f"missing CSVs for suffix '{suffix}': {missing}")
    conv = load_conversion()
    fig_shots(exp1, suffix); fig_forgetting(exp5, suffix); fig_unknown(exp6)
    have_latency = fig_latency(exp1, suffix)
    fig_precision(conv)
    write_summary(exp1, exp2, exp5, exp6, conv, suffix, have_latency)
    print("wrote", [str(p.relative_to(REPO_ROOT)) for p in sorted(FIGS.glob("*.png"))], "and results/summary.md")


if __name__ == "__main__":
    main()
