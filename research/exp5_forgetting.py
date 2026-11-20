"""Experiment 5: does adding classes break old ones?

Classes are added one at a time. After each addition we measure top-1 accuracy on held-out
query images of all EARLIER classes, in three modes:
  replay       head refit on all stored support embeddings (what the app does)
  newest_only  head fit only on the newest class (the Learner's replay=false). A head fit on one class
               can only predict that class, so accuracy on earlier classes is 0 by construction.
  sequential   no replay but a fair test: keep the old head and add the new class using only the new
               class's examples (Logistic: gradient steps on new data, which forgets; nearest class mean:
               the old means are kept, so it cannot forget).
Raw per-trial rows go to results/exp5_forgetting.csv.
"""
import argparse
import csv

import numpy as np

import config
from common import REPO_ROOT, eligible_classes, load_embeddings, mean_ci, split_support_query
from heads import HEADS


def run_trial(X, y, seed, n_classes, k_shot, n_query):
    rng = np.random.default_rng(seed)
    pool = eligible_classes(y, k_shot + n_query)
    classes = rng.choice(pool, size=n_classes, replace=False)  # order = order taught
    support_idx, query_idx = split_support_query(rng, y, classes, k_shot, n_query)
    rows = []
    for head_name, head_cls in HEADS.items():
        for mode in ("replay", "newest_only", "sequential"):
            head = None
            for t in range(1, n_classes + 1):
                taught = classes[:t]
                if mode == "replay":
                    head = head_cls().fit({c: X[support_idx[c]] for c in taught})
                elif mode == "newest_only":
                    head = head_cls().fit({taught[-1]: X[support_idx[taught[-1]]]})
                elif t == 1:
                    head = head_cls().fit({taught[0]: X[support_idx[taught[0]]]})
                else:
                    head = head.add_class(taught[-1], X[support_idx[taught[-1]]])
                if t < 2:
                    continue

                def accuracy(cs):
                    qi = np.concatenate([query_idx[c] for c in cs])
                    pred = head.labels[head.predict_proba(X[qi]).argmax(axis=1)]
                    return float((pred == y[qi]).mean())

                rows.append(dict(head=head_name, mode=mode, seed=seed, n_classes=t,
                                 acc_old=accuracy(taught[:-1]), acc_new=accuracy(taught[-1:]),
                                 acc_all=accuracy(taught)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    p.add_argument("--backbones", nargs="+", default=list(config.BACKBONES))
    p.add_argument("--suffix", default="", help="embedding file suffix, e.g. _limit2000")
    p.add_argument("--seeds", type=int, default=config.N_SEEDS)
    p.add_argument("--k-shot", type=int, default=config.K_SHOT)
    p.add_argument("--n-query", type=int, default=config.N_QUERY)
    p.add_argument("--n-classes", type=int, default=config.EXP5_N_CLASSES)
    p.add_argument("--out", default="exp5_forgetting.csv")
    args = p.parse_args()

    all_rows = []
    for ds in args.datasets:
        for bb in args.backbones:
            data = load_embeddings(ds, bb, args.suffix)
            if data is None:
                print(f"skip {ds}/{bb}: no embedding file for suffix '{args.suffix}'")
                continue
            X, y = data
            for seed in range(args.seeds):
                for row in run_trial(X, y, seed, args.n_classes, args.k_shot, args.n_query):
                    all_rows.append(dict(dataset=ds, backbone=bb, k_shot=args.k_shot, **row))

    if not all_rows:
        raise SystemExit("no data")
    out = REPO_ROOT / config.RESULTS_DIR / args.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(all_rows)} rows, {args.seeds} seeds)\n")

    # Summary: accuracy on earlier classes at the last step (all classes added).
    last = args.n_classes
    modes = ("replay", "newest_only", "sequential")
    print(f"Top-1 accuracy on EARLIER classes after adding class {last} (mean [95% CI], "
          f"{args.seeds} seeds, k={args.k_shot})")
    print(f"{'dataset':<11}{'backbone':<13}{'head':<20}" + "".join(f"{m:<22}" for m in modes))
    for ds in args.datasets:
        for bb in args.backbones:
            for head in HEADS:
                cells = []
                for mode in modes:
                    v = [r["acc_old"] for r in all_rows if r["dataset"] == ds and r["backbone"] == bb
                         and r["head"] == head and r["mode"] == mode and r["n_classes"] == last]
                    if not v:
                        break
                    m, lo, hi = mean_ci(v)
                    cells.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
                if len(cells) == len(modes):
                    print(f"{ds:<11}{bb:<13}{head:<20}" + "".join(f"{c:<22}" for c in cells))


if __name__ == "__main__":
    main()
