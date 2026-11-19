"""Experiment 5: does adding classes break old ones?

Classes are added one at a time. After each addition we measure top-1 accuracy on held-out
query images of all EARLIER classes, with replay (head refit on all stored support embeddings)
and without replay (head fit only on the newest class). Raw per-trial rows go to
results/exp5_forgetting.csv.

Note: a head fit on one class can only ever predict that class, so without replay accuracy on
earlier classes is 0 by construction for both heads.
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
        for replay in (True, False):
            for t in range(2, n_classes + 1):
                taught = classes[:t]
                fit_on = taught if replay else taught[-1:]
                head = head_cls().fit({c: X[support_idx[c]] for c in fit_on})

                def accuracy(cs):
                    qi = np.concatenate([query_idx[c] for c in cs])
                    pred = head.labels[head.predict_proba(X[qi]).argmax(axis=1)]
                    return float((pred == y[qi]).mean())

                rows.append(dict(head=head_name, replay=int(replay), seed=seed, n_classes=t,
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
    print(f"Top-1 accuracy on EARLIER classes after adding class {last} (mean [95% CI], "
          f"{args.seeds} seeds, k={args.k_shot})")
    print(f"{'dataset':<11}{'backbone':<13}{'head':<20}{'replay':<22}{'no replay':<22}")
    for ds in args.datasets:
        for bb in args.backbones:
            for head in HEADS:
                cells = []
                for replay in (1, 0):
                    v = [r["acc_old"] for r in all_rows if r["dataset"] == ds and r["backbone"] == bb
                         and r["head"] == head and r["replay"] == replay and r["n_classes"] == last]
                    if not v:
                        break
                    m, lo, hi = mean_ci(v)
                    cells.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
                if cells:
                    print(f"{ds:<11}{bb:<13}{head:<20}{cells[0]:<22}{cells[1]:<22}")


if __name__ == "__main__":
    main()
