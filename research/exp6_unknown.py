"""Experiment 6: can it say unknown?

Some classes are taught (k support images each); other classes are held out as "unknown".
Query images from both groups are scored. Reported per head:
  - AUROC of max-probability for separating taught-class queries (positive) from unseen-class
    queries. 0.5 = chance.
  - ECE (15 bins) of max-probability on the taught-class queries: confidence vs. top-1 accuracy.
And for PersonalizeCore's threshold rule (max cosine to the class means, threshold =
mean - 2*std of the taught classes' own leave-one-out similarities): AUROC of the cosine score,
the fraction of taught queries accepted and of unseen queries rejected at that threshold.
Raw per-trial rows go to results/exp6_unknown.csv.
"""
import argparse
import csv

import numpy as np

import config
from common import REPO_ROOT, auroc, ece, eligible_classes, load_embeddings, mean_ci, split_support_query
from heads import HEADS, NearestClassMean, suggested_threshold


def run_trial(X, y, seed, n_taught, n_unknown, k_shot, n_query):
    rng = np.random.default_rng(seed)
    pool = eligible_classes(y, k_shot + n_query)
    chosen = rng.choice(pool, size=n_taught + n_unknown, replace=False)
    taught, unseen = chosen[:n_taught], chosen[n_taught:]
    support_idx, query_idx = split_support_query(rng, y, taught, k_shot, n_query)
    _, unseen_idx = split_support_query(rng, y, unseen, 0, n_query)
    support = {c: X[support_idx[c]] for c in taught}
    known_q = np.concatenate([query_idx[c] for c in taught])
    unknown_q = np.concatenate([unseen_idx[c] for c in unseen])
    base = dict(seed=seed)
    rows = []

    for head_name, head_cls in HEADS.items():
        head = head_cls().fit(support)
        p_known = head.predict_proba(X[known_q])
        p_unknown = head.predict_proba(X[unknown_q])
        conf = p_known.max(axis=1)
        correct = head.labels[p_known.argmax(axis=1)] == y[known_q]
        rows.append(dict(base, head=head_name, auroc=auroc(conf, p_unknown.max(axis=1)),
                         ece=ece(conf, correct), acc_known=float(correct.mean()),
                         threshold="", known_accepted="", unknown_rejected=""))

    # The threshold rule is on cosine similarity to the class means, independent of the head.
    ncm = NearestClassMean().fit(support)
    cos_known, cos_unknown = ncm.cosine(X[known_q]).max(axis=1), ncm.cosine(X[unknown_q]).max(axis=1)
    thr = suggested_threshold(support)
    rows.append(dict(base, head="cosine_rule", auroc=auroc(cos_known, cos_unknown), ece="", acc_known="",
                     threshold=thr, known_accepted=float((cos_known >= thr).mean()),
                     unknown_rejected=float((cos_unknown < thr).mean())))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    p.add_argument("--backbones", nargs="+", default=list(config.BACKBONES))
    p.add_argument("--suffix", default="", help="embedding file suffix, e.g. _limit2000")
    p.add_argument("--seeds", type=int, default=config.N_SEEDS)
    p.add_argument("--k-shot", type=int, default=config.K_SHOT)
    p.add_argument("--n-query", type=int, default=config.N_QUERY)
    p.add_argument("--n-taught", type=int, default=config.EXP6_N_TAUGHT)
    p.add_argument("--n-unknown", type=int, default=config.EXP6_N_UNKNOWN)
    p.add_argument("--out", default="exp6_unknown.csv")
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
                for row in run_trial(X, y, seed, args.n_taught, args.n_unknown, args.k_shot, args.n_query):
                    all_rows.append(dict(dataset=ds, backbone=bb, k_shot=args.k_shot, **row))

    if not all_rows:
        raise SystemExit("no data")
    out = REPO_ROOT / config.RESULTS_DIR / args.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(all_rows)} rows, {args.seeds} seeds)\n")

    def cell(ds, bb, head, col):
        v = [float(r[col]) for r in all_rows if r["dataset"] == ds and r["backbone"] == bb
             and r["head"] == head and r[col] != ""]
        m, lo, hi = mean_ci(v)
        return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

    print(f"Unknown detection, {args.n_taught} taught / {args.n_unknown} unseen classes, k={args.k_shot} "
          f"(mean [95% CI], {args.seeds} seeds)")
    print(f"{'dataset':<11}{'backbone':<13}{'head':<20}{'AUROC':<22}{'ECE':<22}{'top-1 (taught)':<22}")
    for ds in args.datasets:
        for bb in args.backbones:
            if not any(r["dataset"] == ds and r["backbone"] == bb for r in all_rows):
                continue
            for head in HEADS:
                print(f"{ds:<11}{bb:<13}{head:<20}{cell(ds, bb, head, 'auroc'):<22}"
                      f"{cell(ds, bb, head, 'ece'):<22}{cell(ds, bb, head, 'acc_known'):<22}")
            print(f"{ds:<11}{bb:<13}{'cosine_rule':<20}{cell(ds, bb, 'cosine_rule', 'auroc'):<22}"
                  f"  threshold {cell(ds, bb, 'cosine_rule', 'threshold')}")
            print(f"{'':<44}taught accepted {cell(ds, bb, 'cosine_rule', 'known_accepted')}   "
                  f"unseen rejected {cell(ds, bb, 'cosine_rule', 'unknown_rejected')}")


if __name__ == "__main__":
    main()
