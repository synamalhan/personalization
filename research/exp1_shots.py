"""Experiment 1: how many photos does a new class need?

For k in SHOTS, run N_EPISODES random episodes (EXP1_N_WAY classes, k support and EXP1_N_QUERY
query examples each) and report top-1 accuracy and macro-F1 with 95% CIs, for both heads on
every available dataset and backbone. Raw per-episode rows: results/exp1_shots.csv.
The same episodes feed experiment 2 (exp2_heads.py), which compares the heads.
"""
import argparse
import csv

import numpy as np

import config
from common import REPO_ROOT, load_embeddings, macro_f1, mean_ci
from episodes import sample_episode
from heads import HEADS


def run_episodes(X, y, k_shot, n_way, n_query, n_episodes):
    """Yield one row per (episode, head). Episode seeds are the same for every k and head."""
    for seed in range(n_episodes):
        ep = sample_episode(y, n_way, k_shot, n_query, seed)
        support = {c: X[ep["support_idx"][ep["support_y"] == c]] for c in ep["classes"]}
        for head_name, head_cls in HEADS.items():
            head = head_cls().fit(support)
            pred = head.labels[head.predict_proba(X[ep["query_idx"]]).argmax(axis=1)]
            yield dict(head=head_name, seed=seed,
                       accuracy=float((pred == ep["query_y"]).mean()),
                       macro_f1=macro_f1(ep["query_y"], pred, ep["classes"]))


def collect(args, quiet=False):
    rows = []
    for ds in args.datasets:
        for bb in args.backbones:
            data = load_embeddings(ds, bb, args.suffix)
            if data is None:
                if not quiet:
                    print(f"skip {ds}/{bb}: no embedding file for suffix '{args.suffix}'")
                continue
            X, y = data
            for k in args.shots:
                try:
                    for row in run_episodes(X, y, k, args.n_way, args.n_query, args.episodes):
                        rows.append(dict(dataset=ds, backbone=bb, k_shot=k, **row))
                except ValueError as e:
                    print(f"skip {ds}/{bb} k={k}: {e}")
    return rows


def add_args(p):
    p.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    p.add_argument("--backbones", nargs="+", default=list(config.BACKBONES))
    p.add_argument("--suffix", default="", help="embedding file suffix, e.g. _limit2000")
    p.add_argument("--shots", nargs="+", type=int, default=list(config.SHOTS))
    p.add_argument("--episodes", type=int, default=config.N_EPISODES)
    p.add_argument("--n-way", type=int, default=config.EXP1_N_WAY)
    p.add_argument("--n-query", type=int, default=config.EXP1_N_QUERY)


def write_csv(rows, name):
    out = REPO_ROOT / config.RESULTS_DIR / name
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(rows)} rows)\n")


def main():
    p = argparse.ArgumentParser()
    add_args(p)
    p.add_argument("--out", default="exp1_shots.csv")
    args = p.parse_args()
    rows = collect(args)
    if not rows:
        raise SystemExit("no data")
    write_csv(rows, args.out)

    print(f"Top-1 accuracy / macro-F1 (mean [95% CI], {args.episodes} episodes, {args.n_way}-way)")
    print(f"{'dataset':<11}{'backbone':<13}{'head':<20}{'k':<4}{'accuracy':<24}{'macro-F1':<24}")
    keys = sorted({(r["dataset"], r["backbone"], r["head"], r["k_shot"]) for r in rows})
    for ds, bb, head, k in keys:
        sel = [r for r in rows if (r["dataset"], r["backbone"], r["head"], r["k_shot"]) == (ds, bb, head, k)]
        cells = []
        for col in ("accuracy", "macro_f1"):
            m, lo, hi = mean_ci([r[col] for r in sel])
            cells.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"{ds:<11}{bb:<13}{head:<20}{k:<4}{cells[0]:<24}{cells[1]:<24}")


if __name__ == "__main__":
    main()
