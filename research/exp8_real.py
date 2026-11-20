"""Experiment 8: real photos taken on different days.

Input: the JSON exported by the app (Settings > Lab > Real-photo test):
    {"backbone": "...", "items": [{"class": str, "day": str, "embedding": [...]}, ...]}
Only embeddings are used; keep personal photos out of the repo.

Protocol (leave-one-day-out): for each day D, TRAIN on all photos from the other days and TEST on the
photos from D. Both heads are evaluated. Reports top-1 accuracy and macro-F1 (mean over held-out days
with a 95% CI across days and over bootstrap resamples of the test photos), and unknown detection:
each class in turn is held out as "unknown" (not taught) and AUROC separates its test photos from
the taught classes' test photos, using max probability. Raw per-fold rows: results/exp8_real.csv.

    python research/exp8_real.py path/to/real_photos_export.json [--out exp8_real.csv]
"""
import argparse
import csv
import json

import numpy as np

import config
from common import REPO_ROOT, auroc, macro_f1, mean_ci
from heads import HEADS


def bootstrap_ci(values, n=2000, seed=0):
    """95% percentile bootstrap CI of the mean of per-photo correctness (for small samples)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(path):
    data = json.load(open(path))
    items = data["items"]
    X = np.array([it["embedding"] for it in items], dtype=np.float32)
    return data.get("backbone", "?"), X, np.array([it["class"] for it in items]), np.array([it["day"] for it in items])


def fit(head_cls, X, y, classes):
    return head_cls().fit({c: X[y == c] for c in classes})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("export_json")
    p.add_argument("--out", default="exp8_real.csv")
    args = p.parse_args()
    backbone, X, y, day = load(args.export_json)
    classes, days = np.unique(y), np.unique(day)
    print(f"backbone {backbone}: {len(X)} photos, {len(classes)} classes, {len(days)} days {list(days)}")
    if len(days) < 2:
        raise SystemExit("need photos from at least 2 different days")

    rows, correct_by_head = [], {h: [] for h in HEADS}
    for held in days:
        train, test = day != held, day == held
        # Only classes seen in training can be predicted; test photos of other classes are skipped.
        taught = [c for c in classes if (y[train] == c).sum() >= 2]
        keep = test & np.isin(y, taught)
        if keep.sum() == 0 or len(taught) < 2:
            print(f"  skip day {held}: not enough training data")
            continue
        for name, head_cls in HEADS.items():
            head = fit(head_cls, X[train], y[train], taught)
            pred = head.labels[head.predict_proba(X[keep]).argmax(axis=1)]
            correct_by_head[name] += list(pred == y[keep])
            rows.append(dict(fold=f"holdout_{held}", head=name, n_test=int(keep.sum()),
                             accuracy=float((pred == y[keep]).mean()),
                             macro_f1=macro_f1(y[keep], pred, taught), auroc="", ))

    # Unknown detection: hold each class out as "unknown" (all days); train on the rest of the days' photos.
    for unknown in classes:
        taught = [c for c in classes if c != unknown and (y == c).sum() >= 2]
        if len(taught) < 2:
            continue
        for held in days:
            train = (day != held) & np.isin(y, taught)
            known_q = (day == held) & np.isin(y, taught)
            unk_q = (day == held) & (y == unknown)
            if known_q.sum() == 0 or unk_q.sum() == 0 or len(np.unique(y[train])) < 2:
                continue
            for name, head_cls in HEADS.items():
                head = fit(head_cls, X[train], y[train], list(np.unique(y[train])))
                a = auroc(head.predict_proba(X[known_q]).max(axis=1), head.predict_proba(X[unk_q]).max(axis=1))
                rows.append(dict(fold=f"unknown_{unknown}_holdout_{held}", head=name, n_test=int(known_q.sum() + unk_q.sum()),
                                 accuracy="", macro_f1="", auroc=a))

    out = REPO_ROOT / config.RESULTS_DIR / args.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "head", "n_test", "accuracy", "macro_f1", "auroc"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(rows)} rows)\n")

    print("Leave-one-day-out (mean over held-out days [95% CI across days]; photo-level bootstrap CI)")
    for name in HEADS:
        acc = [r["accuracy"] for r in rows if r["head"] == name and r["accuracy"] != ""]
        f1 = [r["macro_f1"] for r in rows if r["head"] == name and r["macro_f1"] != ""]
        au = [r["auroc"] for r in rows if r["head"] == name and r["auroc"] != ""]
        m, lo, hi = mean_ci(acc)
        _, blo, bhi = bootstrap_ci(correct_by_head[name]) if correct_by_head[name] else (0, float("nan"), float("nan"))
        line = f"  {name:<20} accuracy {m:.3f} [{lo:.3f},{hi:.3f}] (bootstrap [{blo:.3f},{bhi:.3f}], {len(correct_by_head[name])} test photos)"
        mf, flo, fhi = mean_ci(f1)
        line += f"  macro-F1 {mf:.3f} [{flo:.3f},{fhi:.3f}]"
        if au:
            ma, alo, ahi = mean_ci(au)
            line += f"  unknown AUROC {ma:.3f} [{alo:.3f},{ahi:.3f}] over {len(au)} folds"
        print(line)


if __name__ == "__main__":
    main()
