"""Export parity.json: stored embeddings, queries, and the Python heads' outputs, so the
Swift heads can be checked against them.

Usage:
    python research/export_fixture.py [--dataset flowers102] [--backbone dinov2] [--suffix _limit2000]

All vectors are float32 (written as the exact float32 value). Class names are zero-padded
("c012") so Swift's string sort and Python's integer sort give the same class order.
"""
import argparse
import json

import numpy as np

import config
from common import REPO_ROOT, eligible_classes, load_embeddings
from heads import Logistic, NearestClassMean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="flowers102", choices=config.DATASETS)
    p.add_argument("--backbone", default="dinov2", choices=config.BACKBONES)
    p.add_argument("--suffix", default="_limit2000", help="embedding file suffix ('' for the full run)")
    p.add_argument("--out", default=config.FIXTURE_PATH, help="path relative to the repo root")
    args = p.parse_args()

    data = load_embeddings(args.dataset, args.backbone, args.suffix)
    if data is None:
        raise SystemExit(f"no embedding file for {args.dataset}/{args.backbone}{args.suffix}")
    X, y = data
    X = X.astype(np.float32)

    rng = np.random.default_rng(config.FIXTURE_SEED)
    need = config.FIXTURE_K_SHOT + config.FIXTURE_N_QUERY_TAUGHT
    taught = np.sort(rng.choice(eligible_classes(y, need), size=config.FIXTURE_N_CLASSES, replace=False))
    others = np.setdiff1d(np.unique(y), taught)

    support, query_idx = {}, []
    for c in taught:
        idx = rng.permutation(np.flatnonzero(y == c))
        support[int(c)] = X[idx[:config.FIXTURE_K_SHOT]]
        query_idx += idx[config.FIXTURE_K_SHOT:need].tolist()
    for c in rng.choice(others, size=config.FIXTURE_N_QUERY_OTHER, replace=False):
        query_idx.append(int(rng.choice(np.flatnonzero(y == c))))
    Q = X[query_idx]

    name = lambda c: f"c{int(c):03d}"
    ncm = NearestClassMean().fit(support)
    logistic = Logistic().fit(support)
    ncm_proba, ncm_cos, log_proba = ncm.predict_proba(Q), ncm.cosine(Q), logistic.predict_proba(Q)
    class_names = [name(c) for c in ncm.labels]  # same order for both heads (sorted labels)

    def head_output(proba, **extra):
        return [dict(predicted=class_names[int(row.argmax())],
                     probabilities={n: float(v) for n, v in zip(class_names, row)},
                     **{k: {n: float(v) for n, v in zip(class_names, arr[i])} for k, arr in extra.items()})
                for i, row in enumerate(proba)]

    fixture = dict(
        source=dict(dataset=args.dataset, backbone=args.backbone, suffix=args.suffix,
                    seed=config.FIXTURE_SEED, dtype="float32", dimension=int(X.shape[1])),
        config=dict(
            logistic=dict(steps=config.LOGISTIC_STEPS, learning_rate=config.LOGISTIC_LR,
                          regularization=config.LOGISTIC_REG),
            nearest_class_mean=dict(temperature=config.NCM_TEMPERATURE),
        ),
        classes=class_names,
        stored={name(c): v.tolist() for c, v in support.items()},
        queries=[dict(vector=Q[i].tolist(), true_class=name(y[j]), taught=bool(y[j] in taught))
                 for i, j in enumerate(query_idx)],
        expected=dict(
            nearest_class_mean=head_output(ncm_proba, similarities=ncm_cos),
            logistic=head_output(log_proba),
        ),
    )
    assert not any(np.isnan(v).any() for v in (Q, ncm_proba, log_proba)), "NaN in fixture"

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture))
    print(f"wrote {out.relative_to(REPO_ROOT)}: {out.stat().st_size:,} bytes ({out.stat().st_size / 1024:.1f} KiB)")
    print(f"{len(support)} classes x {config.FIXTURE_K_SHOT} stored, {len(Q)} queries "
          f"({int(sum(q['taught'] for q in fixture['queries']))} taught, dim {X.shape[1]})")
    for head in ("nearest_class_mean", "logistic"):
        acc = np.mean([e["predicted"] == q["true_class"] for e, q in zip(fixture["expected"][head], fixture["queries"])
                       if q["taught"]])
        print(f"  {head}: top-1 on taught queries {acc:.2f}")


if __name__ == "__main__":
    main()
