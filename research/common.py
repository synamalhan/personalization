"""Helpers shared by the experiment scripts."""
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import rankdata

import config

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_embeddings(dataset: str, backbone: str, suffix: str = ""):
    path = REPO_ROOT / config.RESULTS_DIR / f"emb_{dataset}_{backbone}{suffix}.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        return f["X"], f["y"]


def split_support_query(rng, y, classes, k_shot, n_query):
    """For each class: k_shot random support indices and n_query disjoint query indices."""
    support, query = {}, {}
    for c in classes:
        idx = rng.permutation(np.flatnonzero(y == c))
        support[c], query[c] = idx[:k_shot], idx[k_shot:k_shot + n_query]
    return support, query


def eligible_classes(y, need):
    labels, counts = np.unique(y, return_counts=True)
    return labels[counts >= need]


def mean_ci(values, level=config.CI_LEVEL):
    """Mean and t-based confidence interval over independent trials."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), float("nan"), float("nan")
    m, se = v.mean(), stats.sem(v)
    half = se * stats.t.ppf((1 + level) / 2, len(v) - 1)
    return float(m), float(m - half), float(m + half)


def auroc(scores_pos, scores_neg):
    """P(score of a random positive > score of a random negative), ties count half."""
    s = np.concatenate([scores_pos, scores_neg])
    ranks = rankdata(s)
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(confidence, correct, n_bins=config.ECE_BINS):
    """Expected calibration error with equal-width confidence bins."""
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bin_ids == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def macro_f1(y_true, y_pred, labels):
    """Unweighted mean of per-class F1 over `labels` (a class with no true or predicted
    examples contributes 0)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in labels:
        tp = float(((y_pred == c) & (y_true == c)).sum())
        fp = float(((y_pred == c) & (y_true != c)).sum())
        fn = float(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))
