"""Few-shot episode sampler."""
import numpy as np


def sample_episode(y, n_classes, k_shot, n_query, seed):
    """Pick `n_classes` classes, then `k_shot` support and `n_query` query examples per class.

    Query examples are drawn from the class's remaining examples, so support and query never
    overlap. Only classes with at least k_shot + n_query examples are eligible. The result is
    fully determined by the arguments (including `seed`).

    Returns a dict: classes (sorted labels), support_idx, support_y, query_idx, query_y.
    """
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    eligible = labels[counts >= k_shot + n_query]
    if len(eligible) < n_classes:
        raise ValueError(f"only {len(eligible)} classes have >= {k_shot + n_query} examples, "
                         f"need {n_classes}")
    rng = np.random.default_rng(seed)
    classes = np.sort(rng.choice(eligible, size=n_classes, replace=False))
    support, query = [], []
    for c in classes:
        idx = rng.permutation(np.flatnonzero(y == c))
        support.append(idx[:k_shot])
        query.append(idx[k_shot:k_shot + n_query])
    support_idx, query_idx = np.concatenate(support), np.concatenate(query)
    return dict(classes=classes, support_idx=support_idx, support_y=y[support_idx],
                query_idx=query_idx, query_y=y[query_idx])
