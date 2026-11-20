"""NumPy versions of the PersonalizeCore heads. The math mirrors the Swift code exactly.

Inputs are dicts {class_label: array (n, d)} of RAW (un-normalized) embeddings.
"""
import numpy as np

import config


def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / norms


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=-1, keepdims=True)


def normalized_mean(X: np.ndarray) -> np.ndarray:
    """Swift's normalizedMean(of:): sum the RAW vectors, then L2-normalize the sum."""
    return l2_normalize(X.sum(axis=0))


class NearestClassMean:
    """Class means (of raw vectors) L2-normalized; probabilities = softmax(cosine / T)."""

    def __init__(self, temperature: float = config.NCM_TEMPERATURE):
        self.temperature = temperature

    def fit(self, support: dict) -> "NearestClassMean":
        self.labels = np.array(sorted(support))
        self.means = np.stack([normalized_mean(support[c]) for c in self.labels]).astype(np.float32)
        return self

    def cosine(self, X: np.ndarray) -> np.ndarray:
        return l2_normalize(X) @ self.means.T

    def add_class(self, label, X: np.ndarray) -> "NearestClassMean":
        """Sequential learning without replay: keep the old class means, add one more.
        A class mean depends only on its own examples, so old classes cannot be forgotten."""
        new = NearestClassMean(self.temperature)
        new.labels = np.append(self.labels, label)
        new.means = np.vstack([self.means, normalized_mean(X)[None].astype(np.float32)])
        return new

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.cosine(X) / self.temperature)


class Logistic:
    """Softmax regression: L2-normalized inputs, zero init, full-batch gradient descent."""

    def __init__(self, steps=config.LOGISTIC_STEPS, lr=config.LOGISTIC_LR, reg=config.LOGISTIC_REG):
        self.steps, self.lr, self.reg = steps, lr, reg

    def fit(self, support: dict) -> "Logistic":
        self.labels = np.array(sorted(support))
        X = l2_normalize(np.concatenate([support[c] for c in self.labels])).astype(np.float32)
        y = np.concatenate([np.full(len(support[c]), i) for i, c in enumerate(self.labels)])
        k, n, d = len(self.labels), len(X), X.shape[1]
        self.W = np.zeros((k, d), dtype=np.float32)
        self.b = np.zeros(k, dtype=np.float32)
        if k < 2:  # one class: nothing to learn, probability is always 1
            return self
        onehot = np.eye(k, dtype=np.float32)[y]
        for _ in range(self.steps):
            probs = softmax(X @ self.W.T + self.b)
            err = probs - onehot                       # gradient w.r.t. the logits
            self.b -= self.lr * err.sum(axis=0) / n    # biases are not regularized
            self.W -= self.lr * (err.T @ X / n + self.reg * self.W)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(l2_normalize(X) @ self.W.T + self.b)

    def add_class(self, label, X: np.ndarray) -> "Logistic":
        """Sequential learning without replay: keep the old weights, add a zero row for the new class,
        then run the same number of gradient steps on the NEW class's examples only. Nothing in the loss
        keeps the old classes up, so their logits are pushed down: this is catastrophic forgetting."""
        new = Logistic(self.steps, self.lr, self.reg)
        new.labels = np.append(self.labels, label)
        d = self.W.shape[1]
        new.W = np.vstack([self.W, np.zeros((1, d), dtype=np.float32)])
        new.b = np.append(self.b, np.float32(0))
        Xn = l2_normalize(X).astype(np.float32)
        k, n = len(new.labels), len(Xn)
        onehot = np.zeros((n, k), dtype=np.float32)
        onehot[:, -1] = 1
        for _ in range(new.steps):
            err = softmax(Xn @ new.W.T + new.b) - onehot
            new.b -= new.lr * err.sum(axis=0) / n
            new.W -= new.lr * (err.T @ Xn / n + new.reg * new.W)
        return new


def suggested_threshold(support: dict, std_devs: float = config.THRESHOLD_STD_DEVS):
    """PersonalizeCore's rule: leave-one-out cosine of each vector to the mean of the other
    vectors in its class; threshold = mean - k * population std, clamped to [-1, 1].
    None if no class has 2+ vectors."""
    sims = []
    for X in support.values():
        if len(X) < 2:
            continue
        for i in range(len(X)):
            others = np.delete(X, i, axis=0)
            sims.append(float(normalized_mean(others) @ l2_normalize(X[i])))
    if not sims:
        return None
    sims = np.array(sims, dtype=np.float64)
    return float(np.clip(sims.mean() - std_devs * sims.std(), -1.0, 1.0))


HEADS = {"nearest_class_mean": NearestClassMean, "logistic": Logistic}
