"""pytest tests for the sampler, heads and metric helpers. Run: pytest research/tests"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from common import auroc, ece, macro_f1, mean_ci  # noqa: E402
from episodes import sample_episode  # noqa: E402
from heads import Logistic, NearestClassMean, l2_normalize, softmax, suggested_threshold  # noqa: E402

Y = np.repeat(np.arange(10), 30)  # 10 classes x 30 examples


def test_same_seed_same_episode():
    a, b = sample_episode(Y, 5, 3, 7, seed=4), sample_episode(Y, 5, 3, 7, seed=4)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_different_seed_different_episode():
    a, b = sample_episode(Y, 5, 3, 7, seed=1), sample_episode(Y, 5, 3, 7, seed=2)
    assert not np.array_equal(a["support_idx"], b["support_idx"])


def test_no_overlap_and_correct_sizes():
    ep = sample_episode(Y, 4, 5, 10, seed=0)
    assert len(ep["classes"]) == 4
    assert len(ep["support_idx"]) == 4 * 5 and len(ep["query_idx"]) == 4 * 10
    assert set(ep["support_idx"]).isdisjoint(ep["query_idx"])
    assert len(set(ep["support_idx"])) == len(ep["support_idx"])
    assert set(ep["support_y"]) == set(ep["query_y"]) == set(ep["classes"])
    np.testing.assert_array_equal(Y[ep["support_idx"]], ep["support_y"])


def test_too_few_eligible_classes():
    with pytest.raises(ValueError):
        sample_episode(Y, 5, 20, 20, seed=0)  # needs 40 per class, only 30 exist


def _toy():
    """Three well-separated classes in 3-D."""
    return {0: np.array([[1, .1, 0], [2, 0, .2], [3, .2, -.1]], dtype=np.float32),
            1: np.array([[.1, 1, 0], [0, 2, .2], [.2, 3, -.1]], dtype=np.float32),
            2: np.array([[0, .1, 1], [.2, 0, 2], [-.1, .2, 3]], dtype=np.float32)}


@pytest.mark.parametrize("head_cls", [NearestClassMean, Logistic])
def test_heads_classify_training_points(head_cls):
    support = _toy()
    head = head_cls().fit(support)
    for c, X in support.items():
        proba = head.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-5)
        assert (head.labels[proba.argmax(axis=1)] == c).all()


def test_logistic_deterministic_and_zero_steps_uniform():
    a, b = Logistic().fit(_toy()), Logistic().fit(_toy())
    np.testing.assert_array_equal(a.W, b.W)
    uniform = Logistic(steps=0).fit(_toy()).predict_proba(np.array([[1, 0, 0]], dtype=np.float32))
    np.testing.assert_allclose(uniform, 1 / 3, atol=1e-6)


def test_logistic_one_class():
    head = Logistic().fit({7: np.array([[1, 0], [1, .1]], dtype=np.float32)})
    np.testing.assert_allclose(head.predict_proba(np.array([[0, 1]], dtype=np.float32)), [[1.0]])


def test_ncm_mean_is_of_raw_vectors_then_normalized():
    # Raw sum (4,0)+(0,1) = (4,1); normalizing each first would give (1,1)/sqrt2 instead.
    head = NearestClassMean().fit({0: np.array([[4, 0], [0, 1]], dtype=np.float32)})
    np.testing.assert_allclose(head.means[0], np.array([4, 1]) / np.sqrt(17), atol=1e-6)


def test_threshold_rule():
    assert suggested_threshold({0: np.array([[1, 0]], dtype=np.float32)}) is None
    thr = suggested_threshold(_toy())
    assert 0.5 < thr <= 1


def test_metrics():
    assert auroc(np.array([.9, .8]), np.array([.1, .2])) == 1.0
    assert auroc(np.array([.1]), np.array([.1])) == 0.5
    assert ece(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0
    assert ece(np.array([1.0, 1.0]), np.array([0, 0])) == pytest.approx(1.0)
    assert macro_f1([0, 0, 1, 1], [0, 0, 1, 1], [0, 1]) == 1.0
    assert macro_f1([0, 0, 1, 1], [0, 0, 0, 0], [0, 1]) == pytest.approx((2 / 3 + 0) / 2)
    m, lo, hi = mean_ci([1.0, 1.0, 1.0])
    assert (m, lo, hi) == (1.0, 1.0, 1.0)


def test_softmax_and_normalize():
    np.testing.assert_allclose(softmax(np.array([[1000., 1000.]])), [[.5, .5]])
    np.testing.assert_allclose(np.linalg.norm(l2_normalize(np.array([[3., 4.]])), axis=1), 1)
    assert config.SHOTS == (1, 3, 5, 10, 20)


def test_sequential_add_class_ncm_never_forgets_logistic_does():
    toy = _toy()
    old = {0: toy[0], 1: toy[1]}
    ncm = NearestClassMean().fit(old).add_class(2, toy[2])
    assert (ncm.labels[ncm.predict_proba(toy[0]).argmax(axis=1)] == 0).all()  # old class intact
    assert (ncm.labels[ncm.predict_proba(toy[2]).argmax(axis=1)] == 2).all()
    log = Logistic().fit(old).add_class(2, toy[2])
    assert (log.labels[log.predict_proba(toy[2]).argmax(axis=1)] == 2).all()
    # Trained on class 2 alone, the logistic head now answers 2 for old-class inputs too: forgetting.
    assert (log.labels[log.predict_proba(toy[0]).argmax(axis=1)] == 2).mean() > 0.5
