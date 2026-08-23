"""Unit tests for active-learning acquisition functions."""
from __future__ import annotations

import numpy as np

from rnaseq_loop.active import bald, combine_acquisitions, coverage_scores, select_top_k


def test_bald_zero_when_ensemble_agrees():
    # 5 models, 10 pool items, 2 classes; every model puts all mass on class 0.
    probs = np.zeros((5, 10, 2))
    probs[..., 0] = 1.0 - 1e-6
    probs[..., 1] = 1e-6
    scores = bald(probs)
    assert np.all(scores < 1e-5)


def test_bald_positive_when_ensemble_disagrees():
    # Half the ensemble prefers class 0, half prefers class 1 → high MI.
    probs = np.zeros((4, 3, 2))
    probs[:2, :, 0] = 0.9
    probs[:2, :, 1] = 0.1
    probs[2:, :, 0] = 0.1
    probs[2:, :, 1] = 0.9
    scores = bald(probs)
    assert np.all(scores > 0.1)


def test_coverage_prefers_far_from_labeled():
    pool = np.array([[0.0, 0.0], [10.0, 10.0]])
    labeled = np.array([[0.1, 0.0]])
    scores = coverage_scores(pool, labeled)
    # The point at (10,10) is far from labeled → larger score.
    assert scores[1] > scores[0]


def test_coverage_empty_labeled():
    pool = np.random.default_rng(0).normal(size=(5, 3))
    labeled = np.empty((0, 3))
    scores = coverage_scores(pool, labeled)
    assert scores.shape == (5,)
    assert np.all(scores == 1.0)


def test_select_top_k_respects_already_labeled():
    combined = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    picks = select_top_k(combined, k=3, already_labeled={0, 1})
    assert picks == [2, 3, 4]


def test_combine_acquisitions_weighted_zscore():
    scores = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([10.0, 0.0, -10.0]),
    }
    weights = {"a": 1.0, "b": 1.0}
    combined = combine_acquisitions(scores, weights)
    # `a` ascending, `b` descending → combined should be roughly zero-mean, monotone check disabled.
    assert combined.shape == (3,)
    assert abs(combined.mean()) < 1e-9
