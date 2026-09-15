"""Tests for leakage-safe grouped generalization evaluation."""
from __future__ import annotations

import numpy as np

from rnaseq_loop.evaluation import evaluate_group_holdout, select_top_features, summarize_predictions


def test_select_top_features_finds_training_signal():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 12)
    X = rng.normal(size=(24, 8))
    X[:, 5] += y * 4.0
    selected = select_top_features(X, y, top_n=1)
    assert selected.tolist() == [5]


def test_group_holdout_compares_methods_and_saves_each_prediction():
    rng = np.random.default_rng(7)
    groups = np.repeat(["study_a", "study_b", "study_c"], 12)
    y = np.tile(np.repeat([0, 1], 6), 3)
    X = rng.normal(scale=0.25, size=(len(y), 10))
    X[:, 2] += y * 3.0

    methods = ["dummy_majority", "logistic_top", "nearest_centroid_top"]
    predictions, folds = evaluate_group_holdout(
        X, y, groups, top_n=3, methods=methods, random_state=1,
    )

    assert len(predictions) == len(y) * len(methods)
    assert len(folds) == len(np.unique(groups)) * len(methods)
    assert predictions.groupby(["method", "sample"]).size().eq(1).all()
    logistic = folds[folds.method == "logistic_top"]
    assert logistic.balanced_accuracy.mean() > 0.95


def test_summary_has_cluster_confidence_intervals():
    rng = np.random.default_rng(11)
    groups = np.repeat(["a", "b", "c"], 8)
    y = np.tile(np.repeat([0, 1], 4), 3)
    X = rng.normal(size=(24, 5))
    X[:, 0] += 2.5 * y
    predictions, _ = evaluate_group_holdout(
        X, y, groups, top_n=2,
        methods=["dummy_majority", "logistic_top"],
    )
    summary = summarize_predictions(predictions, n_bootstrap=50, random_state=9)
    assert set(summary.method) == {"dummy_majority", "logistic_top"}
    assert summary.balanced_accuracy_ci_low.notna().all()
    assert (summary.balanced_accuracy_ci_low <= summary.balanced_accuracy).all()
    assert (summary.balanced_accuracy <= summary.balanced_accuracy_ci_high).all()
