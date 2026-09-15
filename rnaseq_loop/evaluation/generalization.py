"""Leakage-safe grouped-holdout evaluation and classical baselines.

Feature selection is repeated using only the training portion of every fold.
This is important for leave-one-cell-line/study-out evaluation: selecting genes
once on the complete dataset would leak information from the held-out group.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DEFAULT_METHODS = (
    "dummy_majority",
    "nearest_centroid_top",
    "linear_svm_top",
    "logistic_top",
    "logistic_all",
)


def select_top_features(X: np.ndarray, y: np.ndarray, top_n: int) -> np.ndarray:
    """Return indices with the largest absolute Welch t statistic."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(y) != X.shape[0]:
        raise ValueError("X must be 2-D and y must contain one label per row")
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("feature selection requires exactly two classes")
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = stats.ttest_ind(
            X[y == classes[1]], X[y == classes[0]], axis=0,
            equal_var=False, nan_policy="omit",
        ).statistic
    scores = np.nan_to_num(np.abs(statistic), nan=0.0, posinf=np.finfo(float).max)
    n = min(top_n, X.shape[1])
    return np.argsort(-scores, kind="stable")[:n]


def _estimator(method: str, random_state: int) -> BaseEstimator:
    if method == "dummy_majority":
        return DummyClassifier(strategy="prior")
    if method == "nearest_centroid_top":
        return make_pipeline(StandardScaler(), NearestCentroid())
    if method in {"logistic_top", "logistic_all"}:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=2_000,
                random_state=random_state,
            ),
        )
    if method == "linear_svm_top":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", random_state=random_state),
        )
    raise ValueError(f"Unknown method {method!r}; choose from {DEFAULT_METHODS}")


def _continuous_score(estimator: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Get a positive-class ranking score without calibrating on held-out data."""
    if hasattr(estimator, "predict_proba"):
        probability = estimator.predict_proba(X)
        classes = np.asarray(estimator.classes_)
        positive = int(np.flatnonzero(classes == 1)[0])
        return np.asarray(probability[:, positive], dtype=float)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    # NearestCentroid on older sklearn versions exposes neither API. Its hard
    # predictions still give a valid, if coarse, AUROC baseline.
    return np.asarray(estimator.predict(X), dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, score))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auroc": auc,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate_group_holdout(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    sample_names: Sequence[str] | None = None,
    top_n: int = 100,
    methods: Sequence[str] = DEFAULT_METHODS,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate methods by holding out each group once.

    Returns ``(predictions, fold_metrics)``. Every feature-dependent operation,
    including top-gene selection and scaling, is fitted inside its training fold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    if X.ndim != 2 or len(y) != X.shape[0] or len(groups) != X.shape[0]:
        raise ValueError("X, y, and groups have incompatible shapes")
    if len(np.unique(y)) != 2:
        raise ValueError("grouped evaluation requires exactly two classes")
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")

    names = np.asarray(sample_names if sample_names is not None else [f"sample_{i}" for i in range(len(y))])
    features = np.asarray(feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])])
    if len(names) != len(y) or len(features) != X.shape[1]:
        raise ValueError("sample_names or feature_names has the wrong length")

    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    for held_out in sorted(np.unique(groups)):
        train = groups != held_out
        test = ~train
        if len(np.unique(y[train])) != 2:
            raise ValueError(f"training fold for {held_out!r} does not contain both classes")
        selected = select_top_features(X[train], y[train], top_n)

        for method in methods:
            use = np.arange(X.shape[1]) if method == "logistic_all" else selected
            if method == "dummy_majority":
                use = np.array([selected[0]])
            estimator = _estimator(method, random_state)
            estimator.fit(X[train][:, use], y[train])
            pred = np.asarray(estimator.predict(X[test][:, use]), dtype=int)
            score = _continuous_score(estimator, X[test][:, use])

            fold_metric = _metrics(y[test], pred, score)
            metric_rows.append({
                "method": method,
                "held_out_group": held_out,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "n_features": int(len(use)),
                **fold_metric,
            })
            selected_names = ";".join(features[selected])
            for sample, truth, prediction, value in zip(names[test], y[test], pred, score):
                prediction_rows.append({
                    "sample": str(sample), "held_out_group": held_out,
                    "method": method, "label": int(truth),
                    "prediction": int(prediction), "score": float(value),
                    "selected_features": selected_names,
                })

    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def _cluster_bootstrap(
    frame: pd.DataFrame,
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float]:
    groups = frame["held_out_group"].unique()
    rng = np.random.default_rng(random_state)
    values = []
    by_group = {group: sub for group, sub in frame.groupby("held_out_group", sort=False)}
    for _ in range(n_bootstrap):
        picked = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([by_group[group] for group in picked], ignore_index=True)
        try:
            values.append(metric(boot.label.to_numpy(), boot.prediction.to_numpy(), boot.score.to_numpy()))
        except ValueError:
            continue
    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [0.025, 0.975]).astype(float))


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Pool out-of-fold predictions and add held-out-group bootstrap CIs."""
    required = {"method", "held_out_group", "label", "prediction", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions is missing columns: {sorted(missing)}")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    metric_functions: dict[str, Callable] = {
        "accuracy": lambda y, p, s: accuracy_score(y, p),
        "balanced_accuracy": lambda y, p, s: balanced_accuracy_score(y, p),
        "auroc": lambda y, p, s: roc_auc_score(y, s),
        "macro_f1": lambda y, p, s: f1_score(y, p, average="macro", zero_division=0),
        "mcc": lambda y, p, s: matthews_corrcoef(y, p),
    }
    rows = []
    for method, frame in predictions.groupby("method", sort=False):
        point = _metrics(frame.label.to_numpy(), frame.prediction.to_numpy(), frame.score.to_numpy())
        row: dict[str, float | int | str] = {
            "method": method,
            "n_samples": len(frame),
            "n_held_out_groups": frame.held_out_group.nunique(),
            **point,
        }
        for offset, (name, function) in enumerate(metric_functions.items()):
            low, high = _cluster_bootstrap(
                frame, function, n_bootstrap=n_bootstrap,
                random_state=random_state + offset,
            )
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "auroc"], ascending=False,
    ).reset_index(drop=True)
