"""Aggregate cell-level classifier output into biological-sample predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def aggregate_probabilities(
    predictions: pd.DataFrame,
    *,
    sample_column: str = "sample_id",
    truth_column: str = "senescence_label",
    score_column: str = "probability_senescent",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create one pre-specified probability-based prediction per sample."""
    required = {sample_column, truth_column, score_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    rows = []
    for sample, frame in predictions.groupby(sample_column, sort=True):
        truth = frame[truth_column].astype(str).unique()
        if len(truth) != 1:
            raise ValueError(f"Sample {sample!r} contains multiple true labels")
        scores = frame[score_column].astype(float).to_numpy()
        mean_score = float(np.mean(scores))
        rows.append({
            "sample_id": str(sample),
            "true_label": truth[0],
            "n_cells": len(frame),
            "mean_probability_senescent": mean_score,
            "median_probability_senescent": float(np.median(scores)),
            "q10_probability_senescent": float(np.quantile(scores, 0.1)),
            "q90_probability_senescent": float(np.quantile(scores, 0.9)),
            "fraction_cells_predicted_senescent": float(np.mean(scores >= threshold)),
            "predicted_label": "senescent" if mean_score >= threshold else "proliferating",
            "threshold": threshold,
        })
    return pd.DataFrame(rows)


def sample_metrics(samples: pd.DataFrame) -> dict[str, float | int]:
    """Score aggregated samples; samples, not cells, are the evaluation units."""
    y = (samples["true_label"] == "senescent").astype(int).to_numpy()
    pred = (samples["predicted_label"] == "senescent").astype(int).to_numpy()
    score = samples["mean_probability_senescent"].astype(float).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Sample-level metrics require both biological classes")
    return {
        "n_samples": len(samples),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auroc": float(roc_auc_score(y, score)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
