from __future__ import annotations

import pandas as pd

from rnaseq_loop.sample_aggregation import aggregate_probabilities, sample_metrics


def test_probability_aggregation_uses_samples_as_units():
    cells = pd.DataFrame({
        "sample_id": ["p", "p", "s", "s"],
        "senescence_label": ["proliferating", "proliferating", "senescent", "senescent"],
        "probability_senescent": [0.1, 0.3, 0.7, 0.9],
    })
    samples = aggregate_probabilities(cells)
    assert samples["predicted_label"].tolist() == ["proliferating", "senescent"]
    assert samples["n_cells"].tolist() == [2, 2]
    metrics = sample_metrics(samples)
    assert metrics["n_samples"] == 2
    assert metrics["balanced_accuracy"] == 1.0


def test_aggregation_rejects_mixed_labels_within_sample():
    cells = pd.DataFrame({
        "sample_id": ["mixed", "mixed"],
        "senescence_label": ["proliferating", "senescent"],
        "probability_senescent": [0.2, 0.8],
    })
    try:
        aggregate_probabilities(cells)
    except ValueError as error:
        assert "multiple true labels" in str(error)
    else:
        raise AssertionError("Expected mixed-label sample rejection")
