"""CPU-only tests for deployment artifact helpers."""
from __future__ import annotations

import json

import numpy as np

from rnaseq_loop.deployment import load_label_mapping, prediction_frame


def test_load_json_label_mapping(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"0": "proliferating", "1": "senescent"}))
    assert load_label_mapping(path) == {0: "proliferating", 1: "senescent"}


def test_prediction_frame_outputs_labels_probabilities_and_metadata():
    frame = prediction_frame(
        np.array([[0.8, 0.2], [0.1, 0.9]]),
        {0: "proliferating", 1: "senescent"},
        {"cell_id": ["cell-a", "cell-b"]},
    )
    assert frame.predicted_label.tolist() == ["proliferating", "senescent"]
    assert frame.confidence.tolist() == [0.8, 0.9]
    assert frame.cell_id.tolist() == ["cell-a", "cell-b"]
    assert "probability_senescent" in frame


def test_prediction_frame_rejects_class_count_mismatch():
    try:
        prediction_frame(np.ones((2, 3)), {0: "a", 1: "b"})
    except ValueError as error:
        assert "one column per mapped class" in str(error)
    else:
        raise AssertionError("Expected a class-count validation error")
