from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from rnaseq_loop.pseudobulk import aggregate_raw_counts, log_cpm


def test_raw_counts_are_summed_within_biological_sample():
    counts = sparse.csr_matrix([
        [1, 2, 0],
        [3, 0, 1],
        [0, 4, 2],
    ])
    obs = pd.DataFrame({
        "sample_id": ["sample_a", "sample_a", "sample_b"],
        "senescence_label": ["proliferating", "proliferating", "senescent"],
    })
    bulk, metadata = aggregate_raw_counts(counts, obs)
    np.testing.assert_array_equal(bulk, [[4, 2, 1], [0, 4, 2]])
    assert metadata["n_cells"].tolist() == [2, 1]
    assert metadata["true_label"].tolist() == ["proliferating", "senescent"]


def test_log_cpm_removes_library_size_difference():
    counts = np.array([[1, 3], [10, 30]])
    cpm, transformed = log_cpm(counts)
    np.testing.assert_allclose(cpm[0], cpm[1])
    np.testing.assert_allclose(transformed[0], transformed[1])


def test_pseudobulk_rejects_mixed_labels_within_sample():
    counts = sparse.csr_matrix([[1, 0], [0, 1]])
    obs = pd.DataFrame({
        "sample_id": ["mixed", "mixed"],
        "senescence_label": ["proliferating", "senescent"],
    })
    try:
        aggregate_raw_counts(counts, obs)
    except ValueError as error:
        assert "multiple labels" in str(error)
    else:
        raise AssertionError("Expected mixed-label sample rejection")
