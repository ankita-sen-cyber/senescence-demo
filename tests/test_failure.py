"""Unit tests for slice metrics and stability."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rnaseq_loop.failure import (
    bootstrap_topk_stability,
    counterfactual_probe,
    flag_failing_slices,
    slice_metrics,
)


def _fake_preds(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, size=n)
    score = rng.uniform(size=n)
    # Make the model good overall but bad on assay="bad".
    assay = np.where(rng.uniform(size=n) < 0.3, "bad", "good")
    good = assay == "good"
    score[good] = 0.7 * label[good] + 0.3 * score[good]
    pred = (score > 0.5).astype(int)
    return pd.DataFrame({
        "label": label, "pred": pred, "score": score,
        "assay": assay,
        "dataset_id": rng.choice(["studyA", "studyB", "studyC"], size=n),
        "tissue": rng.choice(["skin", "lung"], size=n),
        "sex": rng.choice(["male", "female"], size=n),
    })


def test_slice_metrics_produces_overall_and_slices():
    df = _fake_preds()
    out = slice_metrics(df, min_slice_size=20)
    assert "OVERALL" in out["slice_col"].values
    assert (out["slice_col"] == "assay").any()


def test_flag_failing_slices_detects_bad_slice():
    df = _fake_preds()
    out = slice_metrics(df, min_slice_size=20)
    bad = flag_failing_slices(out, gap_threshold=-0.05)
    # We forced 'bad' to have random-ish scores → should flag.
    assert (bad["slice_value"] == "bad").any()


def test_bootstrap_topk_stability():
    boots = [
        pd.Series({"A": 3, "B": 2, "C": 1, "D": 0.5}),
        pd.Series({"A": 3, "B": 2.5, "C": 0, "D": 1}),
        pd.Series({"A": 3, "B": 2, "C": 0, "D": 1.5}),
    ]
    stab = bootstrap_topk_stability(boots, k=2)
    # A always in top-2 → freq=1.0
    a_row = stab[stab["gene"] == "A"].iloc[0]
    assert a_row["topk_freq"] == 1.0


def test_counterfactual_probe_flags_pass_fail():
    df = pd.DataFrame({
        "gene": ["G1", "G2", "G3"],
        "attribution": [1.0, 0.9, 0.8],
        "goal_state_shift": [0.5, -0.1, 0.3],
        "pval": [0.01, 0.2, 0.01],
    })
    out = counterfactual_probe(df, top_k=3)
    assert out.loc[out["gene"] == "G1", "passes_counterfactual"].iloc[0]
    assert not out.loc[out["gene"] == "G2", "passes_counterfactual"].iloc[0]  # bad pval
    assert out.loc[out["gene"] == "G3", "passes_counterfactual"].iloc[0]
