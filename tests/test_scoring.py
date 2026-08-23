"""Unit tests for the scoring layer (pure Python — no Geneformer needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rnaseq_loop.scoring import ScoringConfig, rank_targets, zscore


def test_zscore_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(s)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std(ddof=1) - 1.0) < 1e-6


def test_rank_targets_fuses_sources():
    isp = pd.Series({"TP53": 3.0, "MYC": 2.0, "RB1": 1.0, "GAPDH": -1.0})
    ot = pd.Series({"TP53": 0.9, "MYC": 0.6, "RB1": 0.4, "OTHER": 0.8})
    cfg = ScoringConfig(
        weights={"isp_attribution": 0.6, "ot_overall_score": 0.4},
        min_sources=2,
    )
    ranked = rank_targets(
        {"isp_attribution": isp, "ot_overall_score": ot},
        cfg,
    )
    # TP53 has strong signal from both → should be top.
    assert ranked.index[0] == "TP53"
    # OTHER only in OT — should be dropped (min_sources=2).
    assert np.isnan(ranked.loc["OTHER", "target_score"])
