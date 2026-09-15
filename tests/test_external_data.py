from __future__ import annotations

import pandas as pd

from rnaseq_loop.external_data import GSE282425_SAMPLES, _sample_metadata


def test_gse282425_barcode_suffixes_map_to_experimental_samples():
    barcodes = pd.Index(["cell-a-1", "cell-b-2", "cell-c-3", "cell-d-4"])
    obs = _sample_metadata(barcodes)
    assert obs["senescence_label"].tolist() == [
        "senescent", "senescent", "proliferating", "proliferating"
    ]
    assert obs["culture_dimension"].tolist() == ["2D", "3D", "2D", "3D"]
    assert obs["sample_id"].tolist() == [
        "GSM8642882", "GSM8642883", "GSM8642880", "GSM8642881"
    ]
    assert set(obs["study_id"]) == {"GSE282425"}
    assert GSE282425_SAMPLES["1"][1] == "senescent"
