"""Preparation of experimentally labelled fibroblast training data."""
from __future__ import annotations

import gzip
import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


GSE226225_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE226nnn/"
    "GSE226225/suppl/GSE226225_RAW.tar"
)
GSE226225_SHA256 = "d17f787c6176c044f732948fbf2bd2330a0f9e16799f629ad0a66275a9eb7e9f"

# prefix: (label, induction, cells retained). Intermediate ETO days 1/2/4/7
# are intentionally absent because they do not have an unambiguous binary label.
GSE226225_SAMPLES = {
    "GSM7068354_CTRL_2": ("proliferating", "untreated_control", 3500),
    "GSM7068361_ETO_day_0": ("proliferating", "etoposide_day_0", 3500),
    "GSM7068355_RS_1": ("senescent", "replicative_senescence", 1000),
    "GSM7068356_RS_2": ("senescent", "replicative_senescence", 1000),
    "GSM7068357_IR_1": ("senescent", "ionizing_radiation_day_10", 1000),
    "GSM7068358_IR_2": ("senescent", "ionizing_radiation_day_10", 1000),
    "GSM7068359_ETO_1": ("senescent", "etoposide_day_10", 1000),
    "GSM7068360_ETO_2": ("senescent", "etoposide_day_10", 1000),
    "GSM7068366_ETO_day_10": ("senescent", "etoposide_day_10", 1000),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_gse226225(archive: str | Path) -> Path:
    """Download and checksum the official GEO archive when it is absent."""
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or _sha256(archive) != GSE226225_SHA256:
        partial = archive.with_suffix(".tar.part")
        urllib.request.urlretrieve(GSE226225_URL, partial)
        partial.replace(archive)
    observed = _sha256(archive)
    if observed != GSE226225_SHA256:
        raise ValueError(f"GSE226225 checksum mismatch: {observed}")
    return archive


def _read_member(table: tarfile.TarFile, name: str, **kwargs) -> pd.DataFrame:
    member = table.extractfile(name)
    if member is None:
        raise FileNotFoundError(f"Archive member not found: {name}")
    with gzip.GzipFile(fileobj=member) as stream:
        return pd.read_csv(stream, **kwargs)


def prepare_gse226225(
    archive: str | Path,
    output_h5ad: str | Path,
    *,
    seed: int = 42,
) -> dict[str, object]:
    """Build a balanced, raw-count AnnData from unambiguous fibroblast samples."""
    import anndata as ad
    from scipy import sparse
    from scipy.io import mmread

    rng = np.random.default_rng(seed)
    matrices = []
    observations = []
    reference_features: pd.DataFrame | None = None

    with tarfile.open(archive, mode="r") as table:
        for prefix, (label, induction, target_cells) in GSE226225_SAMPLES.items():
            features = _read_member(
                table, f"{prefix}_features.tsv.gz", sep="\t", header=None,
                names=["ensembl_id", "gene_symbol", "feature_type"],
            )
            barcodes = _read_member(
                table, f"{prefix}_barcodes.tsv.gz", header=None, names=["barcode"],
            )["barcode"].astype(str)
            matrix_member = table.extractfile(f"{prefix}_matrix.mtx.gz")
            if matrix_member is None:
                raise FileNotFoundError(f"Matrix missing for {prefix}")
            with gzip.GzipFile(fileobj=matrix_member) as stream:
                counts = mmread(stream).tocsr().T
            if counts.shape != (len(barcodes), len(features)):
                raise ValueError(f"Shape mismatch for {prefix}: {counts.shape}")
            if reference_features is None:
                reference_features = features
            elif not features.equals(reference_features):
                raise ValueError(f"Feature ordering differs for {prefix}")
            if len(barcodes) < target_cells:
                raise ValueError(f"{prefix} has fewer than {target_cells} cells")
            selected = np.sort(rng.choice(len(barcodes), target_cells, replace=False))
            matrices.append(counts[selected])
            selected_barcodes = barcodes.iloc[selected].to_numpy()
            accession = prefix.split("_", 1)[0]
            observations.append(pd.DataFrame({
                "cell_id": [f"{accession}:{barcode}" for barcode in selected_barcodes],
                "senescence_label": label,
                "sample_id": accession,
                "dataset_id": "GSE226225",
                "study_id": "GSE226225",
                "donor_id": "WI38_cell_line",
                "cell_line": "WI-38",
                "cell_type": "fibroblast",
                "tissue": "fetal lung",
                "induction": induction,
                "label_source": "experimental_condition",
            }))

    if reference_features is None:
        raise ValueError("No GSE226225 samples were loaded")
    X = sparse.vstack(matrices, format="csr")
    obs = pd.concat(observations, ignore_index=True)
    obs.index = pd.Index(obs.pop("cell_id"), name="cell_id")
    obs["n_counts"] = np.asarray(X.sum(axis=1)).ravel().astype(np.int64)
    var = reference_features.copy()
    var.index = pd.Index(var["ensembl_id"].astype(str), name="ensembl_id_index")
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.uns["source_accession"] = "GSE226225"
    adata.uns["source_url"] = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226225"
    adata.uns["training_policy"] = (
        "fibroblast-only; unambiguous endpoints; balanced classes; "
        "intermediate etoposide days 1/2/4/7 excluded"
    )
    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad, compression="gzip")
    counts = obs["senescence_label"].value_counts().to_dict()
    return {
        "cells": adata.n_obs,
        "genes": adata.n_vars,
        "samples": obs["sample_id"].nunique(),
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "seed": seed,
    }
