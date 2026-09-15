"""Import independently produced, labelled external single-cell datasets."""
from __future__ import annotations

import hashlib
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


GSE282425_FILES = {
    "GSE282425_barcodes.tsv.gz": "5945b59ae608ebfb2ad55b8527f50a3f254532b6a692c803ce4f49335cb0c681",
    "GSE282425_features.tsv.gz": "8f978c80872e705d9214ca3705d74e101a0f5b305c4495efc379aeb0c361da7c",
    "GSE282425_matrix.mtx.gz": "7132e6e715a176d05d04d8f749ef77e7717c555ad4a5e541a347c20da7c7e7bc",
}
GSE282425_BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE282nnn/GSE282425/suppl"
)
GSE282425_SAMPLES = {
    "1": ("GSM8642882", "senescent", "2D"),
    "2": ("GSM8642883", "senescent", "3D"),
    "3": ("GSM8642880", "proliferating", "2D"),
    "4": ("GSM8642881", "proliferating", "3D"),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_gse282425(raw_dir: str | Path) -> dict[str, Path]:
    """Download the official GEO matrix files and verify fixed checksums."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for filename, expected in GSE282425_FILES.items():
        destination = raw_dir / filename
        if not destination.exists() or sha256(destination) != expected:
            partial = destination.with_suffix(destination.suffix + ".part")
            urllib.request.urlretrieve(f"{GSE282425_BASE_URL}/{filename}", partial)
            partial.replace(destination)
        observed = sha256(destination)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch for {destination}: {observed}, expected {expected}"
            )
        paths[filename] = destination
    return paths


def load_geneformer_symbol_map(path: str | Path) -> dict[str, str]:
    with Path(path).open("rb") as handle:
        mapping = pickle.load(handle)
    if not isinstance(mapping, dict):
        raise TypeError("Geneformer gene-name mapping must be a dictionary")
    return {str(symbol): str(ensembl) for symbol, ensembl in mapping.items()}


def _sample_metadata(barcodes: pd.Index) -> pd.DataFrame:
    suffixes = barcodes.to_series(index=barcodes).str.rsplit("-", n=1).str[-1]
    unknown = sorted(set(suffixes) - set(GSE282425_SAMPLES))
    if unknown:
        raise ValueError(f"Unknown GSE282425 barcode suffixes: {unknown}")
    records = [GSE282425_SAMPLES[suffix] for suffix in suffixes]
    return pd.DataFrame(
        {
            "cell_id": barcodes.astype(str),
            "dataset_id": "GSE282425",
            "study_id": "GSE282425",
            "sample_id": [row[0] for row in records],
            "senescence_label": [row[1] for row in records],
            "culture_dimension": [row[2] for row in records],
            "donor_id": "GSE282425_donor_1",
            "cell_type": "primary human dermal fibroblast",
            "label_source": "experimental_condition",
        },
        index=barcodes,
    )


def convert_gse282425(
    raw_dir: str | Path,
    output_h5ad: str | Path,
    gene_name_map: str | Path,
) -> dict[str, int]:
    """Convert GSE282425's merged 10x matrix into Geneformer-ready AnnData.

    GEO supplies gene symbols but uses ``ENS_ID`` as a placeholder in the first
    feature column. Symbols are therefore mapped with Geneformer's bundled
    human gene-name dictionary. Symbols mapping to the same Ensembl gene are
    summed, while unmapped features are omitted and reported.
    """
    import anndata as ad
    from scipy import io, sparse

    raw_dir = Path(raw_dir)
    barcodes = pd.read_csv(
        raw_dir / "GSE282425_barcodes.tsv.gz", header=None, names=["barcode"]
    )["barcode"].astype(str)
    features = pd.read_csv(
        raw_dir / "GSE282425_features.tsv.gz", sep="\t", header=None,
        names=["source_id", "gene_symbol"],
    )
    counts = io.mmread(raw_dir / "GSE282425_matrix.mtx.gz").tocsr().T
    if counts.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"Matrix shape {counts.shape} does not match "
            f"{len(barcodes)} barcodes and {len(features)} features"
        )

    symbol_map = load_geneformer_symbol_map(gene_name_map)
    mapped = features["gene_symbol"].map(symbol_map)
    keep = mapped.notna().to_numpy()
    retained_ids = mapped[keep].astype(str).to_numpy()
    unique_ids, inverse = np.unique(retained_ids, return_inverse=True)
    projection = sparse.csr_matrix(
        (np.ones(len(inverse), dtype=counts.dtype), (np.arange(len(inverse)), inverse)),
        shape=(len(inverse), len(unique_ids)),
    )
    merged = (counts[:, keep] @ projection).tocsr()

    obs = _sample_metadata(pd.Index(barcodes, name="cell_id"))
    obs["n_counts"] = np.asarray(merged.sum(axis=1)).ravel().astype(np.int64)
    var = pd.DataFrame({"ensembl_id": unique_ids}, index=pd.Index(unique_ids, name="ensembl_id"))
    adata = ad.AnnData(X=merged, obs=obs, var=var)
    adata.uns["source_accession"] = "GSE282425"
    adata.uns["source_url"] = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282425"
    adata.uns["intended_use"] = "independent_test_only"
    adata.uns["label_definition"] = {
        "proliferating": "Early Proliferative (EP) experimental condition",
        "senescent": "Deeply Senescent (DS) experimental condition",
    }
    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_h5ad, compression="gzip")
    return {
        "cells": adata.n_obs,
        "source_features": len(features),
        "mapped_source_features": int(keep.sum()),
        "output_genes": adata.n_vars,
        "unmapped_source_features": int((~keep).sum()),
    }
