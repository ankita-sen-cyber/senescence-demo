"""Tokenize AnnData → Geneformer `.dataset` format.

Geneformer's `TranscriptomeTokenizer` expects:
  - Input directory containing `.loom` or `.h5ad` files.
  - Raw count matrix (no normalization; no HVG selection).
  - `var` column `ensembl_id` with Ensembl gene IDs.
  - `obs` column `n_counts` with total per-cell UMI/read counts.
  - Any extra `obs` columns to carry through are declared in `custom_attr_name_dict`.

Reference: https://geneformer.readthedocs.io/en/latest/geneformer.tokenizer.html
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Mapping

import anndata as ad

from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger(__name__)


def prepare_input_dir(
    adata: ad.AnnData,
    input_dir: str | Path,
    filename: str = "cells.h5ad",
) -> Path:
    """Write a single h5ad into a directory (Geneformer expects a directory)."""
    input_dir = ensure_dir(input_dir)
    # Clear any stale files.
    for p in input_dir.glob("*.h5ad"):
        p.unlink()
    out = input_dir / filename
    adata.write_h5ad(out, compression="gzip")
    log.info(f"Wrote input file {out}")
    return input_dir


def tokenize(
    input_dir: str | Path,
    output_dir: str | Path,
    output_prefix: str,
    custom_attrs: Mapping[str, str] | None = None,
    nproc: int = 8,
    model_version: str = "V2",
) -> Path:
    """
    Run Geneformer's TranscriptomeTokenizer.

    Args:
        input_dir: directory containing .h5ad files.
        output_dir: where the tokenized .dataset will be written.
        output_prefix: prefix for output files (produces `<prefix>.dataset`).
        custom_attrs: mapping of {adata_obs_column: new_dataset_column}.
        nproc: parallel workers.
        model_version: "V1" (30M) or "V2" (104M pretraining).

    Returns:
        Path to the `<prefix>.dataset` directory.
    """
    from geneformer import TranscriptomeTokenizer

    output_dir = ensure_dir(output_dir)
    log.info(
        f"Tokenizing {input_dir} → {output_dir}/{output_prefix}.dataset "
        f"(model_version={model_version})"
    )

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=dict(custom_attrs) if custom_attrs else {},
        nproc=nproc,
        model_version=model_version,
    )
    tk.tokenize_data(
        data_directory=str(input_dir),
        output_directory=str(output_dir),
        output_prefix=output_prefix,
        file_format="h5ad",
    )

    dataset_path = output_dir / f"{output_prefix}.dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Expected tokenized dataset at {dataset_path}")
    log.info(f"Tokenized dataset ready: {dataset_path}")
    return dataset_path


def tokenize_anndata(
    adata: ad.AnnData,
    workdir: str | Path,
    output_prefix: str,
    custom_attrs: Mapping[str, str] | None = None,
    nproc: int = 8,
    model_version: str = "V2",
) -> Path:
    """Convenience wrapper: adata → dataset in one call."""
    workdir = Path(workdir)
    input_dir = workdir / "input"
    output_dir = workdir / "tokenized"
    prepare_input_dir(adata, input_dir)
    ds_path = tokenize(
        input_dir=input_dir,
        output_dir=output_dir,
        output_prefix=output_prefix,
        custom_attrs=custom_attrs,
        nproc=nproc,
        model_version=model_version,
    )
    # Clean up the temporary h5ad now that it's been tokenized.
    shutil.rmtree(input_dir, ignore_errors=True)
    return ds_path
