"""Train and use a deployable Geneformer cell-state classifier.

Heavy GPU dependencies are imported inside functions so the CPU-only package
and its tests remain usable without the Geneformer environment.
"""
from __future__ import annotations

import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rnaseq_loop.train.finetune import FinetuneConfig, _quantize_config, _training_args
from rnaseq_loop.utils import ensure_dir, get_logger, save_json

log = get_logger(__name__)


def load_label_mapping(path: str | Path) -> dict[int, str]:
    """Load an ``id -> class name`` map from JSON or pickle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
    else:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    mapping = {int(key): str(value) for key, value in payload.items()}
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Label-map IDs must be contiguous and start at zero")
    return mapping


def _safe_column(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def prediction_frame(
    probabilities: np.ndarray,
    label_mapping: Mapping[int, str],
    metadata: Mapping[str, Sequence[Any]] | None = None,
) -> pd.DataFrame:
    """Build an auditable prediction table from an N x C probability matrix."""
    probabilities = np.asarray(probabilities, dtype=float)
    mapping = {int(key): str(value) for key, value in label_mapping.items()}
    if probabilities.ndim != 2 or probabilities.shape[1] != len(mapping):
        raise ValueError("probabilities must have one column per mapped class")
    predicted_ids = probabilities.argmax(axis=1)
    frame = pd.DataFrame({
        "row_index": np.arange(len(probabilities)),
        "predicted_class_id": predicted_ids,
        "predicted_label": [mapping[int(index)] for index in predicted_ids],
        "confidence": probabilities.max(axis=1),
    })
    for class_id, label in sorted(mapping.items()):
        frame[f"probability_{_safe_column(label)}"] = probabilities[:, class_id]
    if metadata:
        for name, values in metadata.items():
            if len(values) != len(frame):
                raise ValueError(f"Metadata column {name!r} has the wrong length")
            frame[str(name)] = list(values)
    return frame


def validate_anndata(adata: Any) -> None:
    """Validate Geneformer's raw-count AnnData input contract."""
    if "ensembl_id" not in adata.var.columns:
        raise ValueError("New AnnData must contain var['ensembl_id'] with Ensembl gene IDs")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("New AnnData contains no cells or no genes")
    if adata.var["ensembl_id"].isna().any():
        raise ValueError("var['ensembl_id'] contains missing values")
    values = adata.X.data if hasattr(adata.X, "data") and not isinstance(adata.X, np.ndarray) else np.asarray(adata.X).ravel()
    if len(values) and (not np.isfinite(values).all() or np.min(values) < 0):
        raise ValueError("Expression matrix must contain finite, non-negative raw counts")
    if len(values) and not np.allclose(values, np.rint(values), atol=1e-6):
        raise ValueError("Expression matrix appears normalized; Geneformer requires raw integer counts")


def prepare_new_anndata(
    input_h5ad: str | Path,
    workdir: str | Path,
    prefix: str,
    metadata_columns: Sequence[str] = (),
    *,
    model_version: str = "V2",
    nproc: int = 8,
) -> Path:
    """Validate and tokenize an unseen raw-count h5ad without requiring labels."""
    import anndata as ad
    from scipy import sparse

    from rnaseq_loop.tokenize import tokenize_anndata

    adata = ad.read_h5ad(input_h5ad)
    validate_anndata(adata)
    adata = adata.copy()
    if "n_counts" not in adata.obs.columns:
        totals = np.asarray(adata.X.sum(axis=1)).ravel() if sparse.issparse(adata.X) else np.asarray(adata.X).sum(axis=1)
        adata.obs["n_counts"] = totals
    adata.obs["_input_cell_id"] = adata.obs_names.astype(str)
    custom_attrs = {"_input_cell_id": "cell_id"}
    custom_attrs.update({name: name for name in metadata_columns if name in adata.obs.columns})
    return tokenize_anndata(
        adata=adata,
        workdir=workdir,
        output_prefix=prefix,
        custom_attrs=custom_attrs,
        nproc=nproc,
        model_version=model_version,
    )


def train_deployment_model(
    cfg: FinetuneConfig,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Fine-tune a fresh Geneformer classifier on every labeled training cell.

    Hyperparameters must be chosen before calling this function. Evaluation
    remains the responsibility of held-out/CV runs; this final fit has no eval
    split and therefore does not produce an unbiased performance estimate.
    """
    from datasets import load_from_disk
    from geneformer import Classifier

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Deployment directory is not empty: {output_dir}. "
            "Choose a new versioned output directory."
        )
    ensure_dir(output_dir)
    prepared_dir = ensure_dir(output_dir / "prepared")
    model_dir = output_dir / "model"

    quantize: bool | dict = _quantize_config() if cfg.quantize_4bit_lora else False
    classifier = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": cfg.state_key, "states": cfg.states},
        filter_data=cfg.filter_data,
        training_args=_training_args(cfg),
        freeze_layers=cfg.freeze_layers,
        num_crossval_splits=cfg.num_crossval_splits,
        split_sizes=cfg.split_sizes,
        stratify_splits_col=None,
        forward_batch_size=cfg.forward_batch_size,
        model_version=cfg.model_version,
        quantize=quantize,
        nproc=cfg.nproc,
        ngpu=cfg.ngpu,
    )

    log.info("Preparing all labeled cells for final training")
    classifier.prepare_data(
        input_data_file=cfg.tokenized_dataset,
        output_directory=str(prepared_dir),
        output_prefix=cfg.output_prefix,
        test_size=0,
    )
    labeled = prepared_dir / f"{cfg.output_prefix}_labeled.dataset"
    prepared_mapping = prepared_dir / f"{cfg.output_prefix}_id_class_dict.pkl"
    train_data = load_from_disk(str(labeled))
    label_mapping = load_label_mapping(prepared_mapping)

    log.info(f"Training final Geneformer model on all {len(train_data):,} labeled cells")
    classifier.train_classifier(
        model_directory=cfg.model_directory,
        num_classes=len(label_mapping),
        train_data=train_data,
        eval_data=None,
        output_directory=str(model_dir),
        predict=False,
    )
    shutil.copy2(prepared_mapping, model_dir / "id_class_dict.pkl")
    save_json({str(key): value for key, value in label_mapping.items()}, model_dir / "id_class_dict.json")
    manifest = {
        "artifact_type": "geneformer_cell_classifier",
        "model_version": cfg.model_version,
        "base_model": cfg.model_directory,
        "state_key": cfg.state_key,
        "classes": label_mapping,
        "training_examples": len(train_data),
        "training_policy": "all labeled cells; no internal evaluation split",
        "tokenized_training_dataset": cfg.tokenized_dataset,
        "seed": cfg.seed,
    }
    save_json(manifest, model_dir / "deployment_manifest.json")
    return {"model": model_dir, "prepared": prepared_dir, "manifest": model_dir / "deployment_manifest.json"}


def predict_tokenized(
    model_dir: str | Path,
    tokenized_dataset: str | Path,
    label_map: str | Path,
    *,
    metadata_columns: Sequence[str] = (),
    batch_size: int = 1,
    device: str = "auto",
    model_version: str = "V2",
) -> pd.DataFrame:
    """Predict class probabilities for an unlabeled Geneformer dataset."""
    import torch
    from datasets import load_from_disk
    from geneformer import (
        TOKEN_DICTIONARY_FILE,
        TOKEN_DICTIONARY_FILE_30M,
    )
    from transformers import BertForSequenceClassification

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    mapping = load_label_mapping(label_map)
    dataset = load_from_disk(str(tokenized_dataset))
    if len(dataset) == 0:
        raise ValueError("Tokenized prediction dataset is empty")
    available_metadata = [name for name in metadata_columns if name in dataset.column_names]
    absent = sorted(set(metadata_columns) - set(available_metadata))
    if absent:
        log.warning(f"Ignoring metadata columns absent from tokenized data: {absent}")
    metadata = {name: dataset[name] for name in available_metadata}

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = BertForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=len(mapping),
    ).to(resolved_device)
    model.eval()
    max_length = int(model.config.max_position_embeddings)
    dictionary_file = TOKEN_DICTIONARY_FILE_30M if model_version == "V1" else TOKEN_DICTIONARY_FILE
    with Path(dictionary_file).open("rb") as handle:
        token_dictionary = pickle.load(handle)
    pad_token_id = int(token_dictionary["<pad>"])

    all_probabilities = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        sequences = [
            torch.tensor(dataset[index]["input_ids"][:max_length], dtype=torch.long)
            for index in range(start, stop)
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad_token_id,
        ).to(resolved_device)
        attention_mask = input_ids.ne(pad_token_id).long()
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_probabilities.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
    return prediction_frame(np.concatenate(all_probabilities), mapping, metadata)
