"""Utilities to reattach the human-readable state column onto a prepared dataset.

`Classifier.prepare_data` renames the state key (e.g. `senescence_label`) to
`label` and encodes values as integers. Downstream stages (state embeddings +
in silico perturbation) still filter by the original string state column, so
this helper materializes a sibling dataset with that column re-added.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from rnaseq_loop.utils import get_logger

log = get_logger(__name__)


def resolve_model_directory(model_directory: str | Path) -> str:
    """Return a Geneformer-compatible model dir, resolving fine-tune run layouts.

    If ``model_directory`` already contains ``config.json``, it is returned as-is.
    Otherwise we search under it for a subdirectory containing ``config.json``,
    preferring the top-level ksplit dir over intermediate checkpoint dirs.
    """
    path = Path(model_directory)
    if (path / "config.json").is_file():
        return str(path)

    candidates = sorted(path.glob("**/config.json"))
    non_checkpoint = [c for c in candidates if "checkpoint-" not in c.parent.name]
    ordered = non_checkpoint or candidates
    ksplit_first = sorted(
        ordered, key=lambda c: (0 if c.parent.name.startswith("ksplit") else 1, str(c))
    )
    if ksplit_first:
        chosen = ksplit_first[0].parent
        log.info(f"Resolved model directory {model_directory} -> {chosen}")
        return str(chosen)
    raise FileNotFoundError(
        f"No config.json found under {model_directory}; "
        "provide a valid fine-tuned model path."
    )


def _find_id_class_dict(labeled_dataset: Path) -> Path:
    candidates = list(labeled_dataset.parent.glob("*_id_class_dict.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find *_id_class_dict.pkl next to {labeled_dataset}. "
            "Expected the prepared/ directory produced by Classifier.prepare_data."
        )
    return candidates[0]


def ensure_state_column(
    labeled_dataset: str | Path,
    state_key: str,
    id_class_dict_path: str | Path | None = None,
) -> str:
    """Return a path to a dataset that has `state_key` as a string column.

    If the input already has `state_key`, returns it unchanged. Otherwise a
    sibling dataset with `<original>_with_state.dataset` is produced.
    """
    from datasets import load_from_disk

    labeled_path = Path(labeled_dataset)
    ds = load_from_disk(str(labeled_path))

    if state_key in ds.column_names:
        return str(labeled_path)

    if "label" not in ds.column_names:
        raise KeyError(
            f"Dataset at {labeled_path} has neither '{state_key}' nor 'label'; "
            f"columns are {ds.column_names}."
        )

    icd_path = Path(id_class_dict_path) if id_class_dict_path else _find_id_class_dict(labeled_path)
    with icd_path.open("rb") as f:
        id_class_dict = pickle.load(f)

    def _add(example):
        example[state_key] = id_class_dict[example["label"]]
        return example

    ds = ds.map(_add)

    suffix = labeled_path.name.rstrip("/")
    if suffix.endswith(".dataset"):
        new_name = suffix[: -len(".dataset")] + "_with_state.dataset"
    else:
        new_name = suffix + "_with_state"
    new_path = labeled_path.parent / new_name

    if new_path.exists():
        import shutil

        shutil.rmtree(new_path)

    ds.save_to_disk(str(new_path))
    log.info(f"Reattached state column '{state_key}' -> {new_path}")
    return str(new_path)
