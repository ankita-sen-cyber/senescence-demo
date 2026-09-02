"""Fine-tune Geneformer with the official `Classifier` API.

The Classifier signature (verified against
https://geneformer.readthedocs.io/en/latest/geneformer.classifier.html):

    Classifier(
        classifier: {"cell", "gene"},
        cell_state_dict: dict,            # e.g. {"state_key": "senescence_label",
                                          #        "states": ["senescent", "proliferating"]}
        filter_data: dict | None,
        training_args: dict | None,       # HuggingFace TrainingArguments kwargs
        freeze_layers: int,
        num_crossval_splits: {0, 1, 5},
        split_sizes: {"train": .., "valid": .., "test": ..},
        stratify_splits_col: str | None,
        forward_batch_size: int,
        model_version: {"V1", "V2"},
        quantize: bool | dict,            # {"bnb_config": BitsAndBytesConfig,
                                          #  "peft_config": LoraConfig} for 4/8-bit + LoRA
        nproc: int,
        ngpu: int,
    )

Then:
    Classifier.prepare_data(input_data_file, output_directory, output_prefix, ...)
    Classifier.validate(model_directory, prepared_input_data_file, id_class_dict_file,
                        output_directory, output_prefix, predict_eval=True, ...)
    Classifier.evaluate_saved_model(model_directory, id_class_dict_file, test_data_file,
                                    output_directory, output_prefix, predict=True)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from collections import Counter
from typing import Any

from rnaseq_loop.utils import ensure_dir, get_logger

log = get_logger(__name__)


@dataclass
class FinetuneConfig:
    """Config for a single fine-tuning run."""
    # Data
    tokenized_dataset: str                          # path to <prefix>.dataset
    output_dir: str                                 # runs/<experiment>/finetune
    output_prefix: str = "rnaseq-closed-loop"                   # run name for artifacts

    # Task
    state_key: str = "senescence_label"             # obs column carried through tokenization
    states: list[str] | str = field(default_factory=lambda: ["senescent", "proliferating"])
    filter_data: dict[str, list[Any]] | None = None
    stratify_splits_col: str | None = "dataset_id"   # split by study to detect batch overfitting

    # Model
    model_directory: str = "checkpoints/geneformer-V2-104M"
    model_version: str = "V2"
    freeze_layers: int = 2
    quantize_4bit_lora: bool = False                # enable for 4090 24GB

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 12
    per_device_eval_batch_size: int = 24
    learning_rate: float = 5e-5
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    logging_steps: int = 25
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_macro_f1"
    greater_is_better: bool = True
    seed: int = 0

    # Eval
    num_crossval_splits: int = 1
    split_sizes: dict[str, float] = field(default_factory=lambda: {
        "train": 0.8, "valid": 0.1, "test": 0.1
    })
    forward_batch_size: int = 100

    # Compute
    nproc: int = 8
    ngpu: int = 1


def _training_args(cfg: FinetuneConfig) -> dict[str, Any]:
    """Build the HuggingFace TrainingArguments dict Classifier expects."""
    return {
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "logging_steps": cfg.logging_steps,
        "eval_strategy": cfg.eval_strategy,
        "save_strategy": cfg.save_strategy,
        "load_best_model_at_end": cfg.load_best_model_at_end,
        "metric_for_best_model": cfg.metric_for_best_model,
        "greater_is_better": cfg.greater_is_better,
        "seed": cfg.seed,
        "report_to": [],
    }


def _quantize_config() -> dict[str, Any]:
    """4-bit NF4 quantization + LoRA — fits Geneformer-V2 on a 24GB card."""
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    import torch

    return {
        "bnb_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        "peft_config": LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
        ),
    }


def finetune(cfg: FinetuneConfig) -> dict[str, Path]:
    """
    Full fine-tuning cycle:
      1. Classifier.prepare_data → produces `<prefix>_labeled.dataset` and `_id_class_dict.pkl`
      2. Classifier.validate → cross-validates + saves model
      3. Classifier.evaluate_saved_model (via test split saved by prepare_data)
    """
    from geneformer import Classifier

    out = ensure_dir(cfg.output_dir)
    prep_dir = ensure_dir(out / "prepared")
    run_dir = ensure_dir(out / "runs")
    input_data_file = cfg.tokenized_dataset

    # HF datasets can stratify only by ClassLabel columns.
    # If a string/Value column is requested (e.g. dataset_id), encode it first.
    if cfg.stratify_splits_col:
        from datasets import ClassLabel, load_from_disk

        ds = load_from_disk(cfg.tokenized_dataset)
        strat_col = cfg.stratify_splits_col
        if strat_col not in ds.column_names:
            log.warning(
                f"stratify_splits_col='{strat_col}' not found in dataset; disabling stratification."
            )
            cfg.stratify_splits_col = None
        else:
            counts = Counter(ds[strat_col])
            if counts and min(counts.values()) < 2:
                log.warning(
                    f"stratify_splits_col='{strat_col}' has singleton groups in this pilot run; "
                    "disabling stratification."
                )
                cfg.stratify_splits_col = None
            
            feat = ds.features[strat_col]
            if cfg.stratify_splits_col and not isinstance(feat, ClassLabel):
                log.info(
                    f"Encoding '{strat_col}' as ClassLabel for stratified train/test split."
                )
                ds = ds.class_encode_column(strat_col)
                encoded_path = prep_dir / f"{cfg.output_prefix}_stratify_encoded.dataset"
                if encoded_path.exists():
                    shutil.rmtree(encoded_path)
                ds.save_to_disk(str(encoded_path))
                input_data_file = str(encoded_path)

    quantize: bool | dict = _quantize_config() if cfg.quantize_4bit_lora else False

    log.info(f"Building Classifier (V={cfg.model_version}, freeze={cfg.freeze_layers})")
    cc = Classifier(
        classifier="cell",
        cell_state_dict={"state_key": cfg.state_key, "states": cfg.states},
        filter_data=cfg.filter_data,
        training_args=_training_args(cfg),
        freeze_layers=cfg.freeze_layers,
        num_crossval_splits=cfg.num_crossval_splits,
        split_sizes=cfg.split_sizes,
        stratify_splits_col=cfg.stratify_splits_col,
        forward_batch_size=cfg.forward_batch_size,
        model_version=cfg.model_version,
        quantize=quantize,
        nproc=cfg.nproc,
        ngpu=cfg.ngpu,
    )

    log.info("Preparing labeled dataset")
    try:
        cc.prepare_data(
            input_data_file=input_data_file,
            output_directory=str(prep_dir),
            output_prefix=cfg.output_prefix,
            test_size=0,
        )
    except ValueError as e:
        msg = str(e)
        if cfg.stratify_splits_col and (
            "Minimum class count error" in msg
            or "least populated class" in msg
            or "minimum number of groups" in msg
        ):
            log.warning(
                "Stratified split is not feasible for this small pilot dataset; "
                "retrying without stratification."
            )
            cfg.stratify_splits_col = None
            cc.stratify_splits_col = None
            cc.prepare_data(
                input_data_file=input_data_file,
                output_directory=str(prep_dir),
                output_prefix=cfg.output_prefix,
                test_size=0,
            )
        else:
            raise

    labeled = prep_dir / f"{cfg.output_prefix}_labeled.dataset"
    id_class_dict = prep_dir / f"{cfg.output_prefix}_id_class_dict.pkl"

    log.info("Running Classifier.validate (train + eval + predictions)")
    all_metrics = cc.validate(
        model_directory=cfg.model_directory,
        prepared_input_data_file=str(labeled),
        id_class_dict_file=str(id_class_dict),
        output_directory=str(run_dir),
        output_prefix=cfg.output_prefix,
        predict_eval=True,
    )

    log.info(f"Cross-fold metrics: {all_metrics.get('all_metrics', all_metrics)}")

    return {
        "prepared": prep_dir,
        "runs": run_dir,
        "labeled_dataset": labeled,
        "id_class_dict": id_class_dict,
        "metrics": all_metrics,
    }
