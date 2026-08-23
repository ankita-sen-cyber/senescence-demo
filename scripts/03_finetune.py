"""Fine-tune Geneformer using the official Classifier API."""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import yaml

from rnaseq_loop.train import FinetuneConfig, finetune
from rnaseq_loop.utils import get_logger, set_seed

log = get_logger("finetune")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--experiment", default="rnaseq-closed-loop-senescence")
    args = p.parse_args()

    cfg = FinetuneConfig(**yaml.safe_load(args.config.read_text()))
    set_seed(cfg.seed)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=cfg.output_prefix):
        mlflow.log_params({k: str(v) for k, v in cfg.__dict__.items()})
        result = finetune(cfg)
        log.info(f"Done. Artifacts at {result['runs']}")


if __name__ == "__main__":
    main()
