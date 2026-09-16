"""Aggregate Geneformer cell probabilities into sample-level predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.sample_aggregation import aggregate_probabilities, sample_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    cells = pd.read_csv(args.predictions)
    samples = aggregate_probabilities(cells, threshold=args.threshold)
    metrics = sample_metrics(samples)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output_csv, index=False)
    args.metrics_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(samples.to_string(index=False))
    print("\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
