"""Run in silico perturbation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rnaseq_loop.mechanism import ISPConfig, run_isp, summarize_isp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip perturbation and only aggregate existing ISP pickle outputs.",
    )
    args = p.parse_args()

    top = yaml.safe_load(args.config.read_text())
    isp_yaml = top["isp"]

    # Split ISPConfig fields from run() args.
    run_keys = {"model_directory", "input_data_file", "output_directory", "output_prefix"}
    cfg = ISPConfig(**{k: v for k, v in isp_yaml.items() if k not in run_keys})

    if args.stats_only:
        isp_dir = Path(isp_yaml["output_directory"])
    else:
        isp_dir = run_isp(
            cfg,
            model_directory=isp_yaml["model_directory"],
            input_data_file=isp_yaml["input_data_file"],
            output_directory=isp_yaml["output_directory"],
            output_prefix=isp_yaml["output_prefix"],
        )

    stats_yaml = top["stats"]
    summarize_isp(
        isp_output_dir=stats_yaml["isp_output_dir"],
        stats_output_dir=stats_yaml["stats_output_dir"],
        output_prefix=stats_yaml["output_prefix"],
        cell_states_to_model=isp_yaml["cell_states_to_model"],
        genes_perturbed=isp_yaml["genes_to_perturb"],
        model_version=isp_yaml["model_version"],
    )
    print(f"ISP + stats written under {isp_dir.parent}")


if __name__ == "__main__":
    main()
