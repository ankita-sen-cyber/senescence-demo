"""Fetch Geneformer checkpoint from HuggingFace.

Usage:
  python scripts/00_fetch_geneformer.py --model V2-104M --out checkpoints/geneformer-V2-104M

Notes:
- Geneformer is distributed via the HF repo `ctheodoris/Geneformer`.
- The repo also includes the `geneformer` Python package; install with
  `pip install git+https://huggingface.co/ctheodoris/Geneformer`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


V2_SUBFOLDERS = {
    "V2-104M": "gf-12L-95M-i4096",     # 12-layer, ~95M params, 4096 input len (V2 default)
    "V2-104M-CLcancer": "gf-12L-95M-i4096_CLcancer",
    "V1-30M-6L": "gf-6L-30M-i2048",     # small V1 baseline
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(V2_SUBFOLDERS), default="V2-104M")
    p.add_argument("--out", type=Path, default=Path("checkpoints/geneformer-V2-104M"))
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    subfolder = V2_SUBFOLDERS[args.model]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ctheodoris/Geneformer / {subfolder}  →  {args.out}")

    snapshot_download(
        repo_id="ctheodoris/Geneformer",
        allow_patterns=[f"{subfolder}/*"],
        local_dir=str(args.out.parent),
        local_dir_use_symlinks=False,
    )

    # Move subfolder contents up to `args.out` for a clean model_directory path.
    src = args.out.parent / subfolder
    if src.exists() and src != args.out:
        for f in src.iterdir():
            f.rename(args.out / f.name)
        src.rmdir()

    print(f"Model ready at {args.out}")
    # Also install the geneformer package if not present.
    try:
        import geneformer  # noqa: F401
    except ImportError:
        print("Installing `geneformer` from HuggingFace...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://huggingface.co/ctheodoris/Geneformer",
        ])


if __name__ == "__main__":
    main()
