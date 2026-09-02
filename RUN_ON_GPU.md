# Running the Geneformer pipeline on an RTX 5090

Step-by-step guide to run the full foundation-model pipeline (fine-tuning + in
silico perturbation) on an NVIDIA RTX 5090 (32 GB, Blackwell architecture).

Estimated time for a first run: **1–2 hours** (mostly downloads and fine-tuning).

---

## 0. Prerequisites

The RTX 5090 is a **Blackwell** card (compute capability 12.0). It needs:

- **NVIDIA driver ≥ 570** (for CUDA 12.8)
- **CUDA 12.8+**
- **PyTorch 2.6+** (first version with Blackwell support)
- **Python 3.11**

Check your setup:

```bash
nvidia-smi                     # confirm the card + driver version
python --version               # should be 3.11
```

If `nvidia-smi` shows a driver older than 570, update the driver first.

---

## 1. Create the environment

```bash
cd senescence-demo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 2. Install PyTorch for Blackwell (CUDA 12.8)

Install PyTorch **before** the other packages, from the CUDA 12.8 index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify it sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and `NVIDIA GeForce RTX 5090`.

## 3. Install the rest

```bash
pip install -r requirements.txt -r requirements-gpu.txt
pip install -e ./Geneformer
```

> **Note:** you have 32 GB, so you do **not** need 4-bit quantization. The config
> already sets `quantize_4bit_lora: false`, which also sidesteps `bitsandbytes`
> compatibility issues on Blackwell.

## 4. Download the Geneformer checkpoint

```bash
python scripts/00_fetch_geneformer.py --model V2-104M
```

This downloads `Geneformer-V2-104M` (~104M parameters) into `checkpoints/geneformer-V2-104M`.
The 316M variant is available via `--model V2-316M` if you later scale up.

## 5. Pull data from CELLxGENE Census

```bash
python scripts/01_pull_census.py --config configs/data/senescence.yaml
```

This pulls a single-cell slice (lung tissue by default), scores each cell with a
senescence signature, and labels the top 25% "senescent" and bottom 25%
"proliferating". Output: `data/senescence/census_slice.h5ad`.

> **First-run tip:** the Census is large. The config caps at 20,000 cells — keep
> that for the first run, then raise `max_cells` once everything works.

## 6. Tokenize (AnnData → Geneformer format)

```bash
python scripts/02_tokenize.py \
    --input data/senescence/census_slice.h5ad \
    --workdir data/senescence \
    --prefix senescence
```

Produces `data/senescence/senescence.dataset` (Geneformer's tokenized format).

## 7. Fine-tune the classifier

```bash
python scripts/03_finetune.py --config configs/train/senescence_cls.yaml
```

Fine-tunes Geneformer to classify senescent vs. proliferating cells. On a 5090 this
takes **~30–60 minutes** for the 104M model.

**What to check when it finishes** (in `outputs/senescence/finetune/runs/`):

- `eval_macro_f1` should be **well above 0.5** (chance). Target ≥ 0.8.
- If it collapses to ~0.5, that's a *batch-overfitting* signal — the model is
  learning which dataset a cell came from, not the biology. That's the first
  failure-analysis finding, not a bug.

## 8. Extract state embeddings

```bash
python scripts/04_state_embs.py --config configs/mechanism/senescence.yaml
```

Extracts the embedding of the "senescent" and "proliferating" goal states — needed
for the perturbation step.

## 9. Run in silico perturbation

```bash
python scripts/05_isp.py --config configs/mechanism/senescence.yaml
```

This is the core "wow" step: for each gene, it simulates deleting it and measures
whether the cell shifts toward the goal state. Output: per-gene attribution scores
in `outputs/senescence/isp_stats/`.

## 10. Mechanism inference (pathways + regulators)

```bash
python scripts/06_mechanism.py \
    --isp-stats-csv outputs/senescence/isp_stats/senescence.csv \
    --output-dir outputs/senescence/mechanism/iter1
```

Groups the top attributed genes into pathways and transcription-factor regulators.

## 11. Score targets

```bash
python scripts/07_score_targets.py --config configs/scoring/senescence.yaml
```

Fuses the perturbation scores with external evidence (DepMap essentiality, Open
Targets) into a ranked target shortlist.

## 12. Failure analysis + active learning

```bash
python scripts/08_failure_and_active.py \
    --predictions-pkl outputs/senescence/finetune/runs/ksplit1/predictions.pkl \
    --isp-stats-csv outputs/senescence/isp_stats/senescence.csv \
    --output-report outputs/senescence/failure/iter1_report.json \
    --acquisition-config configs/active/senescence.yaml
```

Produces a failure report and proposes which experiments to run next.

---

## Or: run the whole loop at once

```bash
python scripts/run_loop.py \
    --data-cfg configs/data/senescence.yaml \
    --train-cfg configs/train/senescence_cls.yaml \
    --mech-cfg configs/mechanism/senescence.yaml \
    --score-cfg configs/scoring/senescence.yaml \
    --active-cfg configs/active/senescence.yaml \
    --iterations 2
```

This runs stages 1–3 once, then loops stages 4–12 twice. On later iterations you
can add newly-labeled data and re-run with `--skip-data-and-train` to reuse the
fine-tuned model.

---

## Troubleshooting (Blackwell-specific)

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA error: no kernel image is available` | PyTorch/CUDA too old for Blackwell | Reinstall PyTorch from the `cu128` index (step 2) |
| `torch.cuda.is_available()` is `False` | Driver too old or wrong PyTorch build | Update driver to ≥ 570; reinstall PyTorch |
| `bitsandbytes` import error | bitsandbytes lacks Blackwell support | Not needed — keep `quantize_4bit_lora: false` (you have 32 GB) |
| Out-of-memory during fine-tuning | Batch size too large | Lower `per_device_train_batch_size` in `configs/train/senescence_cls.yaml` |
| `ModuleNotFoundError: geneformer` | Package not installed | `pip install -e ./Geneformer` |
| Checkpoint download fails | Wrong subfolder name | Confirm the model name against the [HF repo](https://huggingface.co/ctheodoris/Geneformer) |

## What "success" looks like

1. Fine-tuned classifier with `eval_macro_f1 ≥ 0.8`.
2. In silico perturbation ranks known senescence genes (CDKN1A, SERPINE1, IL6,
   BCL2L1, MDM2) near the top of the attribution list.
3. The failure report identifies *where* the model is weak (e.g. a specific
   dataset or cell type) and the active-learning step proposes a next experiment.

If you get all three, you have a complete, runnable version of the closed-loop
pipeline — the foundation-model upgrade to the classical demo.
