# RNA-seq Closed-Loop Target Discovery

Turn public RNA-seq data into a shortlist of testable drug/gene targets — using a
**closed loop** that builds a model, figures out *why and where* it fails, fixes
itself, and re-runs.

Built to be runnable two ways:

- **Classical pipeline** — statistics + one small ML model. Runs on a laptop, no GPU.
- **Geneformer pipeline** — a foundation model (transformer) for causal gene
  attribution. Needs an NVIDIA GPU.

---

## What this is (plain language)

Every cell is a tiny factory running ~20,000 "recipes" (genes). RNA-seq takes a
snapshot of which recipes are active. This project compares two states of cells
(e.g. **senescent = old/retired** vs. **young = healthy**) and asks:

> Which genes, if we switched them off or on, would push an old cell back toward
> young?

The answer is a **ranked list of intervention targets** — a hypothesis for a wet-lab
team to test, not a finished result.

## The core idea: a closed loop

Most tools just rank genes once and stop. This project does something different —
it **checks its own work**:

```
1. build a predictive model
2. infer mechanism (genes → pathways → regulators)
3. FAILURE ANALYSIS: where and why does the model break?
4. refine (data / labels / features / model)
5. re-train
6. repeat until the hypothesis is stable
```

The failure-analysis step is the differentiator. It doesn't just say "the model is
76% accurate" — it says "the model is perfect on 3 cell lines and useless on 2, and
here's the biological reason why."

## The pipeline, stage by stage

| # | Stage | What it does | Tool |
|---|---|---|---|
| 1 | Data ingestion | Download public RNA-seq, load into a matrix | CELLxGENE Census / GEO |
| 2 | Preprocessing | Filter low-quality genes, normalize | scanpy |
| 3 | Predictive model | Train a classifier: "senescent or young?" | logistic regression (classical) **or** Geneformer (GPU) |
| 4 | Gene attribution | Which genes drive the prediction? | t-test (classical) **or** in silico perturbation (Geneformer) |
| 5 | Mechanism | Group genes into pathways + upstream regulators | gseapy + decoupler/CollecTRI |
| 6 | Failure analysis | Where/why does the model break? | slice eval, batch detection, stability |
| 7 | Refine + retrain | Fix what failed, run again | the loop (`run_loop.py`) |
| 8 | Target scoring | Rank genes into "inhibit" vs "activate" | custom + DepMap/Open Targets |
| 9 | Next experiment | Propose which wet-lab test to run next | active learning (BALD, coverage) |

## Two ways to run it

### Option A — Classical (CPU, works today)

No GPU needed. Runs in ~1–2 minutes on a laptop.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py      # fetch + decompress the data
python demo_pipeline.py              # full classical pipeline
python generalize.py                 # generalization proof (leave-one-cell-line-out)
python make_figures.py               # regenerate the charts
```

Outputs land in `results/`. A self-contained report is at `report.html`.

### Option B — Geneformer (GPU)

The foundation-model upgrade. Requires an NVIDIA GPU (16 GB+ recommended; an RTX
5090 with 32 GB is more than enough).

```bash
pip install -r requirements.txt -r requirements-gpu.txt
pip install git+https://huggingface.co/ctheodoris/Geneformer

python scripts/00_fetch_geneformer.py --model gc-30M-i2048   # download checkpoint
python scripts/01_pull_census.py --config configs/data/senescence.yaml
python scripts/02_tokenize.py --input data/senescence/census_slice.h5ad --workdir data/senescence --prefix senescence
python scripts/03_finetune.py --config configs/train/senescence_cls.yaml
python scripts/04_state_embs.py --config configs/mechanism/senescence.yaml
python scripts/05_isp.py --config configs/mechanism/senescence.yaml
python scripts/06_mechanism.py --isp-stats-csv outputs/senescence/isp_stats/senescence.csv --output-dir outputs/senescence/mechanism/iter1
python scripts/07_score_targets.py --config configs/scoring/senescence.yaml
python scripts/08_failure_and_active.py --predictions-pkl outputs/senescence/finetune/runs/ksplit1/predictions.pkl --isp-stats-csv outputs/senescence/isp_stats/senescence.csv --output-report outputs/senescence/failure/iter1_report.json --acquisition-config configs/active/senescence.yaml
```

Or run the whole loop at once:

```bash
python scripts/run_loop.py \
    --data-cfg configs/data/senescence.yaml \
    --train-cfg configs/train/senescence_cls.yaml \
    --mech-cfg configs/mechanism/senescence.yaml \
    --score-cfg configs/scoring/senescence.yaml \
    --active-cfg configs/active/senescence.yaml \
    --iterations 2
```

## Docker

Build a GPU image (requires `nvidia-container-toolkit` on the host):

```bash
docker build -t rnaseq-loop .
docker run --gpus all -it \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/outputs:/app/outputs \
  rnaseq-loop
```

**RTX 5090 note:** that card is Blackwell (compute capability 12.0). It needs
PyTorch 2.6+ and CUDA 12.8+. If you hit a "no kernel image" error, bump the base
image in the `Dockerfile` to a CUDA 12.8 build.

## File structure

```
├── demo_pipeline.py        # classical pipeline (CPU)
├── generalize.py           # generalization proof (CPU)
├── make_figures.py         # chart generation
├── rnaseq_loop/            # Geneformer library (GPU)
│   ├── train/finetune.py   #   fine-tune Geneformer classifier
│   ├── mechanism/isp.py    #   in silico perturbation
│   ├── mechanism/tf_activity.py
│   ├── failure/slice_eval.py
│   ├── active/acquisition.py
│   ├── scoring/            #   target scoring + priors
│   └── tokenize/           #   AnnData → Geneformer format
├── scripts/                # CLI entry points (one per stage)
│   ├── download_data.py
│   ├── 00_fetch_geneformer.py … 08_failure_and_active.py
│   └── run_loop.py         # the closed-loop driver
├── configs/                # Hydra YAML configs
├── tests/                  # pytest suite
├── requirements.txt        # CPU deps
├── requirements-gpu.txt    # Geneformer/GPU deps
├── Dockerfile
└── report.html             # self-contained demo report
```

## Results so far (classical pipeline)

On public data ([GSE63577](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63577),
young vs. senescent fibroblasts):

- **16 of 20** known senescence markers and senolytic targets recovered correctly,
  including the drug targets BCL-xL and MDM2.
- **76.7% held-out accuracy** (leave-one-cell-line-out) vs. 50% chance — the model
  generalizes to cell lines it has never seen.
- Failure analysis surfaced a real finding: **senescence is heterogeneous** across
  cell lines (the model is perfect on 3 lines, at chance on 2).

## Glossary (plain-English)

- **Gene** — a "recipe" in a cell's DNA.
- **Gene expression** — whether a recipe is actively being used.
- **RNA-seq** — a snapshot of which recipes are active in a cell.
- **Senescence** — cells that stop dividing and become "zombie" cells (a driver of aging).
- **Senolytic** — a drug that selectively kills senescent cells.
- **Differential expression** — comparing two states to find which genes change.
- **Pathway** — a team of genes working together on one job.
- **Transcription factor (TF)** — a "master switch" that turns many genes on/off.
- **p-value** — probability a result is random chance (lower = more confident).
- **Batch effect** — a technical artifact (e.g. which lab/cell line) masquerading as biology.
- **Foundation model** — a large AI pre-trained on huge data (Geneformer ≈ "ChatGPT for cells").
- **In silico perturbation** — simulating "what if we switched off gene X?" inside the model.
- **Closed loop** — a system that checks its own work, fixes itself, and re-runs.

## References

- Data: [GSE63577](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63577)
- Geneformer: [Theodoris et al., Nature 2023](https://www.nature.com/articles/s41586-023-06139-9) · [model card](https://huggingface.co/ctheodoris/Geneformer)
- Pathways: [MSigDB Hallmark](https://www.gsea-msigdb.org/gsea/msigdb/)
- Regulators: [CollecTRI](https://github.com/saezlab/CollecTRI)
