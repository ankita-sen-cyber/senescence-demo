# Senescence-Reversal Target Discovery — Demo

A CPU-only, closed-loop pipeline that turns public RNA-seq data into a ranked,
mechanism-grounded shortlist of intervention targets. Built as a demo for wet-lab
review — no GPU required, runs on a laptop in minutes.

## What it does

Given young vs. senescent fibroblasts, the pipeline runs a closed loop:

1. **Differential expression** — which genes change between senescent and young cells
2. **Failure analysis** — is the signal real biology or a technical artifact (batch)?
3. **Refinement** — if batch confounds the result, re-run with a batch-aware meta-analysis
4. **Pathway enrichment** — which biological programs are enriched (MSigDB Hallmark)
5. **TF regulon analysis** — which transcription factors drive the program (CollecTRI)
6. **Target scoring** — rank genes into "inhibit" (senolytic-style) vs. "activate" targets

The headline result: **16 of 20 known senescence markers and senolytic targets are
recovered correctly** — including the drug targets BCL-xL (BCL2L1) and MDM2 — without
the model being told they matter.

## The closed loop in action

The pipeline's differentiator is that it *diagnoses its own failure*. On the naive
first pass, cell line (batch) explains more variance than the biological condition
(F = 17.6 vs. 8.2). The failure-analysis layer detects this, and the refined
batch-aware pass sharpens the signal from noise into the canonical senescence program
(p53, IL-6/JAK/STAT3, interferon).

## Data

Public dataset [GSE63577](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63577)
(JenAge project, [PLOS ONE 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154531)):

- 30 samples — 5 fibroblast cell lines (BJ, IMR-90, WI-38, HFF, MRC-5)
- Young proliferating vs. replicatively senescent, 3 biological replicates each

## Quick start

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. download the data (~24 MB)
python scripts/download_data.py

# 3. run the pipeline (CPU, ~1-2 min)
python demo_pipeline.py

# 4. regenerate the figures
python make_figures.py
```

Outputs land in `results/` (CSV tables + PNG figures). A self-contained report with
embedded figures is at `report.html` (open in any browser).

## Outputs

| File | Contents |
|---|---|
| `results/de_iter0_pooled.csv` | Naive differential expression |
| `results/de_iter1_batch_aware.csv` | Refined (batch-aware) differential expression |
| `results/validation_iter1.csv` | Recovery of known markers/senolytics |
| `results/pathway_enrichment.csv` | MSigDB Hallmark GSEA results |
| `results/tf_activity.csv` | TF regulon enrichment (CollecTRI) |
| `results/top_targets.csv` | Ranked target shortlist |
| `results/failure_analysis.json` | Batch-vs-condition variance check |

## Limitations

- Bulk RNA-seq, 5 cell lines — a small in-vitro slice of senescence biology.
- Classical differential expression, not a foundation model. This is the interpretable
  baseline; the Geneformer in-silico-perturbation upgrade (needs GPU) is the next step.
- No wet-lab validation — the shortlist is a hypothesis, not a result.

## References

- Data: [GSE63577](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63577)
- Pathways: [MSigDB Hallmark](https://www.gsea-msigdb.org/gsea/msigdb/)
- Regulons: [CollecTRI](https://github.com/saezlab/CollecTRI)
- Source paper: [PLOS ONE 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154531)
