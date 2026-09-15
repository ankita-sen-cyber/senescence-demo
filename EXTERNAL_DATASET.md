# Independent external dataset: GSE282425

`GSE282425` is reserved as a **test-only study**. It must not be added to the
training data or used to tune thresholds, features, epochs, or other model
settings before its final evaluation.

The study contains 10x 3' single-cell RNA-seq from primary human dermal
fibroblasts from one 42-year-old donor. Its four experimentally defined samples
cross Early Proliferative (EP) or Deeply Senescent (DS) state with 2D or 3D
culture. The barcode suffixes published in the GEO record identify those four
samples, so labels are not inferred from the expression matrix.

Source: [NCBI GEO GSE282425](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282425)
(PubMed 39810225).

## Download and convert

Run from the repository root in the `senescence-gpu` environment:

```bash
python scripts/11_download_external_gse282425.py
```

This downloads the official matrix, barcodes, and features into
`data/external/GSE282425/raw`, verifies their SHA-256 checksums, and writes:

```text
data/external/GSE282425/GSE282425_external.h5ad
data/external/GSE282425/manifest.json
```

The GEO feature file contains `ENS_ID` placeholders rather than real Ensembl
IDs. The converter explicitly maps its gene-symbol column through Geneformer's
V2 gene-name dictionary, combines duplicate mappings, and records how many
features were mapped. The output contains raw integer counts and the required
`var['ensembl_id']` and `obs['n_counts']` fields.

## Run the frozen model

```bash
python scripts/10_predict_new.py \
  --input-h5ad data/external/GSE282425/GSE282425_external.h5ad \
  --model-dir outputs/senescence/deployment/v1/model \
  --output-csv outputs/senescence/predictions/GSE282425_predictions.csv \
  --metadata dataset_id study_id sample_id donor_id cell_type \
             culture_dimension senescence_label label_source \
  --device cuda \
  --batch-size 4
```

The existing `v1` deployment model was trained before this study was imported,
so this run is a genuine study-level external test. Do not interpret it as
donor-level generalization: all four samples come from a single donor. Also
report the 2D and 3D sample results separately because culture is an important
domain shift in this study.

## Reproduce scoring and method comparison

After generating the Geneformer predictions, run:

```bash
python scripts/12_benchmark_external.py
```

Every method is trained only on the existing 70-cell census training matrix;
all 3,097 GSE282425 cells remain test-only. Classical feature selection is fit
on the training cells only. The command writes cell predictions, per-sample
results, and the comparison table to `results/external/GSE282425/`.

The one-shot result for the frozen `v1` model is:

| Method | Accuracy | Balanced accuracy | AUROC | Macro F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Logistic regression, top 100 genes | 0.468 | 0.505 | 0.521 | 0.436 | 0.012 |
| Linear SVM, top 100 genes | 0.463 | 0.504 | 0.541 | 0.421 | 0.012 |
| Training-prior dummy | 0.562 | 0.500 | 0.500 | 0.360 | 0.000 |
| Nearest centroid, top 100 genes | 0.430 | 0.482 | 0.491 | 0.346 | -0.066 |
| Geneformer `v1` | 0.422 | 0.472 | 0.455 | 0.342 | -0.096 |

Geneformer predicted senescence for 2,821 of 3,097 cells. It recovered 1,193
of 1,356 senescent cells but produced 1,628 false positives among 1,741
proliferating cells. This is evidence that the current model does **not**
generalize to this independent fibroblast study. The most likely issue is the
large training-domain mismatch: the current 70 training cells are lung cells
with score-derived labels, whereas GSE282425 contains experimentally labelled
dermal fibroblasts.

Because these labels have now been examined, GSE282425 must not be used to tune
the next model and then reported again as an untouched test. Use it as a fixed
reported benchmark, perform model selection on other training studies, and
reserve another independent labelled study for the final confirmation. The
four samples also come from one donor, so cell-level sample size must not be
mistaken for 3,097 independent biological replicates.
