# Option 1: fibroblast Geneformer with sample-level probability pooling

## Question

Does fibroblast-only, experimentally labelled training improve Geneformer
transfer to the independent GSE282425 dermal-fibroblast study, and can cell
probabilities be pooled into one prediction per biological sample?

## Design

| Component | Implementation |
|---|---|
| Training study | GSE226225 |
| Training biology | WI-38 fetal-lung fibroblasts |
| Training labels | Experimental endpoints |
| Proliferating endpoints | Untreated control and etoposide day 0 |
| Senescent endpoints | Replicative, irradiation day 10, and etoposide day 10 |
| Excluded endpoints | Etoposide days 1, 2, 4, and 7 because the binary state is ambiguous |
| Training size | 14,000 cells from 9 samples |
| Class balance | 7,000 proliferating and 7,000 senescent cells |
| Model | Geneformer V2-104M; first 2 layers frozen |
| Fine-tuning | 1 epoch, batch size 6, learning rate 5e-5, seed 42 |
| External test | All 3,097 cells from GSE282425; never used for training |
| Sample pooling | Arithmetic mean of cell senescence probabilities |
| Decision rule | Fixed threshold of 0.5; not tuned on GSE282425 |

## Before-versus-after comparison

### Cell-level evaluation

| Model | Accuracy | Balanced accuracy | AUROC | Macro F1 | MCC | Cells called senescent |
|---|---:|---:|---:|---:|---:|---:|
| Original mixed-cell model | 0.422 | 0.472 | 0.455 | 0.342 | -0.096 | 2,821 / 3,097 |
| Fibroblast-only model | **0.438** | **0.500** | **0.770** | 0.307 | **-0.008** | 3,085 / 3,097 |

### Sample-level pooled evaluation

| Model | Samples | Accuracy | Balanced accuracy | AUROC | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Original mixed-cell model | 4 | 0.500 | 0.500 | 0.250 | 0.333 |
| Fibroblast-only model | 4 | 0.500 | 0.500 | **0.750** | 0.333 |

## New model: pooled sample predictions

| Sample | True condition | Cells | Mean P(senescent) | Cells above 0.5 | Prediction |
|---|---|---:|---:|---:|---|
| GSM8642880 | Proliferating, 2D | 1,187 | 0.999744 | 100.0% | Senescent |
| GSM8642881 | Proliferating, 3D | 554 | 0.990237 | 98.9% | Senescent |
| GSM8642882 | Senescent, 2D | 1,060 | 0.994479 | 99.4% | Senescent |
| GSM8642883 | Senescent, 3D | 296 | 0.999826 | 100.0% | Senescent |

## Interpretation

Fibroblast-only training materially improved ranking: cell-level AUROC rose
from 0.455 to 0.770 and sample-level AUROC rose from 0.25 to 0.75. This suggests
that the new model learned a more relevant fibroblast senescence signal.

The model is nevertheless not ready for classification at a fixed threshold.
Its probabilities are saturated near one, causing nearly every cell and all
four samples to be classified as senescent. Balanced accuracy therefore remains
at chance. AUROC can improve while thresholded classification fails because
AUROC measures ordering, whereas balanced accuracy depends on calibrated
probabilities and a decision threshold.

The threshold must not be increased using GSE282425 because that would tune the
model on its external test. Calibration requires a separate labelled validation
study, ideally containing multiple proliferating and senescent fibroblast
samples. After calibration is fixed, GSE282425 may remain a reported benchmark,
and another untouched study should provide final confirmation.

The experiment successfully implements Option 1: Geneformer processes cells
individually, and its probabilities are aggregated by biological sample. It is
probability pooling, not raw-count pseudobulking; raw pseudobulk should remain a
separate conventional baseline.
