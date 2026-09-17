# Fibroblast pseudobulk experiment

## Question

Can raw single-cell counts be combined by biological sample and used to train a
sample-level fibroblast senescence classifier that transfers from GSE226225 to
the independent GSE282425 study?

## What was done

This is true pseudobulking, not averaging Geneformer predictions. Raw gene
counts were summed across all fibroblasts belonging to the same sample:

```text
raw counts from all cells in one sample
                 ↓ sum by gene
one pseudobulk gene-expression profile
                 ↓ normalize and classify
one prediction for the sample
```

The training data contained 37,998 cells from 9 GSE226225 samples:

- 2 proliferating samples containing 12,126 cells in total;
- 7 senescent samples containing 25,872 cells in total.

The external benchmark contained 3,097 cells from 4 GSE282425 samples:

- 2 proliferating samples containing 1,741 cells in total;
- 2 senescent samples containing 1,356 cells in total.

All available cells from each selected sample were included in its pseudobulk
profile. This differs from Geneformer fine-tuning, for which cells had been
downsampled to obtain balanced cell classes.

## Leakage-safe analysis

The two studies shared 15,316 Ensembl genes. Counts were converted to counts per
million and transformed as log(1 + CPM). Using GSE226225 only, genes were
required to have at least 1 CPM in at least 2 training samples. This retained
12,680 genes. The 100 genes with the largest absolute Welch t statistic were
then selected using the 9 training samples only.

A class-balanced, L2-regularized logistic-regression classifier was fitted to
the 9 training pseudobulks. The regularization value was fixed at C = 0.1 and
the decision threshold was fixed at 0.5. GSE282425 counts and labels were not
used for feature filtering, feature selection, fitting, or threshold selection.

## External sample predictions

| Sample | True condition | Cells aggregated | P(senescent) | Prediction | Correct |
|---|---|---:|---:|---|---|
| GSM8642880 | Proliferating, 2D | 1,187 | 0.470 | Proliferating | Yes |
| GSM8642881 | Proliferating, 3D | 554 | 0.609 | Senescent | No |
| GSM8642882 | Senescent, 2D | 1,060 | 0.673 | Senescent | Yes |
| GSM8642883 | Senescent, 3D | 296 | 0.806 | Senescent | Yes |

## Comparison with Geneformer probability pooling

| Method | Training unit | Correct samples | Accuracy | Balanced accuracy | AUROC | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Original mixed-cell Geneformer | Cells; probabilities averaged by sample | 2 / 4 | 0.500 | 0.500 | 0.250 | 0.333 |
| Fibroblast-only Geneformer | Cells; probabilities averaged by sample | 2 / 4 | 0.500 | 0.500 | 0.750 | 0.333 |
| **Raw-count pseudobulk logistic regression** | **Biological samples** | **3 / 4** | **0.750** | **0.750** | **1.000** | **0.733** |

The pseudobulk classifier correctly identified both senescent samples and one
of the two proliferating samples. Its sensitivity was 1.000 and its specificity
was 0.500. Unlike the fibroblast Geneformer model, it did not classify every
sample as senescent.

## Conclusion

The professor's pseudobulk suggestion has now been implemented. In this
preliminary external comparison, pseudobulking improved threshold-based
sample classification from 2 of 4 correct samples with Geneformer probability
pooling to 3 of 4. Balanced accuracy increased from 0.500 to 0.750, and AUROC
increased from 0.750 for fibroblast-only Geneformer pooling to 1.000 for the
pseudobulk classifier.

These numbers must be interpreted cautiously. AUROC = 1.000 is based on only
four test samples, because thousands of cells collapse to four experimental
units. The training set also contains only nine samples, including just two
proliferating samples, and all are from the WI-38 cell line. The four external
samples come from one dermal-fibroblast donor. Consequently, the result is
promising but not evidence of production-ready generalization.

GSE282425 remains independent of model fitting, but its outcomes have now been
examined in multiple experiments. A new labelled fibroblast study with multiple
donors should therefore be reserved untouched for final confirmation. More
independent training samples should also be added before treating the
pseudobulk classifier or Geneformer as a deployable product.
