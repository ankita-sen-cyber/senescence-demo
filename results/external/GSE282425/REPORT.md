# Independent external validation of the senescence classifier

## Evaluation design

| Component | Training data | External test data |
|---|---|---|
| Dataset | Current CELLxGENE census slice | GSE282425 (SenPred) |
| Size | 70 cells | 3,097 cells in 4 experimental samples |
| Biological context | Lung; multiple cell types | Primary human dermal fibroblasts |
| Labels | Score-derived labels: 44 proliferating, 26 senescent | Experimental conditions: 1,741 Early Proliferative, 1,356 Deeply Senescent |
| Study separation | Used for model fitting | Entire study excluded from fitting and preprocessing decisions |
| Model execution | Geneformer fine-tuning | Frozen-model inference on an RTX 5090 |

The external data and experimental labels come from
[NCBI GEO GSE282425](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282425).
The deployed model was trained before GSE282425 was downloaded or examined.

## External method comparison

All methods below used the same 70 training cells and the same 3,097 external
test cells. For the classical models, normalization and the selection of the
top 100 genes were fitted using training data only.

| Method | Accuracy | Balanced accuracy | AUROC | Macro F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Geneformer `v1` | 0.422 | 0.472 | 0.455 | 0.342 | -0.096 |
| Logistic regression (top 100 genes) | 0.468 | **0.505** | 0.521 | **0.436** | **0.012** |
| Linear SVM (top 100 genes) | 0.463 | 0.504 | **0.541** | 0.421 | 0.012 |
| Nearest centroid (top 100 genes) | 0.430 | 0.482 | 0.491 | 0.346 | -0.066 |
| Training-prior dummy baseline | **0.562** | 0.500 | 0.500 | 0.360 | 0.000 |

Balanced accuracy and AUROC are the primary measures because the external
classes are unequal. Raw accuracy alone is misleading here: the dummy model
obtains 0.562 simply by predicting the more common proliferating class.

## Geneformer confusion matrix

| Actual condition | Predicted proliferating | Predicted senescent | Total | Class recall |
|---|---:|---:|---:|---:|
| Early Proliferative | 113 | 1,628 | 1,741 | 0.065 specificity |
| Deeply Senescent | 163 | 1,193 | 1,356 | 0.880 sensitivity |
| **Total** | **276** | **2,821** | **3,097** | — |

Geneformer detected most experimentally senescent cells, but it classified
91.1% of all external cells as senescent. Its high sensitivity was therefore
accompanied by a very high false-positive rate.

## Geneformer results by experimental sample

| GEO sample | Condition | Culture | Cells | Correct | Accuracy | Mean senescence probability |
|---|---|---:|---:|---:|---:|---:|
| GSM8642880 | Early Proliferative | 2D | 1,187 | 38 | 0.032 | 0.968 |
| GSM8642881 | Early Proliferative | 3D | 554 | 75 | 0.135 | 0.864 |
| GSM8642882 | Deeply Senescent | 2D | 1,060 | 937 | 0.884 | 0.884 |
| GSM8642883 | Deeply Senescent | 3D | 296 | 256 | 0.865 | 0.863 |

## Interpretation

This experiment establishes that the repository can perform strict
study-level external evaluation. It also shows that the present model does not
yet generalize reliably to independently generated dermal-fibroblast data.
Geneformer and all classical comparators are close to chance, suggesting a
training-domain problem rather than a simple choice of classifier.

The principal limitation is the mismatch between 70 score-labelled lung
training cells and experimentally labelled dermal fibroblasts. The next model
should be trained on several experimentally labelled fibroblast studies, with
complete studies reserved for validation. GSE282425 should remain a fixed
reported benchmark, and a second untouched study should be retained for final
confirmation after model development.

The four external samples originate from one donor. Consequently, the 3,097
cells are not 3,097 independent biological replicates, and narrow cell-level
confidence intervals would overstate certainty.

## Short summary suitable for an email

> The evaluation was performed on an independent labelled scRNA-seq study
> (GSE282425; 3,097 dermal fibroblasts), with complete study-level separation
> from the training data. The frozen Geneformer model achieved a balanced
> accuracy of 0.472 and an AUROC of 0.455. It detected 88.0% of senescent
> cells but achieved only 6.5% specificity for proliferating cells. Classical
> baseline methods trained and tested using the same data separation also
> performed close to chance (best balanced accuracy: 0.505; best AUROC:
> 0.541). These findings indicate that the primary limitation is the small,
> biologically mismatched training dataset rather than the model architecture
> alone. Improved training should therefore use multiple experimentally
> labelled fibroblast studies, while retaining complete independent studies
> for validation and final testing.
