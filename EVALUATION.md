# Generalization and method comparison

The evaluation is designed to answer two separate questions without mixing
them up:

1. **Does the classifier work on biological groups excluded from training?**
   `generalize.py` holds out one complete cell line at a time. Gene selection,
   scaling, and model fitting happen again inside each training fold, so the
   held-out cell line cannot influence the selected signature.
2. **Does it outperform simpler alternatives?** The same folds are used for a
   majority-class control, nearest centroid, linear SVM, top-gene logistic
   regression, and all-gene logistic regression.

This is within-study domain generalization: every test sample is outside its
model's training set, and its cell line was never seen by that model. It is not
independent-cohort validation because all five lines come from GSE63577. A
second study, ideally produced by another laboratory or assay, is still needed
before making an external-validity claim.

## Run

```bash
python scripts/download_data.py
python generalize.py --top-n 100 --bootstrap 2000 --seed 42
```

The command writes:

| Output | Purpose |
|---|---|
| `results/generalization_method_comparison.csv` | Pooled out-of-fold accuracy, balanced accuracy, AUROC, macro-F1, MCC, and 95% intervals |
| `results/generalization_by_group.csv` | Every method's metrics for each held-out cell line |
| `results/generalization_predictions.csv` | Auditable sample-level labels, predictions, scores, and fold-selected genes |
| `results/generalization_leave_one_out.csv` | Compatibility copy of the per-group results |

Confidence intervals resample whole cell lines, rather than pretending the 30
samples are independent. With only five groups these intervals should be
expected to be wide. Balanced accuracy is the primary comparison metric;
AUROC measures ranking quality independently of the classification threshold.

## Methods compared

| Name | Role |
|---|---|
| `dummy_majority` | Non-informative control; predicts the training-fold class prior |
| `nearest_centroid_top` | Simple distance-based expression-signature baseline |
| `linear_svm_top` | Alternative linear discriminative classifier |
| `logistic_top` | Main model using genes selected in the training fold only |
| `logistic_all` | Ablation showing whether top-gene selection helps |

Choose a subset when needed:

```bash
python generalize.py --methods dummy_majority logistic_top logistic_all
```

The reusable `rnaseq_loop.evaluation.evaluate_group_holdout` function accepts
any sample-by-gene matrix. For a multi-study benchmark, harmonize gene IDs and
normalization first, concatenate studies, and pass the study identifier as
`groups`; this produces leave-one-study-out rather than leave-one-cell-line-out
results with the same leakage controls.

## Interpretation guardrails

- Classification performance does not validate the causal target ranking.
  Target claims still need perturbational or wet-lab validation.
- Do not select the best method or `top_n` using these same five held-out
  results and then report them as unbiased. Any tuning requires nested grouped
  cross-validation or a separate validation cohort.
- The existing 76.7% result is a point estimate for top-100 logistic regression
  under the earlier run. Re-run the comparison to generate uncertainty and
  baseline results in the current environment.
