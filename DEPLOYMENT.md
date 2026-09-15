# Training Geneformer for unseen datasets

There are two deliberately separate model stages:

1. **Evaluate and choose settings** with held-out donors, cell types, or studies.
   Never use the final deployment fit to report accuracy.
2. **Train the deployment model** once on all labeled cells after the settings
   are fixed. This maximizes the data available to the model that will receive
   new samples.

The current `outputs/senescence/finetune/.../ksplit1` model is a validation-fold
artifact. `scripts/09_train_final.py` creates the separate all-data artifact.

## 1. Activate the GPU environment

Follow `RUN_ON_GPU.md`, then verify that the active Python—not only
`nvidia-smi`—can use the RTX 5090:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Train a versioned final model

```bash
python scripts/09_train_final.py \
  --config configs/train/senescence_cls.yaml \
  --output-dir outputs/senescence/deployment/v1
```

The output bundle contains:

- `model.safetensors` and `config.json`;
- `id_class_dict.json` and `.pkl`, fixing the class order;
- `deployment_manifest.json`, recording the training dataset and policy;
- the fully prepared labeled training dataset for auditability.

The command refuses to overwrite a non-empty model version. Use `v2`, `v3`,
and so on when the training data or settings change.

## 3. Predict a new unlabeled dataset

The safest input is raw-count single-cell AnnData. It must have:

- a non-negative integer count matrix in `.X`;
- Ensembl gene IDs in `.var['ensembl_id']`;
- cells as rows and genes as columns.

`n_counts` is computed when absent. The command validates that the matrix does
not appear normalized, tokenizes with the same Geneformer version, and retains
the AnnData observation names as `cell_id`.

```bash
python scripts/10_predict_new.py \
  --input-h5ad data/external/new_cells.h5ad \
  --model-dir outputs/senescence/deployment/v1/model \
  --output-csv outputs/senescence/predictions/new_cells.csv \
  --metadata dataset_id donor_id cell_type \
  --device cuda \
  --batch-size 1
```

For an already tokenized dataset:

```bash
python scripts/10_predict_new.py \
  --tokenized-dataset data/external/tokenized/new_cells.dataset \
  --model-dir outputs/senescence/deployment/v1/model \
  --output-csv outputs/senescence/predictions/new_cells.csv \
  --device cuda
```

The CSV contains the predicted label, confidence, and a probability for every
class. Start with batch size 1 on long cell sequences; increase it only after
checking GPU memory.

## What “works on unseen data” requires

Training on all current cells creates a usable artifact, but it does not by
itself establish external generalization. The current labeled pilot contains
too few cells/studies to cover likely laboratory, tissue, platform, and donor
shifts. For a credible unseen-dataset model:

1. combine several independently produced, raw-count training studies;
2. harmonize Ensembl IDs and use the same biological label definition;
3. reserve one complete study as the untouched external test set;
4. choose epochs/settings using only grouped validation within the remaining
   studies;
5. report per-study AUROC, balanced accuracy, calibration, and abstention rate;
6. retrain the versioned deployment model on all eligible labeled training
   studies only after evaluation is complete.

If a new dataset is far outside the training distribution, low confidence is
not sufficient to guarantee that the prediction is safe. Review predictions by
study/cell type and add an explicit out-of-distribution or abstention policy
before operational use.
