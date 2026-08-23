# RNA-seq closed-loop scaffold — architecture at a glance

```
                        ┌──────────────────────────────────────────────┐
                        │  CZ CELLxGENE Census   +   LINCS L1000 pool  │
                        └──────────────────────────────────────────────┘
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ (1) rnaseq_loop.data.census  ─ pull TileDB-SOMA slice → AnnData (raw)    │
   │ (2) rnaseq_loop.data.labels  ─ senescence / age binary label             │
   │ (3) rnaseq_loop.tokenize     ─ TranscriptomeTokenizer → .dataset         │
   └─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ (4) rnaseq_loop.train.finetune ─ Geneformer Classifier V2 (± 4-bit LoRA) │
   │       output: fine-tuned model + per-cell predictions.pkl           │
   └─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ (5) rnaseq_loop.mechanism.state_embs ─ EmbExtractor → state_embs_dict    │
   │ (6) rnaseq_loop.mechanism.isp        ─ InSilicoPerturber (delete/all)    │
   │ (7) rnaseq_loop.mechanism.tf_activity ─ decoupler + CollecTRI + PROGENy  │
   │                                    + GSEA Hallmark                  │
   └─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ (8) rnaseq_loop.scoring.priors ─ Open Targets GraphQL + DepMap CSV       │
   │ (9) rnaseq_loop.scoring.rank   ─ z-score fuse → target_score             │
   └─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ (10) rnaseq_loop.failure.slice_eval ─ per-slice AUROC, gap flag,         │
   │                                   bootstrap top-k stability,        │
   │                                   counterfactual probe              │
   │ (11) rnaseq_loop.active.acquisition ─ BALD + TF-prior disagreement +     │
   │                                   coverage → next-experiment picks  │
   └─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                             wet-lab / scPerturb feedback
                                        │
                                        └── back to (1) with expanded labels
```

## Verified API contracts

- **Geneformer `Classifier`** signature and methods (`prepare_data`, `validate`,
  `evaluate_saved_model`, `plot_predictions`, `quantize={bnb_config, peft_config}`)
  from <https://geneformer.readthedocs.io/en/latest/geneformer.classifier.html>.
- **Geneformer `InSilicoPerturber`** signature (`perturb_type`, `emb_mode="cls_and_gene"`,
  `cell_states_to_model`, `state_embs_dict`, `emb_layer=-1`) from
  <https://geneformer.readthedocs.io/en/latest/geneformer.in_silico_perturber.html>.
- **Geneformer `TranscriptomeTokenizer`** input requirements (`ensembl_id`,
  `n_counts`, `.h5ad` or `.loom` directory) from
  <https://geneformer.readthedocs.io/en/latest/geneformer.tokenizer.html>.
- **CZ CELLxGENE Census** access via `cellxgene-census` with `value_filter`
  syntax from <https://chanzuckerberg.github.io/cellxgene-census/>.
- **decoupler-py** `get_collectri`, `get_progeny`, `run_mlm`, `run_ulm` from
  <https://decoupler-py.readthedocs.io/>.
- **Open Targets Platform GraphQL v4** query shape from
  <https://platform-docs.opentargets.org/>.

## Where to iterate

- **Quality wedge**: `rnaseq_loop.mechanism.isp` + `rnaseq_loop.failure.counterfactual_probe`
  is what makes RNA-seq closed-loop different from a ranker. Invest here first.
- **Compute wedge**: enable `quantize_4bit_lora=True` in
  `configs/train/senescence_cls.yaml` to fit the V2-104M model on a 24 GB card.
- **Data wedge**: swap `configs/data/senescence.yaml` for an oncology config
  (TCGA-BRCA slice via a `Recount3` loader — add `rnaseq-closed-loop/data/recount3.py`).
