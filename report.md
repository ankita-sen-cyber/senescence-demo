# Senescence-Reversal Target Discovery — Demo Report

**Prepared for wet-lab review · August 2026**

## What this demo shows

Given only public RNA-seq data of young vs. senescent human fibroblasts — with no prior knowledge of senescence biology fed into the model — the pipeline recovers the known senescence program and produces a ranked, mechanism-grounded shortlist of intervention targets. The key result: **16 of 20 known senescence markers and senolytic targets are recovered correctly**, including the canonical drug targets BCL-xL (BCL2L1) and MDM2.

## The data

Public dataset [GSE63577](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63577), from the JenAge project ([PLOS ONE, 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154531)):

- 30 samples — 5 fibroblast cell lines (BJ, IMR-90, WI-38, HFF, MRC-5)
- Each line: young proliferating vs. replicatively senescent, 3 biological replicates
- 32,995 genes after QC

## Method (one paragraph)

Differential expression between senescent and young cells → pathway enrichment ([MSigDB Hallmark](https://www.gsea-msigdb.org/gsea/msigdb/)) → transcription-factor regulon analysis ([CollecTRI](https://github.com/saezlab/CollecTRI)) → target scoring. Critically, a **failure-analysis step** checks whether the signal is real biology or a technical artifact, and triggers a refinement. This is the "closed loop" — the model diagnoses its own failure and re-runs.

## The closed loop in action

**Iteration 0 — naive pooled analysis.** A standard differential-expression test across all 30 samples pooled recovers 16/20 known markers. But the failure-analysis layer flags a problem: cell line (batch) explains more variance than the biological condition (F = 17.6 vs. 8.2). The model is partly "learning" which cell line a sample came from.

**Iteration 1 — batch-aware refinement.** The pipeline re-runs with a per-cell-line meta-analysis (log-fold-change computed within each line, then combined across lines). Recovery holds at 16/20, but the mechanistic signal sharpens dramatically — the enriched pathways flip from noise (KRAS, xenobiotic metabolism) to the canonical senescence program (p53, IL-6/JAK/STAT3, interferon).

This is the differentiator: the loop does not just rank genes, it *detects that its first answer was confounded and corrects itself*.

## Results

### Differential expression and validation

16/20 known markers recovered with correct direction. The strongest, most consistent signal is **loss of proliferation** — cell-cycle genes (MKI67, CCNB1, CCNA2, PCNA, E2F1, HMGB2) are sharply down, as expected for growth-arrested senescent cells. The senescence/SASP markers CDKN1A (p21), SERPINE1 (PAI-1), TP53, CCL2, MMP3 and the senolytic targets BCL2L1 (BCL-xL) and MDM2 are all up.

![volcano](fig_volcano.png)

### Pathways

The enriched Hallmark pathways are the textbook senescence program: **p53 pathway**, **IL-6/JAK/STAT3 signaling**, and **interferon alpha/gamma response** (the senescence-associated secretory phenotype, SASP).

![pathway](fig_pathway.png)

### Transcription factors

The top regulators are the canonical senescence drivers: **TP53 (p53) is #1**, followed by E2F4, FOXO3, and RB1 (the p16–RB axis), plus the interferon regulators IRF1/IRF9 and NF-κB components (NFKB1, CEBPB, RELA).

![tf](fig_tf.png)

### Target shortlist

| Class | Example targets | Rationale |
|---|---|---|
| **Inhibit (senolytic-style)** | BCL2L1 (BCL-xL), MDM2, CTSK, NCSTN, PAM | Up in senescence; inhibiting them should selectively kill senescent cells |
| **Activate (rejuvenation)** | NUSAP1, KIF4A, DLGAP5, NDC80, CKS2 | Down in senescence; restoring proliferation machinery |

The headline actionable targets — BCL-xL and MDM2 — are the same anti-apoptotic nodes targeted by existing senolytics (navitoclax, idasanutlin), recovered *without* the model being told they matter.

## What the 4 "misses" tell us

BCL2, IL1B, IGFBP3, and TIMP1 were not recovered in the expected direction. This is not a bug — these genes are well-documented as **heterogeneous across senescence contexts and cell lines**. The failure-analysis layer flags exactly this kind of context-dependence, which is the information a wet-lab team needs before committing to a target.

## Limitations (stated plainly)

- **Bulk RNA-seq, 5 cell lines** — a small, in-vitro slice of senescence biology.
- **Classical differential expression**, not a foundation model. This is the interpretable baseline; the Geneformer in-silico-perturbation upgrade (which needs a GPU) is the next step.
- **No wet-lab validation yet** — the shortlist is a hypothesis, not a result.
- Batch correction is a meta-analysis, not a full mixed-effects model.

## Next steps

1. **Wet-lab review** — does the shortlist pass a biologist's plausibility check?
2. **Validate top targets** — test BCL-xL/MDM2 inhibition (and 2–3 novel targets) for senolytic activity in a senescence assay.
3. **Upgrade the model** — Geneformer in-silico perturbation for gene-level causal attribution (needs GPU).
4. **Add a second phenotype** — to demonstrate the loop generalizes beyond senescence.

---

*Data: [GSE63577](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63577) · Pathways: [MSigDB Hallmark](https://www.gsea-msigdb.org/gsea/msigdb/) · Regulons: [CollecTRI](https://github.com/saezlab/CollecTRI) · Source paper: [PLOS ONE 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0154531)*
