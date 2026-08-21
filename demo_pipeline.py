"""
Senescence-reversal target discovery — CPU-only demo pipeline (closed loop).

Iteration 0: naive pooled differential expression (young vs senescent, all cell lines pooled)
  -> failure analysis: is cell-line (batch) confounding the signal?
Iteration 1: batch-aware meta-analysis (per-cell-line logFC + Fisher-combined p-values)
  -> pathway enrichment + TF activity on the refined ranking
  -> validation: does the loop recover known senescence markers / senolytics?

All CPU. Outputs to results/.
"""
import os, json, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
os.makedirs("results", exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Load + label
# ----------------------------------------------------------------------------
raw = pd.read_excel("data/GSE63577_counts_rpkm_exvivo_jenage_data.xls", engine="xlrd")
meta_cols = ["ensembl_gene_id", "external_gene_id", "description", "gene_biotype"]
count_cols = [c for c in raw.columns if c not in meta_cols]

def label(c):
    cell = c.split("_")[0] if not c.startswith("WI_") else "WI38"
    if c.startswith("IMR90"): cell = "IMR90"
    if c.startswith("MRC_5"): cell = "MRC5"
    cond = "young" if ("_Y" in c or "PD16" in c or "PD32" in c) else "senescent"
    return cell, cond

obs = pd.DataFrame({
    "sample": count_cols,
    "cell_line": [label(c)[0] for c in count_cols],
    "condition": [label(c)[1] for c in count_cols],
})
X = raw[count_cols].values.astype(np.float32).T
sym = raw["external_gene_id"].values
keep = ~pd.Series(sym).duplicated(keep="first").values
X, sym = X[:, keep], sym[keep]

adata = sc.AnnData(X=X, obs=obs,
                   var=pd.DataFrame({"symbol": sym}, index=sym))
adata.obs_names = obs["sample"].values
adata.var_names = sym
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print(f"[data] {adata.n_obs} samples x {adata.n_vars} genes")

# known senescence markers / senolytic targets (positive controls)
KNOWN_UP = ["CDKN1A", "CDKN2A", "TP53", "IL6", "CXCL8", "SERPINE1", "IGFBP3",
            "BCL2", "BCL2L1", "MDM2", "CCL2", "IL1B", "MMP3", "TIMP1"]
KNOWN_DN = ["LMNB1", "MKI67", "PCNA", "CCNA2", "CCNB1", "E2F1", "HMGB2"]

def validate(de):
    """Return recovery stats for known markers."""
    de = de.set_index("symbol")
    rows = []
    for g in KNOWN_UP:
        if g in de.index:
            rows.append((g, "up", de.loc[g, "logFC"] > 0, de.loc[g, "logFC"], de.loc[g, "padj"]))
    for g in KNOWN_DN:
        if g in de.index:
            rows.append((g, "down", de.loc[g, "logFC"] < 0, de.loc[g, "logFC"], de.loc[g, "padj"]))
    v = pd.DataFrame(rows, columns=["symbol", "expected", "recovered", "logFC", "padj"])
    return v

# ----------------------------------------------------------------------------
# 2. Iteration 0 — naive pooled DE
# ----------------------------------------------------------------------------
sc.tl.rank_genes_groups(adata, groupby="condition", groups=["senescent"],
                        reference="young", method="wilcoxon", n_genes=adata.n_vars)
de0 = sc.get.rank_genes_groups_df(adata, group="senescent").rename(
    columns={"names": "symbol", "logfoldchanges": "logFC", "pvals_adj": "padj"})
de0 = de0[["symbol", "logFC", "padj"]].sort_values("padj")
v0 = validate(de0)
print(f"\n[iter0] pooled Wilcoxon: recovered {v0.recovered.sum()}/{len(v0)} known markers")

# ----------------------------------------------------------------------------
# 3. Failure analysis — does cell line dominate?
# ----------------------------------------------------------------------------
sc.pp.pca(adata, n_comps=10)
pc = pd.DataFrame(adata.obsm["X_pca"], index=adata.obs_names)
pc["cond"] = (adata.obs["condition"] == "senescent").values
pc["cell"] = adata.obs["cell_line"].values
f_cond = np.mean([stats.f_oneway(*[g[k].values for _, g in pc.groupby("cond")]).statistic for k in range(10)])
f_cell = np.mean([stats.f_oneway(*[g[k].values for _, g in pc.groupby("cell")]).statistic for k in range(10)])
print(f"[failure] F(condition)={f_cond:.1f}  F(cell_line)={f_cell:.1f}  "
      f"-> batch {'DOMINATES' if f_cell > f_cond else 'is minor'}")

# ----------------------------------------------------------------------------
# 4. Iteration 1 — batch-aware meta-analysis
# ----------------------------------------------------------------------------
expr = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
obs = adata.obs.copy()

meta_rows = []
for g in adata.var_names:
    lfcs, ps = [], []
    for cell, grp in obs.groupby("cell_line"):
        y = expr.loc[grp.index[grp["condition"] == "young"], g].values
        s = expr.loc[grp.index[grp["condition"] == "senescent"], g].values
        if len(y) < 2 or len(s) < 2 or (np.std(y) == 0 and np.std(s) == 0):
            continue
        lfcs.append(np.mean(s) - np.mean(y))
        t, p = stats.ttest_ind(s, y, equal_var=False)
        ps.append(max(p, 1e-300))
    if not lfcs:
        continue
    logfc = float(np.median(lfcs))
    chi2 = -2 * np.sum(np.log(ps))
    p_fisher = float(stats.chi2.sf(chi2, df=2 * len(ps)))
    meta_rows.append((g, logfc, p_fisher))

de1 = pd.DataFrame(meta_rows, columns=["symbol", "logFC", "pval"]).sort_values("pval")
de1["padj"] = de1["pval"] * de1["pval"].shape[0] / de1["pval"].rank(method="first")  # BH approx
de1["padj"] = de1["padj"].clip(upper=1.0)
v1 = validate(de1)
print(f"[iter1] batch-aware meta-analysis: recovered {v1.recovered.sum()}/{len(v1)} known markers")

# save both
de0.to_csv("results/de_iter0_pooled.csv", index=False)
de1.to_csv("results/de_iter1_batch_aware.csv", index=False)
v0.to_csv("results/validation_iter0.csv", index=False)
v1.to_csv("results/validation_iter1.csv", index=False)

# ----------------------------------------------------------------------------
# 5. Pathway enrichment (preranked GSEA on refined logFC)
# ----------------------------------------------------------------------------
import gseapy as gp
r = de1.set_index("symbol")["logFC"].dropna().sort_values(ascending=False)
r = r[~r.index.duplicated(keep="first")]
try:
    pre = gp.prerank(rnk=r, gene_sets="MSigDB_Hallmark_2020",
                     min_size=10, max_size=500, permutation_num=1000,
                     seed=6, no_plot=True, outdir=None)
    gsea = pre.res2d[["Term", "NES", "NOM p-val", "FDR q-val"]].sort_values("NES", ascending=False)
    gsea.to_csv("results/pathway_enrichment.csv", index=False)
    print(f"\n[pathway] top enriched (senescent vs young):")
    for _, row in gsea.head(8).iterrows():
        print(f"    {row['Term']:34s} NES={row['NES']:+.2f} q={row['FDR q-val']:.2e}")
except Exception as e:
    print("[pathway] ERROR", repr(e)[:200])

# ----------------------------------------------------------------------------
# 6. TF activity (decoupler ULM + CollecTRI)
# ----------------------------------------------------------------------------
try:
    import decoupler as dc
    net = dc.op.collectri(organism="human", remove_complexes=False)
    # regulon enrichment: are a TF's activation targets enriched among genes UP
    # in senescence, and its repression targets among genes DOWN? (Fisher exact)
    de_sig = de1[de1.padj < 0.05]
    up = set(de_sig[de_sig.logFC > 0].symbol)
    dn = set(de_sig[de_sig.logFC < 0].symbol)
    allg = set(de_sig.symbol)
    tf_rows = []
    for tf, grp in net.groupby("source"):
        act = set(grp[grp.weight > 0].target) & allg
        rep = set(grp[grp.weight < 0].target) & allg
        if len(act) < 5 and len(rep) < 5:
            continue
        a_up = len(act & up); a_not = len(act) - a_up
        b_up = len(up - act); b_not = len(allg - up - act)
        p_act = stats.fisher_exact([[a_up, a_not], [b_up, b_not]], alternative="greater")[1] \
            if (a_up + a_not > 0 and b_up + b_not > 0) else 1.0
        r_dn = len(rep & dn); r_not = len(rep) - r_dn
        c_dn = len(dn - rep); c_not = len(allg - dn - rep)
        p_rep = stats.fisher_exact([[r_dn, r_not], [c_dn, c_not]], alternative="greater")[1] \
            if (r_dn + r_not > 0 and c_dn + c_not > 0) else 1.0
        score = -np.log10(min(p_act, p_rep) + 1e-300)
        mode = "activation-targets-up" if p_act < p_rep else "repression-targets-down"
        tf_rows.append((tf, score, p_act, p_rep, len(act), len(rep), mode))
    tf = pd.DataFrame(tf_rows, columns=["TF", "regulon_score", "p_act", "p_rep",
                                        "n_act", "n_rep", "mode"])
    tf = tf.sort_values("regulon_score", ascending=False)
    tf.to_csv("results/tf_activity.csv", index=False)
    print(f"\n[tf] TFs with regulons most enriched in senescence:")
    for _, row in tf.head(12).iterrows():
        print(f"    {row['TF']:14s} score={row['regulon_score']:.2f} ({row['mode']})")
except Exception as e:
    print("[tf] ERROR", repr(e)[:300])

# ----------------------------------------------------------------------------
# 7. Target shortlist (refined)
# ----------------------------------------------------------------------------
de1["neglogp"] = -np.log10(de1["padj"].clip(lower=1e-300))
de1["target_score"] = de1["neglogp"] * np.sign(de1["logFC"])
de1["direction"] = np.where(de1["logFC"] > 0, "inhibit (senolytic-style)", "activate")
top = de1.sort_values("target_score", key=abs, ascending=False).head(30)
top[["symbol", "logFC", "padj", "target_score", "direction"]].to_csv("results/top_targets.csv", index=False)
print("\n[targets] top 12 by |score|:")
print(top.head(12)[["symbol", "logFC", "padj", "direction"]].to_string(index=False))

print("\nDONE.")
