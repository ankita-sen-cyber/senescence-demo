"""
Generalization proof: leave-one-cell-line-out cross-validation.

Builds a predictive model (logistic regression on a senescence signature) and
tests whether it can classify senescent vs. young cells in a cell line it has
NEVER seen during training. This is the "reproducible, generalizable hypotheses"
claim, made concrete and falsifiable.

For each of the 5 cell lines:
  - select the top-100 discriminating genes on the OTHER 4 cell lines (no leakage)
  - train a logistic regression on those 4 cell lines
  - predict senescent vs. young in the held-out cell line
"""
import os, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

DATA_PATH = "data/GSE63577_counts_rpkm_exvivo_jenage_data.xls"
if not os.path.exists(DATA_PATH):
    raise SystemExit(f"Data not found: {DATA_PATH}\nRun `python scripts/download_data.py` first.")

raw = pd.read_excel(DATA_PATH, engine="xlrd")
meta_cols = ["ensembl_gene_id", "external_gene_id", "description", "gene_biotype"]
count_cols = [c for c in raw.columns if c not in meta_cols]

def label(c):
    if c.startswith("IMR90"): cell = "IMR90"
    elif c.startswith("MRC_5"): cell = "MRC5"
    elif c.startswith("WI_"): cell = "WI38"
    else: cell = c.split("_")[0]
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

adata = sc.AnnData(X=X, obs=obs, var=pd.DataFrame({"symbol": sym}, index=sym))
adata.obs_names = obs["sample"].values
adata.var_names = sym
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

E = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
y = (adata.obs["condition"] == "senescent").astype(int).values
cell = adata.obs["cell_line"].values

TOP_N = 100
results = []
for held in sorted(set(cell)):
    tr = cell != held
    te = cell == held
    # feature selection WITHIN training fold only (no leakage)
    tstats = []
    for g in E.columns:
        yv = E.loc[tr & (adata.obs["condition"] == "young").values, g].values
        sv = E.loc[tr & (adata.obs["condition"] == "senescent").values, g].values
        if np.std(yv) == 0 and np.std(sv) == 0:
            tstats.append((g, 0.0))
        else:
            tstats.append((g, abs(stats.ttest_ind(sv, yv, equal_var=False).statistic)))
    top = [g for g, _ in sorted(tstats, key=lambda z: -z[1])[:TOP_N]]

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(E.loc[tr, top].values, y[tr])
    pred = clf.predict(E.loc[te, top].values)
    acc = (pred == y[te]).mean()
    results.append((held, int(tr.sum()), int(te.sum()), float(acc)))

res = pd.DataFrame(results, columns=["held_out_cell_line", "n_train", "n_test", "accuracy"])
res.to_csv("results/generalization_leave_one_out.csv", index=False)
print("Leave-one-cell-line-out (logistic regression, top-100 genes):\n")
print(res.to_string(index=False))
print(f"\nMean held-out accuracy: {res.accuracy.mean():.1%}")
print(f"Chance baseline (majority class): 50%")
