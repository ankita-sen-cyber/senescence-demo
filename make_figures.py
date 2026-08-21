"""Generate demo figures: volcano, pathway, TF, validation."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

de = pd.read_csv("results/de_iter1_batch_aware.csv")
gsea = pd.read_csv("results/pathway_enrichment.csv")
tf = pd.read_csv("results/tf_activity.csv")
val = pd.read_csv("results/validation_iter1.csv")

KNOWN_UP = ["CDKN1A","CDKN2A","TP53","IL6","CXCL8","SERPINE1","IGFBP3","BCL2","BCL2L1","MDM2","CCL2","IL1B","MMP3","TIMP1"]
KNOWN_DN = ["LMNB1","MKI67","PCNA","CCNA2","CCNB1","E2F1","HMGB2"]
known = set(KNOWN_UP + KNOWN_DN)

# ---- 1. Volcano ----
from adjustText import adjust_text
fig, ax = plt.subplots(figsize=(7.5, 6))
x = de["logFC"]; y = -np.log10(de["padj"].clip(lower=1e-300))
ax.scatter(x, y, s=4, c="#cccccc", alpha=0.5, linewidths=0)
pts, texts = [], []
for _, r in val.iterrows():
    c = "#2e9e5b" if r.recovered else "#d64545"
    xi = de.loc[de.symbol == r.symbol, "logFC"].values[0]
    yi = -np.log10(max(de.loc[de.symbol == r.symbol, "padj"].values[0], 1e-300))
    ax.scatter(xi, yi, s=70, c=c, edgecolors="black", linewidths=0.6, zorder=3)
    texts.append(ax.text(xi, yi, r.symbol, fontsize=8))
ax.axhline(-np.log10(0.05), color="gray", ls="--", lw=0.8)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("log2 fold change (senescent vs young)")
ax.set_ylabel("-log10 adjusted p-value")
ax.set_title("Differential expression — batch-aware meta-analysis")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='#2e9e5b',markersize=8,label='recovered'),
                   Line2D([0],[0],marker='o',color='w',markerfacecolor='#d64545',markersize=8,label='missed')],
          loc='lower right', frameon=False)
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
fig.tight_layout(); fig.savefig("results/fig_volcano.png", dpi=150); plt.close(fig)

# ---- 2. Pathway ----
fig, ax = plt.subplots(figsize=(7, 4.5))
g = gsea.head(10).iloc[::-1]
colors = ["#2e9e5b" if q < 0.05 else "#9aa0a6" for q in g["FDR q-val"]]
ax.barh(g["Term"], g["NES"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Normalized enrichment score (NES)")
ax.set_title("Hallmark pathways enriched in senescence")
fig.tight_layout(); fig.savefig("results/fig_pathway.png", dpi=150); plt.close(fig)

# ---- 3. TF ----
fig, ax = plt.subplots(figsize=(7, 4.5))
t = tf.head(12).iloc[::-1]
ax.barh(t["TF"], t["regulon_score"], color="#3b6ea5")
ax.set_xlabel("Regulon enrichment score (-log10 p)")
ax.set_title("Transcription factors driving the senescence program")
fig.tight_layout(); fig.savefig("results/fig_tf.png", dpi=150); plt.close(fig)

# ---- 4. Validation summary ----
fig, ax = plt.subplots(figsize=(6, 2.2))
rec = val.recovered.sum(); tot = len(val)
ax.barh([0], [rec], color="#2e9e5b")
ax.barh([0], [tot - rec], left=[rec], color="#d64545")
ax.set_yticks([]); ax.set_xlim(0, tot)
ax.text(rec/2, 0, f"{rec}/{tot} recovered", ha="center", va="center", color="white", fontweight="bold")
ax.text(rec + (tot-rec)/2, 0, f"{tot-rec} missed", ha="center", va="center", color="white", fontweight="bold")
ax.set_title("Recovery of known senescence markers / senolytics")
fig.tight_layout(); fig.savefig("results/fig_validation.png", dpi=150); plt.close(fig)

print("figures written")
