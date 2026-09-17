"""Render the genepilot overview using the reported external evaluation results."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures'
REPORTS = ROOT / 'results/external/GSE282425'
BG, INK, MUTED = '#f5f7fb', '#14243b', '#45566e'
fig = plt.figure(figsize=(16, 16), facecolor=BG)
fig.text(.055,.959,'genepilot', fontsize=30, weight='bold', color=INK)
fig.text(.055,.931,'Recognise senescent cells from gene activity and propose genes for laboratory research.',fontsize=14,color=MUTED)
ax=fig.add_axes([.045,.735,.91,.175]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
def box(x,y,w,h,title,subtitle,color='#e4ecf8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.025',facecolor=color,edgecolor='none'))
    ax.text(x+w/2,y+h*.66,title,ha='center',va='center',fontsize=13,weight='bold',color=INK)
    ax.text(x+w/2,y+h*.28,subtitle,ha='center',va='center',fontsize=11,color=MUTED)
def arrow(a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=2,color='#71829a'))
box(.02,.33,.22,.40,'Public RNA data','Which genes are active?')
box(.32,.33,.23,.40,'Prepare measurements','Match genes and normalise')
box(.65,.56,.32,.35,'Predict cell state','Proliferating or senescent','#dcefe9')
box(.65,.06,.32,.35,'Investigate genes','Candidates for lab experiments','#eee6f7')
arrow((.25,.53),(.30,.53)); arrow((.56,.55),(.64,.72)); arrow((.56,.48),(.64,.24))

fig.text(.055,.716,'Sample-level results · 4 external samples',fontsize=20,weight='bold',color=INK)
fig.text(.055,.691,'Same GSE282425 benchmark. Geneformer averages cell predictions; pseudobulk combines raw counts first.',fontsize=11.5,color=MUTED)
with (REPORTS/'sample_level_method_comparison.csv').open() as f:
    rows=list(csv.DictReader(f))
sample_labels=['Original Geneformer\n2/4 correct','Fibroblast Geneformer\n2/4 correct','Pseudobulk + logistic regression\n3/4 correct']
sample_colors=['#a9b6cb','#6e8cba','#269a83']

def panel(rect, values, labels, colors, title, percent, show_labels):
    a=fig.add_axes(rect,facecolor=BG)
    y=np.arange(len(values)); a.barh(y,values,color=colors,height=.55)
    a.set_ylim(len(values)-.5,-.5); a.set_xlim(0,1.18)
    a.set_yticks(y,labels if show_labels else ['']*len(labels),fontsize=11,color=INK)
    a.set_xticks([0,.5,1], ['0%','50%','100%'] if percent else ['0','0.5','1.0'])
    a.tick_params(axis='both',length=0,pad=8,labelsize=10)
    a.xaxis.grid(True,color='#dee4ed'); a.set_axisbelow(True)
    a.set_title(title,loc='left',fontsize=13,weight='bold',color=INK,pad=14)
    if not percent:
        a.axvline(.5,color='#71829a',lw=1,ls='--',zorder=1)
    for yi,v in enumerate(values):
        label=f'{v*100:.1f}%' if percent else f'{v:.3f}'
        a.text(v+.025,yi,label,va='center',fontsize=10.5,weight='bold',color=INK)
    for s in a.spines.values(): s.set_visible(False)
    return a

for i,(key,title,percent) in enumerate([('accuracy','Accuracy',True),('balanced_accuracy','Balanced accuracy',True),('auroc','AUROC · ranking quality',False)]):
    panel([.28+i*.225,.527,.205,.125],[float(r[key]) for r in rows],sample_labels,sample_colors,title,percent,i==0)

fig.text(.055,.475,'Cell-level results · 3,097 external cells',fontsize=20,weight='bold',color=INK)
fig.text(.055,.451,'Original models trained on 70 mixed lung cells; fibroblast Geneformer trained on 14,000 fibroblasts.',fontsize=11.5,color=MUTED)
# Values transcribed from REPORT.md and FIBROBLAST_OPTION1_REPORT.md.
cell_labels=['Original Geneformer','Logistic regression\nTop 100 genes','Linear SVM\nTop 100 genes','Nearest centroid\nTop 100 genes','Majority-class baseline','Fibroblast Geneformer']
cell_values=np.array([[.422,.472,.455],[.468,.505,.521],[.463,.504,.541],[.430,.482,.491],[.562,.500,.500],[.438,.500,.770]])
cell_colors=['#a9b6cb','#b9c7dc','#b9c7dc','#b9c7dc','#d0d5df','#6e8cba']
for i,(title,percent) in enumerate([('Accuracy',True),('Balanced accuracy',True),('AUROC · ranking quality',False)]):
    panel([.28+i*.225,.198,.205,.215],cell_values[:,i],cell_labels,cell_colors,title,percent,i==0)

fig.text(.055,.156,'Earlier classical study (GSE63577): 16/20 reference markers recovered; reported accuracy 76.7%.',fontsize=12,weight='bold',color=INK)
fig.text(.055,.135,'That accuracy used excluded cell lines within one study; it is not directly comparable to the external results above.',fontsize=11,color=MUTED)
fig.text(.055,.105,'Accuracy = fraction correct. Balanced accuracy = average success across the two classes.',fontsize=11,color=MUTED)
fig.text(.055,.086,'AUROC = ranking quality (0.5: chance; 1.0: perfect ordering). AUROC 1.0 does not mean 100% accuracy.',fontsize=11,color=MUTED)
fig.text(.055,.057,'Preliminary: the 4 external samples come from one donor. Pseudobulk has no individual-cell predictions.',fontsize=11,color=MUTED)
fig.text(.055,.039,'Candidate genes require laboratory validation. More independent samples are needed to confirm performance.',fontsize=11,color=MUTED)
fig.text(.055,.019,'Sources: GSE282425 REPORT.md, FIBROBLAST_OPTION1_REPORT.md, PSEUDOBULK_REPORT.md; EVALUATION.md; report.md.',fontsize=9,color='#71829a')
OUT.mkdir(parents=True,exist_ok=True)
for extension in ('png','pdf'):
    path=OUT/f'pipeline_results_overview.{extension}'
    fig.savefig(path,dpi=180,facecolor=BG)
    print(path)
