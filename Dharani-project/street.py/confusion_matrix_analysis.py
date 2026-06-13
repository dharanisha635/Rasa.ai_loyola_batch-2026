import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

# ── Load results ──────────────────────────────────────────────────
with open("stats_summary.json") as f:
    data = json.load(f)

true_labels  = np.array([1 if d["true_label"] == "Dirty" else 0 for d in data])
pred_labels  = np.array([1 if d["predicted"]  == "Dirty" else 0 for d in data])
raw_scores   = np.array([d["raw_score"] for d in data])   # sigmoid output
confidences  = np.array([d["confidence"] for d in data])

# ── Print classification report ───────────────────────────────────
print("=" * 52)
print("  CLASSIFICATION REPORT")
print("=" * 52)
print(classification_report(true_labels, pred_labels,
                             target_names=["Clean", "Dirty"], digits=4))

# Manual breakdown
cm = confusion_matrix(true_labels, pred_labels)
tn, fp, fn, tp = cm.ravel()
print(f"  True Negatives  (Clean → Clean) : {tn}")
print(f"  False Positives (Clean → Dirty) : {fp}  ← clean street wrongly flagged")
print(f"  False Negatives (Dirty → Clean) : {fn}  ← dirty street missed")
print(f"  True Positives  (Dirty → Dirty) : {tp}")
print()
print(f"  Accuracy   : {(tp+tn)/(tp+tn+fp+fn)*100:.2f}%")
print(f"  Precision  : {tp/(tp+fp)*100:.2f}%  (of predicted Dirty, how many truly Dirty)")
print(f"  Recall     : {tp/(tp+fn)*100:.2f}%  (of actual Dirty, how many caught)")
print(f"  F1 Score   : {2*tp/(2*tp+fp+fn)*100:.2f}%")
fpr, tpr, thresholds = roc_curve(true_labels, raw_scores)
roc_auc = auc(fpr, tpr)
print(f"  ROC-AUC    : {roc_auc:.4f}")
print("=" * 52)

# Threshold analysis
print("\n  THRESHOLD SENSITIVITY")
print("  " + "-" * 46)
print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Accuracy':>10}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds_t = (raw_scores >= t).astype(int)
    cm_t    = confusion_matrix(true_labels, preds_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    prec = tp_t/(tp_t+fp_t) if (tp_t+fp_t) > 0 else 0
    rec  = tp_t/(tp_t+fn_t) if (tp_t+fn_t) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    acc  = (tp_t+tn_t)/len(true_labels)
    print(f"  {t:>10.1f}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}  {acc:>10.4f}")
print()

# ── Plots ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#f6f7fa")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

CLEAN_C = "#4ade80"; DIRTY_C = "#f87171"
TEXT_C  = "#0d0d0d"; GRID_C  = "#2d3748"; BG_C = "#f9fafb"

def style_ax(ax):
    ax.set_facecolor(BG_C)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)

# 1. Confusion matrix heatmap
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(cm, interpolation="nearest",
                cmap=plt.cm.Blues, vmin=0, vmax=cm.max()+2)
ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
ax1.set_xticklabels(["Clean","Dirty"], color=TEXT_C)
ax1.set_yticklabels(["Clean","Dirty"], color=TEXT_C)
ax1.set_xlabel("Predicted label"); ax1.set_ylabel("True label")
ax1.set_title("Confusion matrix")
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else TEXT_C,
                 fontsize=18, fontweight="bold")
ax1.title.set_color(TEXT_C)
ax1.tick_params(colors=TEXT_C)
for spine in ax1.spines.values():
    spine.set_edgecolor(GRID_C)

# 2. ROC curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, color="#818cf8", linewidth=2, label=f"AUC = {roc_auc:.4f}")
ax2.plot([0,1],[0,1], color=GRID_C, linewidth=1, linestyle="--", label="Random")
ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
ax2.set_title("ROC curve")
ax2.legend(fontsize=9, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax2)

# 3. Precision-Recall curve
ax3 = fig.add_subplot(gs[0, 2])
prec_curve, rec_curve, _ = precision_recall_curve(true_labels, raw_scores)
pr_auc = auc(rec_curve, prec_curve)
ax3.plot(rec_curve, prec_curve, color="#34d399", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
ax3.set_title("Precision-Recall curve")
ax3.legend(fontsize=9, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax3)

# 4. Per-class precision / recall / F1 bar chart
ax4 = fig.add_subplot(gs[1, 0])
metrics_clean = [
    tn/(tn+fp) if (tn+fp) > 0 else 0,  # precision
    tn/(tn+fn) if (tn+fn) > 0 else 0,  # recall
    0
]
metrics_dirty = [tp/(tp+fp) if (tp+fp)>0 else 0,
                 tp/(tp+fn) if (tp+fn)>0 else 0, 0]
f1_clean = 2*metrics_clean[0]*metrics_clean[1]/(metrics_clean[0]+metrics_clean[1]+1e-9)
f1_dirty = 2*metrics_dirty[0]*metrics_dirty[1]/(metrics_dirty[0]+metrics_dirty[1]+1e-9)
metrics_clean[2] = f1_clean; metrics_dirty[2] = f1_dirty
x = np.arange(3); w = 0.3
bars1 = ax4.bar(x - w/2, metrics_clean, w, label="Clean", color=CLEAN_C, alpha=0.85, edgecolor="#166534")
bars2 = ax4.bar(x + w/2, metrics_dirty, w, label="Dirty", color=DIRTY_C, alpha=0.85, edgecolor="#7f1d1d")
ax4.set_xticks(x)
ax4.set_xticklabels(["Precision","Recall","F1"])
ax4.set_ylim(0, 1.15); ax4.set_title("Per-class metrics")
ax4.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
for bar in list(bars1)+list(bars2):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f"{bar.get_height():.2f}", ha="center", color=TEXT_C, fontsize=8)
style_ax(ax4)

# 5. Threshold sensitivity
ax5 = fig.add_subplot(gs[1, 1])
ts = [0.3, 0.4, 0.5, 0.6, 0.7]
f1s, accs, precs, recs = [], [], [], []
for t in ts:
    pt = (raw_scores >= t).astype(int)
    cmt = confusion_matrix(true_labels, pt)
    tnt,fpt,fnt,tpt = cmt.ravel()
    pr = tpt/(tpt+fpt) if (tpt+fpt)>0 else 0
    rc = tpt/(tpt+fnt) if (tpt+fnt)>0 else 0
    f1s.append(2*pr*rc/(pr+rc+1e-9))
    accs.append((tpt+tnt)/len(true_labels))
    precs.append(pr); recs.append(rc)
ax5.plot(ts, f1s,  "o-", color="#818cf8", linewidth=2, label="F1")
ax5.plot(ts, accs, "s-", color="#facc15", linewidth=2, label="Accuracy")
ax5.plot(ts, precs,"^-", color=CLEAN_C,  linewidth=1.5, label="Precision")
ax5.plot(ts, recs, "v-", color=DIRTY_C,  linewidth=1.5, label="Recall")
ax5.axvline(0.5, color="white", linewidth=1, linestyle=":", alpha=0.5, label="Default (0.5)")
ax5.set_xlabel("Threshold"); ax5.set_ylabel("Score")
ax5.set_title("Threshold sensitivity")
ax5.legend(fontsize=7, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax5)

# 6. Misclassified images summary
ax6 = fig.add_subplot(gs[1, 2])
wrong = [d for d in data if not d["correct"]]
wrong_confs = [d["confidence"] for d in wrong]
wrong_unc   = [d["uncertainty"] for d in wrong]
wrong_types = [f"{d['true_label']}→{d['predicted']}" for d in wrong]
colors_w    = [DIRTY_C if "Clean→" in t else "#fb923c" for t in wrong_types]
ax6.scatter(wrong_confs, wrong_unc, c=colors_w, s=60, alpha=0.9, zorder=3)
ax6.axhline(0.15, color="#f43f5e", linewidth=1, linestyle="--", alpha=0.7, label="Ambiguous threshold")
for i, d in enumerate(wrong):
    ax6.annotate(d["filename"].split("/")[-1][:10],
                 (wrong_confs[i], wrong_unc[i]),
                 fontsize=6, color=TEXT_C, alpha=0.7,
                 xytext=(4, 4), textcoords="offset points")
ax6.set_xlabel("Confidence"); ax6.set_ylabel("Uncertainty")
ax6.set_title(f"Misclassified images (n={len(wrong)})")
ax6.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax6)

fig.suptitle("Street Cleanliness CNN — Confusion Matrix & Classification Report",
             color=TEXT_C, fontsize=13, fontweight="bold", y=1.01)

plt.savefig("confusion_matrix_analysis.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Plot saved → confusion_matrix_analysis.png")