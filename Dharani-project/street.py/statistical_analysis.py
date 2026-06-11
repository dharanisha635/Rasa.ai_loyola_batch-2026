import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

# ── Load results ─────────────────────────────────────────────────
with open("stats_summary.json") as f:
    data = json.load(f)

confidences  = np.array([d["confidence"]  for d in data])
uncertainties= np.array([d["uncertainty"] for d in data])
raw_scores   = np.array([d["raw_score"]   for d in data])
labels       = np.array([d["predicted"]   for d in data])
true_labels  = np.array([d["true_label"]  for d in data])
correct      = np.array([d["correct"]     for d in data])
ambiguous    = np.array([d["is_ambiguous"]for d in data])

clean_conf = confidences[labels == "Clean"]
dirty_conf = confidences[labels == "Dirty"]

# ── Stats summary ─────────────────────────────────────────────────
print("=" * 52)
print("  CONFIDENCE SCORE DISTRIBUTION ANALYSIS")
print("=" * 52)
print(f"  Total predictions  : {len(data)}")
print(f"  Overall mean conf  : {confidences.mean():.4f}")
print(f"  Overall std        : {confidences.std():.4f}")
print(f"  Min / Max          : {confidences.min():.4f} / {confidences.max():.4f}")
print(f"  Median             : {np.median(confidences):.4f}")
print()
print(f"  Clean predictions  : {(labels=='Clean').sum()}")
print(f"    Mean conf        : {clean_conf.mean():.4f}  Std: {clean_conf.std():.4f}")
print()
print(f"  Dirty predictions  : {(labels=='Dirty').sum()}")
print(f"    Mean conf        : {dirty_conf.mean():.4f}  Std: {dirty_conf.std():.4f}")
print()
print(f"  Ambiguous (unc>0.15): {ambiguous.sum()} / {len(data)}")
print(f"  Accuracy           : {correct.mean()*100:.2f}%")

# Calibration: split into bins, check if confidence ≈ accuracy
print("\n  CALIBRATION CHECK (confidence bin → actual accuracy)")
print("  " + "-"*44)
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
bin_labels = ["0.50–0.60","0.60–0.70","0.70–0.80","0.80–0.90","0.90–1.00"]
bin_accs, bin_confs, bin_counts = [], [], []
for i in range(len(bins)-1):
    mask = (confidences >= bins[i]) & (confidences < bins[i+1])
    if mask.sum() > 0:
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        n    = mask.sum()
        print(f"  {bin_labels[i]}  n={n:2d}  avg_conf={conf:.3f}  accuracy={acc*100:.1f}%")
        bin_accs.append(acc); bin_confs.append(conf); bin_counts.append(n)
    else:
        bin_accs.append(None); bin_confs.append((bins[i]+bins[i+1])/2); bin_counts.append(0)

# Normality test on confidence scores
stat, p = stats.shapiro(confidences)
print(f"\n  Shapiro-Wilk normality test: W={stat:.4f}, p={p:.4f}")
print(f"  Distribution is {'approximately normal' if p > 0.05 else 'NOT normal (skewed)'}")
print("=" * 52)

# ── Plots ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#f7f8fa")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

CLEAN_C = "#4ade80"
DIRTY_C = "#f87171"
TEXT_C  = "#080809"
GRID_C  = "#2d3748"
BG_C    = "#fbfbfc"

def style_ax(ax):
    ax.set_facecolor(BG_C)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)

# 1. Overall confidence histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(confidences, bins=12, color="#818cf8", edgecolor="#312e81", alpha=0.85)
ax1.axvline(confidences.mean(), color="#facc15", linewidth=1.5, linestyle="--", label=f"Mean={confidences.mean():.2f}")
ax1.axvline(np.median(confidences), color="#fb923c", linewidth=1.5, linestyle=":", label=f"Median={np.median(confidences):.2f}")
ax1.set_title("Overall confidence distribution")
ax1.set_xlabel("Confidence"); ax1.set_ylabel("Count")
ax1.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax1)

# 2. Clean vs Dirty confidence side-by-side
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(clean_conf, bins=8, color=CLEAN_C, alpha=0.75, label=f"Clean (n={len(clean_conf)})", edgecolor="#166534")
ax2.hist(dirty_conf, bins=8, color=DIRTY_C, alpha=0.75, label=f"Dirty (n={len(dirty_conf)})", edgecolor="#7f1d1d")
ax2.set_title("Clean vs Dirty confidence")
ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count")
ax2.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax2)

# 3. Uncertainty distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(uncertainties, bins=12, color="#38bdf8", edgecolor="#0c4a6e", alpha=0.85)
ax3.axvline(0.15, color="#f43f5e", linewidth=1.5, linestyle="--", label="Ambiguous threshold (0.15)")
ax3.set_title("Uncertainty distribution (MC Dropout)")
ax3.set_xlabel("Std deviation"); ax3.set_ylabel("Count")
ax3.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax3)

# 4. Calibration curve
ax4 = fig.add_subplot(gs[1, 0])
valid = [(c, a, n) for c, a, n in zip(bin_confs, bin_accs, bin_counts) if a is not None]
if valid:
    vc, va, vn = zip(*valid)
    bars = ax4.bar(range(len(vc)), [a*100 for a in va], color="#a78bfa", edgecolor="#4c1d95", alpha=0.85)
    ax4.plot([b + 0.5 for b in range(len(vc))],
             [c*100 for c in vc], "o--", color="#facc15", linewidth=1.5, markersize=5, label="Avg confidence %")
    ax4.set_xticks(range(len(vc)))
    ax4.set_xticklabels([bin_labels[i] for i, (c,a,n) in enumerate(zip(bin_confs,bin_accs,bin_counts)) if a is not None], rotation=30, ha="right", fontsize=7)
    ax4.set_title("Calibration: confidence vs accuracy")
    ax4.set_ylabel("Accuracy / Confidence %")
    ax4.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax4)

# 5. Confidence vs correct/incorrect scatter
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(confidences[correct],  uncertainties[correct],  color=CLEAN_C, alpha=0.75, s=40, label="Correct", zorder=3)
ax5.scatter(confidences[~correct], uncertainties[~correct], color=DIRTY_C, alpha=0.85, s=55, marker="X", label="Wrong", zorder=4)
ax5.axhline(0.15, color="#f43f5e", linewidth=1, linestyle="--", alpha=0.7)
ax5.set_title("Confidence vs Uncertainty")
ax5.set_xlabel("Confidence"); ax5.set_ylabel("Uncertainty (std)")
ax5.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax5)

# 6. Per-class accuracy bar
ax6 = fig.add_subplot(gs[1, 2])
classes = ["Clean", "Dirty"]
accs = [
    correct[true_labels=="Clean"].mean()*100 if (true_labels=="Clean").sum() else 0,
    correct[true_labels=="Dirty"].mean()*100 if (true_labels=="Dirty").sum() else 0
]
colors = [CLEAN_C, DIRTY_C]
bars = ax6.bar(classes, accs, color=colors, edgecolor=GRID_C, alpha=0.85, width=0.5)
for bar, acc in zip(bars, accs):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{acc:.1f}%", ha="center", color=TEXT_C, fontsize=10, fontweight="bold")
ax6.set_ylim(0, 115)
ax6.set_title("Per-class accuracy")
ax6.set_ylabel("Accuracy %")
style_ax(ax6)

fig.suptitle("Street Cleanliness CNN — Statistical Analysis", color=TEXT_C, fontsize=14, fontweight="bold", y=1.01)

plt.savefig("confidence_analysis.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("\nPlot saved → confidence_analysis.png")