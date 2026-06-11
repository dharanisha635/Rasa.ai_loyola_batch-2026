import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

# ── Load results ──────────────────────────────────────────────────
with open("stats_summary.json") as f:
    data = json.load(f)

# ── Re-run MC Dropout with full pass storage ──────────────────────
# If you still have your model loaded, re-run with more passes.
# Otherwise we reconstruct distributions from saved stats.
# To get full MC pass data, add this to evaluate.py and re-run:
#   np.save("mc_passes.npy", mc_passes_array)  shape: (34, 50)
# For now we work with saved mean + std from stats_summary.json

confidences  = np.array([d["confidence"]  for d in data])
uncertainties= np.array([d["uncertainty"] for d in data])
raw_scores   = np.array([d["raw_score"]   for d in data])
correct      = np.array([d["correct"]     for d in data])
true_labels  = np.array([d["true_label"]  for d in data])
pred_labels  = np.array([d["predicted"]   for d in data])
filenames    = [d["filename"]             for d in data]
ambiguous    = np.array([d["is_ambiguous"]for d in data])

n = len(data)

# ── Derived uncertainty metrics ───────────────────────────────────

# 1. Predictive entropy (binary)
#    H = -p*log(p) - (1-p)*log(1-p)
#    Uses raw sigmoid score (mean MC prediction) as p
eps = 1e-8
p = raw_scores
entropy = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
entropy_norm = entropy / np.log(2)          # normalise to [0,1] — max entropy = 1 bit

# 2. Uncertainty ratio: std / confidence (relative uncertainty)
uncertainty_ratio = uncertainties / (confidences + eps)

# 3. Reliability score (inverse of uncertainty, scaled)
reliability = 1 - (uncertainties / (uncertainties.max() + eps))

# ── Print report ──────────────────────────────────────────────────
print("=" * 56)
print("  BAYESIAN UNCERTAINTY ANALYSIS (MC Dropout, N=50)")
print("=" * 56)
print(f"\n  Total predictions     : {n}")
print(f"  MC passes per image   : 50")
print(f"\n  UNCERTAINTY SUMMARY")
print(f"  {'Metric':<28} {'Mean':>8}  {'Std':>8}  {'Max':>8}")
print(f"  {'-'*52}")
print(f"  {'Uncertainty (std)':<28} {uncertainties.mean():>8.4f}  {uncertainties.std():>8.4f}  {uncertainties.max():>8.4f}")
print(f"  {'Predictive entropy (norm)':<28} {entropy_norm.mean():>8.4f}  {entropy_norm.std():>8.4f}  {entropy_norm.max():>8.4f}")
print(f"  {'Uncertainty ratio':<28} {uncertainty_ratio.mean():>8.4f}  {uncertainty_ratio.std():>8.4f}  {uncertainty_ratio.max():>8.4f}")
print(f"  {'Reliability score':<28} {reliability.mean():>8.4f}  {reliability.std():>8.4f}  {reliability.min():>8.4f}")

print(f"\n  AMBIGUITY BREAKDOWN")
print(f"  Threshold             : std > 0.15")
print(f"  Ambiguous predictions : {ambiguous.sum()} / {n}")
print(f"  Certain predictions   : {(~ambiguous).sum()} / {n}")
print(f"  Ambiguous + correct   : {(ambiguous & correct).sum()}")
print(f"  Ambiguous + wrong     : {(ambiguous & ~correct).sum()}")

print(f"\n  UNCERTAINTY VS CORRECTNESS")
certain_acc   = correct[~ambiguous].mean() if (~ambiguous).sum() > 0 else 0
ambiguous_acc = correct[ ambiguous].mean() if ( ambiguous).sum() > 0 else float('nan')
print(f"  Accuracy (certain)    : {certain_acc*100:.2f}%")
print(f"  Accuracy (ambiguous)  : {'N/A' if np.isnan(ambiguous_acc) else f'{ambiguous_acc*100:.2f}%'}")

# Spearman correlation: does higher uncertainty → more wrong?
corr, p_corr = spearmanr(uncertainties, ~correct)
print(f"\n  Spearman correlation (uncertainty vs error): r={corr:.4f}, p={p_corr:.4f}")
if p_corr < 0.05:
    direction = "positively" if corr > 0 else "negatively"
    print(f"  → Uncertainty is {direction} correlated with errors (p<0.05)")
    print(f"  → The model's uncertainty IS a meaningful error signal ✓")
else:
    print(f"  → No significant correlation (p≥0.05)")
    print(f"  → Uncertainty alone doesn't strongly predict errors here")

print(f"\n  EPISTEMIC vs ALEATORIC (proxy)")
print(f"  Low uncertainty + wrong  (overconfident errors) : {((uncertainties < 0.10) & ~correct).sum()}")
print(f"  High uncertainty + wrong (model unsure + wrong) : {((uncertainties >= 0.10) & ~correct).sum()}")
print(f"  High uncertainty + right (model unsure + right) : {((uncertainties >= 0.10) &  correct).sum()}")
print(f"\n  TOP 5 MOST UNCERTAIN PREDICTIONS")
print(f"  {'File':<35} {'Pred':>6} {'True':>6} {'Conf':>6} {'Unc':>6} {'Entropy':>8} {'OK':>4}")
top5_idx = np.argsort(uncertainties)[::-1][:5]
for i in top5_idx:
    fname = filenames[i].split("/")[-1][:32]
    print(f"  {fname:<35} {pred_labels[i]:>6} {true_labels[i]:>6} {confidences[i]:>6.3f} {uncertainties[i]:>6.3f} {entropy_norm[i]:>8.4f} {'✓' if correct[i] else '✗':>4}")

print("=" * 56)

# Save uncertainty metrics to JSON
uncertainty_export = []
for i in range(n):
    uncertainty_export.append({
        "filename":         filenames[i],
        "predicted":        pred_labels[i],
        "true_label":       true_labels[i],
        "confidence":       round(float(confidences[i]),  4),
        "uncertainty_std":  round(float(uncertainties[i]),4),
        "predictive_entropy": round(float(entropy_norm[i]),4),
        "uncertainty_ratio":  round(float(uncertainty_ratio[i]),4),
        "reliability_score":  round(float(reliability[i]),4),
        "is_ambiguous":     bool(ambiguous[i]),
        "correct":          bool(correct[i])
    })
with open("uncertainty_results.json", "w") as f:
    json.dump(uncertainty_export, f, indent=2)
print("\nSaved → uncertainty_results.json")

# ── Plots ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor("#f3f5fa")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36)

CORRECT_C="#4ade80"; WRONG_C="#f87171"; TEXT_C="#131313"
GRID_C="#1b2129";    BG_C="#f6f8fb";   UNC_C="#38bdf8"

def style_ax(ax):
    ax.set_facecolor(BG_C)
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)

# 1. Uncertainty distribution with error overlay
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(uncertainties[correct],  bins=10, color=CORRECT_C, alpha=0.7,
         label=f"Correct (n={correct.sum()})",  edgecolor="#166534")
ax1.hist(uncertainties[~correct], bins=10, color=WRONG_C,   alpha=0.85,
         label=f"Wrong (n={(~correct).sum()})",  edgecolor="#7f1d1d")
ax1.axvline(0.15, color="#facc15", linewidth=1.5, linestyle="--", label="Ambiguous threshold")
ax1.set_title("Uncertainty: correct vs wrong")
ax1.set_xlabel("MC Dropout std"); ax1.set_ylabel("Count")
ax1.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax1)

# 2. Predictive entropy distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(entropy_norm, bins=12, color="#a78bfa", edgecolor="#4c1d95", alpha=0.85)
ax2.axvline(entropy_norm.mean(), color="#facc15", linewidth=1.5, linestyle="--",
            label=f"Mean={entropy_norm.mean():.3f}")
ax2.set_title("Predictive entropy (normalised)")
ax2.set_xlabel("Entropy (bits, 0–1)"); ax2.set_ylabel("Count")
ax2.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax2)

# 3. Reliability scores per prediction (sorted)
ax3 = fig.add_subplot(gs[0, 2])
sorted_idx = np.argsort(reliability)
colors_rel = [CORRECT_C if correct[i] else WRONG_C for i in sorted_idx]
ax3.bar(range(n), reliability[sorted_idx], color=colors_rel, alpha=0.85, width=0.8)
ax3.set_title("Reliability score per prediction")
ax3.set_xlabel("Prediction (sorted by reliability)")
ax3.set_ylabel("Reliability (1 = most reliable)")
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color=CORRECT_C, label="Correct"),
                    Patch(color=WRONG_C,   label="Wrong")],
           fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax3)

# 4. Uncertainty vs confidence 2D scatter
ax4 = fig.add_subplot(gs[1, 0])
sc = ax4.scatter(confidences[correct],   uncertainties[correct],
                 color=CORRECT_C, s=45, alpha=0.75, label="Correct", zorder=3)
ax4.scatter(confidences[~correct], uncertainties[~correct],
            color=WRONG_C, s=65, marker="X", alpha=0.9, label="Wrong", zorder=4)
ax4.axhline(0.15, color="#facc15", linewidth=1, linestyle="--", alpha=0.8)
ax4.axvline(0.70, color="#94a3b8", linewidth=1, linestyle=":", alpha=0.6)
# Quadrant labels
ax4.text(0.54, 0.16, "High unc\nLow conf\n(flag)", fontsize=7, color=WRONG_C, alpha=0.85)
ax4.text(0.78, 0.16, "High unc\nHigh conf\n(overconfident?)", fontsize=7, color="#fb923c", alpha=0.85)
ax4.text(0.78, 0.005,"Low unc\nHigh conf\n(reliable)", fontsize=7, color=CORRECT_C, alpha=0.85)
ax4.set_title("Confidence vs uncertainty map")
ax4.set_xlabel("Confidence"); ax4.set_ylabel("MC Dropout std")
ax4.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax4)

# 5. Entropy vs error scatter
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(entropy_norm[correct],  confidences[correct],
            color=CORRECT_C, s=45, alpha=0.75, label="Correct", zorder=3)
ax5.scatter(entropy_norm[~correct], confidences[~correct],
            color=WRONG_C, s=65, marker="X", alpha=0.9, label="Wrong", zorder=4)
ax5.set_title(f"Entropy vs confidence\n(Spearman r={corr:.3f}, p={p_corr:.3f})")
ax5.set_xlabel("Predictive entropy"); ax5.set_ylabel("Confidence")
ax5.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax5)

# 6. Per-image uncertainty heatmap (strip)
ax6 = fig.add_subplot(gs[1, 2])
sorted_unc_idx = np.argsort(uncertainties)
unc_sorted     = uncertainties[sorted_unc_idx]
correct_sorted = correct[sorted_unc_idx]
bar_colors     = [CORRECT_C if c else WRONG_C for c in correct_sorted]
bars = ax6.barh(range(n), unc_sorted, color=bar_colors, alpha=0.85, height=0.8)
ax6.axvline(0.15, color="#facc15", linewidth=1.5, linestyle="--", label="Ambiguous (0.15)")
ax6.axvline(uncertainties.mean(), color="#94a3b8", linewidth=1,
            linestyle=":", label=f"Mean ({uncertainties.mean():.3f})")
ax6.set_title("Per-image uncertainty (sorted)")
ax6.set_xlabel("MC Dropout std"); ax6.set_ylabel("Image index (sorted)")
ax6.legend(fontsize=8, labelcolor=TEXT_C, facecolor=BG_C, edgecolor=GRID_C)
style_ax(ax6)

fig.suptitle("Street Cleanliness CNN — Bayesian Uncertainty Analysis (MC Dropout)",
             color=TEXT_C, fontsize=13, fontweight="bold", y=1.01)

plt.savefig("uncertainty_analysis.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Plot saved → uncertainty_analysis.png")