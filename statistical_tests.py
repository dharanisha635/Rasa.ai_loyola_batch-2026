import json
import numpy as np
from scipy.stats import binomtest, chi2
from statsmodels.stats.contingency_tables import mcnemar

# ── Load results ──────────────────────────────────────────────────
with open("stats_summary.json") as f:
    data = json.load(f)

true_labels = np.array([1 if d["true_label"] == "Dirty" else 0 for d in data])
pred_labels = np.array([1 if d["predicted"]  == "Dirty" else 0 for d in data])
correct     = np.array([d["correct"] for d in data])
n           = len(data)
n_correct   = correct.sum()

# ════════════════════════════════════════════════════════
print("=" * 54)
print("  TEST 1 — BINOMIAL TEST")
print("=" * 54)
# H0: model accuracy = 0.5 (random chance)
# H1: model accuracy > 0.5
result_binom = binomtest(n_correct, n, p=0.5, alternative="greater")
print(f"  Correct predictions : {n_correct} / {n}")
print(f"  Observed accuracy   : {n_correct/n*100:.2f}%")
print(f"  Null hypothesis     : accuracy = 50% (random chance)")
print(f"  p-value             : {result_binom.pvalue:.6f}")
if result_binom.pvalue < 0.001:
    verdict = "HIGHLY significant (p < 0.001) ✓"
elif result_binom.pvalue < 0.01:
    verdict = "Very significant (p < 0.01) ✓"
elif result_binom.pvalue < 0.05:
    verdict = "Significant (p < 0.05) ✓"
else:
    verdict = "NOT significant (p >= 0.05)"
print(f"  Verdict             : {verdict}")
print(f"\n  Interpretation: The model performs significantly better")
print(f"  than random guessing. This result would occur by chance")
print(f"  less than {result_binom.pvalue*100:.4f}% of the time.")

# ════════════════════════════════════════════════════════
print("\n" + "=" * 54)
print("  TEST 2 — McNEMAR'S TEST")
print("  (CNN at t=0.5  vs  Baseline: always predict Dirty)")
print("=" * 54)

# Baseline: always predict Dirty (majority/naive classifier)
baseline_preds = np.ones(n, dtype=int)

# McNemar contingency table
# b = baseline correct, CNN wrong
# c = CNN correct, baseline wrong
cnn_correct      = (pred_labels == true_labels)
baseline_correct = (baseline_preds == true_labels)

b = int(( baseline_correct & ~cnn_correct).sum())  # baseline wins
c = int((~baseline_correct &  cnn_correct).sum())  # CNN wins
print(f"\n  Discordant pairs:")
print(f"    CNN correct, baseline wrong (c) : {c}")
print(f"    Baseline correct, CNN wrong (b) : {b}")
print(f"\n  CNN accuracy      : {cnn_correct.mean()*100:.2f}%")
print(f"  Baseline accuracy : {baseline_correct.mean()*100:.2f}%")

# Manual McNemar statistic with continuity correction
if (b + c) == 0:
    print("  Cannot compute McNemar: no discordant pairs.")
else:
    mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
    from scipy.stats import chi2
    p_mcnemar = 1 - chi2.cdf(mcnemar_stat, df=1)
    print(f"  McNemar statistic : {mcnemar_stat:.4f}")
    print(f"  p-value           : {p_mcnemar:.6f}")
    if p_mcnemar < 0.05:
        print(f"  Verdict           : CNN is significantly better than baseline ✓")
    else:
        print(f"  Verdict           : No significant difference from baseline")
    print(f"\n  Interpretation: McNemar's test compares whether the CNN")
    print(f"  makes significantly different errors than a naive baseline")
    print(f"  that always predicts 'Dirty'. The discordant pairs (c={c}, b={b})")
    print(f"  show cases where one method was right and the other wrong.")

# ════════════════════════════════════════════════════════
print("\n" + "=" * 54)
print("  TEST 3 — BOOTSTRAP CONFIDENCE INTERVALS")
print("  (10,000 resamples)")
print("=" * 54)

np.random.seed(42)
N_BOOTSTRAP = 10_000
boot_accuracies  = []
boot_f1_dirty    = []
boot_precision   = []
boot_recall      = []

for _ in range(N_BOOTSTRAP):
    idx          = np.random.choice(n, size=n, replace=True)
    b_true       = true_labels[idx]
    b_pred       = pred_labels[idx]
    b_acc        = (b_true == b_pred).mean()
    tp = ((b_pred == 1) & (b_true == 1)).sum()
    fp = ((b_pred == 1) & (b_true == 0)).sum()
    fn = ((b_pred == 0) & (b_true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    boot_accuracies.append(b_acc)
    boot_f1_dirty.append(f1)
    boot_precision.append(prec)
    boot_recall.append(rec)

boot_accuracies = np.array(boot_accuracies)
boot_f1_dirty   = np.array(boot_f1_dirty)
boot_precision  = np.array(boot_precision)
boot_recall     = np.array(boot_recall)

def ci(arr, level=95):
    lo = (100 - level) / 2
    return np.percentile(arr, lo), np.percentile(arr, 100 - lo)

acc_lo,  acc_hi  = ci(boot_accuracies)
f1_lo,   f1_hi   = ci(boot_f1_dirty)
prec_lo, prec_hi = ci(boot_precision)
rec_lo,  rec_hi  = ci(boot_recall)

print(f"\n  Metric          Point est.   95% CI")
print(f"  " + "-" * 46)
print(f"  Accuracy        {n_correct/n*100:>8.2f}%   [{acc_lo*100:.2f}% – {acc_hi*100:.2f}%]")
print(f"  Dirty F1        {boot_f1_dirty.mean()*100:>8.2f}%   [{f1_lo*100:.2f}% – {f1_hi*100:.2f}%]")
print(f"  Dirty precision {boot_precision.mean()*100:>8.2f}%   [{prec_lo*100:.2f}% – {prec_hi*100:.2f}%]")
print(f"  Dirty recall    {boot_recall.mean()*100:>8.2f}%   [{rec_lo*100:.2f}% – {rec_hi*100:.2f}%]")
print(f"\n  Bootstrap mean accuracy : {boot_accuracies.mean()*100:.2f}%")
print(f"  Bootstrap std           : {boot_accuracies.std()*100:.2f}%")
print(f"\n  Interpretation: With 95% confidence, the true accuracy")
print(f"  of this model on unseen data lies between")
print(f"  {acc_lo*100:.1f}% and {acc_hi*100:.1f}%.")
print("=" * 54)

# ════════════════════════════════════════════════════════
# Save all test results to JSON for dashboard use
test_results = {
    "binomial_test": {
        "n_correct": int(n_correct),
        "n_total": n,
        "accuracy": round(n_correct/n, 4),
        "p_value": round(result_binom.pvalue, 6),
        "significant": bool(result_binom.pvalue < 0.05)
    },
    "mcnemar_test": {
        "cnn_accuracy": round(cnn_correct.mean(), 4),
        "baseline_accuracy": round(baseline_correct.mean(), 4),
        "discordant_b": b,
        "discordant_c": c,
        "p_value": round(p_mcnemar, 6) if (b+c) > 0 else None,
        "cnn_significantly_better": bool(p_mcnemar < 0.05) if (b+c) > 0 else None
    },
    "bootstrap_ci": {
        "n_resamples": N_BOOTSTRAP,
        "accuracy": {
            "point": round(n_correct/n, 4),
            "ci_low": round(acc_lo, 4),
            "ci_high": round(acc_hi, 4)
        },
        "f1_dirty": {
            "point": round(boot_f1_dirty.mean(), 4),
            "ci_low": round(f1_lo, 4),
            "ci_high": round(f1_hi, 4)
        },
        "precision_dirty": {
            "point": round(boot_precision.mean(), 4),
            "ci_low": round(prec_lo, 4),
            "ci_high": round(prec_hi, 4)
        },
        "recall_dirty": {
            "point": round(boot_recall.mean(), 4),
            "ci_low": round(rec_lo, 4),
            "ci_high": round(rec_hi, 4)
        }
    }
}

with open("test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

print("\nSaved → test_results.json")