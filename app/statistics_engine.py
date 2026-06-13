import statistics as stats_lib
from scipy import stats as scipy_stats


def severity_summary(scores: list) -> dict:
    """
    Mean, median, stdev, variance from a list of severity scores.
    Requires at least 2 values for stdev/variance.
    """
    if not scores:
        return {}
    result = {
        "mean":   round(stats_lib.mean(scores), 2),
        "median": round(stats_lib.median(scores), 2),
        "min":    round(min(scores), 2),
        "max":    round(max(scores), 2),
    }
    if len(scores) >= 2:
        result["stdev"]    = round(stats_lib.stdev(scores), 2)
        result["variance"] = round(stats_lib.variance(scores), 2)
    else:
        result["stdev"]    = 0.0
        result["variance"] = 0.0
    return result


def prevalence_rates(disease_counts: list, total: int) -> list:
    """
    Compute prevalence rate % for each disease.
    disease_counts: [{"disease": "Leaf Blight", "count": 42}, ...]
    """
    if total == 0:
        return disease_counts
    for item in disease_counts:
        item["prevalence_rate"] = round((item["count"] / total) * 100, 1)
    return disease_counts


def pearson_correlation(x_values: list, y_values: list) -> dict:
    """
    Compute Pearson r and p-value between two numeric lists.
    Needs at least 3 paired values.
    """
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return {"r": None, "p_value": None, "significant": False}

    r, p = scipy_stats.pearsonr(x_values, y_values)
    return {
        "r":           round(float(r), 3),
        "p_value":     round(float(p), 4),
        "significant": bool(p < 0.05)
    }


def compute_prf_from_feedback(feedback_rows: list) -> dict:
    """
    Compute real Precision, Recall, F1 from user feedback.
    feedback_rows: [(disease, feedback), ...]
    where feedback is 'correct' or 'incorrect'
    Needs at least 5 feedback entries.
    """
    if len(feedback_rows) < 5:
        return {"precision": None, "recall": None, "f1": None, "sample_size": len(feedback_rows)}

    correct   = sum(1 for _, fb in feedback_rows if fb == "correct")
    incorrect = sum(1 for _, fb in feedback_rows if fb == "incorrect")
    total     = correct + incorrect

    # True Positives = correct predictions
    # False Positives = predicted disease but was wrong
    tp = correct
    fp = incorrect
    fn = incorrect  # missed detections estimated same as false positives

    precision = round((tp / (tp + fp)) * 100, 1) if (tp + fp) > 0 else 0
    recall    = round((tp / (tp + fn)) * 100, 1) if (tp + fn) > 0 else 0
    f1        = round(
        2 * precision * recall / (precision + recall), 1
    ) if (precision + recall) > 0 else 0

    return {
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "sample_size": total,
        "correct":     correct,
        "incorrect":   incorrect
    }


def moving_average(values: list, window: int = 7) -> list:
    """Simple moving average over a list of values."""
    if not values:
        return []
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        result.append(round(sum(window_vals) / len(window_vals), 2))
    return result