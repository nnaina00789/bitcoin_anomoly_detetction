"""Evaluation metrics, written to be robust to tied scores."""
import numpy as np
from sklearn.metrics import average_precision_score


def precision_at_k(y_true, scores, k):
    """Precision among the top-k, *fair under ties*.

    If several items share the score at the cut-off, we count the expected
    precision under a random tie-break instead of letting file order decide.
    """
    y_true, scores = np.asarray(y_true, float), np.asarray(scores, float)
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")
    kth = scores[order[k - 1]]
    above = scores > kth
    tied = scores == kth
    n_slots = k - above.sum()
    hits = y_true[above].sum() + n_slots * y_true[tied].mean()
    return hits / k


def recall_at_k(y_true, scores, k):
    y_true = np.asarray(y_true, float)
    return precision_at_k(y_true, scores, k) * min(k, len(y_true)) / max(y_true.sum(), 1)


def wilson_ci(hits, n, z=1.96):
    """95% confidence interval for a proportion (works well for small samples)."""
    if n == 0:
        return (np.nan, np.nan)
    p = hits / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def flag_rate(scores, threshold):
    return float((np.asarray(scores) >= threshold).mean())


def n_ties(scores):
    """How many distinct score values exist (1 per item = no ties)."""
    return int(np.unique(np.asarray(scores)).size)


def evaluate(name, y_true, scores, threshold, ks=(50, 100, 500)):
    """One row of the model comparison table."""
    y_true, scores = np.asarray(y_true, float), np.asarray(scores, float)
    row = {"model": name}
    for k in ks:
        row[f"P@{k}"] = round(precision_at_k(y_true, scores, k), 3)
    row["PR-AUC"] = round(average_precision_score(y_true, scores), 3)
    flagged = scores >= threshold
    row["flag_rate_%"] = round(flagged.mean() * 100, 2)
    row["recall_at_flag_%"] = round(y_true[flagged].sum() / max(y_true.sum(), 1) * 100, 1)
    return row
