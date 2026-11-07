import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def precision_at_k(y_true, scores, k):
    top = np.argsort(scores)[::-1][:k]
    return y_true[top].mean()

def flag_rate(scores, threshold):
    return (scores >= threshold).mean()

def evaluate(name, y_true, scores, threshold, ks=(50, 100, 500)):
    row = {"model": name}
    for k in ks:
        row[f"P@{k}"] = round(precision_at_k(y_true, scores, k), 3)
    row["PR-AUC"] = round(average_precision_score(y_true, scores), 3)
    row["flag_rate_%"] = round(flag_rate(scores, threshold) * 100, 2)
    return row