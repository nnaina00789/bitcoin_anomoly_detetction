"""Feature engineering: drift-robust transforms and graph (money-flow) features."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import TIME


def signed_log(X):
    """Compress huge BTC values so a few whales don't dominate."""
    return np.sign(X) * np.log1p(np.abs(X))


def per_step_percentile(df, cols):
    """Rank every feature *within its own time step* (0 = lowest, 1 = highest).

    Bitcoin traffic drifts over time; ranking inside each time step compares a
    transaction only with its contemporaries, which makes 'unusual' mean
    'unusual for that moment'.
    """
    return df.groupby(TIME)[cols].rank(pct=True).astype("float32")


def build_graph(tx, edges):
    """Sparse directed adjacency matrix A[i, j] = 1 when money flows i -> j."""
    idx = pd.Series(np.arange(len(tx)), index=tx["txId"].to_numpy())
    e = edges[edges["txId1"].isin(idx.index) & edges["txId2"].isin(idx.index)]
    src, dst = idx[e["txId1"]].to_numpy(), idx[e["txId2"]].to_numpy()
    n = len(tx)
    return sp.csr_matrix((np.ones(len(src), dtype="float32"), (src, dst)), shape=(n, n))


def neighbour_mean(A, values):
    """Average of a value over each transaction's direct senders + receivers."""
    U = ((A + A.T) > 0).astype("float32")
    deg = np.asarray(U.sum(axis=1)).ravel()
    out = U @ values
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / deg[:, None] if out.ndim == 2 else out / deg
    return np.nan_to_num(out), deg

