"""The detector.

Two complementary signals, combined into one risk score:

1. **Look-alike burst score (unsupervised, models normal behaviour).**
   Normal Bitcoin traffic is varied: people send different amounts, with
   different numbers of inputs/outputs. Automated criminal operations (mixers,
   peel chains, ransomware cash-outs) produce *bursts of near-identical
   transactions*. For every transaction we measure how many close look-alikes
   it has inside the same time window. An unusually dense crowd of copies is a
   statistically unusual pattern, so it gets a high score.

2. **Family risk (light investigator feedback).** Transactions are grouped into
   50 behaviour families. Investigators check a handful of past cases per family
   (about 500 in total); each family's smoothed illicit rate becomes its risk.

Both signals are converted to percentiles using the *past only* and blended.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import N_CLUSTERS, SEED
from features import signed_log


def prepare(X, fit):
    """Log-compress, standardise and reduce to 20 dimensions (fitted on the past)."""
    Xl = signed_log(X)
    Z = StandardScaler().fit(Xl[fit]).transform(Xl)
    pca = PCA(n_components=20, random_state=SEED).fit(Z[fit])
    return Xl, pca.transform(Z).astype("float32")


def point_unusualness(Xl, fit):
    """Classic outlier score (Isolation Forest): high = rare one-off shape."""
    iso = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xl[fit])
    return -iso.score_samples(Xl)


def global_density(Zp, fit, k=10, n_ref=30000):
    """How crowded the neighbourhood is compared with all past traffic."""
    rng = np.random.default_rng(SEED)
    idx = np.flatnonzero(fit)
    ref = Zp[rng.choice(idx, min(n_ref, len(idx)), replace=False)]
    dist = NearestNeighbors(n_neighbors=k).fit(ref).kneighbors(Zp)[0]
    return -dist.mean(axis=1)


def lookalike_burst(Zp, steps, k=10):
    """Crowding by near-identical transactions *within the same time window*.

    Score = percentile (0..1) of crowding *within its own time window*, where
    crowding is minus the mean distance to the k nearest same-window
    transactions. 0.99 = more near-copies than 99% of its contemporaries.
    Ranking inside each window keeps the score comparable when traffic volume
    changes over time. Uses no labels at all.
    """
    score = np.zeros(len(steps))
    for s in np.unique(steps):
        m = steps == s
        dist = NearestNeighbors(n_neighbors=k + 1).fit(Zp[m]).kneighbors(Zp[m])[0][:, 1:]
        score[m] = rankdata(-dist.mean(axis=1)) / m.sum()
    return score


def families(Zp, fit, k=N_CLUSTERS):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Zp[fit])
    return km.predict(Zp)


def family_risk(cluster, y, pool, per, seed):
    """Smoothed illicit rate per family from `per` investigator checks per family.

    pool: boolean mask of past, already-investigated transactions to sample from.
    Returns (risk per transaction, DataFrame of the verified sample).
    """
    df = pd.DataFrame({"cluster": cluster, "y": y})[pool]
    verified = df.sample(frac=1, random_state=seed).groupby("cluster").head(per)
    prior = verified["y"].mean()
    g = verified.groupby("cluster")["y"]
    risk = (g.sum() + 2 * prior) / (g.count() + 2)     # shrink small samples to prior
    return pd.Series(cluster).map(risk).fillna(prior).to_numpy(), verified


def to_percentile(values, reference):
    """Percentile of each value within a reference (past) distribution, 0..1."""
    ref = np.sort(reference)
    lo = np.searchsorted(ref, values, side="left")
    hi = np.searchsorted(ref, values, side="right")
    return (lo + hi) / (2 * len(ref))


def blend(lookalike_pct, family_pct, w):
    return w * lookalike_pct + (1 - w) * family_pct
