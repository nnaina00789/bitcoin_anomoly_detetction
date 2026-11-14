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


def score_dataframe(df, bundle):
    """Score an arbitrary uploaded DataFrame using the saved model bundle.

    Works with full 182-feature Elliptic++ CSVs OR simplified CSVs (missing
    features are imputed with training medians).
    """
    feats = bundle["feats"]
    full_cols = sum(c in df.columns for c in feats)
    med = bundle["train_medians"]
    cols = {
        c: (pd.to_numeric(df[c], errors="coerce").fillna(med[c]) if c in df.columns
            else np.full(len(df), med[c], dtype="float32"))
        for c in feats
    }
    X = pd.DataFrame(cols, index=df.index).to_numpy("float32")
    Xl = signed_log(X)
    Z = bundle["scaler"].transform(Xl)
    Zp = bundle["pca"].transform(Z).astype("float32")

    # Point unusualness + family risk
    raw_point = -bundle["iso"].score_samples(Xl)
    clusters = bundle["km"].predict(Zp)
    risk = pd.Series(clusters).map(bundle["family_risk_map"]).fillna(bundle["prior"]).to_numpy()

    # Look-alike burst within the uploaded batch (per window if >= 12 rows, else whole batch)
    steps = pd.to_numeric(df.get("Time step", pd.Series(1, index=df.index)), errors="coerce").fillna(1).to_numpy()
    k = min(10, max(1, len(df) - 1))
    raw_look = np.zeros(len(df))
    if len(df) > 1:
        for s in np.unique(steps):
            m = steps == s
            if m.sum() >= 12:
                d = NearestNeighbors(n_neighbors=11).fit(Zp[m]).kneighbors(Zp[m])[0][:, 1:]
                raw_look[m] = rankdata(-d.mean(axis=1)) / m.sum()
            else:
                d = NearestNeighbors(n_neighbors=k + 1).fit(Zp).kneighbors(Zp[m])[0][:, 1:]
                raw_look[m] = rankdata(-d.mean(axis=1)) / len(df)
    else:
        raw_look[:] = 0.5

    pct_look = to_percentile(raw_look, bundle["ref_lookalike"])
    pct_fam = to_percentile(risk, bundle["ref_family"])
    pct_point = to_percentile(raw_point, bundle["ref_point"])
    final = blend(pct_look, pct_fam, bundle["w"])

    amt = pd.to_numeric(df.get("total_BTC", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0).to_numpy()
    pct_amt = to_percentile(amt, bundle["ref_amount"])

    out = pd.DataFrame({
        "txId": df["txId"].values if "txId" in df.columns else np.arange(1, len(df) + 1),
        "Time step": steps.astype(int),
        "total_BTC": amt,
        "risk_score": np.round(final * 100, 1),
        "risk_level": np.select([final >= bundle["thr"], final >= bundle["thr_med"]], ["High", "Medium"], "Low"),
        "lookalike_pct": np.round(pct_look * 100, 1),
        "family_risk_pct": np.round(risk * 100, 1),
        "shape_rarity_pct": np.round(pct_point * 100, 1),
        "amount_pct": np.round(pct_amt * 100, 1),
        "cluster": clusters,
    })
    out["Pattern"] = np.select(
        [pct_look >= 0.90, risk > bundle["prior"] * 1.5, pct_point >= 0.97],
        ["🔁 Twin Burst", "👥 Risky Group", "⚡ Rare Shape"],
        default="📊 Normal",
    )
    reasons = []
    for i in range(len(out)):
        r = []
        if pct_look[i] >= 0.90:
            r.append(f"Dense look-alike burst (top {100 - pct_look[i] * 100:.0f}%).")
        if risk[i] > bundle["prior"] * 1.5:
            r.append(f"Matches high-risk behaviour group #{clusters[i]}.")
        if pct_point[i] >= 0.97:
            r.append("Unusual transaction shape vs normal traffic.")
        if pct_amt[i] >= 0.99:
            r.append(f"Large BTC transfer ({amt[i]:,.2f} BTC).")
        reasons.append(" | ".join(r) if r else "Looks like ordinary traffic.")
    out["Why Flagged"] = reasons
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True), full_cols

