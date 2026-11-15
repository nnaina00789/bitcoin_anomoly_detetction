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
    n_rows = len(df)
    feats = bundle["feats"]
    full_cols = sum(c in df.columns for c in feats)
    if n_rows == 0:
        return pd.DataFrame(columns=[
            "txId", "Time step", "total_BTC", "risk_score", "risk_level",
            "lookalike_pct", "family_risk_pct", "shape_rarity_pct", "amount_pct",
            "cluster", "Pattern", "Why Flagged",
        ]), full_cols

    med = bundle["train_medians"]
    X = np.empty((n_rows, len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        m_val = float(med[c])
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32, copy=True)
            bad = ~np.isfinite(col)
            if bad.any():
                col[bad] = m_val
            X[:, j] = col
        else:
            X[:, j] = m_val

    Xl = signed_log(X).astype(np.float32, copy=False)
    Z = bundle["scaler"].transform(Xl)
    Zp = bundle["pca"].transform(Z).astype(np.float32, copy=False)
    del Z

    # Point unusualness + family risk
    raw_point = -bundle["iso"].score_samples(Xl)
    del Xl
    clusters = bundle["km"].predict(Zp)
    risk = pd.Series(clusters).map(bundle["family_risk_map"]).fillna(bundle["prior"]).to_numpy()

    # Look-alike burst within the uploaded batch (per window if >= 12 rows, else whole batch)
    if "Time step" in df.columns:
        steps = pd.to_numeric(df["Time step"], errors="coerce").fillna(1).to_numpy()
    else:
        steps = np.ones(n_rows, dtype=np.int32)
    k = min(10, max(1, n_rows - 1))
    raw_look = np.zeros(n_rows, dtype=np.float32)
    if n_rows > 1:
        for s in np.unique(steps):
            m = steps == s
            cnt = int(m.sum())
            if cnt >= 12:
                d = NearestNeighbors(n_neighbors=11).fit(Zp[m]).kneighbors(Zp[m])[0][:, 1:]
                raw_look[m] = rankdata(-d.mean(axis=1)) / cnt
            else:
                d = NearestNeighbors(n_neighbors=k + 1).fit(Zp).kneighbors(Zp[m])[0][:, 1:]
                raw_look[m] = rankdata(-d.mean(axis=1)) / n_rows
    else:
        raw_look[:] = 0.5

    pct_look = to_percentile(raw_look, bundle["ref_lookalike"])
    pct_fam = to_percentile(risk, bundle["ref_family"])
    pct_point = to_percentile(raw_point, bundle["ref_point"])
    final = blend(pct_look, pct_fam, bundle["w"])

    if "total_BTC" in df.columns:
        amt = pd.to_numeric(df["total_BTC"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        amt = np.where(np.isfinite(amt), amt, 0.0)
    else:
        amt = np.zeros(n_rows, dtype=np.float32)
    pct_amt = to_percentile(amt, bundle["ref_amount"])

    out = pd.DataFrame({
        "txId": df["txId"].astype(str).values if "txId" in df.columns else np.arange(1, n_rows + 1).astype(str),
        "Time step": steps.astype(np.int32),
        "total_BTC": np.round(amt, 4),
        "risk_score": np.round(final * 100, 1).astype(np.float32),
        "risk_level": np.select([final >= bundle["thr"], final >= bundle["thr_med"]], ["High", "Medium"], "Low"),
        "lookalike_pct": np.round(pct_look * 100, 1).astype(np.float32),
        "family_risk_pct": np.round(risk * 100, 1).astype(np.float32),
        "shape_rarity_pct": np.round(pct_point * 100, 1).astype(np.float32),
        "amount_pct": np.round(pct_amt * 100, 1).astype(np.float32),
        "cluster": clusters.astype(np.int16),
    })
    out["Pattern"] = np.select(
        [pct_look >= 0.90, risk > bundle["prior"] * 1.5, pct_point >= 0.97],
        ["Twin Burst", "Risky Group", "Rare Shape"],
        default="Normal",
    )
    reasons = []
    prior_cut = bundle["prior"] * 1.5
    for i in range(n_rows):
        r = []
        if pct_look[i] >= 0.90:
            r.append(f"Dense look-alike burst (top {100 - pct_look[i] * 100:.0f}%).")
        if risk[i] > prior_cut:
            r.append(f"Matches high-risk behaviour group #{clusters[i]}.")
        if pct_point[i] >= 0.97:
            r.append("Unusual transaction shape vs normal traffic.")
        if pct_amt[i] >= 0.99:
            r.append(f"Large BTC transfer ({amt[i]:,.2f} BTC).")
        reasons.append(" | ".join(r) if r else "Looks like ordinary traffic.")
    out["Why Flagged"] = reasons
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True), full_cols


def score_csv_stream(source, bundle, chunk_size: int = 15000):
    """Stream-score a potentially huge CSV in low-memory chunks so large files
    never exceed cloud container RAM limits."""
    import gc

    if hasattr(source, "seek"):
        source.seek(0)
    parts = []
    full_cols = 0
    reader = pd.read_csv(source, chunksize=chunk_size, low_memory=True)
    for chunk in reader:
        scored_chunk, fc = score_dataframe(chunk, bundle)
        full_cols = max(full_cols, fc)
        parts.append(scored_chunk)
        del chunk
        gc.collect()
    if not parts:
        return score_dataframe(pd.DataFrame(), bundle)
    combined = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    combined = combined.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return combined, full_cols

