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


_GENERIC_COL_ALIASES = {
    "txId": ("tx_id", "txid", "tx_hash", "txhash", "hash", "transaction_id", "transaction_hash", "id"),
    "Time step": ("time_step", "timestep", "block_height", "block_number", "block_id", "block", "window", "step"),
    "total_BTC": (
        "total_btc", "amount", "amount_btc", "value", "btc", "output_value", "output_val",
        "out_val", "value_out", "total_output", "sent_btc", "volume", "input_value", "in_val",
    ),
    "fees": ("fee", "tx_fee", "transaction_fee", "fee_btc", "fees_btc", "fee_sat", "miner_fee", "fee_amount"),
    "size": ("tx_size", "bytes", "weight", "vsize", "virtual_size", "block_size"),
    "num_input_addresses": ("inputs", "n_inputs", "input_count", "vin_sz", "num_inputs", "in_degree", "n_in"),
    "num_output_addresses": ("outputs", "n_outputs", "output_count", "vout_sz", "num_outputs", "out_degree", "n_out"),
    "in_BTC_total": ("input_value", "in_val", "value_in", "total_input"),
    "out_BTC_total": ("output_value", "out_val", "value_out", "total_output"),
}


def _map_generic_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Map common Bitcoin CSV column names when native Elliptic++ headers are absent."""
    norm_to_actual = {
        c.strip().lower().replace(" ", "_").replace("-", "_"): c
        for c in df.columns
    }
    mapped_df = df.copy(deep=False)
    mapped_count = 0
    for target, aliases in _GENERIC_COL_ALIASES.items():
        if target in mapped_df.columns:
            continue
        for alias in aliases:
            if alias in norm_to_actual:
                src_col = norm_to_actual[alias]
                mapped_df[target] = mapped_df[src_col]
                if target not in ("txId", "Time step"):
                    mapped_count += 1
                break

    # Convert satoshis to BTC automatically if values are clearly in satoshis
    if "total_BTC" in mapped_df.columns and "total_BTC" not in df.columns:
        v = pd.to_numeric(mapped_df["total_BTC"], errors="coerce")
        pos = v[v > 0]
        if len(pos) > 0 and float(pos.median()) > 500_000.0:
            mapped_df["total_BTC"] = v / 1e8
    if "fees" in mapped_df.columns and "fees" not in df.columns:
        v = pd.to_numeric(mapped_df["fees"], errors="coerce")
        pos = v[v > 0]
        if len(pos) > 0 and float(pos.median()) > 50.0:
            mapped_df["fees"] = v / 1e8

    # Derive related Elliptic++ summary features when generic columns were mapped
    if mapped_count > 0 and "total_BTC" in mapped_df.columns:
        tot = pd.to_numeric(mapped_df["total_BTC"], errors="coerce").fillna(0.0)
        n_in = (
            pd.to_numeric(mapped_df["num_input_addresses"], errors="coerce").fillna(1.0).clip(lower=1.0)
            if "num_input_addresses" in mapped_df.columns
            else pd.Series(1.0, index=mapped_df.index)
        )
        n_out = (
            pd.to_numeric(mapped_df["num_output_addresses"], errors="coerce").fillna(2.0).clip(lower=1.0)
            if "num_output_addresses" in mapped_df.columns
            else pd.Series(2.0, index=mapped_df.index)
        )
        derived = {
            "in_txs_degree": n_in,
            "out_txs_degree": n_out,
            "in_BTC_total": tot,
            "out_BTC_total": tot,
            "in_BTC_mean": tot / n_in,
            "out_BTC_mean": tot / n_out,
            "in_BTC_max": tot / (n_in ** 0.5),
            "out_BTC_max": tot / (n_out ** 0.5),
            "in_BTC_min": tot / n_in,
            "out_BTC_min": tot / n_out,
            "in_BTC_median": tot / n_in,
            "out_BTC_median": tot / n_out,
        }
        for k_col, series_val in derived.items():
            if k_col not in mapped_df.columns:
                mapped_df[k_col] = series_val
                mapped_count += 1

    return mapped_df, mapped_count


def score_dataframe(df, bundle):
    """Score an arbitrary uploaded DataFrame using the saved model bundle.

    Works with full 182-feature Elliptic++ CSVs, simplified CSVs, or external
    Bitcoin transaction CSVs with different column names.
    """
    n_rows = len(df)
    feats = bundle["feats"]
    native_cols = sum(c in df.columns for c in feats)
    if native_cols < 20 and n_rows > 0:
        df, mapped_cols = _map_generic_columns(df)
        full_cols = native_cols + mapped_cols if native_cols == 0 else native_cols
    else:
        full_cols = native_cols

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

    # Look-alike burst within the uploaded batch
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
                Zm = Zp[m]
                if cnt > 2048:
                    stride = max(1, cnt // 2048)
                    d = NearestNeighbors(n_neighbors=11).fit(Zm[::stride]).kneighbors(Zm)[0][:, 1:]
                else:
                    d = NearestNeighbors(n_neighbors=11).fit(Zm).kneighbors(Zm)[0][:, 1:]
                d_mean = d.mean(axis=1)
                rank_part = rankdata(-d_mean) / cnt
                # Preserve burst detection when an uploaded sample has a high concentration of tight clones
                dist_sim = np.exp(-d_mean / 15.0)
                raw_look[m] = np.maximum(rank_part, dist_sim)
            else:
                d = NearestNeighbors(n_neighbors=k + 1).fit(Zp).kneighbors(Zp[m])[0][:, 1:]
                d_mean = d.mean(axis=1)
                rank_part = rankdata(-d_mean) / n_rows
                dist_sim = np.exp(-d_mean / 15.0)
                raw_look[m] = np.maximum(rank_part, dist_sim)
    else:
        raw_look[:] = 0.5

    pct_look = to_percentile(raw_look, bundle["ref_lookalike"])
    pct_fam = to_percentile(risk, bundle["ref_family"])
    pct_point = to_percentile(raw_point, bundle["ref_point"])

    # When an external CSV has very few native Elliptic++ features (< 20), compute
    # empirical multivariate anomaly scores across all numeric columns in `df`
    if native_cols < 20 and n_rows >= 5:
        skip_cols = {"txId", "Time step"}
        num_arrays = []
        for col_name in df.columns:
            if col_name in skip_cols or len(num_arrays) >= 16:
                continue
            s_num = pd.to_numeric(df[col_name], errors="coerce")
            valid_ratio = float(s_num.notna().mean())
            if valid_ratio >= 0.5:
                arr = s_num.fillna(s_num.median() if s_num.notna().any() else 0.0).to_numpy(dtype=np.float32)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                if float(np.std(arr)) > 1e-9:
                    num_arrays.append(signed_log(arr))
        if num_arrays:
            M = np.column_stack(num_arrays)
            meds = np.median(M, axis=0)
            mads = np.median(np.abs(M - meds), axis=0) * 1.4826 + 1e-5
            Z_emp = np.abs(M - meds) / mads
            rms_dev = np.sqrt(np.mean(Z_emp ** 2, axis=1))
            max_dev = np.max(Z_emp, axis=1)
            jitter = (np.arange(n_rows, dtype=np.float64) % 997) * 1e-7
            emp_point = (rankdata(0.6 * rms_dev + 0.4 * max_dev + jitter) - 0.5) / n_rows
            emp_look = (rankdata(0.5 * rms_dev + 0.5 * np.mean(Z_emp, axis=1) + jitter) - 0.5) / n_rows
            emp_fam_raw = (rankdata(0.6 * max_dev + 0.4 * rms_dev + jitter) - 0.5) / n_rows
            pct_point = np.maximum(pct_point, emp_point)
            pct_look = emp_look
            pct_fam = emp_fam_raw
            clusters = (np.argmax(Z_emp, axis=1) % 20).astype(np.int16)
            risk = np.where(emp_fam_raw >= 0.88, bundle["prior"] * 2.4, bundle["prior"] * 0.8)

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

