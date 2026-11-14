"""Load the Elliptic++ transaction data (cached as Parquet after the first run)."""
import pandas as pd

from config import CACHE_DIR, CLASSES_CSV, EDGES_CSV, FEATURES_CSV, TIME, AMOUNT, TRAIN_END  # noqa: F401

SPLIT = TRAIN_END  # kept so the older notebooks still import cleanly


def load():
    """Return (transactions DataFrame, list of feature column names).

    Column ``y``: 1 = illicit, 0 = licit, NaN = unknown (never investigated).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "transactions.parquet"
    if cache.exists():
        tx = pd.read_parquet(cache)
    else:
        tx = pd.read_csv(FEATURES_CSV).merge(pd.read_csv(CLASSES_CSV), on="txId")
        tx["y"] = tx["class"].astype(str).map({"1": 1.0, "2": 0.0})  # 3 = unknown
        tx.to_parquet(cache, index=False)
    feats = [c for c in tx.columns if c not in ("txId", TIME, "class", "y")]
    return tx, feats


def load_edges():
    """Directed money-flow edges: txId1 (sender tx) -> txId2 (receiver tx)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "edges.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    e = pd.read_csv(EDGES_CSV)
    e.to_parquet(cache, index=False)
    return e
