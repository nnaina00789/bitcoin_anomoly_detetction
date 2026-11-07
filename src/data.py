import pandas as pd
from pathlib import Path

TIME, AMOUNT, SPLIT = "Time step", "total_BTC", 34

def load():
    files = {f.name: f for f in Path("../data").rglob("*.csv")}
    tx = pd.read_csv(files["txs_features.csv"]).merge(
        pd.read_csv(files["txs_classes.csv"]), on="txId")
    tx["y"] = tx["class"].astype(str).map({"1": 1, "2": 0})
    feats = [c for c in tx.columns if c not in ("txId", TIME, "class", "y")]
    return tx, feats