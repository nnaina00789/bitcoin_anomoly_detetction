"""Shared paths and settings. Every other module imports from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "elliptic_bitcoin_dataset"
CACHE_DIR = ROOT / "data" / "cache"
RESULTS_DIR = ROOT / "results"

# Elliptic++ files (the richer version of the Elliptic dataset)
FEATURES_CSV = DATA_DIR / "txs_features.csv"
CLASSES_CSV = DATA_DIR / "txs_classes.csv"
EDGES_CSV = DATA_DIR / "txs_edgelist.csv"

TIME, AMOUNT = "Time step", "total_BTC"

# Time-based split: we only ever learn from the past and test on the future.
TRAIN_END = 34      # steps 1-34  -> learning period
VAL_START = 25      # steps 25-34 -> validation (used to choose settings)
TEST_START = 35     # steps 35-49 -> final, untouched test period

SEED = 42
N_CLUSTERS = 50          # number of behaviour "families"
LABELS_PER_CLUSTER = 10  # analyst verification budget per family (~500 labels)
FLAG_QUANTILE = 0.99     # flag the top 1% most suspicious (calibrated on the past)

