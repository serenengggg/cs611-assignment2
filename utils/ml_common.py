"""Helpers shared by the training, inference and monitoring stages (pandas: the gold tables are small)."""
import glob
import json
import os
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from utils.config import GOLD, MODEL_BANK

ID_COLS = ["loan_id", "Customer_ID", "snapshot_date"]
REGISTRY = os.path.join(MODEL_BANK, "registry.json")


def to_date(s):
    return s if isinstance(s, date) else date.fromisoformat(str(s)[:10])


def add_months(d, n):
    return to_date(d) + relativedelta(months=n)


# ---------- gold table I/O (hive-style snapshot_date=YYYY-MM-DD partitions, same layout Spark writes) ----------
def list_partitions(table):
    root = os.path.join(GOLD, table)
    return sorted(p.split("snapshot_date=")[1] for p in glob.glob(os.path.join(root, "snapshot_date=*")))


def read_gold(table, start=None, end=None):
    """Read partitions with start <= snapshot_date <= end (inclusive, ISO strings or dates)."""
    frames = []
    for d in list_partitions(table):
        if (start and to_date(d) < to_date(start)) or (end and to_date(d) > to_date(end)):
            continue
        df = pd.read_parquet(os.path.join(GOLD, table, f"snapshot_date={d}"))
        df["snapshot_date"] = to_date(d)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_gold_partition(df, table, snapshot_date):
    path = os.path.join(GOLD, table, f"snapshot_date={to_date(snapshot_date)}")
    os.makedirs(path, exist_ok=True)
    for f in glob.glob(os.path.join(path, "*")):
        os.remove(f)
    df.drop(columns=["snapshot_date"], errors="ignore").to_parquet(os.path.join(path, "part-0.parquet"), index=False)
    print(f"  wrote {path} ({len(df)} rows)")


def labelled_dataset(app_start, app_end, labels_known_by):
    """Join features (application month in [app_start, app_end]) to labels observed on or before labels_known_by."""
    feats = read_gold("feature_store", app_start, app_end)
    labels = read_gold("label_store", end=labels_known_by)
    if feats.empty or labels.empty:
        return pd.DataFrame()
    return feats.merge(labels[["loan_id", "label"]], on="loan_id", how="inner")


# ---------- model registry ----------
def load_registry():
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            return json.load(f)
    return {"models": [], "events": []}


def save_registry(reg):
    os.makedirs(MODEL_BANK, exist_ok=True)
    with open(REGISTRY, "w") as f:
        json.dump(reg, f, indent=2, default=str)


def champion_as_of(reg, snapshot_date):
    """The model that was serving production on snapshot_date (latest promotion on or before that date)."""
    live = [m for m in reg["models"] if m.get("promoted_on") and to_date(m["promoted_on"]) <= to_date(snapshot_date)]
    return max(live, key=lambda m: m["promoted_on"]) if live else None


def load_model(model_version):
    import joblib
    d = os.path.join(MODEL_BANK, model_version)
    with open(os.path.join(d, "metadata.json")) as f:
        meta = json.load(f)
    return joblib.load(os.path.join(d, "model.joblib")), meta


# ---------- metrics ----------
def psi(expected, actual, bins=10):
    """Population Stability Index with bins taken from the reference (expected) distribution's quantiles."""
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_stat(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))
