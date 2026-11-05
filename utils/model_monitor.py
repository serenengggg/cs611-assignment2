"""MONITOR: stability and performance of production predictions -> gold.model_monitoring (one row per application cohort).

Run once per DAG run:  python -m utils.model_monitor --snapshotdate YYYY-MM-DD

  * Stability   (no labels needed, measured the same month the cohort is scored):
      PSI of the score distribution and of the model's top features vs the model's OOT reference sample.
  * Performance (needs labels, measured LABEL_MOB months later when the cohort's labels mature):
      AUC / Gini / KS, precision & recall of the review flag, actual vs predicted default rate.
"""
import argparse
import os

import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from utils import config as C
from utils.ml_common import add_months, ks_stat, load_model, psi, read_gold, to_date, write_gold_partition

TABLE = "model_monitoring"
N_FEATURE_PSI = 5


def _status(value, amber, red, higher_is_better):
    if value is None or pd.isna(value):
        return None
    if higher_is_better:
        return "RED" if value < red else "AMBER" if value < amber else "GREEN"
    return "RED" if value > red else "AMBER" if value > amber else "GREEN"


def _existing_row(cohort):
    rows = read_gold(TABLE, cohort, cohort)
    return rows.iloc[0].to_dict() if not rows.empty else {}


def stability(cohort):
    preds = read_gold("model_predictions", cohort, cohort)
    if preds.empty:
        return None
    version = preds["model_version"].iloc[0]
    _, meta = load_model(version)
    ref = pd.read_parquet(os.path.join(C.MODEL_BANK, version, "reference_oot.parquet"))
    feats = read_gold("feature_store", cohort, cohort)
    top = list(meta["top_features"])[:N_FEATURE_PSI]
    feat_psi = {f: psi(ref[f], feats[f]) for f in top}
    worst = max(feat_psi, key=lambda k: feat_psi[k])
    psi_score = psi(ref["score"], preds["default_proba"])
    return {
        "model_version": version, "n_scored": len(preds),
        "mean_pd": float(preds["default_proba"].mean()), "flag_rate": float(preds["flag_for_review"].mean()),
        "psi_score": psi_score, "max_feature_psi": feat_psi[worst], "max_psi_feature": worst,
        **{f"psi_{f}": v for f, v in feat_psi.items()},
        "stability_status": _status(max(psi_score, feat_psi[worst]), C.PSI_AMBER, C.PSI_RED, higher_is_better=False),
        "stability_measured_on": str(cohort),
    }


def performance(cohort, run_date):
    preds = read_gold("model_predictions", cohort, cohort)
    labels = read_gold("label_store", run_date, run_date)
    if preds.empty or labels.empty:
        return None
    df = preds.merge(labels[["loan_id", "label"]], on="loan_id", how="inner")
    if df["label"].nunique() < 2:
        return None
    y, p, flag = df["label"], df["default_proba"], df["flag_for_review"]
    auc = roc_auc_score(y, p)
    return {
        "n_labelled": len(df), "default_rate": float(y.mean()), "auc": auc, "gini": 2 * auc - 1,
        "ks": ks_stat(y, p), "precision_flag": precision_score(y, flag, zero_division=0),
        "recall_flag": recall_score(y, flag, zero_division=0),
        "performance_status": _status(auc, C.AUC_AMBER, C.AUC_RED, higher_is_better=True),
        "performance_measured_on": str(run_date),
    }


def monitor(run_date):
    # 1) stability of the cohort scored this month
    stab = stability(run_date)
    if stab:
        row = {**_existing_row(run_date), **stab}
        write_gold_partition(pd.DataFrame([row]), TABLE, run_date)
        print(f"stability {run_date}: PSI score={stab['psi_score']:.3f} max feature PSI={stab['max_feature_psi']:.3f} "
              f"({stab['max_psi_feature']}) -> {stab['stability_status']}")
    # 2) performance of the cohort whose labels matured this month
    cohort = add_months(run_date, -C.LABEL_MOB)
    perf = performance(cohort, run_date)
    if perf:
        row = {**_existing_row(cohort), **perf}
        write_gold_partition(pd.DataFrame([row]), TABLE, cohort)
        print(f"performance cohort {cohort}: AUC={perf['auc']:.3f} Gini={perf['gini']:.3f} "
              f"default rate={perf['default_rate']:.3f} -> {perf['performance_status']}")
    if not stab and not perf:
        print(f"{run_date}: nothing to monitor yet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    monitor(to_date(ap.parse_args().snapshotdate))
