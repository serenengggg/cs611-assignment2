"""TRAIN: decide whether to (re)train, train candidate models, store the best in the model bank, champion/challenger.

Run once per DAG run:  python -m utils.model_train --snapshotdate YYYY-MM-DD

Leakage guards
  - Only labels OBSERVED on or before the run date are used (label_store partition <= run date), so the newest
    application month in training is run_date - LABEL_MOB months.
  - Out-of-time (OOT) test = the most recent OOT_MONTHS labelled application months, never seen in fitting.
  - Imputation / scaling live inside the sklearn Pipeline and are fitted on the training split only.
"""
import argparse
import json
import os
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import config as C
from utils.ml_common import (ID_COLS, add_months, champion_as_of, labelled_dataset, load_model, load_registry,
                             read_gold, save_registry, to_date)

# Feature-set ablation on the first two training windows (HistGB, OOT AUC):
#   all 73 = 0.817 / 0.827, without clickstream = 0.786 / 0.814, without occupation = 0.815 / 0.828.
# -> clickstream stays; the 15 occupation dummies (univariate AUC ~0.50) are dropped as pure noise / drift surface.
# loan_to_annual_income is 10000 / Annual_Income (every loan is $10k), i.e. a duplicate of Annual_Income.
OCCUPATIONS = ["lawyer", "architect", "engineer", "accountant", "scientist", "teacher", "mechanic", "media_manager",
               "developer", "entrepreneur", "journalist", "doctor", "musician", "manager", "writer"]
EXCLUDED_FEATURES = [f"occ_{o}" for o in OCCUPATIONS] + ["loan_to_annual_income"]


def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS + ["label"] + EXCLUDED_FEATURES]


def candidates():
    return {
        "logistic_regression": (
            Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0]},
        ),
        "random_forest": (
            Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("model", RandomForestClassifier(n_jobs=2, class_weight="balanced_subsample",
                                                       random_state=C.RANDOM_STATE))]),
            {"model__n_estimators": [200, 400], "model__max_depth": [4, 6, 8, None],
             "model__min_samples_leaf": [5, 10, 25], "model__max_features": ["sqrt", 0.3]},
        ),
        "hist_gradient_boosting": (
            Pipeline([("model", HistGradientBoostingClassifier(random_state=C.RANDOM_STATE))]),
            {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [3, 4, 6],
             "model__max_iter": [100, 200, 400], "model__l2_regularization": [0.0, 1.0, 5.0],
             "model__min_samples_leaf": [20, 50]},
        ),
    }


def should_train(reg, run_date):
    """Governance SOP: initial build, 6-monthly scheduled refresh, or alert-triggered refresh."""
    if run_date < to_date(C.FIRST_TRAIN_DATE):
        return False, f"before first training date {C.FIRST_TRAIN_DATE}"
    champ = champion_as_of(reg, run_date)
    if champ is None:
        return True, "initial model build"
    attempts = [to_date(m["trained_on"]) for m in reg["models"] if to_date(m["trained_on"]) < run_date]
    last_attempt = max(attempts)
    if run_date >= add_months(champ["trained_on"], C.SCHEDULED_RETRAIN_MONTHS) and run_date >= add_months(last_attempt, C.SCHEDULED_RETRAIN_MONTHS):
        return True, f"scheduled {C.SCHEDULED_RETRAIN_MONTHS}-monthly refresh"
    mon = read_gold("model_monitoring")
    if not mon.empty and run_date >= add_months(last_attempt, 3):
        mon = mon.reindex(columns=[*mon.columns, *{"performance_measured_on", "performance_status"} - set(mon.columns)])
        last_month = str(add_months(run_date, -1))
        recent = mon[(mon["performance_measured_on"] == last_month) | (mon["stability_measured_on"] == last_month)]
        if ((recent["performance_status"] == "RED") | (recent["stability_status"] == "RED")).any():
            return True, "monitoring RED alert"
    return False, "champion healthy, no refresh due"


def train(run_date):
    reg = load_registry()
    version = f"credit_default_{run_date:%Y_%m_%d}"
    # idempotent re-run: forget anything this run produced before
    reg["models"] = [m for m in reg["models"] if m["model_version"] != version]
    reg["events"] = [e for e in reg["events"] if e["run_date"] != str(run_date)]

    do_train, reason = should_train(reg, run_date)
    print(f"train decision for {run_date}: {do_train} ({reason})")
    if not do_train:
        if run_date >= to_date(C.FIRST_TRAIN_DATE):
            reg["events"].append({"run_date": str(run_date), "event": "no_retrain", "reason": reason})
            save_registry(reg)
        return

    # ---- data windows (application months) ----
    last_app = add_months(run_date, -C.LABEL_MOB)                  # newest application month with a matured label
    first_app = add_months(last_app, -(C.TRAIN_WINDOW_MONTHS - 1))
    oot_start = add_months(last_app, -(C.OOT_MONTHS - 1))
    data = labelled_dataset(first_app, last_app, labels_known_by=run_date)
    feats = feature_columns(data)
    intime = data[data["snapshot_date"] < oot_start]
    oot = data[data["snapshot_date"] >= oot_start]
    train_df, test_df = train_test_split(intime, test_size=C.TEST_SIZE, stratify=intime["label"],
                                         random_state=C.RANDOM_STATE)
    print(f"windows: train/test apps {intime.snapshot_date.min()}..{intime.snapshot_date.max()} "
          f"({len(train_df)}/{len(test_df)}), OOT {oot_start}..{last_app} ({len(oot)})")

    # ---- fit + tune candidates (3-fold CV on train split), pick best by CV AUC ----
    results, fitted = [], {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.RANDOM_STATE)
    for name, (pipe, grid) in candidates().items():
        search = RandomizedSearchCV(pipe, grid, n_iter=8, scoring="roc_auc", cv=cv, random_state=C.RANDOM_STATE, n_jobs=1)
        search.fit(train_df[feats], train_df["label"])
        m = search.best_estimator_
        aucs = {k: roc_auc_score(d["label"], m.predict_proba(d[feats])[:, 1])
                for k, d in [("train", train_df), ("test", test_df), ("oot", oot)]}
        results.append({"algorithm": name, "best_params": search.best_params_, "cv_auc": search.best_score_,
                        **{f"{k}_auc": v for k, v in aucs.items()}})
        fitted[name] = m
        print(f"  {name:24s} cv={search.best_score_:.4f} train={aucs['train']:.4f} test={aucs['test']:.4f} oot={aucs['oot']:.4f}")
    best = max(results, key=lambda r: r["cv_auc"])  # selection on CV only; test/OOT stay untouched for reporting
    model = fitted[best["algorithm"]]

    # ---- reference data for monitoring + explainability ----
    oot_scores = model.predict_proba(oot[feats])[:, 1]
    imp = permutation_importance(model, test_df[feats], test_df["label"], scoring="roc_auc", n_repeats=5,
                                 random_state=C.RANDOM_STATE, n_jobs=1)
    importance = (pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).head(15).round(5).to_dict())

    # ---- champion / challenger on the same OOT window ----
    champ = champion_as_of(reg, run_date)
    champ_auc = None
    if champ is not None:
        cmodel, cmeta = load_model(champ["model_version"])
        champ_auc = roc_auc_score(oot["label"], cmodel.predict_proba(oot[cmeta["features"]])[:, 1])
    promote = champ_auc is None or best["oot_auc"] >= champ_auc
    print(f"challenger OOT AUC {best['oot_auc']:.4f} vs champion {champ_auc} -> {'PROMOTE' if promote else 'REJECT'}")

    # ---- write artefacts to the model bank ----
    out = os.path.join(C.MODEL_BANK, version)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    joblib.dump(model, os.path.join(out, "model.joblib"))
    ref = oot[["loan_id", "snapshot_date"] + feats].copy()
    ref["score"] = oot_scores
    ref.to_parquet(os.path.join(out, "reference_oot.parquet"), index=False)
    meta = {
        "model_version": version, "algorithm": best["algorithm"], "best_params": best["best_params"],
        "trained_on": str(run_date), "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_reason": reason, "label_definition": f"{C.LABEL_DPD}dpd_{C.LABEL_MOB}mob",
        "train_window": [str(intime.snapshot_date.min()), str(intime.snapshot_date.max())],
        "oot_window": [str(oot_start), str(last_app)],
        "n_train": len(train_df), "n_test": len(test_df), "n_oot": len(oot),
        "default_rate_train": float(train_df["label"].mean()),
        "metrics": {k: round(best[k], 4) for k in ["cv_auc", "train_auc", "test_auc", "oot_auc"]},
        "oot_gini": round(2 * best["oot_auc"] - 1, 4),
        "decision_threshold": float(np.quantile(oot_scores, C.DECISION_THRESHOLD_QUANTILE)),
        "candidates": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in results],
        "features": feats, "top_features": importance,
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    reg["models"].append({
        "model_version": version, "algorithm": best["algorithm"], "trained_on": str(run_date),
        "oot_auc": round(best["oot_auc"], 4), "champion_oot_auc_same_window": None if champ_auc is None else round(champ_auc, 4),
        "status": "champion" if promote else "rejected_challenger",
        "promoted_on": str(run_date) if promote else None,
    })
    if promote and champ is not None:
        for m in reg["models"]:
            if m["model_version"] == champ["model_version"]:
                m["status"], m["retired_on"] = "retired", str(run_date)
    reg["events"].append({"run_date": str(run_date), "event": "promoted" if promote else "challenger_rejected",
                          "reason": reason, "model_version": version})
    save_registry(reg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    train(to_date(ap.parse_args().snapshotdate))
