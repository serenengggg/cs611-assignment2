"""INFERENCE: score this month's loan applications with the production (champion) model -> gold.model_predictions.

Run once per DAG run:  python -m utils.model_inference --snapshotdate YYYY-MM-DD
"""
import argparse

import pandas as pd

from utils.ml_common import champion_as_of, load_model, load_registry, read_gold, to_date, write_gold_partition


def infer(run_date):
    champ = champion_as_of(load_registry(), run_date)
    if champ is None:
        print(f"{run_date}: no production model yet, nothing to score")
        return
    feats = read_gold("feature_store", run_date, run_date)
    if feats.empty:
        print(f"{run_date}: no applications this month")
        return
    model, meta = load_model(champ["model_version"])
    proba = model.predict_proba(feats[meta["features"]])[:, 1]
    out = pd.DataFrame({
        "loan_id": feats["loan_id"], "Customer_ID": feats["Customer_ID"],
        "model_version": meta["model_version"], "default_proba": proba.round(6),
        "decision_threshold": meta["decision_threshold"],
        "flag_for_review": (proba >= meta["decision_threshold"]).astype(int),
        "scored_at": str(run_date),
    })
    print(f"{run_date}: scored {len(out)} applications with {meta['model_version']} "
          f"(mean PD {proba.mean():.3f}, flagged {out.flag_for_review.mean():.1%})")
    write_gold_partition(out, "model_predictions", run_date)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    infer(to_date(ap.parse_args().snapshotdate))
