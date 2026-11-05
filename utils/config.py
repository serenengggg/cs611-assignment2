"""Central configuration: paths, label definition, and ML governance parameters."""
import os

ROOT = os.environ.get("PIPELINE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "data")
DATAMART = os.path.join(ROOT, "datamart")
BRONZE = os.path.join(DATAMART, "bronze")
SILVER = os.path.join(DATAMART, "silver")
GOLD = os.path.join(DATAMART, "gold")
MODEL_BANK = os.path.join(ROOT, "model_bank")
CHARTS = os.path.join(GOLD, "monitoring_charts")

# source name -> raw csv
SOURCES = {
    "clickstream": "feature_clickstream.csv",
    "attributes": "features_attributes.csv",
    "financials": "features_financials.csv",
    "lms_loan_daily": "lms_loan_daily.csv",
}

# Label definition (same as Lab 2 / Assignment 1): default = >= LABEL_DPD days past due at LABEL_MOB months on book
LABEL_MOB = 6
LABEL_DPD = 30

# ---- ML lifecycle ----
# DAG backfill window: applications 2023-01 .. 2025-01, labels for the last cohort mature 6 months later.
PIPELINE_START = "2023-01-01"
PIPELINE_END = "2025-07-01"

# Initial training on 2024-01-01: applications 2023-01..2023-07 have matured (6-month) labels by then.
FIRST_TRAIN_DATE = "2024-01-01"
TRAIN_WINDOW_MONTHS = 12  # rolling window of labelled application months used for training
OOT_MONTHS = 2            # most recent labelled months held out as out-of-time test
TEST_SIZE = 0.2           # random hold-out inside the in-time window
RANDOM_STATE = 611
SCHEDULED_RETRAIN_MONTHS = 6  # calendar refresh cadence (in addition to alert-triggered retrains)

# Monitoring thresholds (governance SOP)
AUC_AMBER, AUC_RED = 0.72, 0.68     # performance on matured cohorts
PSI_AMBER, PSI_RED = 0.10, 0.25     # population stability of scores / features
DECISION_THRESHOLD_QUANTILE = 0.80  # flag top-20% riskiest applicants for manual review
