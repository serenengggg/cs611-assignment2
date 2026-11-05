"""DATA: one monthly snapshot through the medallion datamart (raw csv -> bronze -> silver -> gold feature/label store).

    python -m utils.data_pipeline --snapshotdate YYYY-MM-DD [--layer bronze|silver|gold|all]
"""
import argparse

from utils.spark_io import get_spark
from utils.bronze_processing import run_bronze
from utils.silver_processing import run_silver
from utils.gold_processing import run_gold

LAYERS = {"bronze": run_bronze, "silver": run_silver, "gold": run_gold}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)
    ap.add_argument("--layer", default="all", choices=[*LAYERS, "all"])
    args = ap.parse_args()
    spark = get_spark(f"datamart_{args.layer}_{args.snapshotdate}")
    for name, fn in LAYERS.items():
        if args.layer in (name, "all"):
            fn(spark, args.snapshotdate)
    spark.stop()
