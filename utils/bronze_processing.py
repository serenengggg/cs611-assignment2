"""BRONZE: raw data landed as-is (all columns as string), one snapshot month per DAG run, plus ingestion metadata."""
import os
from pyspark.sql import functions as F

from utils.config import DATA_DIR, BRONZE, SOURCES
from utils.spark_io import write_partition


def build_bronze_partition(spark, name, snapshot_date):
    df = (
        spark.read.option("header", True).option("inferSchema", False)
        .csv(os.path.join(DATA_DIR, SOURCES[name]))
        .filter(F.col("snapshot_date") == snapshot_date)
        .withColumn("_source_file", F.lit(SOURCES[name]))
        .withColumn("_ingested_at", F.current_timestamp())
    )
    n = df.count()
    if n:
        write_partition(df, os.path.join(BRONZE, name), snapshot_date, "csv")
    return n


def run_bronze(spark, snapshot_date):
    print(f"== BRONZE {snapshot_date} ==")
    for name in SOURCES:
        print(f"{name}: {build_bronze_partition(spark, name, snapshot_date)} rows")
