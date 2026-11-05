"""Small Spark helpers shared by all layers."""
import os
import shutil
from pyspark.sql import SparkSession


def get_spark(app_name="medallion_pipeline"):
    spark = (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def partition_path(table_path, snapshot_date):
    return os.path.join(table_path, f"snapshot_date={snapshot_date}")


def write_partition(df, table_path, snapshot_date, fmt="parquet"):
    """Idempotently (re)write ONE snapshot partition of a table, so each monthly DAG run only touches its own month.

    Readers load the table root and get snapshot_date back through partition discovery.
    """
    path = partition_path(table_path, snapshot_date)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(table_path, exist_ok=True)
    writer = df.drop("snapshot_date").coalesce(1).write.mode("overwrite").format(fmt)
    if fmt == "csv":
        writer = writer.option("header", True)
    writer.save(path)
    print(f"  wrote {path}")


def read_table(spark, table_path, fmt="parquet", up_to=None, only=None):
    """Read a partitioned table; optionally restrict to snapshot_date <= up_to or == only."""
    from pyspark.sql import functions as F
    reader = spark.read.option("basePath", table_path)
    if fmt == "csv":
        reader = reader.option("header", True)
    if only is not None:
        path = partition_path(table_path, only)
        if not os.path.exists(path):
            return None
        df = reader.format(fmt).load(path)
    else:
        if not os.path.exists(table_path):
            return None
        df = reader.format(fmt).load(table_path)
    df = df.withColumn("snapshot_date", F.col("snapshot_date").cast("date"))
    if up_to is not None:
        df = df.filter(F.col("snapshot_date") <= F.lit(up_to).cast("date"))
    return df
