"""SILVER: cleaned, typed, validated, de-duplicated tables (one per bronze source).

Fix known dirty patterns (stray underscores, sentinels), cast to proper types,
null-out values outside plausible business ranges,
drop PII (Name, SSN) that has no modelling value.
"""
import os
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

from utils.config import BRONZE, SILVER
from utils.spark_io import write_partition, read_table


def read_bronze(spark, name, snapshot_date):
    return read_table(spark, os.path.join(BRONZE, name), "csv", only=snapshot_date)


def _num(col):
    """'52312.68_' / '__10000__' / '_' -> double (null when not numeric)."""
    return F.regexp_replace(F.col(col), "_", "").cast("double")


def _in_range(col, lo, hi):
    return F.when((F.col(col) >= lo) & (F.col(col) <= hi), F.col(col))


def clean_attributes(df: DataFrame) -> DataFrame:
    df = df.withColumn("Age", _num("Age"))
    return (
        df.withColumn("Age", _in_range("Age", 18, 100).cast("int"))
        .withColumn("Occupation", F.when(F.col("Occupation").rlike("^_+$"), "Unknown").otherwise(F.col("Occupation")))
        .select("Customer_ID", "Age", "Occupation", "snapshot_date")  # Name, SSN dropped (PII)
        .dropDuplicates(["Customer_ID", "snapshot_date"])
    )


def clean_financials(df: DataFrame) -> DataFrame:
    for c in ["Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
              "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
              "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
              "Amount_invested_monthly", "Monthly_Balance"]:
        df = df.withColumn(c, _num(c))

    # plausible-range rules (values outside are data-entry errors / sentinels -> null)
    ranges = {
        "Annual_Income": (0, 250000), "Monthly_Inhand_Salary": (0, 20000),
        "Num_Bank_Accounts": (0, 20), "Num_Credit_Card": (0, 20), "Interest_Rate": (1, 40),
        "Num_of_Loan": (0, 10), "Delay_from_due_date": (-10, 100), "Num_of_Delayed_Payment": (0, 30),
        "Num_Credit_Inquiries": (0, 30), "Total_EMI_per_month": (0, 2000),
        "Amount_invested_monthly": (0, 2000),  # 10000 is a sentinel
        "Monthly_Balance": (0, 3000),          # -3.3e26 is a sentinel
    }
    for c, (lo, hi) in ranges.items():
        df = df.withColumn(c, _in_range(c, lo, hi))
    for c in ["Num_of_Loan", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Delay_from_due_date",
              "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]:
        df = df.withColumn(c, F.col(c).cast("int"))

    # categoricals
    df = (
        df.withColumn("Credit_Mix", F.when(F.col("Credit_Mix").isin("Good", "Standard", "Bad"), F.col("Credit_Mix")).otherwise("Unknown"))
        .withColumn("Payment_of_Min_Amount", F.when(F.col("Payment_of_Min_Amount").isin("Yes", "No"), F.col("Payment_of_Min_Amount")).otherwise("Unknown"))
        .withColumn("Payment_Behaviour", F.when(F.col("Payment_Behaviour").rlike("^(Low|High)_spent_"), F.col("Payment_Behaviour")).otherwise("Unknown"))
        .withColumn("Type_of_Loan", F.coalesce(F.col("Type_of_Loan"), F.lit("No Loan")))
    )
   
    df = df.withColumn(
        "Credit_History_Age_Months",
        (F.regexp_extract("Credit_History_Age", r"(\d+)\s+Years", 1).cast("int") * 12
         + F.regexp_extract("Credit_History_Age", r"(\d+)\s+Months", 1).cast("int")),
    ).drop("Credit_History_Age")
    return df.dropDuplicates(["Customer_ID", "snapshot_date"])


def clean_clickstream(df: DataFrame) -> DataFrame:
    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", F.col(f"fe_{i}").cast("int"))
    return df.dropDuplicates(["Customer_ID", "snapshot_date"])


def clean_lms(df: DataFrame) -> DataFrame:
    for c in ["tenure", "installment_num"]:
        df = df.withColumn(c, F.col(c).cast("int"))
    for c in ["loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance"]:
        df = df.withColumn(c, F.col(c).cast("double"))
    return (
        df.withColumn("loan_start_date", F.col("loan_start_date").cast("date"))
        .dropDuplicates(["loan_id", "installment_num"])
    )


CLEANERS = {
    "attributes": clean_attributes,
    "financials": clean_financials,
    "clickstream": clean_clickstream,
    "lms_loan_daily": clean_lms,
}


def run_silver(spark, snapshot_date):
    print(f"== SILVER {snapshot_date} ==")
    for name, fn in CLEANERS.items():
        df = read_bronze(spark, name, snapshot_date)
        if df is None:
            print(f"{name}: no bronze partition, skipped")
            continue
        df = fn(df.drop("_source_file", "_ingested_at"))
        write_partition(df, os.path.join(SILVER, name), snapshot_date)
