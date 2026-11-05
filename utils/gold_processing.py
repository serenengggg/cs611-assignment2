"""GOLD: ML-ready tables.

* label_store   - one row per loan: default flag (>= LABEL_DPD days past due at LABEL_MOB months on book).
* feature_store - one row per loan application, using ONLY information known on the application date
                  (the snapshot_date of the attributes/financials rows == loan start date).

Leakage guards
  - Nothing from lms_loan_daily is used as a feature (it describes the loan's future).
  - Clickstream is joined "as of" the application date: latest snapshot <= application date.
  - No global statistics (mean/std imputation, scaling) are computed here as that belongs in the
    training pipeline, fitted on the train split only.
"""
import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from utils.config import SILVER, GOLD, LABEL_MOB, LABEL_DPD
from utils.spark_io import write_partition, read_table

LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
    "Mortgage Loan", "Payday Loan", "Personal Loan", "Student Loan", "Not Specified",
]


def read_silver(spark, name, **kw):
    return read_table(spark, os.path.join(SILVER, name), **kw)


# Label store (partitioned by label observation date = loan_start + LABEL_MOB months)
def build_label_store(spark, snapshot_date):
    lms = read_silver(spark, "lms_loan_daily", only=snapshot_date)
    if lms is None:
        return None
    lms = lms.filter(F.col("installment_num") == LABEL_MOB)
    missed = F.ceil(F.col("overdue_amt") / F.col("due_amt"))
    lms = lms.withColumn("installments_missed", F.when(F.col("overdue_amt") > 0, missed).otherwise(F.lit(0)).cast("int"))
    lms = lms.withColumn("first_missed_date", F.when(F.col("installments_missed") > 0, F.add_months("snapshot_date", -F.col("installments_missed"))))
    lms = lms.withColumn("dpd", F.when(F.col("installments_missed") > 0, F.datediff("snapshot_date", "first_missed_date")).otherwise(F.lit(0)).cast("int"))
    return lms.select(
        "loan_id", "Customer_ID",
        F.when(F.col("dpd") >= LABEL_DPD, 1).otherwise(0).cast("int").alias("label"),
        F.lit(f"{LABEL_DPD}dpd_{LABEL_MOB}mob").alias("label_def"),
        "loan_start_date",
        "snapshot_date",  # label observation date = loan_start + LABEL_MOB months
    )


# Feature store (partitioned by application date)
def build_feature_store(spark, snapshot_date):
    attr = read_silver(spark, "attributes", only=snapshot_date)
    fin = read_silver(spark, "financials", only=snapshot_date)
    click = read_silver(spark, "clickstream", up_to=snapshot_date)  # history only, never the future
    if attr is None or fin is None:
        return None

    df = attr.join(fin, ["Customer_ID", "snapshot_date"], "inner")  # same application snapshot
    df = df.withColumn(
        "loan_id", F.concat_ws("_", F.col("Customer_ID"), F.date_format("snapshot_date", "yyyy_MM_dd"))
    )

    # Clickstream: latest snapshot on or before the application date
    fe_cols = [f"fe_{i}" for i in range(1, 21)]
    c = click.select("Customer_ID", F.col("snapshot_date").alias("click_date"), *fe_cols)
    joined = df.select("Customer_ID", "snapshot_date").join(
        c, (df.Customer_ID == c.Customer_ID) & (c.click_date <= df.snapshot_date), "left"
    ).select(df.Customer_ID, df.snapshot_date, *[c[x] for x in ["click_date"] + fe_cols])
    w = Window.partitionBy("Customer_ID", "snapshot_date").orderBy(F.col("click_date").desc_nulls_last())
    latest = (
        joined.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")
        .withColumn("click_months_lag", F.months_between("snapshot_date", "click_date").cast("int"))
        .withColumn("has_clickstream", F.col("click_date").isNotNull().cast("int"))
        .drop("click_date")
    )
    df = df.join(latest, ["Customer_ID", "snapshot_date"], "left")

    # Engineered features (row-level, deterministic; nulls propagate)
    df = (
        df.withColumn("debt_to_income", F.col("Outstanding_Debt") / F.col("Annual_Income"))
        .withColumn("emi_to_salary", F.col("Total_EMI_per_month") / F.col("Monthly_Inhand_Salary"))
        .withColumn("invest_to_salary", F.col("Amount_invested_monthly") / F.col("Monthly_Inhand_Salary"))
        .withColumn("balance_to_salary", F.col("Monthly_Balance") / F.col("Monthly_Inhand_Salary"))
        .withColumn("loan_to_annual_income", F.lit(10000.0) / F.col("Annual_Income"))
        .withColumn("credit_history_years", F.col("Credit_History_Age_Months") / 12.0)
    )
    for t in LOAN_TYPES:
        df = df.withColumn(
            "has_" + t.lower().replace(" ", "_").replace("-", "_"),
            F.col("Type_of_Loan").contains(t).cast("int"),
        )
    df = df.withColumn("no_loan_history", (F.col("Type_of_Loan") == "No Loan").cast("int")).drop("Type_of_Loan")

    # Categorical encoding (fixed vocabularies -> no fitting on the data)
    df = (
        df.withColumn("credit_mix_score", F.when(F.col("Credit_Mix") == "Bad", 0).when(F.col("Credit_Mix") == "Standard", 1)
                      .when(F.col("Credit_Mix") == "Good", 2))
        .withColumn("pays_min_amount", F.when(F.col("Payment_of_Min_Amount") == "Yes", 1).when(F.col("Payment_of_Min_Amount") == "No", 0))
        .withColumn("spend_level", F.when(F.col("Payment_Behaviour").startswith("Low"), 0).when(F.col("Payment_Behaviour").startswith("High"), 1))
        .withColumn("payment_size", F.when(F.col("Payment_Behaviour").endswith("Small_value_payments"), 0)
                    .when(F.col("Payment_Behaviour").endswith("Medium_value_payments"), 1)
                    .when(F.col("Payment_Behaviour").endswith("Large_value_payments"), 2))
        .drop("Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour", "Credit_History_Age_Months")
    )
    for occ in ["Lawyer", "Architect", "Engineer", "Accountant", "Scientist", "Teacher", "Mechanic", "Media_Manager",
                "Developer", "Entrepreneur", "Journalist", "Doctor", "Musician", "Manager", "Writer"]:
        df = df.withColumn("occ_" + occ.lower(), (F.col("Occupation") == occ).cast("int"))
    df = df.drop("Occupation")

    first = ["loan_id", "Customer_ID", "snapshot_date"]
    return df.select(*first, *[x for x in df.columns if x not in first])


def run_gold(spark, snapshot_date):
    print(f"== GOLD {snapshot_date} ==")
    labels = build_label_store(spark, snapshot_date)
    if labels is not None and labels.count():
        write_partition(labels, os.path.join(GOLD, "label_store"), snapshot_date)
    features = build_feature_store(spark, snapshot_date)
    if features is not None:
        write_partition(features, os.path.join(GOLD, "feature_store"), snapshot_date)
    else:
        print("feature_store: no applications this month, skipped")
