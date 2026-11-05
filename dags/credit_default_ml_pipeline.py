"""End-to-end monthly ML pipeline for loan-default prediction at application time.

One DAG run = one month (logical date = snapshot date). Backfilled from 2023-01 to 2025-07 with catchup:

  datamart:  bronze -> silver -> gold (feature store + label store)       [PySpark, Assignment 1 datamart]
  ml:        model_train (only when the governance SOP says so; champion/challenger -> model_bank/)
             -> model_inference (champion scores this month's applications -> gold.model_predictions)
             -> model_monitor   (stability now, performance when labels mature -> gold.model_monitoring)
             -> model_visualize (datamart/gold/monitoring_charts/*.png)
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup

RUN = "cd /opt/airflow && python -m utils.{module} --snapshotdate {{{{ ds }}}}"

default_args = {
    "owner": "credit-risk-ds",
    "depends_on_past": True,           # month N needs month N-1's datamart / model state
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="credit_default_ml_pipeline",
    description="Datamart -> train -> infer -> monitor -> visualise, monthly",
    default_args=default_args,
    schedule="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 7, 1),
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=False,     # backfills automatically on `docker-compose up`
    tags=["cs611", "credit-risk"],
) as dag:
    start = EmptyOperator(task_id="start")

    with TaskGroup("datamart") as datamart:
        bronze = BashOperator(task_id="bronze", bash_command=RUN.format(module="data_pipeline") + " --layer bronze")
        silver = BashOperator(task_id="silver", bash_command=RUN.format(module="data_pipeline") + " --layer silver")
        gold = BashOperator(task_id="gold_feature_label_store",
                            bash_command=RUN.format(module="data_pipeline") + " --layer gold")
        bronze >> silver >> gold

    with TaskGroup("ml") as ml:
        train = BashOperator(task_id="model_train", bash_command=RUN.format(module="model_train"))
        infer = BashOperator(task_id="model_inference", bash_command=RUN.format(module="model_inference"))
        monitor = BashOperator(task_id="model_monitor", bash_command=RUN.format(module="model_monitor"))
        visualize = BashOperator(task_id="model_visualize", bash_command=RUN.format(module="model_visualize"))
        train >> infer >> monitor >> visualize

    end = EmptyOperator(task_id="end")

    start >> datamart >> ml >> end
