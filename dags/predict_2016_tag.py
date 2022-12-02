from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

import predict as predict

with DAG(
    dag_id="predict_2016_tag",
    schedule_interval="@daily",
    start_date=datetime(2020, 1, 1),
    catchup=False,
) as dag:
    extract_task = PythonOperator(
        task_id="extract",
        python_callable=predict.extract,
    )
    load_model_task = PythonOperator(
        task_id="load_model",
        python_callable=predict.load_model,
    )
    get_days_task = PythonOperator(
        task_id="get_days",
        python_callable=predict.get_days,
    )

    predict_week_task = PythonOperator(
        task_id="predict_week",
        python_callable=predict.predict_week,
        provide_context=True,
    )

    [extract_task, load_model_task, get_days_task] >> predict_week_task
