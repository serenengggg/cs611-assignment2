FROM apache/airflow:2.10.5-python3.11

USER root
# Java is required by PySpark (default-java resolves to OpenJDK 17 on amd64 and arm64)
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless procps && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

USER airflow
COPY requirements.txt /requirements.txt
# pin airflow itself so the extra packages can never up/downgrade it
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
