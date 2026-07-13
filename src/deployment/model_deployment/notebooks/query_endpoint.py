# Databricks notebook source
# MAGIC %md
# MAGIC # Test the serving endpoint
# MAGIC
# MAGIC **Session:** Model Serving & Consumption
# MAGIC
# MAGIC Test harness for the endpoint the deployment job created. It sends real transactions
# MAGIC through the endpoint one at a time, as a payment gateway would call the model per
# MAGIC transaction.
# MAGIC
# MAGIC Each call sets `client_request_id` to the transaction id, a correlation id the AI
# MAGIC Gateway logs in its own column, so the Monitoring session can join these predictions to
# MAGIC confirmed-fraud labels later. The transaction id is metadata, not a model input; the
# MAGIC payload is the raw transaction spine (`card_id`, `client_id`, `amount`,
# MAGIC `transaction_hour`, `mcc`, `use_chip`), and the endpoint looks up the entity features
# MAGIC and runs the on-demand functions per request.
# MAGIC
# MAGIC Docs: [Query serving endpoints](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/score-model-serving-endpoints)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "adoption_workshop")
dbutils.widgets.text("gold_schema", "fraud_gold")
dbutils.widgets.text("ml_schema", "")
dbutils.widgets.text("num_requests", "50")

catalog_name = dbutils.widgets.get("catalog_name")
gold_schema = dbutils.widgets.get("gold_schema")
ml_schema = dbutils.widgets.get("ml_schema")
num_requests = int(dbutils.widgets.get("num_requests"))

from pyspark.sql import functions as F

# Interactive fallback: derive the same per-user name the bundle uses (dev_<short>_fraud).
if not ml_schema:
    _user = spark.range(1).select(F.current_user()).first()[0]
    _short = "".join(c if c.isalnum() else "_" for c in _user.split("@")[0])
    ml_schema = f"dev_{_short}_fraud"

# One endpoint per user, named after the personal schema (matches the deployment job).
endpoint_name = f"{ml_schema}_fraud"
source_table = f"{catalog_name}.{gold_schema}.transactions_enriched"
print(f"Endpoint: {endpoint_name}")
print(f"Sample:   {num_requests} rows from {source_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build a sample of live transactions
# MAGIC Take a sample of real transactions and keep only the request spine (the raw fields a
# MAGIC live caller would send). `amount` is cast to double to match the feature tables and
# MAGIC the on-demand function signatures.

# COMMAND ----------

from pyspark.sql import functions as F

sample = (
    spark.read.table(source_table)
    .select(
        "transaction_id",
        "card_id",
        "client_id",
        F.col("amount").cast("double").alias("amount"),
        "transaction_hour",
        "mcc",
        "use_chip",
    )
    .na.drop()
    .limit(num_requests)
    .collect()
)
print(f"Prepared {len(sample)} transactions to score.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Score each transaction through the endpoint
# MAGIC POST one record per call to the endpoint's `invocations` URL, tagging each with
# MAGIC `client_request_id` = the transaction id so Monitoring can join labels later.

# COMMAND ----------

import requests

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = ctx.apiUrl().get()
token = ctx.apiToken().get()
invocations_url = f"{host}/serving-endpoints/{endpoint_name}/invocations"


def score_live(row):
    record = {
        "card_id": int(row["card_id"]),
        "client_id": int(row["client_id"]),
        "amount": float(row["amount"]),
        "transaction_hour": int(row["transaction_hour"]),
        "mcc": int(row["mcc"]),
        "use_chip": row["use_chip"],
    }
    body = {"client_request_id": str(row["transaction_id"]), "dataframe_records": [record]}
    resp = requests.post(
        invocations_url,
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["predictions"][0]


results = [(int(r["transaction_id"]), score_live(r)) for r in sample]
print(f"Scored {len(results)} live transactions (client_request_id = transaction_id). Sample:")
for tid, pred in results[:5]:
    print(f"  transaction_id={tid} -> prediction={pred}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC - The endpoint scored brand-new transactions: entity features looked up by key,
# MAGIC   transaction features computed on demand.
# MAGIC - Every call carried `client_request_id` = transaction id, the correlation key the
# MAGIC   AI Gateway logs so `process_serving_logs` can join labels for the online monitor.
# MAGIC - AI Gateway delivery lags (up to ~1 hour), so these calls appear in the inference
# MAGIC   table shortly after this runs.
