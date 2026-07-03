# Databricks notebook source
# MAGIC %md
# MAGIC # Batch inference: the weekly fraud report
# MAGIC
# MAGIC **Session:** Model Serving & Consumption / Monitoring & Retraining
# MAGIC
# MAGIC Score a batch of transactions with the governed `@champion` model and append the
# MAGIC results to an inference table. This is the batch prediction path (a scheduled "weekly
# MAGIC report"), and the table it writes is what the batch monitor profiles in the
# MAGIC Monitoring session.
# MAGIC
# MAGIC - **Model (personal):** `dev.<you>_fraud.fraud_detection@champion`
# MAGIC - **Inference table (personal):** `dev.<you>_fraud.fraud_predictions`
# MAGIC
# MAGIC To be usable by Lakehouse Monitoring's InferenceLog analysis, every row needs four
# MAGIC things beyond the features: a prediction, the model version that produced it, a
# MAGIC timestamp, and (when it arrives) the ground-truth label. We write all four so the
# MAGIC monitor can track both drift and quality over time.
# MAGIC
# MAGIC Docs: [Batch inference with Feature Engineering](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/score-batch)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC Install the Feature Engineering client so `score_batch` can resolve the feature
# MAGIC lookups the model was trained with (no train/serve skew).

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.0" --upgrade
# MAGIC %pip install -q databricks-feature-engineering

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dev")
dbutils.widgets.text("gold_schema", "fraud_gold")
dbutils.widgets.text("user_schema", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_alias", "champion")

catalog_name = dbutils.widgets.get("catalog_name")
gold_schema = dbutils.widgets.get("gold_schema")
user_schema = dbutils.widgets.get("user_schema")

from pyspark.sql import functions as F

# Interactive fallback: jobs pass the resolved personal schema; running standalone derives
# the same per-user name the bundle uses (dev_<short>_fraud) so runs stay isolated.
if not user_schema:
    _user = spark.range(1).select(F.current_user()).first()[0]
    _short = "".join(c if c.isalnum() else "_" for c in _user.split("@")[0])
    user_schema = f"dev_{_short}_fraud"

model_alias = dbutils.widgets.get("model_alias")
# The batch-inference job passes the full three-level model name; fall back for interactive runs.
model_name = dbutils.widgets.get("model_name") or f"{catalog_name}.{user_schema}.fraud_detection"

# Read the transactions to score from the shared gold source; write predictions to your schema.
source_table = f"{catalog_name}.{gold_schema}.transactions_enriched"
predictions_table = f"{catalog_name}.{user_schema}.fraud_predictions"

print(f"Model:       {model_name}@{model_alias}")
print(f"Score from:  {source_table}")
print(f"Write to:    {predictions_table}")

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

# Resolve the champion alias to a concrete version so we can stamp every scored row with
# the model version that produced it (the monitor's model_id_col groups metrics by this).
client = MlflowClient()
champion = client.get_model_version_by_alias(model_name, model_alias)
model_version = str(champion.version)
print(f"{model_alias} -> version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pick the batch to score
# MAGIC In production this would be the newly-arrived, not-yet-scored transactions. Build the
# MAGIC same request spine the model was trained on: the raw transaction keys and fields
# MAGIC (`card_id`, `client_id`, `amount`, `transaction_hour`, `mcc`, `use_chip`). The Feature
# MAGIC Store looks up the card/client entity features and runs the on-demand functions, so we
# MAGIC pass exactly what a live caller would send. `amount` is cast to double to match the
# MAGIC feature tables and the on-demand function signatures (no train/serve skew).

# COMMAND ----------

# The request spine mirrors the training spine (raw transaction keys and fields, no label).
# Building it is plain Spark, so it is provided; scoring with the champion is the step you
# fill in below.
to_score = (
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
    .limit(20000)
)
print(f"Scoring {to_score.count():,} transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Score with the champion model
# MAGIC `fe.score_batch` loads the `@champion` model, joins the features it was trained on,
# MAGIC and returns the input rows plus a `prediction` column.

# COMMAND ----------

# TODO: score the batch with the champion model
# HINT: fe.score_batch(model_uri=f"models:/{model_name}@{model_alias}", df=to_score).
# HINT: the result has the lookup key plus a "prediction" column (the monitor's prediction_col).
# <-- Your code here

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Shape the inference table and persist
# MAGIC Add the three columns the monitor needs alongside the prediction: the model version,
# MAGIC a scoring timestamp, and the ground-truth label. Then append.
# MAGIC
# MAGIC > The label (`is_fraud`) is joined here for the workshop so quality metrics compute
# MAGIC > immediately. In the real world fraud is confirmed later (chargebacks), so you would
# MAGIC > backfill the label into this table when it arrives; drift is watched in the meantime.

# COMMAND ----------

# Shape the inference table the monitor needs: the fraud score, a 0/1 class, the model
# version, a scoring timestamp, and the label. This assembly is provided; the timestamp is
# spread across the last 14 days so the daily monitor sees several windows (in production it
# would just be current_timestamp()).
labels = spark.read.table(source_table).select(
    "transaction_id", F.col("is_fraud").cast("int").alias("is_fraud")
)

scored_at = F.to_timestamp(
    F.date_sub(F.current_date(), (F.col("transaction_id") % F.lit(14)).cast("int"))
)

predictions = (
    scored.withColumn("model_version", F.lit(model_version))
    .withColumn("scored_at", scored_at)
    .join(labels, on="transaction_id", how="left")
    .select(
        "transaction_id",
        # The model serves a fraud PROBABILITY. Keep it as fraud_score and threshold it to a
        # 0/1 class for the classification monitor (prediction must match the INT label type).
        F.col("prediction").alias("fraud_score"),
        (F.col("prediction") >= F.lit(0.5)).cast("int").alias("prediction"),
        "model_version",
        "scored_at",
        "is_fraud",
    )
)

# TODO: append the predictions to the monitored inference table
# HINT: predictions.write.mode("append").option("mergeSchema", "true").saveAsTable(predictions_table)
# HINT: append (not overwrite) so the table grows each run and the monitor sees history.
# <-- Your code here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC - Batch scoring loads `@champion`; promotion is an alias swap, so this job never changes.
# MAGIC - Each row carries `prediction`, `model_version`, `scored_at`, and `is_fraud`: exactly
# MAGIC   what the InferenceLog monitor needs to track drift and quality over time.
# MAGIC - This is the batch path; the serving endpoint captures the online path. The
# MAGIC   Monitoring session profiles both.

# COMMAND ----------

display(spark.read.table(predictions_table).limit(10))
