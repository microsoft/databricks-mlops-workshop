# Databricks notebook source
# MAGIC %md
# MAGIC # Seed feature (data) drift (demo helper)
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC A small, safe helper so you can *see* feature drift inside the workshop instead of
# MAGIC waiting days for it to happen naturally. It reuses the batch-inference scoring path but
# MAGIC scores a **deliberately shifted slice** and stamps it with **today's** timestamp, so the
# MAGIC most recent monitor window looks different from the previous one.
# MAGIC
# MAGIC It writes to **your own** tables (the per-user `dev_<you>_fraud` schema), exactly like
# MAGIC batch inference, so every attendee can run it independently:
# MAGIC - appends the drifted batch to `dev.<you>_fraud.fraud_predictions`
# MAGIC - refreshes your `batch_monitor`, so `_drift_metrics` picks up the spike
# MAGIC
# MAGIC After it finishes, re-run `monitoring.py` section 4 to see the `amount` drift, and run
# MAGIC `feature_drift_check.py` to watch it return `is_drift_violated = True`.
# MAGIC
# MAGIC Run **after** you have run batch inference at least once (so there is a normal baseline
# MAGIC in the earlier windows to drift away from) and the monitor exists.
# MAGIC
# MAGIC Docs: [Batch inference with Feature Engineering](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/score-batch)

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.0" --upgrade
# MAGIC %pip install -q databricks-feature-engineering databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dev")
dbutils.widgets.text("gold_schema", "fraud_gold")
dbutils.widgets.text("user_schema", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("amount_multiplier", "6.0", label="Scale amount by this to force drift")
dbutils.widgets.text("num_rows", "5000", label="How many drifted rows to append")
dbutils.widgets.text("refresh_monitor", "true", label="Refresh the batch monitor after writing")

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
model_name = dbutils.widgets.get("model_name") or f"{catalog_name}.{user_schema}.fraud_detection"
amount_multiplier = float(dbutils.widgets.get("amount_multiplier"))
num_rows = int(dbutils.widgets.get("num_rows"))
refresh_monitor = dbutils.widgets.get("refresh_monitor").strip().lower() == "true"

source_table = f"{catalog_name}.{gold_schema}.transactions_enriched"
predictions_table = f"{catalog_name}.{user_schema}.fraud_predictions"

print(f"Model:       {model_name}@{model_alias}")
print(f"Write to:    {predictions_table}")
print(f"Drift:       amount x{amount_multiplier} on {num_rows} rows, stamped today")

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

client = MlflowClient()
champion = client.get_model_version_by_alias(model_name, model_alias)
model_version = str(champion.version)
print(f"{model_alias} -> version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build a deliberately drifted batch
# MAGIC Pick a skewed slice (younger clients) and inflate `amount`, so several monitored
# MAGIC features shift at once: `amount` directly, and the on-demand amount ratios and
# MAGIC `current_age`/income columns via the slice. This is the "world changed" we want the
# MAGIC monitor to catch.

# COMMAND ----------

# A skewed subpopulation so entity features (age/income/credit) also move, plus an inflated
# amount so the amount distribution clearly separates from the normal baseline windows.
to_score = (
    spark.read.table(source_table)
    .where(F.col("current_age") < F.lit(35))
    .select(
        "transaction_id",
        "card_id",
        "client_id",
        (F.col("amount").cast("double") * F.lit(amount_multiplier)).alias("amount"),
        "transaction_hour",
        "mcc",
        "use_chip",
    )
    .limit(num_rows)
)
print(f"Scoring {to_score.count():,} drifted transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Score and append with today's timestamp
# MAGIC Same `score_batch` path as batch inference, but every row is stamped with the current
# MAGIC timestamp so it lands in the newest monitor window, and we log the same feature columns
# MAGIC so the monitor can profile their drift.

# COMMAND ----------

scored = fe.score_batch(
    model_uri=f"models:/{model_name}@{model_alias}",
    df=to_score,
)

labels = spark.read.table(source_table).select(
    "transaction_id", F.col("is_fraud").cast("int").alias("is_fraud")
)
entity_features = spark.read.table(source_table).select(
    "transaction_id",
    F.col("credit_score").cast("int").alias("credit_score"),
    F.col("credit_limit").cast("double").alias("credit_limit"),
    F.col("yearly_income").cast("double").alias("yearly_income"),
    F.col("current_age").cast("int").alias("current_age"),
)

drifted = (
    scored.withColumn("model_version", F.lit(model_version))
    .withColumn("scored_at", F.current_timestamp())
    .join(labels, on="transaction_id", how="left")
    .join(entity_features, on="transaction_id", how="left")
    .select(
        "transaction_id",
        F.col("prediction").alias("fraud_score"),
        (F.col("prediction") >= F.lit(0.5)).cast("int").alias("prediction"),
        "model_version",
        "scored_at",
        "is_fraud",
        F.col("amount").cast("double").alias("amount"),
        "credit_score",
        "credit_limit",
        "yearly_income",
        "current_age",
    )
)

(drifted.write.mode("append").option("mergeSchema", "true").saveAsTable(predictions_table))
print(f"Appended {drifted.count():,} drifted rows to {predictions_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Refresh the batch monitor
# MAGIC Recompute the profile/drift metrics so the new window shows up. This takes a few
# MAGIC minutes; when it finishes, `monitoring.py` section 4 and `feature_drift_check.py` will
# MAGIC reflect the drift.

# COMMAND ----------

if refresh_monitor:
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    run = w.quality_monitors.run_refresh(table_name=predictions_table)
    print(f"Refresh started for {predictions_table}: refresh_id={run.refresh_id}")
    print("When it completes, re-run monitoring.py section 4 and feature_drift_check.py.")
else:
    print("Skipped refresh. Refresh the batch monitor manually to see the drift.")
