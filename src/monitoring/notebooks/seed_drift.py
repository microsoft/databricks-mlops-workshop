# Databricks notebook source
# MAGIC %md
# MAGIC # Seed feature (data) drift (demo helper)
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC Demo helper to make feature drift visible during the session instead of waiting for it
# MAGIC to occur naturally. The batch inference table is already scored, so this samples
# MAGIC existing predictions, inflates the amount-driven features, and stamps them with the
# MAGIC current time, so the most recent monitor window differs from the previous one.
# MAGIC
# MAGIC It writes to the per-user `dev_<you>_fraud` schema, so each attendee runs it
# MAGIC independently:
# MAGIC
# MAGIC - appends the drifted batch to `dev.<you>_fraud.fraud_predictions`;
# MAGIC - refreshes `batch_monitor` so `_drift_metrics` reflects the change.
# MAGIC
# MAGIC Prerequisites: run batch inference at least once (to establish a baseline) and ensure
# MAGIC the monitor exists. Afterwards, re-run `monitoring.py` section 4 to view the `amount`
# MAGIC drift and `feature_drift_check.py` to confirm `is_drift_violated = True`.
# MAGIC
# MAGIC Docs: [Batch inference with Feature Engineering](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/score-batch)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dev")
dbutils.widgets.text("ml_schema", "")
dbutils.widgets.text("amount_multiplier", "6.0", label="Scale amount by this to force drift")
dbutils.widgets.text("num_rows", "5000", label="How many drifted rows to append")
dbutils.widgets.text("refresh_monitor", "true", label="Refresh the batch monitor after writing")

from pyspark.sql import functions as F

catalog_name = dbutils.widgets.get("catalog_name")
ml_schema = dbutils.widgets.get("ml_schema")

# Interactive fallback: jobs pass the resolved personal schema; running standalone derives
# the same per-user name the bundle uses (dev_<short>_fraud) so runs stay isolated.
if not ml_schema:
    _user = spark.range(1).select(F.current_user()).first()[0]
    _short = "".join(c if c.isalnum() else "_" for c in _user.split("@")[0])
    ml_schema = f"dev_{_short}_fraud_ml"

amount_multiplier = float(dbutils.widgets.get("amount_multiplier"))
num_rows = int(dbutils.widgets.get("num_rows"))
refresh_monitor = dbutils.widgets.get("refresh_monitor").strip().lower() == "true"

predictions_table = f"{catalog_name}.{ml_schema}.fraud_predictions"

print(f"Table: {predictions_table}")
print(f"Drift: amount x{amount_multiplier} on {num_rows} rows, stamped now")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build a drifted batch from already-scored rows
# MAGIC The batch inference table is already fully scored with every feature column, so the
# MAGIC model is not re-run. Take a sample of existing predictions, inflate the amount-driven
# MAGIC features (`amount` and the two ratios that scale with it), and stamp them with the
# MAGIC current time so they land in the newest monitor window. This simulates a change in the
# MAGIC input distribution for the monitor to catch.

# COMMAND ----------

# Build the drifted sample by selecting from the same table (keeps the schema identical, no
# mergeSchema needed) and scaling the amount-driven features. Stage it to a separate table
# first so the append does not read from the table it is writing to; serverless does not allow
# cache/checkpoint, and a staging table is the clean way to break that dependency.
stage_table = f"{catalog_name}.{ml_schema}._seed_drift_stage"
(
    spark.read.table(predictions_table)
    .limit(num_rows)
    .withColumn("amount", F.col("amount") * F.lit(amount_multiplier))
    .withColumn(
        "amount_to_income_ratio", F.col("amount_to_income_ratio") * F.lit(amount_multiplier)
    )
    .withColumn(
        "amount_to_credit_limit_ratio",
        F.col("amount_to_credit_limit_ratio") * F.lit(amount_multiplier),
    )
    .withColumn("scored_at", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(stage_table)
)

drifted = spark.read.table(stage_table)
drifted.write.mode("append").saveAsTable(predictions_table)
n = drifted.count()
spark.sql(f"DROP TABLE IF EXISTS {stage_table}")
print(f"Appended {n:,} drifted rows to {predictions_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Refresh the batch monitor
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
