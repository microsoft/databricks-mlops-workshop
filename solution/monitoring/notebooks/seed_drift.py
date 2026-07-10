# Databricks notebook source
# MAGIC %md
# MAGIC # Seed feature (data) drift (demo helper)
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC A small, safe helper so you can *see* feature drift inside the workshop instead of
# MAGIC waiting days for it to happen naturally. The batch inference table is already fully
# MAGIC scored, so instead of running the model again this takes a sample of existing
# MAGIC predictions, inflates the amount-driven features, and stamps them with **now**, so the
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

dbutils.widgets.text("catalog_name", "dev")
dbutils.widgets.text("user_schema", "")
dbutils.widgets.text("amount_multiplier", "6.0", label="Scale amount by this to force drift")
dbutils.widgets.text("num_rows", "5000", label="How many drifted rows to append")
dbutils.widgets.text("refresh_monitor", "true", label="Refresh the batch monitor after writing")

from pyspark.sql import functions as F

catalog_name = dbutils.widgets.get("catalog_name")
user_schema = dbutils.widgets.get("user_schema")

# Interactive fallback: jobs pass the resolved personal schema; running standalone derives
# the same per-user name the bundle uses (dev_<short>_fraud) so runs stay isolated.
if not user_schema:
    _user = spark.range(1).select(F.current_user()).first()[0]
    _short = "".join(c if c.isalnum() else "_" for c in _user.split("@")[0])
    user_schema = f"dev_{_short}_fraud"

amount_multiplier = float(dbutils.widgets.get("amount_multiplier"))
num_rows = int(dbutils.widgets.get("num_rows"))
refresh_monitor = dbutils.widgets.get("refresh_monitor").strip().lower() == "true"

predictions_table = f"{catalog_name}.{user_schema}.fraud_predictions"

print(f"Table: {predictions_table}")
print(f"Drift: amount x{amount_multiplier} on {num_rows} rows, stamped now")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build a drifted batch from already-scored rows
# MAGIC The batch inference table is already fully scored with every feature column, so we do
# MAGIC not re-run the model. Take a sample of existing predictions, inflate the amount-driven
# MAGIC features (`amount` and the two ratios that scale with it), and stamp them with the
# MAGIC current time so they land in the newest monitor window. This is the "world changed" we
# MAGIC want the monitor to catch.

# COMMAND ----------

# Build the drifted sample by selecting from the same table (keeps the schema identical, no
# mergeSchema needed) and scaling the amount-driven features. We stage it to a separate table
# first so the append does not read from the table it is writing to; serverless does not allow
# cache/checkpoint, and a staging table is the clean way to break that dependency.
stage_table = f"{catalog_name}.{user_schema}._seed_drift_stage"
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
