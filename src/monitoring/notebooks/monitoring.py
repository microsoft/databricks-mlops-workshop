# Databricks notebook source
# MAGIC %md
# MAGIC # Monitoring: refresh and inspect the fraud monitors
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC Two Lakehouse Monitors were created declaratively by the bundle: one over the
# MAGIC batch report table and one over the online serving table. This lab refreshes them and
# MAGIC reads the metric tables they produce, so you can see model quality and drift over time
# MAGIC and compare the two prediction distributions.
# MAGIC
# MAGIC Each monitor emits two Delta tables plus a dashboard:
# MAGIC - `..._profile_metrics`: summary stats and model-quality metrics per window / model.
# MAGIC - `..._drift_metrics`: how the data / predictions drift window-over-window.
# MAGIC
# MAGIC - **Batch table:** `dev.<you>_fraud.fraud_predictions` (has labels, so drift and quality)
# MAGIC - **Online table:** `dev.<you>_fraud.fraud_serving_inference` (labels lag, drift-led)
# MAGIC
# MAGIC Docs: [Lakehouse Monitoring](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/)

# COMMAND ----------

# MAGIC %pip install -q --upgrade databricks-sdk
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog_name", "dev")
dbutils.widgets.text("user_schema", "")

catalog_name = dbutils.widgets.get("catalog_name")
user_schema = dbutils.widgets.get("user_schema")

from pyspark.sql import functions as F

# Interactive fallback: jobs pass the resolved personal schema; running standalone derives
# the same per-user name the bundle uses (dev_<short>_fraud) so runs stay isolated.
if not user_schema:
    _user = spark.range(1).select(F.current_user()).first()[0]
    _short = "".join(c if c.isalnum() else "_" for c in _user.split("@")[0])
    user_schema = f"dev_{_short}_fraud"

batch_table = f"{catalog_name}.{user_schema}.fraud_predictions"
online_table = f"{catalog_name}.{user_schema}.fraud_serving_inference"

print(f"Batch monitor table:  {batch_table}")
print(f"Online monitor table: {online_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Refresh the monitors
# MAGIC The bundle created the monitors on a schedule, but in a live lab we refresh on demand
# MAGIC so the metric tables reflect the rows we just scored. A refresh recomputes the
# MAGIC profile and drift metrics; it can take a few minutes.

# COMMAND ----------

# A Lakehouse Monitor recomputes its profile and drift metrics on refresh. The bundle
# created the monitors on a schedule; here we refresh on demand so the metric tables reflect
# the rows we just scored. The client and the guard loop are provided.
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

for table in [batch_table, online_table]:
    if spark.catalog.tableExists(table) and spark.read.table(table).limit(1).count() > 0:
        try:
            # TODO: refresh the monitor for this table
            # HINT: w.quality_monitors.run_refresh(table_name=table) starts a refresh and
            # HINT:   returns a run whose run.refresh_id you can print.
            # <-- Your code here
            print(f"Refresh started for {table}: refresh_id={run.refresh_id}")
        except Exception as exc:
            print(f"(could not refresh {table}: {exc})")
    else:
        print(f"Skipping {table}: table missing or empty.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model quality over time (batch)
# MAGIC The batch table has ground-truth labels, so its `_profile_metrics` table carries
# MAGIC model-quality metrics per time window. Read the whole-table rows for the real model
# MAGIC versions and watch the quality trend; this is what the retraining trigger checks.

# COMMAND ----------

# The monitor writes a _profile_metrics table with model-quality metrics per time window.
# The point is that monitoring produces a queryable metrics table you can trend and alert on.
# Classification metrics are STRUCTs, so we dereference the fields (f1_score.macro etc.) with
# F.col.
from pyspark.sql import functions as F

profile_metrics = f"{batch_table}_profile_metrics"
if spark.catalog.tableExists(profile_metrics):
    quality = (
        spark.read.table(profile_metrics)
        .where(
            (F.col("column_name") == ":table")
            & F.col("slice_key").isNull()
            & (F.col("model_version") != "*")
            & (F.col("log_type") == "INPUT")
        )
        .select(
            F.col("window.start").alias("window_start"),
            "model_version",
            F.col("count"),
            F.round("accuracy_score", 4).alias("accuracy"),
            F.round("f1_score.macro", 4).alias("f1_macro"),
            F.round("precision.macro", 4).alias("precision_macro"),
            F.round("recall.macro", 4).alias("recall_macro"),
        )
        .orderBy("window_start")
    )
    display(quality)
else:
    print(f"{profile_metrics} not ready yet; re-run after the refresh completes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Drift over time (batch + online)
# MAGIC The `_drift_metrics` table quantifies how the predictions and features shift
# MAGIC window-over-window. Because online ad-hoc traffic is skewed differently from the
# MAGIC weekly batch, comparing the two drift profiles is exactly the signal we care about.

# COMMAND ----------

# The _drift_metrics table quantifies how the prediction distribution shifts window-over-
# window. Comparing the batch and online drift profiles is the signal the retraining trigger
# watches.
for label, table in [("batch", batch_table), ("online", online_table)]:
    drift_metrics = f"{table}_drift_metrics"
    if spark.catalog.tableExists(drift_metrics):
        print(f"--- {label} prediction drift ---")
        drift = (
            spark.read.table(drift_metrics)
            .where(
                (F.col("column_name") == "prediction")
                & F.col("slice_key").isNull()
                & (F.col("drift_type") == "CONSECUTIVE")
            )
            .select(
                F.col("window.start").alias("window_start"),
                F.round("js_distance", 4).alias("js_distance"),
                F.round("wasserstein_distance", 4).alias("wasserstein_distance"),
            )
            .orderBy("window_start")
        )
        display(drift)
    else:
        print(f"{drift_metrics} not ready (no data or refresh pending).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC - **Two monitors, one picture:** batch (quality + drift) and online (drift-led) cover
# MAGIC   both prediction distributions.
# MAGIC - The auto-generated dashboard (linked from each table in Catalog Explorer) is the
# MAGIC   shareable view of everything above.
# MAGIC - The retraining job reads the batch `_profile_metrics` quality trend and only
# MAGIC   triggers training when it degrades; retraining then picks up the latest labelled data.
