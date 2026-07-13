# Databricks notebook source
# MAGIC %md
# MAGIC # Monitoring: refresh and inspect the fraud monitors
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC Interactive dashboard. It refreshes the two Lakehouse Monitors and displays the metric
# MAGIC tables they produce; it makes no decisions and is not part of the retraining job. The
# MAGIC automated decisions run headless in two job tasks that read the same tables:
# MAGIC `metric_violation_check.py` (model quality) and `feature_drift_check.py` (feature drift).
# MAGIC
# MAGIC The bundle creates two monitors:
# MAGIC
# MAGIC - **Batch table:** `dev.<you>_fraud.fraud_predictions` (labels present, drift and quality).
# MAGIC - **Online table:** `dev.<you>_fraud.fraud_serving_inference` (labels lag, drift-led).
# MAGIC
# MAGIC Each monitor emits two Delta tables plus a dashboard:
# MAGIC
# MAGIC - `..._profile_metrics`: summary statistics and model-quality metrics per window / model.
# MAGIC - `..._drift_metrics`: data and prediction drift, window over window.
# MAGIC
# MAGIC Docs: [Lakehouse Monitoring](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "adoption_workshop")
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
# MAGIC
# MAGIC The monitors run on a schedule, but here they are refreshed on demand so the metric
# MAGIC tables reflect the rows just scored. A refresh recomputes the profile and drift metrics
# MAGIC and can take a few minutes.

# COMMAND ----------

# A Lakehouse Monitor recomputes its profile and drift metrics on refresh. The monitors run
# on a schedule; here they are refreshed on demand so the metric tables reflect the rows just
# scored. The client and the guard loop are provided.
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

for table in [batch_table, online_table]:
    if spark.catalog.tableExists(table) and spark.read.table(table).limit(1).count() > 0:
        try:
            # TODO: refresh the monitor for this table
            # HINT: w.quality_monitors.run_refresh(table_name=table) starts a refresh and
            # HINT:   returns a run with a run.refresh_id to print.
            # <-- Your code here
            print(f"Refresh started for {table}: refresh_id={run.refresh_id}")
        except Exception as exc:
            print(f"(could not refresh {table}: {exc})")
    else:
        print(f"Skipping {table}: table missing or empty.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model quality over time (batch)
# MAGIC
# MAGIC The batch table has ground-truth labels, so its `_profile_metrics` table carries
# MAGIC model-quality metrics per window. This reads the whole-table rows for the real model
# MAGIC versions to show the quality trend, the signal the retraining trigger checks.

# COMMAND ----------

# The monitor writes a _profile_metrics table with model-quality metrics per time window,
# a queryable table for trending and alerting. Classification metrics are STRUCTs, so the
# fields (f1_score.macro etc.) are dereferenced with F.col.
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
# MAGIC
# MAGIC The `_drift_metrics` table quantifies how predictions and features shift window over
# MAGIC window. Online traffic is distributed differently from the batch, so comparing the two
# MAGIC drift profiles is a useful signal.

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
                F.round("population_stability_index", 4).alias("population_stability_index"),
                F.round("wasserstein_distance", 4).alias("wasserstein_distance"),
            )
            .orderBy("window_start")
        )
        display(drift)
    else:
        print(f"{drift_metrics} not ready (no data or refresh pending).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature (data) drift
# MAGIC
# MAGIC The same `_drift_metrics` table has a row per input feature, so input distribution
# MAGIC shift can be tracked, not just prediction drift. `population_stability_index` (PSI) is
# MAGIC populated for numeric features; Lakehouse Monitoring leaves `js_distance` null for
# MAGIC numeric columns and fills it only for categoricals. The retraining job's feature-drift
# MAGIC check thresholds these same values.

# COMMAND ----------

features_to_watch = [
    "amount",
    "credit_score",
    "credit_limit",
    "yearly_income",
    "current_age",
    "num_cards_issued",
    "is_night",
    "is_online",
    "is_high_risk_mcc",
    "amount_to_income_ratio",
    "amount_to_credit_limit_ratio",
]
batch_drift = f"{batch_table}_drift_metrics"
if spark.catalog.tableExists(batch_drift):
    feature_drift = (
        spark.read.table(batch_drift)
        .where(
            F.col("column_name").isin(features_to_watch)
            & F.col("slice_key").isNull()
            & (F.col("drift_type") == "CONSECUTIVE")
        )
        .select(
            F.col("window.start").alias("window_start"),
            "column_name",
            F.round("population_stability_index", 4).alias("population_stability_index"),
            F.round("wasserstein_distance", 4).alias("wasserstein_distance"),
        )
        .orderBy("window_start", "column_name")
    )
    display(feature_drift)
else:
    print(f"{batch_drift} not ready yet; re-run after the refresh completes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC
# MAGIC - Two monitors cover both prediction distributions: batch (quality and drift) and
# MAGIC   online (drift-led).
# MAGIC - Section 3 tracks prediction drift (model output); section 4 tracks feature drift
# MAGIC   (model inputs). Both come from the same `_drift_metrics` table.
# MAGIC - The auto-generated dashboard, linked from each table in Catalog Explorer, is the
# MAGIC   shareable view.
# MAGIC - The retraining job acts on two signals: the batch quality trend (`_profile_metrics`,
# MAGIC   label-based) and sustained feature drift (`_drift_metrics`). Either can trigger
# MAGIC   training, which retrains on the latest labelled data.
