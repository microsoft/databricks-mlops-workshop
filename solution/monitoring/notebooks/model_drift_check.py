# Databricks notebook source
# MAGIC %md
# MAGIC # Model drift check (instructor-provided)
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC Automated task in the scheduled `retraining_job`. It reads the monitor's
# MAGIC `..._profile_metrics` table, decides whether model quality has degraded across recent
# MAGIC windows, and publishes the boolean `is_model_drift_violated` as a task value. A downstream
# MAGIC `condition_task` reads that value and triggers `model_training_job` only when quality
# MAGIC has degraded.
# MAGIC
# MAGIC Related notebooks:
# MAGIC
# MAGIC - `feature_drift_check.py`: the same pattern for feature drift (`..._drift_metrics`).
# MAGIC - `monitoring.py`: the interactive dashboard that displays these metrics.
# MAGIC
# MAGIC Parameters:
# MAGIC
# MAGIC - `table_name_under_monitor`: the inference table the monitor profiles.
# MAGIC - `metric_to_monitor`: quality metric column in `..._profile_metrics` (e.g. `f1_score`).
# MAGIC - `model_drift_threshold`: retrain if the metric drops below this.
# MAGIC - `num_evaluation_windows` / `num_violation_windows`: how many recent windows to check,
# MAGIC   and how many must be in violation before retraining.
# MAGIC
# MAGIC Docs: [Lakehouse Monitoring metric tables](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/monitor-output)

# COMMAND ----------

dbutils.widgets.text(
    "table_name_under_monitor", "", label="Inference table under monitor (three-level name)"
)
dbutils.widgets.text("metric_to_monitor", "f1_score.macro", label="Quality metric to monitor")
dbutils.widgets.text(
    "model_drift_threshold", "0.20", label="Retrain if the metric drops below this"
)
dbutils.widgets.text("num_evaluation_windows", "5", label="Number of recent windows to check")
dbutils.widgets.text("num_violation_windows", "2", label="Windows that must violate to trigger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The violation query
# MAGIC
# MAGIC Lakehouse Monitoring writes a `<table>_profile_metrics` Delta table. For an
# MAGIC InferenceLog monitor with a `label_col`, that table holds model-quality metrics
# MAGIC (accuracy, precision, recall, f1, ...) computed per time `window`, per
# MAGIC `model_version`, and per `slice`. It reads the whole-table rows for the real model
# MAGIC versions and ask: of the last `num_evaluation_windows` windows, did at least
# MAGIC `num_violation_windows` fall **below** the quality threshold, and is the most recent
# MAGIC window also below it?
# MAGIC
# MAGIC Fraud quality (e.g. `f1_score.macro`) is "higher is better", so a violation means the
# MAGIC metric dropped **below** `model_drift_threshold`. (For a drift or error metric
# MAGIC where "higher is worse", flip `<` to `>`.) Requiring the most recent window to be in
# MAGIC violation avoids retraining on a dip that has already recovered.
# MAGIC
# MAGIC The row filters mean:
# MAGIC - `column_name = ":table"` and `slice_key IS NULL` -> the metric for the whole table
# MAGIC   within a granularity (not a single feature column or data slice),
# MAGIC - `log_type = "INPUT"` -> the primary table metrics, not baseline-table metrics,
# MAGIC - `model_version != "*"` -> a concrete model version, not the all-versions aggregate
# MAGIC   (the model-id column is named after the monitor's `model_id_col`, here `model_version`).
# MAGIC
# MAGIC The classification quality metrics are **structs**: pass `f1_score.macro` (or
# MAGIC `.weighted`), not the bare `f1_score`. `F.col(metric_to_monitor)` dereferences the
# MAGIC struct field and is aliased to `metric_value`.

# COMMAND ----------

from pyspark.sql import functions as F

table_name_under_monitor = dbutils.widgets.get("table_name_under_monitor")
metric_to_monitor = dbutils.widgets.get("metric_to_monitor")
model_drift_threshold = float(dbutils.widgets.get("model_drift_threshold"))
num_evaluation_windows = int(dbutils.widgets.get("num_evaluation_windows"))
num_violation_windows = int(dbutils.widgets.get("num_violation_windows"))

profile_metrics_table = f"{table_name_under_monitor}_profile_metrics"

# If the monitor hasn't produced metrics yet (first run, or never refreshed), there is
# nothing to evaluate, so report "not violated" and the job does not retrain.
if not spark.catalog.tableExists(profile_metrics_table):
    print(f"{profile_metrics_table} does not exist yet; treating as not violated.")
    dbutils.jobs.taskValues.set("is_model_drift_violated", False)
    dbutils.notebook.exit("no_metrics")

# COMMAND ----------

# The most recent num_evaluation_windows whole-table quality metrics for real model versions,
# newest first. metric_to_monitor can be a struct field (e.g. f1_score.macro), so F.col
# dereferences it.
metric_value = F.col(metric_to_monitor)
recent_metrics = (
    spark.read.table(profile_metrics_table)
    .where(
        (F.col("column_name") == ":table")
        & F.col("slice_key").isNull()
        & (F.col("model_version") != "*")
        & (F.col("log_type") == "INPUT")
        & metric_value.isNotNull()
    )
    .select(metric_value.alias("metric_value"), F.col("window"))
    .orderBy(F.col("window").desc())
    .limit(num_evaluation_windows)
    .collect()
)

# Violation: at least num_violation_windows of those windows are below the threshold and the
# most recent window is also below it (so a recovered dip does not trigger retraining).
# Fraud quality is "higher is better"; for a drift/error metric flip < to >.
windows_in_violation = sum(1 for row in recent_metrics if row["metric_value"] < model_drift_threshold)
latest_in_violation = bool(recent_metrics) and recent_metrics[0]["metric_value"] < model_drift_threshold
is_model_drift_violated = windows_in_violation >= num_violation_windows and latest_in_violation
print(f"is_model_drift_violated = {is_model_drift_violated}")

# Publish the decision for the downstream condition_task in the retraining job.
dbutils.jobs.taskValues.set("is_model_drift_violated", is_model_drift_violated)
