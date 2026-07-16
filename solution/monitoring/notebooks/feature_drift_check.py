# Databricks notebook source
# MAGIC %md
# MAGIC # Feature-drift violation check (instructor-provided)
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC Automated task in the scheduled `retraining_job`. It reads the monitor's
# MAGIC `..._drift_metrics` table, decides whether one or more features have drifted enough for
# MAGIC long enough to act on, and publishes the boolean `is_drift_violated` as a task value. A
# MAGIC `condition_task` reads that value and, with the quality check, decides whether to retrain.
# MAGIC
# MAGIC Related notebooks:
# MAGIC
# MAGIC - `model_drift_check.py`: the same pattern for model quality (`..._profile_metrics`).
# MAGIC - `monitoring.py`: the interactive dashboard that displays these metrics.
# MAGIC
# MAGIC Quality and drift live in separate tables: `..._profile_metrics` holds quality metrics
# MAGIC as structs keyed by `:table` / `model_version`, while `..._drift_metrics` holds one row
# MAGIC per feature column.
# MAGIC
# MAGIC Parameters:
# MAGIC - `table_name_under_monitor`: the inference table the monitor profiles.
# MAGIC - `features_to_monitor`: comma-separated feature columns to watch (e.g. `amount,credit_score`).
# MAGIC - `drift_metric`: the drift column to threshold (default `population_stability_index`;
# MAGIC   Lakehouse Monitoring populates it for numeric features, where `js_distance` is null).
# MAGIC - `drift_violation_threshold`: drift is a "higher is worse" metric, so a value **above**
# MAGIC   this counts as a violation.
# MAGIC - `num_evaluation_windows` / `num_violation_windows`: how many recent windows to look at,
# MAGIC   and how many must be in violation before action is taken.
# MAGIC
# MAGIC Note on acting on feature drift: retraining only helps once fresh labels arrive, so in
# MAGIC production feature drift typically raises an alert first, and the label-based quality
# MAGIC check gates retraining. Here drift is wired into the same trigger (OR-ed with the
# MAGIC quality check) to demonstrate the mechanism end to end.
# MAGIC
# MAGIC Docs: [Monitor metric tables](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/monitor-output)

# COMMAND ----------

dbutils.widgets.text(
    "table_name_under_monitor", "", label="Inference table under monitor (three-level name)"
)
dbutils.widgets.text(
    "features_to_monitor",
    "amount,credit_score,credit_limit,yearly_income,current_age,num_cards_issued,is_night,is_online,is_high_risk_mcc,amount_to_income_ratio,amount_to_credit_limit_ratio",
    label="Comma-separated feature columns to watch",
)
dbutils.widgets.text(
    "drift_metric", "population_stability_index", label="Drift metric column to threshold"
)
dbutils.widgets.text(
    "drift_violation_threshold", "0.2", label="Flag drift when the metric rises above this"
)
dbutils.widgets.text("num_evaluation_windows", "5", label="Number of recent windows to check")
dbutils.widgets.text("num_violation_windows", "1", label="Windows that must violate to trigger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The violation query
# MAGIC
# MAGIC Lakehouse Monitoring writes a `<table>_drift_metrics` Delta table with one row per
# MAGIC column, per time `window`, per drift comparison. For each monitored feature, it checks
# MAGIC the whole-population rows and ask: of the last `num_evaluation_windows` windows, did at
# MAGIC least `num_violation_windows` have `drift_metric` **above** `drift_violation_threshold`,
# MAGIC and is the most recent window also above it?
# MAGIC
# MAGIC The row filters mean:
# MAGIC - `column_name` is one of the monitored features (not `:table`),
# MAGIC - `slice_key IS NULL` -> the whole population, not a single data slice,
# MAGIC - `drift_type = "CONSECUTIVE"` -> window-over-window drift (there is no baseline table; a
# MAGIC   baseline-vs-training comparison would use `drift_type = "BASELINE"`).
# MAGIC
# MAGIC The table can carry more than one row per window (one per `model_version` plus a `*`
# MAGIC aggregate), so the max metric per window is used to get one value each.
# MAGIC
# MAGIC `population_stability_index` (PSI) measures how far a feature's distribution has moved
# MAGIC between the previous window and the current one: both windows are cut into the same
# MAGIC bins, and PSI sums `(curr% - prev%) * ln(curr% / prev%)` over the bins. It is 0 when the
# MAGIC two distributions match and grows as they pull apart, so "higher is worse". A common
# MAGIC reading is >0.1 a moderate shift and >0.25 a significant shift.
# MAGIC
# MAGIC PSI is thresholded (not `js_distance`) because Lakehouse Monitoring leaves `js_distance`
# MAGIC null for numeric columns and only fills it for categoricals, so PSI is the drift knob
# MAGIC that fires on numeric features like `amount`. If **any** monitored feature is
# MAGIC in violation, `is_drift_violated` is set.

# COMMAND ----------

from pyspark.sql import functions as F

table_name_under_monitor = dbutils.widgets.get("table_name_under_monitor")
features_to_monitor = [c.strip() for c in dbutils.widgets.get("features_to_monitor").split(",") if c.strip()]
drift_metric = dbutils.widgets.get("drift_metric")
drift_violation_threshold = float(dbutils.widgets.get("drift_violation_threshold"))
num_evaluation_windows = int(dbutils.widgets.get("num_evaluation_windows"))
num_violation_windows = int(dbutils.widgets.get("num_violation_windows"))

drift_metrics_table = f"{table_name_under_monitor}_drift_metrics"

# If the monitor hasn't produced drift metrics yet (first run, or never refreshed), there is
# nothing to evaluate, so report "not violated" and the job does not retrain.
if not spark.catalog.tableExists(drift_metrics_table):
    print(f"{drift_metrics_table} does not exist yet; treating as not violated.")
    dbutils.jobs.taskValues.set("is_drift_violated", False)
    dbutils.notebook.exit("no_metrics")

# COMMAND ----------

metric_value = F.col(drift_metric)
violated_features = []

for feature in features_to_monitor:
    # The most recent num_evaluation_windows drift values for this feature, newest first.
    recent = (
        spark.read.table(drift_metrics_table)
        .where(
            (F.col("column_name") == feature)
            & F.col("slice_key").isNull()
            & (F.col("drift_type") == "CONSECUTIVE")
            & metric_value.isNotNull()
        )
        .groupBy(F.col("window"))
        .agg(F.max(metric_value).alias("metric_value"))
        .orderBy(F.col("window").desc())
        .limit(num_evaluation_windows)
        .collect()
    )
    if not recent:
        continue
    # Violation: at least num_violation_windows above the threshold AND the most recent window
    # is also above it (so a spike that has already settled is ignored). Drift is "higher is
    # worse", so the comparison uses >.
    windows_in_violation = sum(1 for row in recent if row["metric_value"] > drift_violation_threshold)
    latest_in_violation = recent[0]["metric_value"] > drift_violation_threshold
    if windows_in_violation >= num_violation_windows and latest_in_violation:
        violated_features.append((feature, round(recent[0]["metric_value"], 4)))

is_drift_violated = len(violated_features) > 0
print(f"features in drift violation: {violated_features}")
print(f"is_drift_violated = {is_drift_violated}")

# Publish the decision for the downstream condition_task in the retraining job.
dbutils.jobs.taskValues.set("is_drift_violated", is_drift_violated)
