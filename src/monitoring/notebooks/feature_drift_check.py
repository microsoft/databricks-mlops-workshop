# Databricks notebook source
# MAGIC %md
# MAGIC # Feature-drift violation check (instructor-provided)
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC A sibling of `metric_violation_check.py`, but for **feature (data) drift** instead of
# MAGIC model quality. It reads the monitor's `..._drift_metrics` table and decides whether the
# MAGIC input distribution of one or more features has shifted enough, for long enough, to act
# MAGIC on. It publishes the boolean `is_drift_violated` as a job task value; a `condition_task`
# MAGIC in the retraining job reads it and (together with the quality check) decides whether to
# MAGIC retrain.
# MAGIC
# MAGIC Why a separate notebook from the quality check: quality metrics live in
# MAGIC `..._profile_metrics` as structs keyed by `:table`/`model_version`, while drift lives in
# MAGIC `..._drift_metrics` with one row **per feature column**. Different table, different shape.
# MAGIC
# MAGIC Parameters:
# MAGIC - `table_name_under_monitor`: the inference table the monitor profiles.
# MAGIC - `features_to_monitor`: comma-separated feature columns to watch (e.g. `amount,credit_score`).
# MAGIC - `drift_metric`: the drift column to threshold (default `js_distance`, in [0, 1]).
# MAGIC - `drift_violation_threshold`: drift is a "higher is worse" metric, so a value **above**
# MAGIC   this counts as a violation.
# MAGIC - `num_evaluation_windows` / `num_violation_windows`: how many recent windows to look at,
# MAGIC   and how many must be in violation before we act.
# MAGIC
# MAGIC A note on acting on feature drift: retraining only helps once **fresh labels** arrive, so
# MAGIC in production feature drift usually raises an **alert** first, and the label-based quality
# MAGIC check is what actually gates retraining. Here we wire drift into the same retraining
# MAGIC trigger (OR-ed with the quality check) to show the mechanism end-to-end.
# MAGIC
# MAGIC Docs: [Monitor metric tables](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/monitor-output)

# COMMAND ----------

dbutils.widgets.text(
    "table_name_under_monitor", "", label="Inference table under monitor (three-level name)"
)
dbutils.widgets.text(
    "features_to_monitor",
    "amount,credit_score,credit_limit,yearly_income,current_age",
    label="Comma-separated feature columns to watch",
)
dbutils.widgets.text("drift_metric", "js_distance", label="Drift metric column to threshold")
dbutils.widgets.text(
    "drift_violation_threshold", "0.2", label="Flag drift when the metric rises above this"
)
dbutils.widgets.text("num_evaluation_windows", "5", label="Number of recent windows to check")
dbutils.widgets.text("num_violation_windows", "2", label="Windows that must violate to trigger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The violation query
# MAGIC
# MAGIC Lakehouse Monitoring writes a `<table>_drift_metrics` Delta table with one row per
# MAGIC column, per time `window`, per drift comparison. For each monitored feature we look at
# MAGIC the whole-population rows and ask: of the last `num_evaluation_windows` windows, did at
# MAGIC least `num_violation_windows` have `drift_metric` **above** `drift_violation_threshold`,
# MAGIC and is the most recent window also above it?
# MAGIC
# MAGIC The row filters mean:
# MAGIC - `column_name` is one of the monitored features (not `:table`),
# MAGIC - `slice_key IS NULL` -> the whole population, not a single data slice,
# MAGIC - `drift_type = "CONSECUTIVE"` -> window-over-window drift (we have no baseline table; a
# MAGIC   baseline-vs-training comparison would use `drift_type = "BASELINE"`),
# MAGIC - `log_type = "INPUT"` -> the primary table, not baseline-table metrics.
# MAGIC
# MAGIC `js_distance` (Jensen-Shannon) is in [0, 1] and works for both numeric and categorical
# MAGIC columns, so it is a good single drift knob. If **any** monitored feature is in violation,
# MAGIC we flag `is_drift_violated`.

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
# nothing to evaluate, so report "not violated" and the job simply doesn't retrain.
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
            & (F.col("log_type") == "INPUT")
            & metric_value.isNotNull()
        )
        .select(metric_value.alias("metric_value"), F.col("window"))
        .orderBy(F.col("window").desc())
        .limit(num_evaluation_windows)
        .collect()
    )
    if not recent:
        continue
    # Violation: at least num_violation_windows above the threshold AND the most recent window
    # is also above it (so we don't act on a spike that has already settled). Drift is "higher
    # is worse", so we compare with >.
    windows_in_violation = sum(1 for row in recent if row["metric_value"] > drift_violation_threshold)
    latest_in_violation = recent[0]["metric_value"] > drift_violation_threshold
    if windows_in_violation >= num_violation_windows and latest_in_violation:
        violated_features.append((feature, round(recent[0]["metric_value"], 4)))

is_drift_violated = len(violated_features) > 0
print(f"features in drift violation: {violated_features}")
print(f"is_drift_violated = {is_drift_violated}")

# Publish the decision for the downstream condition_task in the retraining job.
dbutils.jobs.taskValues.set("is_drift_violated", is_drift_violated)
