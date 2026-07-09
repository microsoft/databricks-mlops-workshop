# Databricks notebook source
# MAGIC %md
# MAGIC # Feature-drift violation check (instructor-provided)
# MAGIC
# MAGIC **Session:** Monitoring & Retraining
# MAGIC
# MAGIC **What this notebook is:** an automated job task (like `metric_violation_check.py`), not
# MAGIC a lab you open. It runs headless inside the scheduled `retraining_job` and emits a
# MAGIC true/false. Contrast `monitoring.py`, which is the human-facing dashboard that just
# MAGIC *displays* the same drift numbers for a person to read.
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
# MAGIC - `drift_metric`: the drift column to threshold (default `population_stability_index`;
# MAGIC   Lakehouse Monitoring populates it for numeric features, where `js_distance` is null).
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
# MAGIC column, per time `window`, per drift comparison. For each monitored feature we look at
# MAGIC the whole-population rows and ask: of the last `num_evaluation_windows` windows, did at
# MAGIC least `num_violation_windows` have `drift_metric` **above** `drift_violation_threshold`,
# MAGIC and is the most recent window also above it?
# MAGIC
# MAGIC The row filters mean:
# MAGIC - `column_name` is one of the monitored features (not `:table`),
# MAGIC - `slice_key IS NULL` -> the whole population, not a single data slice,
# MAGIC - `drift_type = "CONSECUTIVE"` -> window-over-window drift (we have no baseline table; a
# MAGIC   baseline-vs-training comparison would use `drift_type = "BASELINE"`).
# MAGIC
# MAGIC The table can carry more than one row per window (one per `model_version` plus a `*`
# MAGIC aggregate), so we take the max metric per window to get one value each.
# MAGIC
# MAGIC `population_stability_index` (PSI) is populated for numeric features (Lakehouse
# MAGIC Monitoring leaves `js_distance` null for numeric columns and only fills it for
# MAGIC categoricals), so PSI is the drift knob that actually fires on features like `amount`.
# MAGIC A common reading is >0.1 moderate, >0.25 significant. If **any** monitored feature is in
# MAGIC violation, we flag `is_drift_violated`.

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
