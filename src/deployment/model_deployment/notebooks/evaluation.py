# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job: Evaluation task
# MAGIC
# MAGIC **Session:** Model Promotion Lifecycle
# MAGIC
# MAGIC This is the first task of the **MLflow 3 deployment job**. A deployment job is a
# MAGIC Lakeflow Job that is **connected to a Unity Catalog registered model** and
# MAGIC **auto-triggers on every new model version** (Databricks injects the job-level
# MAGIC parameters `model_name` and `model_version`). The three tasks are:
# MAGIC
# MAGIC 1. **Evaluation** (this notebook): score the new version and record its metrics on
# MAGIC    the model-version page so an approver can decide.
# MAGIC 2. **Approval_Check**: gate, auto-approved in dev, human-approved in staging/prod.
# MAGIC 3. **Deployment**: promote the version to `@champion` and serve it.
# MAGIC
# MAGIC The version being evaluated is the one that triggered the job, so there is no
# MAGIC champion/challenger bookkeeping here. The version is evaluated and gated against a
# MAGIC metric floor (and the current champion, if any).
# MAGIC See [MLflow deployment jobs](https://learn.microsoft.com/azure/databricks/mlflow/deployment-job).

# COMMAND ----------

# The deployment job injects model_name + model_version as JOB-level parameters.
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("metric", "roc_auc")
dbutils.widgets.text("baseline", "0.65")

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
metric = dbutils.widgets.get("metric")
baseline = float(dbutils.widgets.get("baseline"))

assert model_name and model_version, (
    "model_name and model_version are injected by the deployment job."
)
print(f"Evaluating {model_name} version {model_version} on '{metric}' (floor {baseline}).")

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
fe = FeatureEngineeringClient()

# Independent evaluation: score the REGISTERED artifact on a fresh, labelled holdout rather
# than trusting the metrics training self-reported. fe.score_batch replays the exact feature
# lookups + on-demand functions recorded at training, and the served model returns a fraud
# probability, so a real ROC AUC can be computed. In production this holdout would be a
# curated, leakage-controlled evaluation table; here it is sampled from the shared gold source.
catalog_name = model_name.split(".")[0]
gold_table = f"{catalog_name}.fraud_gold.transactions_enriched"

eval_spine = (
    spark.read.table(gold_table)
    .select(
        "card_id",
        "client_id",
        F.col("amount").cast("double").alias("amount"),
        "transaction_hour",
        "mcc",
        "use_chip",
        F.col("is_fraud").cast("int").alias("is_fraud"),
    )
    .where(F.col("is_fraud").isNotNull())
    .orderBy(F.rand(7))  # different seed than training, so mostly-unseen rows
    .limit(50000)
)
# Note: no .cache(), because serverless compute rejects PERSIST. The seeded rand(7) sample is
# deterministic over the immutable gold table, so candidate and champion score the same rows.


def score_holdout(version):
    """Score a model version on the holdout; return a pandas DF of is_fraud + fraud score."""
    scored = fe.score_batch(model_uri=f"models:/{model_name}/{version}", df=eval_spine)
    return scored.select("is_fraud", F.col("prediction").cast("double").alias("score")).toPandas()


candidate_pdf = score_holdout(model_version)
candidate_score = float(roc_auc_score(candidate_pdf["is_fraud"], candidate_pdf["score"]))

# Fair comparison: re-score the current champion on the SAME holdout. If there is no champion
# yet (first model), gate against the metric floor instead.
try:
    champion = client.get_model_version_by_alias(model_name, "champion")
    champ_pdf = score_holdout(champion.version)
    bar = float(roc_auc_score(champ_pdf["is_fraud"], champ_pdf["score"]))
    bar_label = f"champion (v{champion.version})"
except Exception:
    bar, bar_label = baseline, "baseline"

print(f"candidate {metric}={candidate_score:.4f}  vs  {bar_label}={bar:.4f}")

# Surface the decision inputs on the model version so the approver sees them in the UI.
client.set_model_version_tag(model_name, model_version, "eval_metric", metric)
client.set_model_version_tag(model_name, model_version, "eval_score", f"{candidate_score:.4f}")
client.set_model_version_tag(model_name, model_version, "eval_bar", f"{bar:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix for the approver
# MAGIC
# MAGIC Threshold the candidate's fraud score at 0.5 and show the confusion matrix on the
# MAGIC holdout, logged to an MLflow run so it sits alongside the metrics the approver reviews
# MAGIC before approving.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Naive 0.5 cut-off, on purpose: it shows the low-precision operating point the training
# notebook discusses. The gate itself uses ROC AUC, which is threshold-independent.
preds = (candidate_pdf["score"] >= 0.5).astype(int)
cm = confusion_matrix(candidate_pdf["is_fraud"], preds, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"]).plot(
    cmap="Blues", ax=ax, colorbar=False
)
ax.set_title(f"{model_name.split('.')[-1]} v{model_version} (holdout, threshold 0.5)")
fig.tight_layout()
with mlflow.start_run(run_name=f"evaluation_v{model_version}") as ev_run:
    mlflow.log_figure(fig, "confusion_matrix.png")
    mlflow.log_metric("holdout_roc_auc", candidate_score)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ROC and precision-recall curves
# MAGIC
# MAGIC The confusion matrix is a single operating point; these curves show the full threshold
# MAGIC trade-off:
# MAGIC
# MAGIC - the ROC curve's area is the gate metric (threshold-independent);
# MAGIC - on rare-fraud data, the precision-recall curve (and its average precision) shows how
# MAGIC   much precision is given up as recall increases.
# MAGIC
# MAGIC Both are logged to the same evaluation run, alongside the confusion matrix.

# COMMAND ----------

from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
)

avg_precision = float(average_precision_score(candidate_pdf["is_fraud"], candidate_pdf["score"]))
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4))
RocCurveDisplay.from_predictions(
    candidate_pdf["is_fraud"], candidate_pdf["score"], name=f"v{model_version}", ax=ax_roc
)
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)  # random-guess baseline
ax_roc.set_title(f"ROC (AUC={candidate_score:.3f})")
PrecisionRecallDisplay.from_predictions(
    candidate_pdf["is_fraud"], candidate_pdf["score"], name=f"v{model_version}", ax=ax_pr
)
ax_pr.set_title(f"Precision-Recall (AP={avg_precision:.3f})")
fig.tight_layout()
with mlflow.start_run(run_id=ev_run.info.run_id):
    mlflow.log_figure(fig, "roc_pr_curves.png")
    mlflow.log_metric("holdout_average_precision", avg_precision)
plt.show()

# COMMAND ----------

# Gate: fail the task (and therefore the deployment) if the new version does not clear the
# bar. A failed evaluation stops the version from ever reaching Approval/Deployment.
if candidate_score < bar:
    raise ValueError(
        f"Evaluation failed: {metric}={candidate_score:.4f} < {bar_label}={bar:.4f}. "
        "This version will not be approved or deployed."
    )

print(f"Evaluation passed: {metric}={candidate_score:.4f} >= {bar:.4f}.")
