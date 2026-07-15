# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job: Approval_Check task
# MAGIC
# MAGIC **Session:** Model Promotion Lifecycle
# MAGIC
# MAGIC The middle task of the deployment job. It decides whether the freshly-evaluated
# MAGIC version may proceed to **Deployment**, using a Unity Catalog tag on the model version
# MAGIC as the signal (the mechanism MLflow 3 deployment jobs use):
# MAGIC
# MAGIC - **dev** (`auto_approve=true`): the task sets the approval tag itself and passes, so
# MAGIC   the whole loop runs unattended.
# MAGIC - **staging / prod** (`auto_approve=false`): the task fails until a human reviews the
# MAGIC   evaluation metrics on the model-version page and clicks **Approve** (which sets the
# MAGIC   tag `<task-name>=Approved` and repairs the run). The task then passes and the run
# MAGIC   resumes into Deployment.
# MAGIC
# MAGIC The task key must start with `approval` for the UI Approve button to wire up, and the
# MAGIC tag key is the task name (passed in as `approval_tag_name`).

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.0" --upgrade

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# The deployment job injects the FULL three-level model name (catalog.schema.model) into
# `model_name`, plus the `model_version` it is gating. Both are empty when this notebook is
# run interactively, so the fallback below rebuilds them.
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("catalog_name", "dev")
dbutils.widgets.text("ml_schema", "")
dbutils.widgets.text("approval_tag_name", "")
dbutils.widgets.text("auto_approve", "false")

uc_model_name = dbutils.widgets.get("model_name").strip()
model_version = dbutils.widgets.get("model_version").strip()
catalog_name = dbutils.widgets.get("catalog_name")
ml_schema = dbutils.widgets.get("ml_schema")
# The deployment job passes the task name here (e.g. "Approval_Check") as the tag key.
tag_name = dbutils.widgets.get("approval_tag_name") or "Approval_Check"
auto_approve = dbutils.widgets.get("auto_approve").strip().lower() == "true"

# Fail fast on a half-configured deployment job: model_name and model_version are injected
# together. Both present = job; both empty = interactive run (derived below). Exactly one
# present is a misconfiguration that would silently gate the wrong model or version.
if bool(uc_model_name) != bool(model_version):
    raise ValueError(
        "Only one of model_name / model_version was provided. Pass both (deployment job) "
        "or neither (interactive run)."
    )

from pyspark.sql import functions as F

# Interactive fallback: rebuild the personal schema and the full model name, and default to
# the latest version, so the notebook can be run by hand (outside the deployment job).
if not ml_schema:
    _user = spark.range(1).select(F.current_user()).first()[0]
    _short = "".join(c if c.isalnum() else "_" for c in _user.split("@")[0])
    ml_schema = f"dev_{_short}_fraud"
if not uc_model_name:
    uc_model_name = f"{catalog_name}.{ml_schema}.fraud_detection"
if not model_version:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    _versions = [
        int(mv.version)
        for mv in MlflowClient().search_model_versions(f"name='{uc_model_name}'")
    ]
    if not _versions:
        raise ValueError(
            f"No model versions found for {uc_model_name!r}. Provide model_version explicitly."
        )
    model_version = str(max(_versions))
print(
    f"Approval check for {uc_model_name} v{model_version} | tag='{tag_name}' | auto_approve={auto_approve}"
)

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

# In dev, auto-approve: set the tag so this run (and the UI) reflect an approved state, then
# pass. In staging/prod, a human must have set the tag to "Approved".
if auto_approve:
    client.set_model_version_tag(uc_model_name, model_version, tag_name, "Approved")
    print(f"Auto-approved (dev): set {tag_name}=Approved on v{model_version}.")
else:
    mv = client.get_model_version(uc_model_name, model_version)
    status = (mv.tags or {}).get(tag_name)
    print(f"Current approval tag {tag_name}={status!r}")
    if status != "Approved":
        raise ValueError(
            f"Version {model_version} is not approved. An approver must review the "
            f"evaluation metrics on the model-version page and click Approve "
            f"(sets {tag_name}=Approved), then repair this run."
        )
    print(f"Approved: {tag_name}=Approved. Proceeding to deployment.")
