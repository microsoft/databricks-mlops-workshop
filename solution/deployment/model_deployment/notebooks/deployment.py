# Databricks notebook source
# MAGIC %md
# MAGIC # Deployment job: Deployment task
# MAGIC
# MAGIC **Session:** Model Serving & Consumption
# MAGIC
# MAGIC The final task of the deployment job. It runs only after Evaluation passed and the
# MAGIC Approval_Check cleared (auto in dev, human in staging/prod). It:
# MAGIC
# MAGIC 1. **Promotes** the approved version to `@champion` (and drops `@challenger`), so batch
# MAGIC    consumers that load `models:/...@champion` pick it up on their next run.
# MAGIC 2. **Publishes** the card/client entity features to the online store (idempotent).
# MAGIC 3. **Serves** the approved version on the Mosaic AI endpoint (AI Gateway inference
# MAGIC    table + usage tracking), creating the endpoint on first run or updating it later.
# MAGIC    The serving API needs a concrete version number, which the deployment job passes in.
# MAGIC
# MAGIC Docs: [Model Serving](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/)

# COMMAND ----------

# The deployment job injects the FULL three-level model name (catalog.schema.model) into
# `model_name`, plus the `model_version` that triggered it. Both are empty when this notebook
# is run interactively, so the fallback below rebuilds them.
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("catalog_name", "adoption_workshop")
dbutils.widgets.text("ml_schema", "")

uc_model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
catalog_name = dbutils.widgets.get("catalog_name")
ml_schema = dbutils.widgets.get("ml_schema")

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
    model_version = str(
        max(int(mv.version) for mv in MlflowClient().search_model_versions(f"name='{uc_model_name}'"))
    )

endpoint_name = f"{ml_schema}_fraud"
online_store_name = "fraud-workshop-online"
card_feature_table = f"{catalog_name}.{ml_schema}.card_features"
client_feature_table = f"{catalog_name}.{ml_schema}.client_features"

print(f"Deploying {uc_model_name} v{model_version} -> endpoint {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Promote the approved version to `@champion`
# MAGIC Point the `champion` alias at the version the job is deploying and remove the
# MAGIC `challenger` alias. Promotion is an alias reassignment: instant and reversible.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

client.set_registered_model_alias(uc_model_name, "champion", model_version)
try:
    client.delete_registered_model_alias(uc_model_name, "challenger")
except Exception:
    pass  # there may be no challenger alias (e.g. first deployment)
print(f"@champion -> version {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Publish entity features to the online store
# MAGIC The served model looks card/client features up by key at request time, so the offline
# MAGIC feature tables must be published to the low-latency online store (idempotent).

# COMMAND ----------

import time

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()


def wait_online_store(name, minutes=20):
    for _ in range(minutes * 3):
        s = fe.get_online_store(name=name)
        if s is not None and str(getattr(s, "state", "")).upper().endswith("AVAILABLE"):
            return s
        time.sleep(20)
    raise TimeoutError(f"online store {name} did not become AVAILABLE in time")


if fe.get_online_store(name=online_store_name) is None:
    print(f"Creating online store {online_store_name} (this can take several minutes)...")
    fe.create_online_store(name=online_store_name, capacity="CU_1")
store = wait_online_store(online_store_name)

for offline in (card_feature_table, client_feature_table):
    fe.publish_table(
        online_store=store,
        source_table_name=offline,
        online_table_name=f"{offline}_online",
    )
print("Published card_features and client_features to the online store.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Serve the approved version
# MAGIC Serve the concrete `model_version` with scale-to-zero and an AI Gateway inference
# MAGIC table + usage tracking. Create the endpoint on first run, update the served entity on
# MAGIC later runs.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

served_entities = [
    ServedEntityInput(
        entity_name=uc_model_name,
        entity_version=model_version,
        scale_to_zero_enabled=True,
        workload_size="Small",
    )
]
inference_table = AiGatewayInferenceTableConfig(
    catalog_name=catalog_name,
    schema_name=ml_schema,
    table_name_prefix="fraud_serving",
    enabled=True,
)
usage_tracking = AiGatewayUsageTrackingConfig(enabled=True)
ai_gateway = AiGatewayConfig(
    inference_table_config=inference_table,
    usage_tracking_config=usage_tracking,
)

existing = next((e for e in w.serving_endpoints.list() if e.name == endpoint_name), None)
if existing is None:
    print(f"Creating endpoint {endpoint_name} (this can take a few minutes)...")
    w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(name=endpoint_name, served_entities=served_entities),
        ai_gateway=ai_gateway,
    )
else:
    print(f"Updating endpoint {endpoint_name} to version {model_version}...")
    w.serving_endpoints.update_config_and_wait(name=endpoint_name, served_entities=served_entities)
    w.serving_endpoints.put_ai_gateway(
        name=endpoint_name,
        inference_table_config=inference_table,
        usage_tracking_config=usage_tracking,
    )
print(f"Endpoint {endpoint_name} now serving version {model_version}.")
