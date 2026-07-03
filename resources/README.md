# Resources

Declarative Automation Bundle (DAB) definitions (formerly Databricks Asset Bundles);
one file per job/resource. `databricks.yml` includes these; `databricks bundle deploy`
reconciles them into the workspace.

| File | Description |
|---|---|
| `schemas-resource.yml` | Per-user ML workspace schema (base name `fraud`; dev mode → `dev_<user>_fraud`). |
| `ml-artifacts-resource.yml` | The MLflow experiment for the fraud project. |
| `data-ingestion-workflow.yml` | Instructor-only job: land raw files into Bronze, then build Silver. |
| `feature-engineering-workflow.yml` | Build the card/client entity feature tables and on-demand feature functions. |
| `model-training-workflow.yml` | Train and register the model (auto-triggers the connected deployment job). |
| `deployment-job-workflow.yml` | MLflow 3 deployment job: evaluate → approve → deploy each new model version. |
| `batch-inference-workflow.yml` | Score new data with the registered model and write predictions. |
| `serving-log-processing-workflow.yml` | Unpack the serving endpoint's inference table for the online monitor. |
| `monitoring-resource.yml` | Lakehouse Monitors (batch + online) and the metric-triggered retraining job. |
