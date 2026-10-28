# Resources

Declarative Automation Bundle (DAB, formerly Databricks Asset Bundle) definitions, one file per job/resource. [`databricks.yml`](../databricks.yml) selects the includes; `databricks bundle deploy` reconciles enabled resources into the workspace.

| File | Description |
|---|---|
| [schemas-resource.yml](schemas-resource.yml) | ML schema: shared `fraud_ml` on clean targets; `dev_<user>_fraud` on personal. |
| [ml-artifacts-resource.yml](ml-artifacts-resource.yml) | Fraud project's MLflow experiment. |
| [data-ingestion-workflow.yml](data-ingestion-workflow.yml) | Instructor-only ingestion: landing → Bronze → Silver → Gold. |
| [feature-engineering-workflow.yml](feature-engineering-workflow.yml) | Card/client entity feature tables and on-demand feature functions. |
| [model-training-workflow.yml](model-training-workflow.yml) | Train and register a model, triggering its connected deployment job. |
| [deployment-job-workflow.yml](deployment-job-workflow.yml) | MLflow 3: evaluate → approve → deploy each new model version. |
| [batch-inference-workflow.yml](batch-inference-workflow.yml) | Score new data with the registered model and write predictions. |
| [serving-log-processing-workflow.yml](serving-log-processing-workflow.yml) | Unpack serving inference logs for the online monitor. |
| [monitoring-resource.yml](monitoring-resource.yml) | Batch/online Lakehouse Monitors and metric-triggered retraining. |
