# Deployment

| Folder | Description |
|---|---|
| `model_deployment/` | The MLflow deployment job (evaluate → approve → promote to `@champion` and update the serving endpoint) plus a manual endpoint-test lab. |
| `batch_inference/` | Batch scoring with the `@champion` model (the scheduled batch prediction path). |
