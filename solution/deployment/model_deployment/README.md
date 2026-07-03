# Model deployment

| File | Description |
|---|---|
| `notebooks/evaluation.py` | Deployment-job task 1: score the new model version and record its metrics on the model-version page. |
| `notebooks/approval.py` | Deployment-job task 2: check the approval tag to decide whether the version may be promoted (auto in dev, human in staging/prod). |
| `notebooks/deployment.py` | Deployment-job task 3: promote the approved version to `@champion`, publish features, and update the serving endpoint. |
| `notebooks/query_endpoint.py` | Manual lab: replay real transactions through the live serving endpoint to test it. |
