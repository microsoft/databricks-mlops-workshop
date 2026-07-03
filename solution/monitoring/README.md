# Monitoring

| File | Description |
|---|---|
| `notebooks/monitoring.py` | Refresh the batch and online Lakehouse Monitors and inspect the profile/drift metric tables they produce. |
| `notebooks/process_serving_logs.py` | Unpack the serving endpoint's raw inference-table payloads into typed rows so the online monitor can profile them. |
| `notebooks/metric_violation_check.py` | Retraining-trigger task: run the violation SQL over the monitor's profile metrics to decide whether model quality has degraded across recent windows. |
