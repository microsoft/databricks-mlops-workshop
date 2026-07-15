# Monitoring

| File | Description |
|---|---|
| `notebooks/monitoring.py` | Refresh the batch and online Lakehouse Monitors and inspect the profile/drift metric tables they produce. |
| `notebooks/process_serving_logs.py` | Unpack the serving endpoint's raw inference-table payloads into typed rows so the online monitor can profile them. |
| `notebooks/model_drift_check.py` | Retraining-trigger task: run the drift SQL over the monitor's profile metrics to decide whether model quality has degraded across recent windows. |
| `notebooks/feature_drift_check.py` | Retraining-trigger task: threshold the monitor's `_drift_metrics` to decide whether sustained feature (data) drift on any watched column should trigger retraining. |
| `notebooks/seed_drift.py` | Demo helper (attendee-runnable): sample already-scored predictions, inflate the amount-driven features, append them to your own `fraud_predictions` with today's timestamp, and refresh your batch monitor so feature drift is visible during the session. |
