# Monitoring

| File | Description |
|---|---|
| [monitoring.py](notebooks/monitoring.py) | Refresh batch/online Lakehouse Monitors and inspect their profile/drift metric tables. |
| [process_serving_logs.py](notebooks/process_serving_logs.py) | Unpack raw serving inference payloads into typed rows for online profiling. |
| [model_drift_check.py](notebooks/model_drift_check.py) | Run drift SQL over profile metrics; trigger retraining if model quality degrades across recent windows. |
| [feature_drift_check.py](notebooks/feature_drift_check.py) | Threshold `_drift_metrics`; trigger retraining on sustained data drift in any watched column. |
| [seed_drift.py](notebooks/seed_drift.py) | Attendee demo: sample scored predictions, inflate amount-driven features, append to your own `fraud_predictions` with today's timestamp, and refresh the batch monitor to show feature drift. |
