# Source

Each stage has its notebook entrypoints in `<stage>/notebooks/`.

| Folder | Description |
|---|---|
| `exploration/` | Optional read-only EDA to understand the data before the labs. |
| `data_ingestion/` | Instructor-only setup: land raw files into Bronze and build Silver/Gold. |
| `feature_engineering/` | Build the card/client entity feature tables and on-demand feature functions. |
| `training/` | Train, track and register the fraud detection model. |
| `deployment/` | Promote and serve the champion model (deployment job + endpoint) and run batch inference. |
| `monitoring/` | Refresh the Lakehouse Monitors, process serving logs, and check for model drift that triggers retraining. |
