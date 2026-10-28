# Source

Each stage has its notebook entrypoints in `<stage>/notebooks/`.

| Folder | Description |
|---|---|
| [exploration/](exploration/) | Optional read-only EDA before the labs. |
| [data_ingestion/](data_ingestion/) | Instructor-only raw-file ingestion into Bronze, then Silver/Gold. |
| [feature_engineering/](feature_engineering/) | Build card/client entity tables and on-demand feature functions. |
| [training/](training/) | Train, track, and register the fraud model. |
| [deployment/](deployment/) | Promote and serve the champion via deployment job/endpoint; run batch inference. |
| [monitoring/](monitoring/) | Refresh Lakehouse Monitors, process serving logs, and trigger retraining on drift. |
