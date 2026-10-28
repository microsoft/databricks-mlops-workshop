# Data ingestion

Instructor-only setup: run once before the workshop. No TODO gaps; copied verbatim into `src/`.

| File | Description |
|---|---|
| [load_landing_to_bronze.py](notebooks/load_landing_to_bronze.py) | Load raw fraud files from the landing volume into Bronze Delta tables with schemas and audit columns. |
| [load_bronze_to_silver.py](notebooks/load_bronze_to_silver.py) | Clean, type, and conform Bronze into Silver; build the denormalised Gold consumption table. |
