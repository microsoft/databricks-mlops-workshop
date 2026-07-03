# Data ingestion

Instructor-only setup notebooks (run once before the workshop; no TODO gaps, so the generator copies them verbatim into `src/`).

| File | Description |
|---|---|
| `notebooks/load_landing_to_bronze.py` | Ingest the raw fraud source files from the landing volume into Bronze Delta tables with schemas and audit columns. |
| `notebooks/load_bronze_to_silver.py` | Clean, type and conform the Bronze tables into Silver, then build the denormalized Gold consumption table. |
