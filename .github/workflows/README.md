# Workflows

GitHub Actions pipelines for the workshop repo.

| File | Description |
|---|---|
| `ci.yml` | Continuous integration: validate the bundle and check `src/` is in sync with `solution/` on every PR. |
| `cd.yml` | Continuous deployment: deploy the Declarative Automation Bundle to dev (push to `dev`), staging (`main`) and prod (release). |
