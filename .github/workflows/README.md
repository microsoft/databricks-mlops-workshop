# Workflows

GitHub Actions pipelines for the workshop. **Automatic deployments are intentionally disabled in this public repo**; enable them when setting up a real environment.

| File | Description |
|---|---|
| [ci.yml](ci.yml) | Checks `src/` matches generated `solution/` output on PRs and pushes to `main`. Linting is disabled; PR bundle validation is not configured. |
| [cd.yml](cd.yml) | Validates and deploys the bundle via manual dispatch. Commented triggers support dev (`dev`), staging (`main`), and prod (`release/**` or `vX.Y.Z` tags). |

For real deployments, configure per-environment service principals, OIDC, GitHub Environments, and a production reviewer; restore lint/PR validation and branch protection, then enable CD's push/tag triggers. See the [CI/CD setup](../../README.md#cicd-promoting-the-bundle-across-environments) for exact settings.
