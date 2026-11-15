# Publishing PyMC Bot on PyPI

The distribution name is `pymc-bot` (import name `pymc_bot`); the version comes from
`pyproject.toml`. PyPI versions cannot be overwritten or reused, including after deletion.
This workflow does not publish automatically on pushes or releases. A successful build or a
GitHub push is not proof of a PyPI upload.

## One-time setup

1. Create a PyPI account, enable 2FA, and ensure the name `pymc-bot` is available and belongs to
   you before uploading. PyPI may reject a name or a token even when its project page does not exist.
2. Create a PyPI API token. For the first upload, an account-scoped token may be needed because
   the project does not exist yet. After the first upload, replace it with a project-scoped token
   for `pymc-bot` and revoke the broader token.
3. In the GitHub repository, open **Settings → Secrets and variables → Actions → New repository
   secret** and name it `PYPI_API_TOKEN`. Paste the token there, not in code, logs, or chat.
   Protect the repository and restrict who can run publishing workflows. GitHub may require
   additional permissions or environment approval depending on repository settings.
4. The workflow must be present on the repository's default branch before GitHub exposes its
   manual trigger. Review and merge the changes through your normal process first.

## Publish

1. Review the package version in `pyproject.toml` and set a new, unused version for each release.
   The initial configured version is `0.1.0`, independent of the Fabric mod version.
2. In **Actions → Publish Python package → Run workflow**, select the intended ref. Leave
   **Upload to PyPI** off for a validation-only run. The workflow builds the sdist and wheel,
   checks metadata, confirms runtime assets are included and notebooks are excluded, tests the
   installed wheel outside the checkout, and saves the archives as a workflow artifact.
3. After reviewing the ref, CI and artifacts, run it again with **Upload to PyPI** enabled and
   type `pymc-bot` in the confirmation field. The workflow uses `PYPI_API_TOKEN` and uploads both
   distributions. It does not bypass an existing version or retry with a different version.
4. Verify the workflow and the project page on PyPI. An upload can partially succeed; inspect
   PyPI before retrying, and bump the version if necessary.

After a successful upload, users can run `python -m pip install pymc-bot`, `pymc-bot setup`, and
`pymc-bot serve`. The Mineflayer bridge requires Node.js and separately installed JavaScript
packages. PyPI contains the Python backend and bundled runtime assets, not the Fabric mod jar,
model weights, or training notebooks. `pymc-bot setup` downloads runtime components as needed;
review its output and use `pymc-bot setup --check` for diagnostics.
